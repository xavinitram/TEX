"""LANG-L4 — the per-pixel scalar oracle.

**What this is.** An evaluator that runs a TEX program **one pixel at a time**, in plain
sequential Python, with **no masking of any kind** — because for a single pixel there is
nothing to mask. `break` is a Python `break`, `continue` a Python `continue`, `return` a
Python exception that unwinds one call. Every condition is a single boolean, so the only
control flow here is Python's own.

**Why it exists.** The interpreter is the oracle every other tier must match bit-exactly,
which means an error in the interpreter's masking rules is an error everywhere: codegen is
required to reproduce it. Comparing the two tiers cannot catch a defect they *share*
(`docs/masked-control-flow.md` §5, divergence site 6 — and §0's table is that failure
already shipped). So the masking rules need an answer computed a second way, by something
that does not mask.

**Independence, stated exactly, because a shared mistake is the thing this exists to
catch.** This module shares with `tex_runtime/interpreter.py`:

  * **torch's elementwise kernels.** Values are 0-dim fp32 tensors and `a + b` is
    `torch.Tensor.__add__`, so IEEE rounding matches the vectorised run exactly rather
    than drifting by fp64-vs-fp32 and producing false mismatches. torch is the substrate
    both sit on, not a TEX helper.
  * **`TEXStdlib.get_functions()`** for stdlib *leaf* calls (`sin`, `clamp`, `fetch`, …).
    These are pixel-local pure functions: a bug in one is a bug in a leaf, not in the
    masking rules, and re-deriving thirty of them bit-exactly here would manufacture false
    failures rather than find real ones.

It shares **nothing else** — no AST evaluator, no dispatch table, no assignment path, no
live mask, no loop driver, no call frame. Every rule `docs/masked-control-flow.md` §1
states (M1–M7) is re-derived here from Python's own semantics rather than expressed.

**Coverage is explicit, never silent.** A construct this module does not implement raises
`OracleUnsupported`, which a caller reports as a skip. An oracle that quietly guessed at a
node it did not understand would be worse than no oracle.
"""
from __future__ import annotations

import math
import torch

from TEX_Wrangle.tex_compiler import ast_nodes as A
from TEX_Wrangle.tex_compiler.types import CHANNEL_MAP
from TEX_Wrangle.tex_runtime.stdlib import TEXStdlib

F32 = torch.float32

_FOOTMAP = None


def _footprint_of(name):
    """The registry's ROI-1 footprint for a stdlib name, or `'point'`. Read from
    `tex_roi._footmap()` so "is this a gather" has one definition here too."""
    global _FOOTMAP
    if _FOOTMAP is None:
        from TEX_Wrangle import tex_roi
        _FOOTMAP = tex_roi._footmap()
    return _FOOTMAP.get(name, "point")


class OracleUnsupported(Exception):
    """This program uses a construct the scalar oracle does not implement."""


class _Return(Exception):
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


# The oracle's own cap, deliberately the same number the interpreter uses, so a
# never-terminating pixel is reported here too instead of hanging the sweep.
MAX_PASSES = 1024
MAX_DEPTH = 64


def _t(v) -> torch.Tensor:
    if isinstance(v, torch.Tensor):
        return v
    return torch.scalar_tensor(float(v), dtype=F32)


def _num(v) -> float:
    if isinstance(v, torch.Tensor):
        return float(v.reshape(-1)[0].item())
    return float(v)


def _truth(v) -> bool:
    """`docs/masked-control-flow.md` §5 divergence site 1, at one pixel: a condition is
    taken when it is strictly greater than 0.5 for a float, and non-zero for a bool.

    Written from the rule, not from the implementation. The vectorised side must derive
    its mask through ONE shared helper; this is the scalar reading of the same rule."""
    if isinstance(v, torch.Tensor):
        if v.dtype is torch.bool:
            return bool(v.reshape(-1)[0].item())
        return float(v.reshape(-1)[0].item()) > 0.5
    return float(v) > 0.5


class ScalarOracle:
    """Evaluates `program` at ONE pixel `(b, y, x)` of a `(B, H, W)` cook.

    `bindings` holds the WHOLE tensors (as the engine receives them); a plain `@A` read is
    indexed down to this pixel, while a gather (`@A[i, j]`, `@A(u, v)`) is handed the whole
    image exactly as the engine hands it, so the oracle's taps read what the cook's taps
    read.

    `scatter` is an optional dict the sweeper owns: `{name: tensor}` written in place by
    `@T[x, y] op= v`. M5 gates a scatter BY SOURCE, and one pixel is one source, so a
    scatter reaching this evaluator simply happens — the sweeper drives the pixels in
    row-major order, which is the order M5 names."""

    def __init__(self, program, bindings, b, y, x, B, H, W,
                 latent_channels=0, scatter=None, probes=None):
        self.program = program
        self.raw_bindings = {k: self._as_tensor_binding(v) for k, v in bindings.items()}
        self.b, self.y, self.x = b, y, x
        self.B, self.H, self.W = B, H, W
        self.latent_channels = latent_channels
        self.scatter = scatter
        self.probes = probes
        self.stdlib = TEXStdlib.get_functions()
        self.env: dict = {}
        self.fns: dict = {}
        self.outputs: dict = {}
        self.depth = 0
        self._seed_builtins()

    # ── set-up ──────────────────────────────────────────────────────────────
    def _seed_builtins(self):
        b, y, x, B, H, W = self.b, self.y, self.x, self.B, self.H, self.W
        e = self.env
        e["ix"] = _t(x)
        e["iy"] = _t(y)
        e["u"] = _t(x) / _t(max(W - 1, 1))
        e["v"] = _t(y) / _t(max(H - 1, 1))
        e["iw"] = _t(W)
        e["ih"] = _t(H)
        e["px"] = _t(1.0 / max(W, 1))
        e["py"] = _t(1.0 / max(H, 1))
        e["fi"] = _t(b)
        e["fn"] = _t(B)
        e["PI"] = _t(math.pi)
        e["TAU"] = _t(math.tau)
        e["E"] = _t(math.e)
        e["ic"] = _t(self.latent_channels)

    def _pixel_of(self, value):
        """A whole binding narrowed to THIS pixel. Singleton axes broadcast, exactly as
        they do in the vectorised cook."""
        if isinstance(value, str):
            return value
        t = value if isinstance(value, torch.Tensor) else _t(value)
        if t.dim() == 0:
            return t
        if t.dim() >= 3:
            bi = self.b if t.shape[0] > 1 else 0
            yi = self.y if t.shape[1] > 1 else 0
            xi = self.x if t.shape[2] > 1 else 0
            out = t[bi, yi, xi]
            return out
        if t.dim() == 1:                       # a vec param
            return t
        raise OracleUnsupported(f"binding of rank {t.dim()}")

    @staticmethod
    def _as_tensor_binding(value):
        if isinstance(value, (list, tuple)) and not isinstance(value, str):
            return torch.tensor([float(v) for v in value], dtype=F32)
        return value

    # ── driver ──────────────────────────────────────────────────────────────
    def run(self) -> dict:
        for stmt in self.program.statements:
            if type(stmt) is A.FunctionDef:
                self.fns[stmt.name] = stmt
        for stmt in self.program.statements:
            self._stmt(stmt)
        return self.outputs

    # ── statements ──────────────────────────────────────────────────────────
    def _stmt(self, n):
        cls = type(n)
        if cls is A.VarDecl:
            return self._var_decl(n)
        if cls is A.Assignment:
            return self._assign(n)
        if cls is A.IfElse:
            if _truth(self._eval(n.condition)):
                for s in n.then_body:
                    self._stmt(s)
            else:
                for s in n.else_body or ():
                    self._stmt(s)
            return None
        if cls is A.ForLoop:
            return self._for(n)
        if cls is A.WhileLoop:
            return self._while(n)
        if cls is A.BreakStmt:
            raise _Break()
        if cls is A.ContinueStmt:
            raise _Continue()
        if cls is A.ReturnStmt:
            raise _Return(self._eval(n.value) if n.value is not None else _t(0.0))
        if cls is A.ExprStatement:
            self._eval(n.expr)
            return None
        if cls is A.FunctionDef:
            self.fns[n.name] = n
            return None
        if cls is A.ParamDecl:
            return None
        if cls is A.ArrayDecl:
            return self._array_decl(n)
        raise OracleUnsupported(f"statement {cls.__name__}")

    def _var_decl(self, n):
        if n.initializer is not None:
            v = self._eval(n.initializer)
        else:
            v = self._default_for(n.type_name)
        self.env[n.name] = self._coerce_decl(n.type_name, v)

    def _array_decl(self, n):
        if (n.element_type_name or "").lower() == "string":
            raise OracleUnsupported("string array")
        if n.initializer is not None:
            if type(n.initializer) is A.ArrayLiteral:
                vals = [self._eval(e) for e in n.initializer.elements]
            else:
                src = self._eval(n.initializer)
                if not isinstance(src, list):
                    raise OracleUnsupported("array copy from non-array")
                vals = list(src)
        else:
            vals = [_t(0.0) for _ in range(int(n.size or 0))]
        self.env[n.name] = vals

    @staticmethod
    def _default_for(type_name):
        tn = (type_name or "float").lower()
        if tn in ("vec2", "vec3", "vec4"):
            return [_t(0.0)] * int(tn[-1])
        if tn == "string":
            return ""
        return _t(0.0)

    @staticmethod
    def _coerce_decl(type_name, v):
        tn = (type_name or "").lower()
        if tn in ("vec2", "vec3") and isinstance(v, list):
            return v[: int(tn[-1])]
        return v

    def _for(self, n):
        if n.init is not None:
            self._stmt(n.init)
        passes = 0
        while True:
            if passes >= MAX_PASSES:
                raise OracleLoopCap(f"for loop exceeded {MAX_PASSES} passes")
            if n.condition is not None and not _truth(self._eval(n.condition)):
                break
            try:
                for s in n.body:
                    self._stmt(s)
            except _Break:
                break
            except _Continue:
                pass
            if n.update is not None:
                self._stmt(n.update)
            passes += 1

    def _while(self, n):
        passes = 0
        while True:
            if passes >= MAX_PASSES:
                raise OracleLoopCap(f"while loop exceeded {MAX_PASSES} passes")
            if not _truth(self._eval(n.condition)):
                break
            try:
                for s in n.body:
                    self._stmt(s)
            except _Break:
                break
            except _Continue:
                pass
            passes += 1

    # ── assignment ──────────────────────────────────────────────────────────
    def _assign(self, n):
        target = n.target
        cls = type(target)
        if cls is A.BindingIndexAccess:
            return self._scatter(n)
        value = self._eval(n.value)
        if cls is A.Identifier:
            cur = self.env.get(target.name)
            if isinstance(cur, list) and not isinstance(value, (list, str)):
                value = [value] * len(cur)
            elif isinstance(cur, list) and isinstance(value, list):
                value = value[: len(cur)] if len(value) > len(cur) else value
            self.env[target.name] = value
            return None
        if cls is A.BindingRef:
            self.outputs[target.name] = value
            return None
        if cls is A.ChannelAccess:
            return self._channel_assign(target, value)
        if cls is A.ArrayIndexAccess:
            arr = self._lookup_array(target.array)
            idx = int(_num(self._eval(target.index)))
            idx = max(0, min(idx, len(arr) - 1))
            arr[idx] = value
            return None
        raise OracleUnsupported(f"assignment target {cls.__name__}")

    def _lookup_array(self, node):
        if type(node) is not A.Identifier:
            raise OracleUnsupported("array target that is not a name")
        arr = self.env.get(node.name)
        if not isinstance(arr, list):
            raise OracleUnsupported("array write to a non-array")
        return arr

    def _channel_assign(self, target, value):
        obj = target.object
        chans = target.channels
        if type(obj) is A.Identifier:
            base = self.env.get(obj.name)
            store, key = self.env, obj.name
        elif type(obj) is A.BindingRef:
            base = self.outputs.get(obj.name)
            store, key = self.outputs, obj.name
        else:
            raise OracleUnsupported("channel write to a computed target")
        if not isinstance(base, list):
            if len(chans) == 1 and CHANNEL_MAP.get(chans) == 0:
                store[key] = value          # `m.r = v` on a scalar means `m = v`
                return None
            base = [base if isinstance(base, torch.Tensor) else _t(0.0)] * 4
        base = list(base)
        vals = value if isinstance(value, list) else [value] * len(chans)
        for i, ch in enumerate(chans):
            idx = CHANNEL_MAP[ch]
            if idx < len(base):
                base[idx] = vals[i] if i < len(vals) else vals[-1]
        store[key] = base
        return None

    def _scatter(self, n):
        """M5 at one pixel: this source contributes, because a source that reached this
        statement is by definition live."""
        if self.scatter is None:
            raise OracleUnsupported("scatter write without a scatter buffer")
        target = n.target
        name = target.binding.name
        buf = self.scatter.get(name)
        if buf is None:
            raise OracleUnsupported(f"scatter into unallocated @{name}")
        value = self._eval(n.value)
        coords = [self._eval(a) for a in target.args]
        px = int(max(0, min(math.floor(_num(coords[0])), self.W - 1)))
        py = int(max(0, min(math.floor(_num(coords[1])), self.H - 1)))
        pb = int(max(0, min(int(_num(coords[2])), self.B - 1))) if len(coords) == 3 else self.b
        vals = value if isinstance(value, list) else [value]
        if buf.dim() == 4:
            cur = buf[pb, py, px]
            new = torch.stack([_t(v) for v in vals])[: buf.shape[-1]]
        else:
            cur = buf[pb, py, px]
            new = _t(vals[0])
        op = n.op
        if op is None:
            buf[pb, py, px] = new
        elif op == "+":
            buf[pb, py, px] = cur + new
        elif op == "-":
            buf[pb, py, px] = cur - new
        elif op == "*":
            buf[pb, py, px] = cur * new
        elif op == "/":
            buf[pb, py, px] = cur / torch.where(new == 0, _t(1e-8), new)
        return None

    # ── expressions ─────────────────────────────────────────────────────────
    def _eval(self, n):
        cls = type(n)
        if cls is A.NumberLiteral:
            return _t(n.value)
        if cls is A.StringLiteral:
            return n.value
        if cls is A.Identifier:
            if n.name not in self.env:
                raise OracleUnsupported(f"undefined name {n.name}")
            return self.env[n.name]
        if cls is A.BindingRef:
            if n.name in self.outputs:
                return self.outputs[n.name]     # a binding this program has written
            raw = self.raw_bindings.get(n.name)
            if raw is None:
                raise OracleUnsupported(f"unbound @{n.name}")
            v = self._pixel_of(raw)
            if isinstance(v, torch.Tensor) and v.dim() == 1:
                return [v[i] for i in range(v.shape[0])]
            return v
        if cls is A.ChannelAccess:
            return self._channel_read(n)
        if cls is A.BinOp:
            return self._binop(n)
        if cls is A.UnaryOp:
            return self._unary(n)
        if cls is A.TernaryOp:
            return (self._eval(n.true_expr) if _truth(self._eval(n.condition))
                    else self._eval(n.false_expr))
        if cls is A.FunctionCall:
            return self._call(n)
        if cls is A.VecConstructor:
            return self._vec(n)
        if cls is A.CastExpr:
            return self._cast(n)
        if cls is A.ArrayIndexAccess:
            arr = self._eval(n.array)
            if not isinstance(arr, list):
                raise OracleUnsupported("index of a non-array")
            idx = int(_num(self._eval(n.index)))
            return arr[max(0, min(idx, len(arr) - 1))]
        if cls is A.ArrayLiteral:
            return [self._eval(e) for e in n.elements]
        if cls is A.BindingIndexAccess:
            return self._tap(n, "fetch")
        if cls is A.BindingSampleAccess:
            return self._tap(n, "sample")
        raise OracleUnsupported(f"expression {cls.__name__}")

    def _tap(self, n, kind):
        image = self.raw_bindings.get(n.binding.name)
        if image is None:
            raise OracleUnsupported(f"unbound @{n.binding.name}")
        args = [_t(_num(self._eval(a))) for a in n.args]
        if len(args) == 3:
            fn = self.stdlib["fetch_frame" if kind == "fetch" else "sample_frame"]
            out = fn(image, args[2], args[0], args[1])
        else:
            out = self.stdlib[kind](image, args[0], args[1])
        return self._spread(out)

    def _spread(self, out):
        """Narrow a stdlib answer to this pixel's value.

        A stdlib function is vectorised: handed 0-dim coordinates it still answers at the
        grid's rank, broadcasting the one tap it made (`fetch(img, 1.0, 1.0)` returns the
        whole `[B,H,W,C]`, every position holding `img[b,1,1]`), and handed a whole image
        — which is what a gather gets — it answers a whole frame. Either way THIS pixel's
        value is the one at `(b, y, x)`, so that is what a single-pixel evaluation reads.
        A 1-D answer is this pixel's vector (or its array, for `sort` of a local);
        anything left wider means the call reached past this pixel, and the oracle refuses
        it rather than guessing."""
        if not isinstance(out, torch.Tensor):
            return out
        if out.dim() >= 3:
            out = self._pixel_of(out)
        if not isinstance(out, torch.Tensor) or out.dim() == 0:
            return out
        r = out.reshape(-1)
        if r.numel() == 1:
            return r[0]
        if out.dim() == 1:
            return [r[i] for i in range(r.numel())]
        if out.shape[-1] in (2, 3, 4) and out.numel() == out.shape[-1]:
            return [r[i] for i in range(r.numel())]
        raise OracleUnsupported("a builtin returned more than this pixel's value")

    def _channel_read(self, n):
        base = self._eval(n.object)
        chans = n.channels
        if not isinstance(base, list):
            if len(chans) == 1 and CHANNEL_MAP.get(chans) == 0:
                return base
            raise OracleUnsupported(f"channel .{chans} of a scalar")
        picked = [base[CHANNEL_MAP[c]] if CHANNEL_MAP[c] < len(base) else _t(0.0)
                  for c in chans]
        return picked[0] if len(picked) == 1 else picked

    def _binop(self, n):
        op = n.op
        if op == "&&":
            return _t(1.0 if (_truth(self._eval(n.left)) and _truth(self._eval(n.right)))
                      else 0.0)
        if op == "||":
            return _t(1.0 if (_truth(self._eval(n.left)) or _truth(self._eval(n.right)))
                      else 0.0)
        left = self._eval(n.left)
        right = self._eval(n.right)
        if isinstance(left, str) or isinstance(right, str):
            if op == "+":
                return f"{left}{right}"
            raise OracleUnsupported(f"string operator {op}")
        return self._apply(op, left, right)

    def _apply(self, op, left, right):
        if isinstance(left, list) or isinstance(right, list):
            ln = len(left) if isinstance(left, list) else 0
            rn = len(right) if isinstance(right, list) else 0
            n = max(ln, rn)
            lv = left if isinstance(left, list) else [left] * n
            rv = right if isinstance(right, list) else [right] * n
            if len(lv) != len(rv):
                m = min(len(lv), len(rv))
                lv, rv = lv[:m], rv[:m]
            return [self._apply(op, a, b) for a, b in zip(lv, rv)]
        a, b = _t(left), _t(right)
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            return a / torch.where(b == 0, _t(1e-8), b)
        if op == "%":
            return torch.remainder(a, torch.where(b == 0, _t(1e-8), b))
        if op == "<":
            return _t(1.0) if float(a.item()) < float(b.item()) else _t(0.0)
        if op == ">":
            return _t(1.0) if float(a.item()) > float(b.item()) else _t(0.0)
        if op == "<=":
            return _t(1.0) if float(a.item()) <= float(b.item()) else _t(0.0)
        if op == ">=":
            return _t(1.0) if float(a.item()) >= float(b.item()) else _t(0.0)
        if op == "==":
            return _t(1.0) if float(a.item()) == float(b.item()) else _t(0.0)
        if op == "!=":
            return _t(1.0) if float(a.item()) != float(b.item()) else _t(0.0)
        raise OracleUnsupported(f"operator {op}")

    def _unary(self, n):
        v = self._eval(n.operand)
        if n.op == "-":
            return [-_t(e) for e in v] if isinstance(v, list) else -_t(v)
        if n.op == "+":
            return v
        if n.op == "!":
            return _t(0.0) if _truth(v) else _t(1.0)
        raise OracleUnsupported(f"unary {n.op}")

    def _vec(self, n):
        out = []
        for a in n.args:
            v = self._eval(a)
            out.extend(v if isinstance(v, list) else [v])
        size = n.size
        if len(out) == 1:
            out = out * size
        while len(out) < size:
            out.append(_t(1.0) if len(out) == 3 else _t(0.0))
        return out[:size]

    def _cast(self, n):
        v = self._eval(n.expr)
        tt = (n.target_type or "").lower()
        if tt == "string":
            raise OracleUnsupported("string cast")
        if tt == "int":
            if isinstance(v, list):
                raise OracleUnsupported("int cast of a vector")
            return torch.trunc(_t(v))
        if tt == "float":
            return _t(v) if not isinstance(v, list) else v[0]
        raise OracleUnsupported(f"cast to {tt}")

    def _call(self, n):
        name = n.name
        fd = self.fns.get(name)
        if fd is not None:
            return self._call_user(fd, n)
        if name == "debug_print":
            if self.probes is not None:
                self.probes.append((self.b, self.y, self.x,
                                    tuple(_num(self._eval(a)) if type(a) is not A.StringLiteral
                                          else a.value for a in n.args)))
            return _t(0.0)
        fn = self.stdlib.get(name)
        if fn is None:
            raise OracleUnsupported(f"function {name}")
        args = []
        nonlocal_fn = _footprint_of(name) != "point"
        for a in n.args:
            if nonlocal_fn and type(a) is A.BindingRef:
                raw = self.raw_bindings.get(a.name)
                if isinstance(raw, torch.Tensor) and raw.dim() >= 3:
                    # A gather is handed the WHOLE image, exactly as the engine hands it:
                    # narrowing it to this pixel would make `fetch(@A, ix+1, iy)` read a
                    # 1x1 image. Decided from the registry's footprint (ROI-1), not from a
                    # name list, so the oracle cannot drift from what the engine calls
                    # non-pixel-local.
                    args.append(raw)
                    continue
            v = self._eval(a)
            if isinstance(v, list):
                args.append(torch.stack([_t(e) for e in v]))
            elif isinstance(v, str):
                args.append(v)
            else:
                args.append(_t(v))
        try:
            out = fn(*args)
        except OracleUnsupported:
            raise
        except Exception as exc:                                  # noqa: BLE001
            raise OracleUnsupported(f"{name}() refused a scalar call: {exc}") from exc
        return self._spread(out)

    def _call_user(self, fd, call):
        """M4 at one pixel: a call runs, a `return` unwinds it, and a body that falls off
        the end answers the language's zero default."""
        if self.depth >= MAX_DEPTH:
            raise OracleUnsupported("call depth")
        args = [self._eval(a) for a in call.args]
        saved = self.env
        self.env = dict(saved)
        for (ptype, pname), val in zip(fd.params, args):
            self.env[pname] = self._coerce_decl(ptype, val)
        self.depth += 1
        try:
            for s in fd.body:
                self._stmt(s)
            return _t(0.0)
        except _Return as r:
            return r.value
        finally:
            self.depth -= 1
            self.env = saved


class OracleLoopCap(Exception):
    """A pixel that never terminates — the scalar reading of E6010."""


def sweep(program, bindings, B, H, W, output_names, latent_channels=0,
          scatter_init=None):
    """Run the oracle over every pixel of a `(B, H, W)` grid, in ROW-MAJOR order (M5's
    stated compaction order), and reassemble the per-pixel answers into whole tensors.

    Returns `(outputs, skipped)`: `outputs` maps each name in `output_names` to a
    `[B, H, W]` or `[B, H, W, C]` tensor, `skipped` is None or the `OracleUnsupported`
    reason the sweep could not be completed. A scatter output is returned from
    `scatter_init`'s buffer, written by the pixels in source order."""
    scatter = {k: v.clone() for k, v in (scatter_init or {}).items()}
    per_pixel = {name: [] for name in output_names}
    probes: list = []
    for b in range(B):
        for y in range(H):
            for x in range(W):
                ev = ScalarOracle(program, bindings, b, y, x, B, H, W,
                                  latent_channels=latent_channels,
                                  scatter=scatter, probes=probes)
                out = ev.run()
                for name in output_names:
                    if name in scatter:
                        continue
                    if name not in out:
                        raise OracleUnsupported(f"@{name} was never assigned")
                    per_pixel[name].append(out[name])
    results = {}
    for name in output_names:
        if name in scatter:
            results[name] = scatter[name]
            continue
        vals = per_pixel[name]
        if vals and isinstance(vals[0], list):
            C = len(vals[0])
            flat = torch.stack([torch.stack([_t(c) for c in v[:C]]) for v in vals])
            results[name] = flat.reshape(B, H, W, C)
        elif vals and isinstance(vals[0], str):
            results[name] = vals[0]
        else:
            results[name] = torch.stack([_t(v) for v in vals]).reshape(B, H, W)
    return results, probes
