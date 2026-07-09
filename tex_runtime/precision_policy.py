"""
PR-LP2 — `precision="auto"` classifier (doc 28 §2.3).

fp16 is the only measured low-precision lever, and it wins only in a specific region:
CUDA + resolution >= 1024^2 + a **pointwise, accuracy-robust** program. This module
resolves `auto` to `fp16` or `fp32` with a human reason (recorded in tier_trace).

Eligibility = `is_tile_safe(program)` (pointwise: no sampling/fetch/blur/reduction —
those are the fp16 losers: lens 1.9e-2, and the img_* reductions; and no scatter, which
is coordinate-sensitive 0.23) **AND** no image-derived threshold (a comparison / step /
smoothstep / branch whose operand traces to an input image — fp16 rounding flips the
threshold, e.g. `halftone`'s `smoothstep(luma(@image)+/-e, dot)`).

Over-declining is always safe (fp32 is exact and the 1.0x baseline). The gate aims to
decline every inaccurate program, but **fp16 accuracy is a condition-number property**,
not merely a "smooth, no branch, no fragile fn" one (doc 32 C1): a continuous function of
an image value scaled by a large constant, catastrophic cancellation, or loop-accumulation
of image lineage all amplify fp16's input-quantization error past the 8-bit quantum while
staying finite. So the hazard test *also* declines constant magnitude amplification of
image lineage (`_amplifies_image`, threshold `_AMP`) and `for`-loop accumulation of an
image-tainted variable. Runtime params are the honest residual limit — the gate cannot see
a `@A * $huge` factor statically; the value-keyed first-cook finiteness net (tex_node) still
catches a param-induced NaN, but a wrong-but-finite runtime blow-up is out of static reach.
"""
import math

import torch

from ..tex_memory import is_tile_safe  # single source for pointwise-ness (M-4)
from ..tex_compiler.ast_nodes import iter_child_nodes as _children

_MIN_FP16_PX = 1024 * 1024          # measured: fp16 win starts at 1024^2 (0.96x at 512^2)
_COMPARE_OPS = frozenset({">", "<", ">=", "<=", "==", "!="})
# C1 (doc 32): decline a constant magnitude amplification of image lineage at or above
# this factor. At |K|>=4 an fp16 operand reaches the [4,8) binade whose ULP (2^-8 ~= 3.9e-3)
# already equals the 8-bit quantum — so a single cancellation there can exceed it, and a
# continuous fn (sin/cos/tan) of such an argument samples a high-derivative region. Below 4
# the amplified error stays under the quantum. Verified by the adversarial accuracy sweep.
_AMP = 3.5                          # decline image-lineage gain at/above this (measured: a
                                    # single sin(@A.r*3) at gain 3 is accurate; (@A.r+1)*3.9
                                    # at gain 3.9 is not — the safe budget sits between)
_MAG_MAX = 6.0                      # an image-carrying value reaching this magnitude loses
                                    # > the 8-bit quantum (fp16 ULP at [4,8) is already 2^-8)
_FP16_MAX = 65504.0                 # a literal beyond fp16's range can't be represented
# Builtin math constants are constants for folding/amplification purposes (so `@A*TAU*TAU`
# is seen as an amplification, not an opaque Identifier product). Coordinate builtins
# (u/v/ix/iy) are deliberately NOT here — they are fp32 (invariant #4) and never image-taint.
_BUILTIN_CONSTS = {"PI": math.pi, "TAU": 2.0 * math.pi, "E": math.e}
# Builtins that carry a LARGE runtime magnitude — multiplying image lineage by one amplifies
# fp16's error the same way a big constant does (doc 32 round 2: `sin(@A.r * iw)` where iw is
# the image width ~1024). They are fp32 themselves, but the PRODUCT `@A.r*iw` is fp16.
_BUILTIN_MAG = {"iw": 8192.0, "ih": 8192.0, "ix": 8192.0, "iy": 8192.0, "fi": 4096.0,
                "fn": 4096.0}

# fp16-fragile stdlib functions — a half-ULP wrecks them (measured in the 114-sweep:
# posterize floor -> 0.33, film_vignette acos/sqrt/smoothstep -> 0.63, hue_shift
# fract -> NaN). Declined on ANY lineage, because fp16 *intermediates* lose precision
# even for coordinate-derived values (u/v stay fp32, but `float d = ...` does not).
#   - discontinuous: a fp16 rounding flips the branch/level
#   - domain-restricted: a fp16 overshoot leaves the domain -> NaN (acos/asin |x|>1, sqrt/log<0)
# NOTE: pow/spow/sdiv are NOT here — the grade-class headline needs pow([0,1], 0.45),
# which is exact enough; the rare pow/division NaN is caught by execute()'s runtime
# finiteness fallback instead (can't be told apart statically from grade's safe pow).
_FP16_FRAGILE_FNS = frozenset({
    "floor", "round", "ceil", "fract", "trunc", "mod", "sign",
    "step", "smoothstep",
    "acos", "asin", "sqrt", "inversesqrt", "log", "log2",
    # magnitude-amplifying / near-singular — a fp16 mantissa is too coarse above ~1
    # (measured in the fp16 fuzzer: exp-towers 0.02-0.08, smin/smax blends 0.4)
    "exp", "exp2", "smin", "smax",
    # ill-conditioned near a singularity the static gate can't prove the args avoid
    # (doc 32 adversarial sweep): tan (poles), atan2 (origin — angle undefined),
    # normalize (zero vector), hypot / sdiv (near-zero → sqrt-/divide-sensitive),
    # sinh/cosh (exp-like growth). Single-arg atan is bounded+smooth and stays eligible.
    "tan", "atan2", "normalize", "hypot", "sdiv", "sinh", "cosh",
    # arr_sum grows unboundedly over image-filled elements the flow-analysis can't see
    # inside the array (arr_avg/min/max/median stay bounded by the input range).
    "arr_sum",
})
# Bounded-range fns (accepted) whose magnitude is capped regardless of argument, so a big
# argument doesn't propagate as a big output magnitude (their fp16 *error* from a big
# argument is caught by the gain/argument-magnitude checks instead).
_BOUNDED_FNS = frozenset({"sin", "cos", "tanh", "atan"})


def _walk(node):
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(_children(n))


def _output_names(program) -> set:
    """Names of @-bindings written by the program (so they aren't treated as image
    *inputs* for taint)."""
    outs = set()
    for n in _walk(program):
        if n.__class__.__name__ == "Assignment":
            t = n.target
            tc = t.__class__.__name__
            if tc == "BindingRef":
                outs.add(t.name)
            elif tc == "ChannelAccess" and t.object.__class__.__name__ == "BindingRef":
                outs.add(t.object.name)
            elif tc == "BindingIndexAccess" and t.binding.__class__.__name__ == "BindingRef":
                outs.add(t.binding.name)
    return outs


def _reads_image(node, tainted, out_names) -> bool:
    """True if this expression subtree reads an input image wire binding (@A, not an
    output, not a $param) or an image-tainted variable."""
    for n in _walk(node):
        cls = n.__class__.__name__
        if cls == "BindingRef" and n.kind == "wire" and n.name not in out_names:
            return True
        if cls == "Identifier" and n.name in tainted:
            return True
    return False


def _image_tainted_vars(program, out_names) -> set:
    """Variables whose value derives from an input image, via a fixed-point forward
    scan (handles chains like `col=@A; lum=luma(col); thr=lum;`). Fixed-point (not a
    single pass) so out-of-order/looped assignments can't under-taint (which would be
    the unsafe direction — accepting an inaccurate program)."""
    tainted: set = set()
    for _ in range(8):  # converges in <=depth passes; 8 is a generous cap
        grew = False
        for n in _walk(program):
            cls = n.__class__.__name__
            if cls == "VarDecl" and n.initializer is not None:
                if n.name not in tainted and _reads_image(n.initializer, tainted, out_names):
                    tainted.add(n.name); grew = True
            elif cls == "Assignment" and n.target.__class__.__name__ == "Identifier":
                if n.target.name not in tainted and _reads_image(n.value, tainted, out_names):
                    tainted.add(n.target.name); grew = True
        if not grew:
            break
    return tainted


def _const_eval(node, consts=None):
    """Fold a constant numeric subtree to a float, else None. Handles literals, unary
    +/-, the builtin math constants, +-*/ of constants, AND scalar variables that the
    forward pass has proven constant (`consts`, doc 32 round 2) — so a matrix built from a
    constant variable (`float s=1.4; mat3(s,s,…)`) folds and its row-L1 gain is recovered.
    Folding also means the gate reaches the SAME verdict on the raw and the const-folded
    AST (raw `pow(c, 1.0/2.2)` vs optimized `pow(c, 0.4545)` must not disagree)."""
    cls = node.__class__.__name__
    if cls == "NumberLiteral":
        return float(node.value)
    if cls == "Identifier":
        if node.name in _BUILTIN_CONSTS:
            return _BUILTIN_CONSTS[node.name]
        if consts is not None and node.name in consts:
            return consts[node.name]
        return None
    if cls == "UnaryOp":
        v = _const_eval(node.operand, consts)
        if v is None:
            return None
        return -v if node.op == "-" else v
    if cls == "BinOp" and node.op in ("+", "-", "*", "/"):
        l = _const_eval(node.left, consts)
        r = _const_eval(node.right, consts)
        if l is None or r is None:
            return None
        if node.op == "+":
            return l + r
        if node.op == "-":
            return l - r
        if node.op == "*":
            return l * r
        return l / r if r != 0.0 else None
    return None


def _const_magnitude(node, consts=None):
    """Max |constant| in a scalar or vector-of-constants node, else None."""
    if node.__class__.__name__ == "VecConstructor":
        vals = [_const_eval(a, consts) for a in node.args]
        if not vals or any(v is None for v in vals):
            return None
        return max(abs(v) for v in vals)
    v = _const_eval(node, consts)
    return abs(v) if v is not None else None


def _vec_l1(node, consts=None):
    """L1 norm (sum of |components|) of a constant vector/scalar, else None — the exact
    gain/magnitude weight for `dot(image, const_vec)` (a sum of products, not a max)."""
    if node.__class__.__name__ == "VecConstructor":
        vals = [_const_eval(a, consts) for a in node.args]
        if not vals or any(v is None for v in vals):
            return None
        return sum(abs(v) for v in vals)
    v = _const_eval(node, consts)
    return abs(v) if v is not None else None


def _vec_vals(node, consts=None):
    """Constant component values of a scalar/vector node as a list, else None (broadcasts
    a scalar to a 1-list). Used to fold `fit` bounds that may be vectors (doc 32 round 2)."""
    if node.__class__.__name__ == "VecConstructor":
        vs = [_const_eval(a, consts) for a in node.args]
        return vs if None not in vs else None
    v = _const_eval(node, consts)
    return [v] if v is not None else None


def _mat_max_row_l1(node, consts=None):
    """Max row L1 of a constant matrix (an over-estimate of its operator gain on a vector),
    else None. `mat3(3.9×9)*@A.rgb` amplifies each channel by a row sum ~11.7, invisible to
    a per-BinOp const check — so a matrix's 'gain' is taken as this row norm. A single-arg
    `mat3(k)` is a k·identity, so its row L1 is |k|."""
    if node.__class__.__name__ != "MatConstructor":
        return None
    n = node.size
    vals = [_const_eval(a, consts) for a in node.args]
    if any(v is None for v in vals):
        return None
    if len(vals) == 1:            # mat(k) == k * identity
        return abs(vals[0])
    if len(vals) != n * n:
        return None
    return max(sum(abs(vals[r * n + c]) for c in range(n)) for r in range(n))


def _gm(node, vg, vm, out_names, hz, consts):
    """Return `(image_gain, magnitude)` for an expression and set `hz[0]` if any
    image-carrying subexpression reaches magnitude >= _MAG_MAX.

    - **gain** = the factor by which fp16's input error on image lineage is multiplied —
      the condition-number estimate the flat "smooth ⇒ safe" axiom lacked (doc 32 C1). It
      tracks amplification ASSEMBLED FROM SUB-THRESHOLD STEPS: `sin(@A.r*3*3)` is
      `sin(9·@A.r)` (gain 9), image×image squaring (`x*x`), `/0.3/0.3`, `@A.r*PI*PI`.
    - **magnitude** = an upper bound on |value|. A transient high magnitude on image
      lineage (`(@A.r + 60000) - 60000`, net gain ~1) quantizes the image detail below
      fp16's ULP — invisible to gain alone, so peak image-magnitude is flagged separately.

    `vg`/`vm` carry per-variable gain/magnitude and `consts` per-variable constant values
    from the flow-sensitive forward pass (so `mat3(s,…)` with a constant `s` folds)."""
    cls = node.__class__.__name__
    cv = _const_eval(node, consts)
    if cv is not None:
        return 0.0, abs(cv)
    if cls == "BindingRef":
        img = node.kind == "wire" and node.name not in out_names
        return (1.0 if img else 0.0), 1.0
    if cls == "Identifier":
        if node.name in _BUILTIN_MAG:
            return 0.0, _BUILTIN_MAG[node.name]
        return vg.get(node.name, 0.0), vm.get(node.name, 1.0)
    if cls == "ChannelAccess":
        return _gm(node.object, vg, vm, out_names, hz, consts)
    if cls == "UnaryOp":
        return _gm(node.operand, vg, vm, out_names, hz, consts)
    if cls == "BinOp":
        lg, lm = _gm(node.left, vg, vm, out_names, hz, consts)
        rg, rm = _gm(node.right, vg, vm, out_names, hz, consts)
        op = node.op
        if op == "*":
            lc, rc = _const_eval(node.left, consts), _const_eval(node.right, consts)
            g = rg * abs(lc) if lc is not None else lg * abs(rc) if rc is not None else lg + rg
            m = lm * rm
        elif op == "/":
            rc = _const_eval(node.right, consts)
            if rc is not None and rc != 0.0:
                g, m = lg / abs(rc), lm / abs(rc)
            else:
                g = m = float("inf")          # image / image expr: near-singular
        elif op in ("+", "-", "%"):
            g, m = lg + rg, lm + rm           # cancellation (-) / additive: error + mag add
        else:
            g, m = max(lg, rg), max(lm, rm)   # comparisons handled as hazards elsewhere
        if g > 0.0 and m >= _MAG_MAX:
            hz[0] = True
        return g, m
    if cls == "FunctionCall":
        parts = [_gm(a, vg, vm, out_names, hz, consts) for a in node.args]
        g = max((p[0] for p in parts), default=0.0)
        name = node.name
        if name in _BOUNDED_FNS:
            m = 2.0                           # sin/cos/tanh <=1, atan <=pi/2 — bounded
        elif name in ("pow", "spow") and len(node.args) >= 2:
            k = _const_magnitude(node.args[1], consts)
            bg, bm = parts[0]
            if k is not None:
                g = max(g, bg * k)
                m = bm ** min(k, 8.0)
            else:
                m = bm
        elif name == "dot" and len(parts) == 2:
            # dot is a SUM of products, not a max: dot(@A.rgb, vec3(3.9)) ~ 3*3.9. Weight
            # the image operand's gain/mag by the const operand's L1 norm (doc 32 sweep).
            (ga, ma), (gb, mb) = parts
            la, lb = _vec_l1(node.args[0], consts), _vec_l1(node.args[1], consts)
            if lb is not None:
                g, m = ga * lb, ma * lb
            elif la is not None:
                g, m = gb * la, mb * la
            else:
                g, m = (ga + gb) * 4.0, ma * mb * 4.0
        elif name in ("length", "distance"):
            # sqrt(sum of squares) over up to 4 channels: ~2x the operand gain/mag
            if name == "distance" and len(parts) == 2:
                gg, mm = parts[0][0] + parts[1][0], parts[0][1] + parts[1][1]
            else:
                gg, mm = parts[0]
            g, m = 2.0 * gg, 2.0 * mm
        elif name in ("cross", "reflect", "refract") and len(parts) >= 2:
            (ga, ma), (gb, mb) = parts[0], parts[1]
            g, m = 2.0 * (ga * mb + gb * ma), 2.0 * ma * mb
        elif name == "fit" and len(node.args) >= 5:
            # fit(x, inLo, inHi, outLo, outHi) = outLo + (x-inLo)/(inHi-inLo)*(outHi-outLo):
            # a narrow input band amplifies by |outHi-outLo|/|inHi-inLo| (doc 32: fit(@A,
            # 0.49, 0.51, 0, 1) = 50x). Fold the four bounds (scalar OR vector) and take the
            # worst-case per-component factor.
            ig, im = parts[0]
            inlo, inhi = _vec_vals(node.args[1], consts), _vec_vals(node.args[2], consts)
            outlo, outhi = _vec_vals(node.args[3], consts), _vec_vals(node.args[4], consts)
            factor = None
            if None not in (inlo, inhi, outlo, outhi):
                n = max(len(inlo), len(inhi), len(outlo), len(outhi))
                get = lambda v, i: v[i] if i < len(v) else v[-1]
                facs = []
                for i in range(n):
                    din = abs(get(inhi, i) - get(inlo, i))
                    if din == 0.0:
                        facs = [float("inf")]; break
                    facs.append(abs(get(outhi, i) - get(outlo, i)) / din)
                factor = max(facs)
            if factor is not None:
                g, m = ig * factor, im * factor
            else:
                g, m = max((p[0] for p in parts), default=0.0), max((p[1] for p in parts), default=1.0)
        else:                                 # abs/min/max/clamp/lerp/mix/...: arg mag
            m = max((p[1] for p in parts), default=1.0)
        if g > 0.0 and m >= _MAG_MAX:
            hz[0] = True
        return g, m
    if cls == "MatConstructor":
        # A constant matrix's gain proxy is its max row L1 — so `m*@A.rgb` inherits the
        # fan-in amplification a per-BinOp check misses. With `consts`, a matrix built from
        # a constant VARIABLE (`float s=1.4; mat3(s,…)`) folds too (doc 32 round 2).
        rl1 = _mat_max_row_l1(node, consts)
        if rl1 is not None:
            return rl1, rl1
        parts = [_gm(a, vg, vm, out_names, hz, consts) for a in node.args]
        return (max((p[0] for p in parts), default=0.0),
                max((p[1] for p in parts), default=0.0))
    if cls == "VecConstructor":
        parts = [_gm(a, vg, vm, out_names, hz, consts) for a in node.args]
        return (max((p[0] for p in parts), default=0.0),
                max((p[1] for p in parts), default=0.0))
    if cls == "CastExpr":
        return _gm(node.expr, vg, vm, out_names, hz, consts)
    g = m = 0.0
    for ch in _children(node):
        cg, cm = _gm(ch, vg, vm, out_names, hz, consts)
        g, m = max(g, cg), max(m, cm)
    return g, m


def _amplification_hazard(program, out_names) -> bool:
    """Forward, flow-sensitive pass over the program: track each variable's image (gain,
    magnitude) in program order, flag when a value written to an OUTPUT binding has gain
    >= _AMP, or when any image-carrying subexpression reaches magnitude >= _MAG_MAX.
    Flow-sensitive (not a saturating fixed-point) so a single re-scale `c = c*1.5` reads as
    gain 1.5, not a diverging self-loop — loops that assign image lineage are already
    declined by the for-accum / while hazards, so there are no back-edges to diverge on.
    `consts` const-propagates scalar variables so a matrix built from them folds."""
    vg: dict = {}
    vm: dict = {}
    consts: dict = {}
    hz = [False]

    def _bind(name, node):
        vg[name], vm[name] = _gm(node, vg, vm, out_names, hz, consts)
        cv = _const_eval(node, consts)
        if cv is not None:
            consts[name] = cv
        else:
            consts.pop(name, None)            # reassigned to non-const -> no longer constant

    def process(stmts) -> bool:
        for s in stmts:
            cls = s.__class__.__name__
            if cls == "VarDecl":
                if s.initializer is not None:
                    _bind(s.name, s.initializer)
                else:
                    vg[s.name], vm[s.name] = 0.0, 1.0
            elif cls == "Assignment":
                t = s.target
                tcls = t.__class__.__name__
                if tcls == "Identifier":
                    if s.op:                  # compound (x += v): x = x + v, never constant
                        g, m = _gm(s.value, vg, vm, out_names, hz, consts)
                        vg[t.name] = g + vg.get(t.name, 0.0)
                        vm[t.name] = m + vm.get(t.name, 1.0)
                        consts.pop(t.name, None)
                    else:
                        _bind(t.name, s.value)
                elif tcls == "ArrayIndexAccess" and t.array.__class__.__name__ == "Identifier":
                    # a[i] = expr — carry the element's gain/mag onto the array name so an
                    # arr_*/reduction over it inherits image lineage (doc 32: the analysis
                    # doesn't see inside arrays otherwise). Compound array stores accumulate.
                    g, m = _gm(s.value, vg, vm, out_names, hz, consts)
                    an = t.array.name
                    if s.op:
                        g += vg.get(an, 0.0)
                        m += vm.get(an, 1.0)
                    vg[an] = max(vg.get(an, 0.0), g)
                    vm[an] = max(vm.get(an, 1.0), m)
                    consts.pop(an, None)
                else:
                    g, _m = _gm(s.value, vg, vm, out_names, hz, consts)
                    if g >= _AMP:             # write to an @output binding, amplified
                        return True
            elif cls == "IfElse":
                bg, bm, bc = dict(vg), dict(vm), dict(consts)
                if process(s.then_body):
                    return True
                tg, tm, tc = dict(vg), dict(vm), dict(consts)   # then-branch results
                vg.clear(); vg.update(bg); vm.clear(); vm.update(bm)
                consts.clear(); consts.update(bc)
                if process(s.else_body):                        # else results land in vg/vm/consts
                    return True
                for k in set(tg) | set(vg):                     # gain/mag: merge by max
                    vg[k] = max(tg.get(k, 0.0), vg.get(k, 0.0))
                for k in set(tm) | set(vm):
                    vm[k] = max(tm.get(k, 0.0), vm.get(k, 0.0))
                merged = {k: v for k, v in tc.items() if consts.get(k) == v}  # const iff both agree
                consts.clear(); consts.update(merged)
            elif cls in ("ForLoop", "WhileLoop"):
                if process(s.body):
                    return True
            if hz[0]:
                return True
        return hz[0]

    return process(program.statements)


def _pow_is_safe(node) -> bool:
    """`pow(base, k)` is fp16-safe only when k is a small POSITIVE CONSTANT (grade's
    `pow(c, 0.4545)`): the output then stays bounded for a bounded base. A variable /
    negative / large exponent can blow a near-zero base up (measured: 1.7e3) or NaN a
    negative one — indistinguishable from grade's safe pow except by the exponent."""
    if len(node.args) < 2:
        return False
    exp = node.args[1]
    if exp.__class__.__name__ == "VecConstructor":
        vals = [_const_eval(a) for a in exp.args]
    else:
        vals = [_const_eval(exp)]
    if not vals or any(v is None for v in vals):
        return False
    return all(0.0 < v <= 4.0 for v in vals)


def _for_accumulates_image(loop, tainted) -> bool:
    """True if a `for` loop body assigns to an image-tainted variable — loop
    accumulation of image lineage (`for(...) acc += @A.r;`) compounds fp16 error across
    iterations while staying finite (doc 32 C1). A loop over coordinates (`u`/`v`, not
    image-tainted) or pure counters is untouched; over-declining an image loop is safe."""
    for stmt in loop.body:
        for m in _walk(stmt):
            if (m.__class__.__name__ == "Assignment"
                    and m.target.__class__.__name__ == "Identifier"
                    and m.target.name in tainted):
                return True
    return False


def _has_fp16_hazard(program, out_names) -> bool:
    """Decline fp16 if the program has a fp16-fragile function anywhere, an unsafe
    `pow`, a data-dependent branch (if/ternary/while — a fp16 value decides control
    flow), an image-lineage comparison (an image thresholded directly, `@A.r>0.5`),
    an out-of-fp16-range literal, a `for` loop that accumulates image lineage (C1), or an
    image-lineage amplification of gain >= _AMP assembled anywhere (C1: `@A*40`,
    `@A/0.0002`, but also `sin(@A*3*3)`, `x*x` squaring, `/0.3/0.3`, `@A*PI*PI`)."""
    tainted = _image_tainted_vars(program, out_names)
    if _amplification_hazard(program, out_names):
        return True
    user_fns = {n.name for n in _walk(program) if n.__class__.__name__ == "FunctionDef"}
    for n in _walk(program):
        cls = n.__class__.__name__
        if cls == "FunctionCall":
            if n.name in _FP16_FRAGILE_FNS:
                return True
            if n.name in ("pow", "spow") and not _pow_is_safe(n):
                return True
            # F1 (doc 33): the gain forward-pass never enters user-function BODIES, so
            # amplification assembled inside `float amp(float x){ return x*50; }` is invisible
            # to it (the _walk-based fragile/branch checks DO descend, so those are caught).
            # Over-decline: a user-fn call carrying image lineage -> fp32. User functions are
            # rare in the pointwise fp16 target, so keeping the gate honest beats eligibility.
            if n.name in user_fns and any(_reads_image(a, tainted, out_names) for a in n.args):
                return True
        elif cls in ("IfElse", "TernaryOp", "WhileLoop"):
            return True  # a fp16 value steering control flow -> unstable output
        elif cls == "NumberLiteral":
            # a constant fp16 can't represent (|v| > 65504) becomes inf in the fp16 lane
            if not (-_FP16_MAX <= n.value <= _FP16_MAX):
                return True
        elif cls == "ForLoop":
            if _for_accumulates_image(n, tainted):
                return True
        elif cls == "BinOp" and n.op in _COMPARE_OPS:
            # image-lineage comparison (integer for-loop counters `i < N` are not
            # image-tainted, so bounded loops stay eligible)
            if _reads_image(n.left, tainted, out_names) or _reads_image(n.right, tainted, out_names):
                return True
    return False


def apply_tf32_profile(enable: bool = True):
    """PR-LP6 — opt-in TF32 for fp32 matmul/conv on Ampere+ (sm_80+). Returns a restore
    callable. Default OFF. **No-op on Turing (sm_75)** — no TF32 hardware, so it's inert
    here (verified: results bit-identical). The 1.5–2.5x Ampere claim is NOT shipped as a
    planning input until measured on sm_80+; this ships the Turing-safe half only — a
    profile a host/user can flip, with an honest restore. It's a *precision* trade (TF32
    keeps run-to-run determinism), never a determinism trade."""
    prev_mm = torch.get_float32_matmul_precision()
    prev_cudnn = bool(getattr(torch.backends.cudnn, "allow_tf32", False))

    def restore():
        torch.set_float32_matmul_precision(prev_mm)
        try:
            torch.backends.cudnn.allow_tf32 = prev_cudnn
        except Exception:
            pass

    if enable:
        torch.set_float32_matmul_precision("high")
        try:
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
    return restore


def resolve_auto_precision(program, spatial_px: int, device_type: str):
    """Return (precision, reason) for `precision="auto"`. precision is "fp16" only when the
    condition-number gate proves the program accurate in fp16; "fp32" (with the declining
    reason) otherwise — the gate over-declines rather than risk accuracy."""
    if device_type != "cuda":
        return "fp32", "auto->fp32: CPU (fp16 is slower on CPU)"
    if spatial_px < _MIN_FP16_PX:
        return "fp32", "auto->fp32: <1024^2 (fp16 gain only above)"
    if not is_tile_safe(program):
        return "fp32", "auto->fp32: sampling/fetch/reduction/scatter (fp16-unsafe)"
    if _has_fp16_hazard(program, _output_names(program)):
        return "fp32", "auto->fp32: fp16-fragile fn, data branch, or image-lineage amplification"
    return "fp16", "auto->fp16: gate-verified accurate (smooth, bounded condition number)"
