"""
TEX Interpreter — tree-walking evaluator that executes TEX AST on PyTorch tensors.

The interpreter maintains a "spatial context" derived from image inputs:
  - All operations are vectorized across [B, H, W]
  - Scalar values broadcast to the spatial context when mixed with images
  - Vector values have shape [B, H, W, C] where C is 3 or 4
  - Channel access indexes into the last dimension

Execution model:
  1. Determine spatial context (B, H, W) from image inputs
  2. Create coordinate tensors (ix, iy, u, v, iw, ih)
  3. Walk the AST, evaluating each node to a tensor
  4. Return the value of @OUT
"""
from __future__ import annotations
import torch
import math
from ..tex_compiler.ast_nodes import (
    ASTNode, Program, VarDecl, Assignment, IfElse, ForLoop, WhileLoop, ExprStatement,
    BreakStmt, ContinueStmt, FunctionDef, ReturnStmt,
    BinOp, UnaryOp, TernaryOp, FunctionCall, Identifier, BindingRef,
    ChannelAccess, NumberLiteral, StringLiteral, VecConstructor, CastExpr, SourceLoc,
    ArrayDecl, ArrayIndexAccess, ArrayLiteral, MatConstructor, ParamDecl,
    BindingIndexAccess, BindingSampleAccess,
    try_extract_static_range,
)
from ..tex_compiler.type_checker import TEXType, CHANNEL_MAP, TYPE_NAME_MAP
from .stdlib import TEXStdlib, SAFE_EPSILON, VEC_CHANNELS

# Hard limit on for-loop iterations to prevent infinite loops
MAX_LOOP_ITERATIONS = 1024

# Hard limit on user function call depth to prevent stack overflow
MAX_CALL_DEPTH = 64

# Default builtin values for scalar (no-spatial-context) mode
_SCALAR_BUILTIN_DEFAULTS = {"ix": 0.0, "iy": 0.0, "u": 0.0, "v": 0.0,
                            "iw": 1.0, "ih": 1.0, "px": 1.0, "py": 1.0,
                            "fi": 0.0, "fn": 1.0}

_VEC_CHANNELS = VEC_CHANNELS  # alias for local use



class InterpreterError(Exception):
    def __init__(self, message: str, loc: SourceLoc = None, *, source: str = "",
                 code: str = "E6000", hint: str = ""):
        self.loc = loc
        self.diagnostic = None
        self._raw_message = message
        self._source = source
        self._code = code
        self._hint = hint
        prefix = f"[{loc}] " if loc else ""
        super().__init__(f"{prefix}{message}")

    def _build_diagnostic(self):
        if self.diagnostic is not None:
            return
        from ..tex_compiler.diagnostics import make_diagnostic
        self.diagnostic = make_diagnostic(
            code=self._code,
            message=self._raw_message,
            loc=self.loc,
            source=self._source,
            hint=self._hint,
            phase="runtime",
        )


class _Break(Exception):
    """Internal signal for break statements inside for loops."""
    pass


class _Continue(Exception):
    """Internal signal for continue statements inside for loops."""
    pass


class _ReturnSignal(Exception):
    """Internal signal to unwind the call stack on return."""
    __slots__ = ("value",)
    def __init__(self, value):
        self.value = value


class Interpreter:
    """
    Evaluates a TEX AST against concrete tensor inputs.

    Usage:
        interp = Interpreter()
        result = interp.execute(ast, bindings, type_map)

    The interpreter is reusable across executions. State is fully reset
    at the start of each execute() call. The dispatch tables and stdlib
    function registry are built once per instance and reused.
    """

    # ── Class-level caches (built once, shared across all instances) ──
    _stdlib_functions: dict[str, callable] | None = None

    @classmethod
    def _get_stdlib(cls) -> dict[str, callable]:
        if cls._stdlib_functions is None:
            cls._stdlib_functions = TEXStdlib.get_functions()
        return cls._stdlib_functions

    def __init__(self):
        self.env: dict[str, torch.Tensor] = {}
        self.bindings: dict[str, torch.Tensor] = {}
        self.type_map: dict[int, TEXType] = {}
        self.functions: dict[str, callable] = self._get_stdlib()
        self.device: torch.device = torch.device("cpu")
        self.spatial_shape: tuple[int, int, int] | None = None  # (B, H, W)
        self._array_meta: dict[str, int] = {}  # var_name -> array size
        self._source: str = ""  # Original source code for diagnostics
        self._literal_cache: dict[tuple[float, str], torch.Tensor] = {}  # (value, device) -> tensor
        self._inplace_ready: set[str] = set()  # vars that have been cloned for in-place ops
        self._user_functions: dict[str, FunctionDef] = {}
        self._call_depth: int = 0
        self._builtins_cache_key: tuple | None = None
        self._builtins_cache_env: dict[str, torch.Tensor] = {}
        self._assigned_vars_cache: dict[int, tuple[set[str], set[str]]] = {}

        # ── Dispatch tables (O(1) lookup, built once per instance) ──
        self._stmt_dispatch: dict[type, callable] = {
            VarDecl: self._exec_var_decl,
            ArrayDecl: self._exec_array_decl,
            Assignment: self._exec_assignment,
            IfElse: self._exec_if_else,
            ForLoop: self._exec_for_loop,
            WhileLoop: self._exec_while_loop,
            ExprStatement: lambda node: self._eval(node.expr),
            ParamDecl: lambda node: None,
            FunctionDef: self._exec_function_def,
            ReturnStmt: self._exec_return_stmt,
        }
        self._eval_dispatch: dict[type, callable] = {
            NumberLiteral: self._eval_number_literal,
            StringLiteral: lambda node: node.value,
            Identifier: self._eval_identifier,
            BindingRef: self._eval_binding,
            ChannelAccess: self._eval_channel_access,
            BinOp: self._eval_binop,
            UnaryOp: self._eval_unary,
            TernaryOp: self._eval_ternary,
            FunctionCall: self._eval_function_call,
            VecConstructor: self._eval_vec_constructor,
            MatConstructor: self._eval_mat_constructor,
            CastExpr: self._eval_cast,
            ArrayIndexAccess: self._eval_array_index,
            ArrayLiteral: self._eval_array_literal,
            BindingIndexAccess: self._eval_binding_index,
            BindingSampleAccess: self._eval_binding_sample,
        }

    def execute(
        self,
        program: Program,
        bindings: dict[str, torch.Tensor | float | int],
        type_map: dict[int, TEXType],
        device: torch.device | str = "cpu",
        latent_channel_count: int = 0,
        output_names: list[str] | None = None,
        precision: str = "fp32",
        used_builtins: frozenset[str] | None = None,
        source: str = "",
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Execute a TEX program.

        Args:
            program: Parsed and type-checked AST
            bindings: @ binding values (name -> tensor or scalar)
            type_map: AST node id -> TEXType from type checker
            device: Target device
            latent_channel_count: Channel count for latent inputs
            output_names: If provided, return dict of named outputs instead of single @OUT
            precision: "fp32" (default), "fp16", or "bf16" for reduced precision
            used_builtins: Pre-computed frozenset of builtin names (from cache).
                If None, will be computed by walking the AST.

        Returns:
            If output_names is None: the value of @OUT (backward compat)
            If output_names is provided: dict mapping name -> value for each output
        """
        self._source = source
        # Always use inference_mode — eliminates gradient tracking overhead
        # even for internal tensor ops (coordinate builtins, constants, etc.)
        with torch.inference_mode():
            return self._execute_inner(program, bindings, type_map, device,
                                       latent_channel_count, output_names,
                                       precision, used_builtins=used_builtins)

    def execute_tiled(
        self,
        program: Program,
        bindings: dict[str, torch.Tensor | float | int],
        type_map: dict[int, TEXType],
        device: torch.device | str = "cpu",
        latent_channel_count: int = 0,
        output_names: list[str] | None = None,
        precision: str = "fp32",
        tile_height: int = 2048,
        used_builtins: frozenset[str] | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Execute with tiling for large images to prevent OOM.

        Splits images into horizontal strips of `tile_height` rows.
        Coordinate builtins (u, v, ix, iy) reflect full-image positions.
        Sampling functions receive the FULL source image so neighbourhood
        lookups work correctly across tile boundaries.

        For images smaller than tile_height, falls through to regular execute.
        """
        # Find spatial dims from bindings
        full_H = full_W = B = 0
        for val in bindings.values():
            if isinstance(val, torch.Tensor) and val.dim() >= 3:
                B, full_H, full_W = val.shape[0], val.shape[1], val.shape[2]
                break
        if full_H <= tile_height:
            return self.execute(program, bindings, type_map, device,
                                latent_channel_count, output_names, precision,
                                used_builtins=used_builtins)

        # Process in horizontal strips
        tiles: list[torch.Tensor] = []
        tile_dicts: list[dict] = []  # for multi-output mode
        for y_start in range(0, full_H, tile_height):
            y_end = min(y_start + tile_height, full_H)

            # Slice spatial bindings to the tile region for OUTPUT computation,
            # but keep full-resolution bindings available for sampling.
            # Strategy: slice bindings that are consumed as @OUT source,
            # but sampling functions always receive the full image.
            # We achieve this by slicing bindings and passing tile_offset.
            tile_bindings = {}
            for name, val in bindings.items():
                if isinstance(val, torch.Tensor) and val.dim() >= 3:
                    tile_bindings[name] = val[:, y_start:y_end]
                else:
                    tile_bindings[name] = val

            # Execute tile with offset info for correct coordinate generation
            with torch.inference_mode():
                result = self._execute_inner(
                    program, tile_bindings, type_map, device,
                    latent_channel_count, output_names, precision,
                    tile_offset=(y_start, full_H, full_W),
                    used_builtins=used_builtins,
                )

            if isinstance(result, dict):
                tile_dicts.append(result)
            else:
                tiles.append(result)

        if tile_dicts:
            # Multi-output: concatenate each output along H
            merged = {}
            for key in tile_dicts[0]:
                vals = [d[key] for d in tile_dicts]
                if isinstance(vals[0], torch.Tensor):
                    merged[key] = torch.cat(vals, dim=1)
                else:
                    merged[key] = vals[0]
            return merged

        return torch.cat(tiles, dim=1)

    # Precision dtype mapping
    _PRECISION_DTYPES = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }

    def _execute_inner(
        self,
        program: Program,
        bindings: dict[str, torch.Tensor | float | int],
        type_map: dict[int, TEXType],
        device: torch.device | str,
        latent_channel_count: int,
        output_names: list[str] | None,
        precision: str = "fp32",
        tile_offset: tuple[int, int, int] | None = None,
        used_builtins: frozenset[str] | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self._device_str = str(self.device)  # Cache for literal cache key
        self.type_map = type_map
        self.latent_channel_count = latent_channel_count
        self._dtype = self._PRECISION_DTYPES.get(precision, torch.float32)
        self.env = {}
        self.bindings = {}
        self._array_meta = {}
        self._inplace_ready.clear()
        self._literal_cache.clear()
        self._user_functions.clear()
        self._assigned_vars_cache.clear()

        # Process bindings — skip redundant .to() when already on target device/dtype
        target_device = self.device
        target_dtype = self._dtype
        needs_dtype_cast = target_dtype != torch.float32
        for name, value in bindings.items():
            if isinstance(value, str):
                self.bindings[name] = value  # strings pass through as Python str
            elif isinstance(value, torch.Tensor):
                t = value
                if t.device != target_device:
                    t = t.to(target_device)
                if needs_dtype_cast and t.is_floating_point() and t.dtype != target_dtype:
                    t = t.to(target_dtype)
                self.bindings[name] = t
            elif isinstance(value, (list, tuple)):
                # Vec param defaults / converted widget values (e.g. [0.5, 0.3, 0.1])
                # Shape as [1, 1, 1, C] (channel-last spatial) for correct broadcasting
                t = torch.tensor(value, dtype=target_dtype, device=target_device)
                if t.dim() == 1 and t.shape[0] in (2, 3, 4):
                    t = t.view(1, 1, 1, -1)
                self.bindings[name] = t
            else:
                self.bindings[name] = torch.scalar_tensor(float(value), dtype=target_dtype, device=target_device)

        # Determine spatial context from image inputs
        self.spatial_shape = self._determine_spatial_shape()

        # Create built-in coordinate variables (lazy — only what the program uses)
        self._create_builtins(program, tile_offset=tile_offset, used_builtins=used_builtins)

        # Execute statements via tree-walking interpreter
        for stmt in program.statements:
            self._exec_stmt(stmt)

        # Collect outputs — skip precision conversion for the common fp32 case
        needs_upcast = self._dtype != torch.float32

        # Multi-output mode
        if output_names is not None:
            results = {}
            for name in output_names:
                if name in self.bindings:
                    val = self.bindings[name]
                    if needs_upcast and isinstance(val, torch.Tensor) and val.is_floating_point():
                        val = val.float()
                    results[name] = val
                else:
                    raise InterpreterError(
                        f"Output @{name} was not assigned by the TEX program",
                        source=self._source, code="E6001",
                        hint=f"Add an assignment like '@{name} = <expr>;' to your program.",
                    )
            return results

        # Single-output backward compat
        if "OUT" not in self.bindings:
            raise InterpreterError(
                "This program needs at least one output assignment",
                source=self._source, code="E6001",
                hint="Add '@OUT = <expr>;' or use named outputs like '@result = <expr>;'.",
            )

        val = self.bindings["OUT"]
        if needs_upcast and isinstance(val, torch.Tensor) and val.is_floating_point():
            val = val.float()
        return val

    def _determine_spatial_shape(self) -> tuple[int, int, int] | None:
        """Find the spatial dimensions from image inputs."""
        for name, value in self.bindings.items():
            if name == "OUT":
                continue
            if isinstance(value, str):
                continue  # strings have no spatial shape
            if isinstance(value, torch.Tensor) and value.dim() >= 3:
                # IMAGE: [B, H, W, C] or MASK: [B, H, W]
                return (value.shape[0], value.shape[1], value.shape[2])
        return None

    def _create_builtins(self, program: Program, tile_offset: tuple[int, int, int] | None = None,
                         used_builtins: frozenset[str] | None = None):
        """Create built-in variables lazily — only allocate what the program uses.

        Builtins use compact broadcast-friendly shapes instead of full
        [B, H, W] expansion. PyTorch broadcasts automatically in ops.

        When tile_offset is provided (y_start, full_H, full_W), coordinates
        reflect position in the full image rather than the tile.
        """
        used = used_builtins if used_builtins is not None else _collect_identifiers(program)

        # Cache builtins: reuse tensors when spatial config hasn't changed
        cache_key = (self.spatial_shape, self._device_str, self._dtype, tile_offset, used, self.latent_channel_count)
        if cache_key == self._builtins_cache_key:
            self.env.update(self._builtins_cache_env)
            return

        if self.spatial_shape:
            B, H, W = self.spatial_shape

            # For tiled execution, use full image dimensions for coordinate normalization
            if tile_offset is not None:
                y_start, full_H, full_W = tile_offset
            else:
                y_start, full_H, full_W = 0, H, W

            # ix: pixel x-coordinate [0, W-1] — compact shape [1, 1, W]
            # u/v use expand() which creates a view (no memory copy) at [B,H,W]
            # for compatibility with torch.stack in sampling functions.
            if "ix" in used or "u" in used:
                ix = torch.arange(W, dtype=self._dtype, device=self.device).view(1, 1, W)
                if "ix" in used:
                    self.env["ix"] = ix
                if "u" in used:
                    self.env["u"] = (ix / max(full_W - 1, 1)).expand(B, H, W)

            # iy: pixel y-coordinate — offset by y_start for tiling
            if "iy" in used or "v" in used:
                iy = torch.arange(y_start, y_start + H, dtype=self._dtype, device=self.device).view(1, H, 1)
                if "iy" in used:
                    self.env["iy"] = iy
                if "v" in used:
                    self.env["v"] = (iy / max(full_H - 1, 1)).expand(B, H, W)

            # iw, ih: FULL image dimensions (not tile dimensions)
            if "iw" in used:
                self.env["iw"] = torch.scalar_tensor(float(full_W), dtype=self._dtype, device=self.device)
            if "ih" in used:
                self.env["ih"] = torch.scalar_tensor(float(full_H), dtype=self._dtype, device=self.device)

            # px, py: pixel step in UV space (1/width, 1/height)
            if "px" in used:
                self.env["px"] = torch.scalar_tensor(1.0 / max(full_W, 1), dtype=self._dtype, device=self.device)
            if "py" in used:
                self.env["py"] = torch.scalar_tensor(1.0 / max(full_H, 1), dtype=self._dtype, device=self.device)

            if "fi" in used:
                self.env["fi"] = torch.arange(B, dtype=self._dtype, device=self.device).view(B, 1, 1)
            if "fn" in used:
                self.env["fn"] = torch.scalar_tensor(float(B), dtype=self._dtype, device=self.device)
        else:
            # No spatial context — pure scalar mode (only create what's used)
            for name, val in _SCALAR_BUILTIN_DEFAULTS.items():
                if name in used:
                    self.env[name] = torch.scalar_tensor(val, device=self.device)

        if "PI" in used:
            self.env["PI"] = torch.scalar_tensor(math.pi, dtype=self._dtype, device=self.device)
        if "TAU" in used:
            self.env["TAU"] = torch.scalar_tensor(math.tau, dtype=self._dtype, device=self.device)
        if "E" in used:
            self.env["E"] = torch.scalar_tensor(math.e, dtype=self._dtype, device=self.device)
        if "ic" in used:
            self.env["ic"] = torch.scalar_tensor(float(self.latent_channel_count), dtype=self._dtype, device=self.device)

        # Store only builtin tensors in cache (not user variables)
        self._builtins_cache_key = cache_key
        self._builtins_cache_env = {k: v for k, v in self.env.items() if k in _BUILTIN_NAMES}

    # -- Statement execution --------------------------------------------

    def _exec_stmt(self, node: ASTNode):
        # Fast path: dispatch table lookup (O(1) instead of isinstance chain)
        handler = self._stmt_dispatch.get(type(node))
        if handler is not None:
            handler(node)
            return
        # Break/continue are rare — check after dispatch table
        if isinstance(node, BreakStmt):
            raise _Break()
        if isinstance(node, ContinueStmt):
            raise _Continue()
        raise InterpreterError(
            f"Expected a recognized statement, but found '{type(node).__name__}'",
            node.loc, source=self._source, code="E6002",
            hint="This node type is not supported as a statement.",
        )

    def _exec_var_decl(self, node: VarDecl):
        if node.initializer:
            value = self._eval(node.initializer)
        else:
            # Default initialize based on type
            declared = self.type_map.get(id(node), TEXType.FLOAT)
            value = self._default_value(declared)
        self.env[node.name] = value
        # New declaration invalidates in-place readiness (value may be aliased)
        self._inplace_ready.discard(node.name)

    def _exec_array_decl(self, node: ArrayDecl):
        """Execute an array declaration."""
        is_vec = TYPE_NAME_MAP.get(node.element_type_name, TEXType.FLOAT).is_vector
        is_string = node.element_type_name == "string"

        # -- String arrays: Python list, not tensor -----------------------
        if is_string:
            if node.initializer:
                if isinstance(node.initializer, ArrayLiteral):
                    value = [self._eval(elem) for elem in node.initializer.elements]
                else:
                    src = self._eval(node.initializer)
                    value = list(src) if isinstance(src, list) else [str(src)]
                size = len(value)
            else:
                size = node.size
                value = [""] * size
            self.env[node.name] = value
            self._array_meta[node.name] = size
            return

        # -- Tensor arrays (float, int, vec3, vec4) -----------------------
        if node.initializer:
            if isinstance(node.initializer, ArrayLiteral):
                elements = [self._eval(elem) for elem in node.initializer.elements]
                size = len(elements)
                spatial = self.spatial_shape
                if spatial:
                    expanded = [_ensure_spatial(e, spatial) for e in elements]
                else:
                    expanded = [e if isinstance(e, torch.Tensor) else torch.scalar_tensor(float(e), dtype=self._dtype, device=self.device) for e in elements]
                # For vec3/vec4 arrays, promote elements to consistent channel count
                if is_vec:
                    channels = TYPE_NAME_MAP[node.element_type_name].channels
                    expanded = [self._promote_to_channels(e, channels) for e in expanded]
                value = torch.stack(expanded, dim=-1)
                # For vec arrays, stacking puts channels before N: [..., C, N]
                # Transpose to [..., N, C]
                if is_vec:
                    value = value.transpose(-2, -1)
            else:
                # Array copy from another variable
                value = self._eval(node.initializer).clone()
                size = value.shape[-2] if is_vec else value.shape[-1]
        else:
            # Zero-initialized array
            size = node.size
            if is_vec:
                channels = TYPE_NAME_MAP[node.element_type_name].channels
                if self.spatial_shape:
                    B, H, W = self.spatial_shape
                    value = torch.zeros(B, H, W, size, channels, dtype=self._dtype, device=self.device)
                else:
                    value = torch.zeros(size, channels, dtype=self._dtype, device=self.device)
            else:
                if self.spatial_shape:
                    B, H, W = self.spatial_shape
                    value = torch.zeros(B, H, W, size, dtype=self._dtype, device=self.device)
                else:
                    value = torch.zeros(size, dtype=self._dtype, device=self.device)

        self.env[node.name] = value
        self._array_meta[node.name] = size

    def _promote_to_channels(self, tensor: torch.Tensor, channels: int) -> torch.Tensor:
        """Ensure a tensor has the correct number of channels for a vec array element."""
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.scalar_tensor(float(tensor), dtype=self._dtype, device=self.device)
        if tensor.dim() >= 1 and tensor.shape[-1] in _VEC_CHANNELS:
            c = tensor.shape[-1]
            if c == channels:
                return tensor
            if c < channels:
                # Pad with 1.0 (alpha) to reach target channels
                pad = torch.ones(*tensor.shape[:-1], channels - c, device=tensor.device)
                return torch.cat([tensor, pad], dim=-1)
            # Truncate extra channels
            return tensor[..., :channels]
        # Scalar — broadcast to vec
        return tensor.unsqueeze(-1).expand(*tensor.shape, channels)

    # In-place operation mapping: op -> (tensor_method, is_commutative)
    _INPLACE_OPS = {
        "+": ("add_", True),
        "-": ("sub_", False),
        "*": ("mul_", True),
        "/": ("div_", False),
    }

    def _ensure_inplace_ready(self, name: str) -> torch.Tensor:
        """Ensure a variable is safe for in-place mutation (clone-on-first-write)."""
        current = self.env[name]
        if name not in self._inplace_ready:
            current = current.clone()
            self.env[name] = current
            self._inplace_ready.add(name)
        return current

    def _exec_assignment(self, node: Assignment):
        target = node.target

        # In-place optimization: x = x OP expr or x = expr OP x
        # Reuses x's memory instead of allocating a new tensor
        if isinstance(target, Identifier) and isinstance(node.value, BinOp):
            rhs = node.value
            op_info = self._INPLACE_OPS.get(rhs.op)
            if op_info is not None:
                method_name, commutative = op_info
                name = target.name
                is_div = rhs.op == "/"
                # Pattern 1: x = x OP expr
                if isinstance(rhs.left, Identifier) and rhs.left.name == name:
                    current = self.env.get(name)
                    if current is not None and current.__class__ is torch.Tensor:
                        other_val = self._eval(rhs.right)
                        if other_val.__class__ is torch.Tensor and (
                            other_val.dim() == 0 or other_val.shape == current.shape
                        ):
                            current = self._ensure_inplace_ready(name)
                            if is_div:
                                # Safe in-place division: x /= where(rhs==0, eps, rhs)
                                current.div_(torch.where(other_val == 0, SAFE_EPSILON, other_val))
                            else:
                                getattr(current, method_name)(other_val)
                            return
                # Pattern 2: x = expr OP x (only for commutative ops)
                elif commutative and isinstance(rhs.right, Identifier) and rhs.right.name == name:
                    current = self.env.get(name)
                    if current is not None and current.__class__ is torch.Tensor:
                        other_val = self._eval(rhs.left)
                        if other_val.__class__ is torch.Tensor and (
                            other_val.dim() == 0 or other_val.shape == current.shape
                        ):
                            current = self._ensure_inplace_ready(name)
                            getattr(current, method_name)(other_val)
                            return

        value = self._eval(node.value)

        if isinstance(target, Identifier):
            self.env[target.name] = value

        elif isinstance(target, BindingRef):
            self.bindings[target.name] = value

        elif isinstance(target, ChannelAccess):
            self._exec_channel_assign(target, value)

        elif isinstance(target, ArrayIndexAccess):
            self._exec_array_index_assign(target, value)

        elif isinstance(target, BindingIndexAccess):
            self._exec_scatter_write(target, value, op=node.op)

        else:
            raise InterpreterError(
                "This assignment target is not supported",
                node.loc, source=self._source, code="E6003",
                hint="Assignments work with variables, @bindings, channels (.r), array indices ([i]), and scatter writes (@OUT[x,y]).",
            )

    def _exec_channel_assign(self, target: ChannelAccess, value: torch.Tensor):
        """Handle assignment to a channel: `@A.r = expr;` or `color.rgb = expr;`"""
        base = self._eval(target.object)
        channels = target.channels

        if len(channels) == 1:
            idx = CHANNEL_MAP.get(channels)
            if idx is None:
                raise InterpreterError(
                    f"Expected a known channel name, but found '.{channels}'",
                    target.loc, source=self._source, code="E6004",
                    hint="Use one of: .r, .g, .b, .a (or .x, .y, .z, .w).",
                )
            result = base.clone()
            if result.dim() >= 1 and result.shape[-1] > idx:
                result[..., idx] = _ensure_spatial(value, result.shape[:-1])
            else:
                raise InterpreterError(
                    f"This tensor (shape {result.shape}) does not have a channel at index {idx}",
                    target.loc, source=self._source, code="E6004",
                    hint="The target needs enough channels for this assignment. Check if the variable is a vec3 or vec4.",
                )
        else:
            # Multi-channel assignment: .rgb, .xy, .rgba, etc.
            indices = [CHANNEL_MAP[ch] for ch in channels]
            result = base.clone()
            val = _ensure_spatial(value, result.shape[:-1]) if self.spatial_shape else value
            val_is_multi = isinstance(val, torch.Tensor) and val.dim() >= 1 and val.shape[-1] > 1
            for i, idx in enumerate(indices):
                if result.dim() >= 1 and result.shape[-1] > idx:
                    result[..., idx] = val[..., i] if val_is_multi else val

        # Write back to the correct location
        if isinstance(target.object, Identifier):
            self.env[target.object.name] = result
        elif isinstance(target.object, BindingRef):
            self.bindings[target.object.name] = result
        else:
            raise InterpreterError(
                "This channel assignment needs a variable or @binding as its target",
                target.loc, source=self._source, code="E6004",
                hint="Assign to a named variable's channel, e.g. 'color.r = 1.0;' or '@OUT.r = 1.0;'.",
            )

    def _exec_array_index_assign(self, target: ArrayIndexAccess, value):
        """Handle: arr[i] = expr;"""
        array = self._eval(target.array)
        index = self._eval(target.index)

        # String array (Python list)
        if isinstance(array, list):
            idx_int = max(0, min(int(round(index.item() if isinstance(index, torch.Tensor) else float(index))), len(array) - 1))
            result = list(array)
            result[idx_int] = value if isinstance(value, str) else str(value)
            if isinstance(target.array, Identifier):
                self.env[target.array.name] = result
            return

        # Vector array: dim 5 (spatial) or 2 (non-spatial) → [..., N, C]
        if array.dim() in (2, 5):
            arr_size = array.shape[-2]
            idx = _safe_array_index(index, arr_size)
            result = array.clone()

            if result.dim() == 2:
                # Non-spatial vector array [N, C]
                val_t = value if isinstance(value, torch.Tensor) else torch.scalar_tensor(float(value), dtype=self._dtype, device=self.device)
                result[idx] = val_t
            elif idx.dim() == 0:
                # Constant index: [..., N, C] → assign vec to [..., C]
                spatial_shape = result.shape[:-2]  # [B, H, W]
                result[..., idx.item(), :] = _ensure_spatial(value, spatial_shape)
            else:
                # Per-pixel assignment via scatter on dim=-2
                C = result.shape[-1]
                idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(*idx.shape, 1, C)
                if idx_exp.shape[:3] != result.shape[:3]:
                    idx_exp = idx_exp.expand(result.shape[:3] + (1, C))
                val_spatial = _ensure_spatial(value, result.shape[:-2])
                val_exp = val_spatial.unsqueeze(-2)
                result.scatter_(-2, idx_exp, val_exp)

        else:
            # Scalar array: dim 4 (spatial) or 1 (non-spatial) → [..., N]
            arr_size = array.shape[-1]
            idx = _safe_array_index(index, arr_size)
            result = array.clone()

            if result.dim() == 1:
                result[idx] = value if isinstance(value, torch.Tensor) else torch.scalar_tensor(float(value), dtype=self._dtype, device=self.device)
            elif idx.dim() == 0:
                spatial_shape = result.shape[:-1]
                result[..., idx.item()] = _ensure_spatial(value, spatial_shape)
            else:
                idx_expanded = idx.unsqueeze(-1)
                if idx_expanded.shape[:3] != result.shape[:3]:
                    idx_expanded = idx_expanded.expand(result.shape[:3] + (1,))
                val_spatial = _ensure_spatial(value, result.shape[:-1])
                val_expanded = val_spatial.unsqueeze(-1)
                result.scatter_(-1, idx_expanded, val_expanded)

        # Write back to the correct location
        if isinstance(target.array, Identifier):
            self.env[target.array.name] = result
        elif isinstance(target.array, BindingRef):
            self.bindings[target.array.name] = result
        else:
            raise InterpreterError(
                "This array index assignment needs a variable or @binding as its target",
                target.loc, source=self._source, code="E6005",
                hint="Assign to a named array element, e.g. 'arr[i] = value;'.",
            )

    def _exec_scatter_write(self, target: BindingIndexAccess, value, op=None):
        """Handle @OUT[px, py] = value or @OUT[px, py] += value (scatter write)."""
        name = target.binding.name
        args = [self._eval(a) for a in target.args]
        px, py = args[0], args[1]
        frame = args[2] if len(args) == 3 else None

        # Determine channel count from value
        if isinstance(value, torch.Tensor) and value.dim() >= 1 and value.shape[-1] in _VEC_CHANNELS:
            C = value.shape[-1]
        else:
            C = 1

        # Get or create output buffer — must be at least [B, H, W] for scatter indexing
        buf = self.bindings.get(name)
        if self.spatial_shape:
            B, H, W = self.spatial_shape
        else:
            B, H, W = 1, 1, 1

        needs_new_buf = (buf is None or not isinstance(buf, torch.Tensor) or buf.dim() < 3)
        if needs_new_buf:
            # Existing buffer too small — create spatial buffer, preserving old value if possible
            if C > 1:
                new_buf = torch.zeros(B, H, W, C, dtype=self._dtype, device=self.device)
            else:
                new_buf = torch.zeros(B, H, W, dtype=self._dtype, device=self.device)
            if isinstance(buf, torch.Tensor):
                new_buf[...] = buf
            buf = new_buf
            self.bindings[name] = buf
        elif not buf.is_contiguous():
            # Clone expanded tensors (e.g. from vec3(1.0) broadcast) so index_put_ works
            buf = buf.clone()
            self.bindings[name] = buf

        # Clamp coordinates
        ix_t = torch.clamp(torch.floor(_ensure_spatial(px, (B, H, W)) if self.spatial_shape else px).long(), 0, W - 1)
        iy_t = torch.clamp(torch.floor(_ensure_spatial(py, (B, H, W)) if self.spatial_shape else py).long(), 0, H - 1)
        val = _ensure_spatial(value, (B, H, W)) if self.spatial_shape else value

        # Build flat indices
        if frame is not None:
            batch_idx = torch.clamp(
                (_ensure_spatial(frame, (B, H, W)) if self.spatial_shape else frame).long(),
                0, B - 1
            )
        else:
            batch_idx = torch.arange(B, device=self.device).view(B, 1, 1).expand(B, H, W)

        flat_b = batch_idx.contiguous().reshape(-1)
        flat_y = iy_t.contiguous().reshape(-1)
        flat_x = ix_t.contiguous().reshape(-1)

        if C > 1 and buf.dim() == 4:
            flat_v = val.contiguous().reshape(-1, C)
        else:
            flat_v = val.reshape(-1) if isinstance(val, torch.Tensor) and val.dim() > 0 else val

        idx = (flat_b, flat_y, flat_x)
        if op is None:
            buf[idx] = flat_v
        elif op == "+":
            buf.index_put_(idx, flat_v, accumulate=True)
        elif op == "-":
            buf.index_put_(idx, -flat_v, accumulate=True)
        elif op == "*":
            buf[idx] *= flat_v
        elif op == "/":
            buf[idx] /= torch.where(flat_v == 0, SAFE_EPSILON, flat_v)

    @staticmethod
    def _collect_assigned_vars(stmts: list[ASTNode]) -> tuple[set[str], set[str]]:
        """Collect variable names and binding names assigned in a statement list.

        Returns (env_vars, binding_names) — recursively walks into nested
        if/else and loop bodies to find all possible assignments.
        """
        env_vars: set[str] = set()
        binding_names: set[str] = set()
        for stmt in stmts:
            if isinstance(stmt, Assignment):
                target = stmt.target
                if isinstance(target, Identifier):
                    env_vars.add(target.name)
                elif isinstance(target, BindingRef):
                    binding_names.add(target.name)
                elif isinstance(target, ChannelAccess):
                    obj = target.object
                    if isinstance(obj, Identifier):
                        env_vars.add(obj.name)
                    elif isinstance(obj, BindingRef):
                        binding_names.add(obj.name)
                elif isinstance(target, ArrayIndexAccess):
                    arr = target.array
                    if isinstance(arr, Identifier):
                        env_vars.add(arr.name)
                    elif isinstance(arr, BindingRef):
                        binding_names.add(arr.name)
            elif isinstance(stmt, VarDecl):
                env_vars.add(stmt.name)
            elif isinstance(stmt, ArrayDecl):
                env_vars.add(stmt.name)
            elif isinstance(stmt, IfElse):
                e1, b1 = Interpreter._collect_assigned_vars(stmt.then_body)
                e2, b2 = Interpreter._collect_assigned_vars(stmt.else_body)
                env_vars |= e1 | e2
                binding_names |= b1 | b2
            elif isinstance(stmt, ForLoop):
                e, b = Interpreter._collect_assigned_vars(stmt.body)
                env_vars |= e
                binding_names |= b
            elif isinstance(stmt, WhileLoop):
                e, b = Interpreter._collect_assigned_vars(stmt.body)
                env_vars |= e
                binding_names |= b
        return env_vars, binding_names

    def _exec_if_else(self, node: IfElse):
        """
        Execute if/else.

        Scalar conditions (0-dim tensors, e.g. loop counters): use short-circuit
        evaluation — only execute the true branch. This is required for
        break/continue to work correctly inside if blocks.

        Spatial conditions (B, H, W tensors): use vectorized both-branch
        evaluation with torch.where merging. Uses selective cloning — only
        variables actually assigned in branches are cloned/merged.
        """
        cond = self._eval(node.condition)

        # Scalar condition: short-circuit (supports break/continue)
        if cond.dim() == 0:
            if cond.item() > 0.5:
                for stmt in node.then_body:
                    self._exec_stmt(stmt)
            elif node.else_body:
                for stmt in node.else_body:
                    self._exec_stmt(stmt)
            return

        # Spatial condition: vectorized both-branch evaluation
        self._exec_spatial_if(node, cond)

    def _snapshot_vars(
        self, keys: set[str], source: dict,
    ) -> dict[str, Any]:
        """Clone tensor values (or copy non-tensors) for the given keys."""
        snap = {}
        for k in keys:
            v = source.get(k)
            if v is not None:
                snap[k] = v.clone() if isinstance(v, torch.Tensor) else v
        return snap

    @staticmethod
    def _merge_branch_vars(
        cond_bool: torch.Tensor, cond_scalar_box: list,
        target: dict, keys: set[str],
        then_vals: dict, else_vals: dict,
    ) -> None:
        """Merge then/else branch values into *target* using torch.where.

        *cond_scalar_box* is a single-element list used as a lazy cache for
        the majority-vote scalar (needed for string merges).
        """
        for key in keys:
            then_val = then_vals.get(key)
            else_val = else_vals.get(key)
            if then_val is not None and else_val is not None:
                if isinstance(then_val, torch.Tensor) and isinstance(else_val, torch.Tensor):
                    target[key] = _tensor_where(cond_bool, then_val, else_val)
                elif isinstance(then_val, str) or isinstance(else_val, str):
                    if not cond_scalar_box:
                        cond_scalar_box.append(
                            cond_bool.float().mean().item() > 0.5
                        )
                    target[key] = then_val if cond_scalar_box[0] else else_val
                else:
                    target[key] = then_val
            elif then_val is not None:
                target[key] = then_val
            elif else_val is not None:
                target[key] = else_val

    def _exec_spatial_if(self, node: IfElse, cond: torch.Tensor):
        """Execute a spatial (vectorized) if/else with torch.where merging."""
        # Selective cloning: only snapshot variables that are assigned in branches.
        # Cache by node id — AST is immutable, so results never change.
        node_id = id(node)
        cached = self._assigned_vars_cache.get(node_id)
        if cached is not None:
            modified_env, modified_bindings = cached
        else:
            modified_env, modified_bindings = self._collect_assigned_vars(node.then_body)
            if node.else_body:
                e2, b2 = self._collect_assigned_vars(node.else_body)
                modified_env |= e2
                modified_bindings |= b2
            self._assigned_vars_cache[node_id] = (modified_env, modified_bindings)

        # Snapshot only modified variables that already exist
        env_snapshot = self._snapshot_vars(modified_env, self.env)
        bindings_snapshot = self._snapshot_vars(modified_bindings, self.bindings)
        has_arrays = bool(self._array_meta)
        if has_arrays:
            meta_snapshot = dict(self._array_meta)

        # Execute then-branch and capture modified state
        for stmt in node.then_body:
            self._exec_stmt(stmt)
        then_env = {k: self.env.get(k) for k in modified_env}
        then_bindings = {k: self.bindings.get(k) for k in modified_bindings}
        if has_arrays:
            then_meta = dict(self._array_meta)

        # Restore snapshot and execute else-branch
        self.env.update(env_snapshot)
        self.bindings.update(bindings_snapshot)
        if has_arrays:
            self._array_meta = dict(meta_snapshot)

        if node.else_body:
            for stmt in node.else_body:
                self._exec_stmt(stmt)
            else_env = {k: self.env.get(k) for k in modified_env}
            else_bindings = {k: self.bindings.get(k) for k in modified_bindings}
        else:
            else_env = env_snapshot
            else_bindings = bindings_snapshot

        # Merge using torch.where (tensors) or scalar majority-vote (strings)
        cond_bool = (cond > 0.5) if cond.is_floating_point() else cond.bool()
        cond_scalar_box: list = []  # lazy cache for string merge
        self._merge_branch_vars(
            cond_bool, cond_scalar_box,
            self.env, modified_env, then_env, else_env,
        )
        self._merge_branch_vars(
            cond_bool, cond_scalar_box,
            self.bindings, modified_bindings, then_bindings, else_bindings,
        )

        # Merge array metadata from then-branch
        if has_arrays:
            self._array_meta.update(then_meta)

    def _loop_cond_true(self, cond_node: ASTNode) -> bool:
        """Evaluate a loop condition and return True if the loop should continue."""
        cond = self._eval(cond_node)
        if cond.dim() == 0:
            return cond.item() > 0.5
        return (cond > 0.5).any().item()

    def _exec_loop_body(self, body: list[ASTNode]) -> bool:
        """Execute loop body statements. Returns True if break was hit."""
        try:
            for stmt in body:
                self._exec_stmt(stmt)
        except _Break:
            return True
        except _Continue:
            pass
        return False

    def _raise_loop_limit(self, loop_type: str, loc):
        raise InterpreterError(
            f"This {loop_type} loop ran {MAX_LOOP_ITERATIONS} iterations without finishing",
            loc, source=self._source, code="E6010",
            hint=f"Loops are capped at {MAX_LOOP_ITERATIONS} iterations to prevent hangs. "
                 "Make sure your loop condition will eventually become false.",
        )

    def _exec_for_loop(self, node: ForLoop):
        """
        Execute a bounded for loop sequentially.

        Each iteration runs the loop body as vectorized tensor operations.
        The loop variable is a scalar that gets updated each iteration.
        Hard limit of MAX_LOOP_ITERATIONS to prevent infinite loops.

        Optimization: for fully static loops like `for (int i = 0; i < N; i++)`
        where init, condition, and update are all literal-based, pre-compute
        the iteration range as Python range() — zero .item() GPU→CPU syncs.
        """
        # Try fully static loop: pre-compute range() from init/cond/update literals
        static_range = self._try_extract_static_range(node)

        if static_range is not None:
            loop_var, iter_range = static_range

            if len(iter_range) > MAX_LOOP_ITERATIONS:
                raise InterpreterError(
                    f"This for loop would run {len(iter_range)} iterations, which exceeds the limit of {MAX_LOOP_ITERATIONS}",
                    node.loc, source=self._source, code="E6010",
                    hint=f"Loops are capped at {MAX_LOOP_ITERATIONS} iterations to prevent hangs. "
                         "Consider reducing your range or processing in smaller batches.",
                )

            # Pre-allocate all loop variable tensors at once.
            # torch.arange + unbind is faster than per-iteration torch.tensor().
            dtype = self._dtype
            device = self.device
            n = len(iter_range)
            start = iter_range.start
            step = iter_range.step
            loop_tensors = torch.arange(start, start + n * step, step,
                                        dtype=dtype, device=device)
            # unbind(0) returns a tuple of 0-d tensors — no per-iteration allocation
            loop_values = loop_tensors.unbind(0)
            env = self.env
            inplace_discard = self._inplace_ready.discard

            for val in loop_values:
                env[loop_var] = val
                inplace_discard(loop_var)
                if self._exec_loop_body(node.body):
                    break
        else:
            # General case — execute init, evaluate condition each iteration
            self._exec_stmt(node.init)
            iteration = 0
            while iteration < MAX_LOOP_ITERATIONS:
                if not self._loop_cond_true(node.condition):
                    break
                if self._exec_loop_body(node.body):
                    break
                self._exec_stmt(node.update)
                iteration += 1

            if iteration >= MAX_LOOP_ITERATIONS:
                self._raise_loop_limit("for", node.loc)

    def _try_extract_static_range(self, node: ForLoop) -> tuple[str, range] | None:
        """Try to extract a fully static loop as a Python range().

        Delegates to the shared ``try_extract_static_range`` in ast_nodes
        and wraps the (var, start, stop, step) tuple into a range object.
        """
        result = try_extract_static_range(node)
        if result is None:
            return None
        loop_var, start, end, step = result
        try:
            return (loop_var, range(start, end, step))
        except ValueError:
            return None

    def _exec_while_loop(self, node: WhileLoop):
        """Execute a bounded while loop. Hard limit of MAX_LOOP_ITERATIONS."""
        iteration = 0
        while iteration < MAX_LOOP_ITERATIONS:
            if not self._loop_cond_true(node.condition):
                break
            if self._exec_loop_body(node.body):
                break
            iteration += 1

        if iteration >= MAX_LOOP_ITERATIONS:
            self._raise_loop_limit("while", node.loc)

    # -- Expression evaluation ------------------------------------------

    def _eval(self, node: ASTNode) -> torch.Tensor | str:
        # Fast path: dispatch table lookup (O(1) instead of isinstance chain)
        handler = self._eval_dispatch.get(type(node))
        if handler is not None:
            return handler(node)
        raise InterpreterError(
            f"Expected a recognized expression, but found '{type(node).__name__}'",
            node.loc, source=self._source, code="E6002",
            hint="This node type is not supported as an expression.",
        )

    def _eval_number_literal(self, node: NumberLiteral) -> torch.Tensor:
        key = (node.value, self._device_str, self._dtype)
        cached = self._literal_cache.get(key)
        if cached is not None:
            return cached
        t = torch.scalar_tensor(node.value, dtype=self._dtype, device=self.device)
        self._literal_cache[key] = t
        return t

    def _eval_array_literal(self, node: ArrayLiteral) -> torch.Tensor:
        elements = [self._eval(elem) for elem in node.elements]
        if self.spatial_shape:
            expanded = [_ensure_spatial(e, self.spatial_shape) for e in elements]
        else:
            expanded = [e if isinstance(e, torch.Tensor) else torch.scalar_tensor(float(e), dtype=self._dtype, device=self.device) for e in elements]
        return torch.stack(expanded, dim=-1)

    def _eval_identifier(self, node: Identifier) -> torch.Tensor:
        val = self.env.get(node.name)
        if val is not None:
            return val
        raise InterpreterError(
            f"Variable '{node.name}' is not defined",
            node.loc, source=self._source, code="E6020",
            hint=f"Make sure '{node.name}' is declared before it is used. Check for typos in the variable name.",
        )

    def _eval_binding(self, node: BindingRef) -> torch.Tensor:
        val = self.bindings.get(node.name)
        if val is not None:
            return val
        raise InterpreterError(
            f"Input '@{node.name}' is not connected",
            node.loc, source=self._source, code="E6021",
            hint=f"Make sure a node output is wired to the '@{node.name}' input, or check for typos in the binding name.",
        )

    def _eval_channel_access(self, node: ChannelAccess) -> torch.Tensor:
        base = self._eval(node.object)
        channels = node.channels

        if len(channels) == 1:
            idx = CHANNEL_MAP.get(channels)
            if idx is None:
                raise InterpreterError(
                    f"Expected a known channel name, but found '.{channels}'",
                    node.loc, source=self._source, code="E6030",
                    hint="Use one of: .r, .g, .b, .a (or .x, .y, .z, .w).",
                )
            if base.dim() >= 1 and base.shape[-1] > idx:
                return base[..., idx]
            raise InterpreterError(
                f"This value (shape {base.shape}) does not have a channel at '.{channels}' (index {idx})",
                node.loc, source=self._source, code="E6030",
                hint="The value may be a scalar or have fewer channels than expected. "
                     "Check if you need a vec3 or vec4 here.",
            )

        # Multi-channel swizzle
        indices = [CHANNEL_MAP[ch] for ch in channels if ch in CHANNEL_MAP]
        if len(indices) != len(channels):
            raise InterpreterError(
                f"Expected valid channel names in swizzle, but '.{channels}' contains unrecognized letters",
                node.loc, source=self._source, code="E6030",
                hint="Swizzle channels must be from: r, g, b, a (or x, y, z, w).",
            )
        # Fast path: contiguous slice (e.g. .rgb on vec4 = first 3 channels)
        if indices == list(range(indices[0], indices[0] + len(indices))):
            return base[..., indices[0]:indices[0] + len(indices)]
        return torch.stack([base[..., i] for i in indices], dim=-1)

    def _eval_array_index(self, node: ArrayIndexAccess) -> torch.Tensor | str:
        """Evaluate array[index] with clamped bounds."""
        array = self._eval(node.array)
        index = self._eval(node.index)

        # String array (Python list)
        if isinstance(array, list):
            idx_int = max(0, min(int(round(index.item() if isinstance(index, torch.Tensor) else float(index))), len(array) - 1))
            return array[idx_int]

        # Vector array: dim 5 (spatial) or 2 (non-spatial) → [..., N, C]
        if array.dim() in (2, 5):
            arr_size = array.shape[-2]
            idx = _safe_array_index(index, arr_size)

            if array.dim() == 2:
                # Non-spatial: [N, C]
                return array[idx]

            if idx.dim() == 0:
                # Constant index: [..., N, C] → [..., C]
                return array[..., idx.item(), :]

            # Per-pixel index: gather on dim=-2
            C = array.shape[-1]
            idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(*idx.shape, 1, C)
            if idx_exp.shape[:3] != array.shape[:3]:
                idx_exp = idx_exp.expand(array.shape[:3] + (1, C))
            return torch.gather(array, dim=-2, index=idx_exp).squeeze(-2)

        # Scalar array: dim 4 (spatial) or 1 (non-spatial) → [..., N]
        arr_size = array.shape[-1]
        idx = _safe_array_index(index, arr_size)

        if array.dim() == 1:
            return array[idx]

        if idx.dim() == 0:
            return array[..., idx.item()]

        idx_expanded = idx.unsqueeze(-1)
        if idx_expanded.shape[:3] != array.shape[:3]:
            idx_expanded = idx_expanded.expand(array.shape[:3] + (1,))
        return torch.gather(array, dim=-1, index=idx_expanded).squeeze(-1)

    def _eval_binop(self, node: BinOp) -> torch.Tensor | str:
        left = self._eval(node.left)
        right = self._eval(node.right)
        op = node.op

        # Fast path: both are tensors (vast majority of cases)
        if left.__class__ is torch.Tensor and right.__class__ is torch.Tensor:
            # Matrix operations: only check type_map when operands could be matrices.
            # Matrices have a last dim of 3 or 4 AND a second-to-last dim of 3 or 4,
            # so skip the expensive type_map lookup for scalars and simple vectors.
            if op == "*" and left.dim() >= 2:
                ld_last = left.shape[-1]
                if ld_last in (3, 4) and left.shape[-2] in (3, 4):
                    lt = self.type_map.get(id(node.left))
                    if lt is not None and lt.is_matrix:
                        rt = self.type_map.get(id(node.right))
                        if rt is not None:
                            if rt.is_matrix:
                                return torch.matmul(left, right)
                            if rt.is_vector:
                                return torch.matmul(left, right.unsqueeze(-1)).squeeze(-1)

            # Ensure compatible shapes.
            # Fast path: same shape — skip all checks (most common case in loops)
            ld = left.dim()
            rd = right.dim()
            if ld != rd:
                if ld == 0 or rd == 0:
                    pass  # PyTorch broadcasts 0-d scalars natively
                elif ld == 3 and rd == 4:
                    left = left.unsqueeze(-1)
                elif ld == 4 and rd == 3:
                    right = right.unsqueeze(-1)
                else:
                    left, right = _broadcast_pair(left, right)
            elif ld >= 1 and left.shape[-1] != right.shape[-1]:
                left, right = _broadcast_pair(left, right)

            # Inline operator dispatch (avoids lambda call overhead)
            if op == "+":
                return left + right
            if op == "*":
                return left * right
            if op == "-":
                return left - right
            if op == "/":
                # Avoid division by zero: replace 0 with SAFE_EPSILON (fewer temporaries than multiply+add)
                return left / torch.where(right == 0, SAFE_EPSILON, right)
            if op == "<":
                return (left < right).float()
            if op == ">":
                return (left > right).float()
            if op == "<=":
                return (left <= right).float()
            if op == ">=":
                return (left >= right).float()
            if op == "==":
                return (left == right).float()
            if op == "!=":
                return (left != right).float()
            if op == "&&":
                return ((left > 0.5) & (right > 0.5)).float()
            if op == "||":
                return ((left > 0.5) | (right > 0.5)).float()
            if op == "%":
                return torch.fmod(left, torch.where(right == 0, SAFE_EPSILON, right))
            raise InterpreterError(
                f"Operator '{op}' is not supported",
                node.loc, source=self._source, code="E6040",
                hint="Supported operators: +, -, *, /, %, ==, !=, <, >, <=, >=, &&, ||.",
            )

        # Slow path: string operations
        if isinstance(left, str) or isinstance(right, str):
            if isinstance(left, str) and isinstance(right, str):
                if op == "+":
                    return left + right
                elif op == "==":
                    return torch.scalar_tensor(1.0 if left == right else 0.0, dtype=self._dtype, device=self.device)
                elif op == "!=":
                    return torch.scalar_tensor(1.0 if left != right else 0.0, dtype=self._dtype, device=self.device)
                else:
                    raise InterpreterError(
                        f"Operator '{op}' is not supported for strings",
                        node.loc, source=self._source, code="E6040",
                        hint="Strings support '+' (concatenation), '==' and '!=' (comparison).",
                    )
            raise InterpreterError(
                f"Expected matching types for '{op}', but found a mix of string and numeric",
                node.loc, source=self._source, code="E6040",
                hint="Both sides of the operator must be the same type. "
                     "Use string() or float() to convert.",
            )

    def _eval_unary(self, node: UnaryOp) -> torch.Tensor:
        operand = self._eval(node.operand)
        if node.op == "-":
            return -operand
        elif node.op == "!":
            return (operand <= 0.5).float()
        raise InterpreterError(
            f"Unary operator '{node.op}' is not supported",
            node.loc, source=self._source, code="E6040",
            hint="Supported unary operators: '-' (negate) and '!' (logical not).",
        )

    def _eval_ternary(self, node: TernaryOp) -> torch.Tensor | str:
        cond = self._eval(node.condition)

        # Scalar condition: short-circuit — only evaluate the taken branch
        if isinstance(cond, torch.Tensor) and cond.dim() == 0:
            return self._eval(node.true_expr) if cond.item() > 0.5 else self._eval(node.false_expr)

        true_val = self._eval(node.true_expr)
        false_val = self._eval(node.false_expr)

        # String ternary — use scalar condition to pick branch
        if isinstance(true_val, str) or isinstance(false_val, str):
            if isinstance(cond, torch.Tensor):
                cond_scalar = cond.float().mean().item() if cond.dim() > 0 else cond.item()
            else:
                cond_scalar = float(cond)
            return true_val if cond_scalar > 0.5 else false_val

        cond_bool = cond > 0.5
        return _tensor_where(cond_bool, true_val, false_val)

    def _exec_function_def(self, node: FunctionDef):
        """Register a user-defined function (no execution yet)."""
        self._user_functions[node.name] = node

    def _exec_return_stmt(self, node: ReturnStmt):
        value = self._eval(node.value)
        raise _ReturnSignal(value)

    def _call_user_function(self, func_def: FunctionDef, call_node: FunctionCall):
        self._call_depth += 1
        if self._call_depth > MAX_CALL_DEPTH:
            self._call_depth -= 1
            raise InterpreterError(
                f"Maximum function call depth ({MAX_CALL_DEPTH}) exceeded — "
                f"possible infinite recursion in '{func_def.name}()'.",
                call_node.loc, source=self._source, code="E6060",
                hint="Check for functions that call themselves without a base case.",
            )

        args = [self._eval(arg) for arg in call_node.args]

        # Save and replace environment
        saved_env = self.env
        self.env = dict(saved_env)  # shallow copy — inherits builtins

        # Bind parameters
        for (ptype, pname), arg_val in zip(func_def.params, args):
            self.env[pname] = arg_val

        # Execute body, catch ReturnSignal
        try:
            for stmt in func_def.body:
                self._exec_stmt(stmt)
        except _ReturnSignal as ret:
            result = ret.value
        else:
            result = torch.scalar_tensor(0.0, dtype=self._dtype, device=self.device)
        finally:
            self.env = saved_env
            self._call_depth -= 1

        return result

    def _eval_binding_index(self, node: BindingIndexAccess):
        """@Image[ix, iy] → fetch, @Image[ix, iy, frame] → fetch_frame"""
        image = self._eval(node.binding)
        args = [self._eval(a) for a in node.args]
        if len(args) == 3:
            return self.functions["fetch_frame"](image, args[2], args[0], args[1])
        return self.functions["fetch"](image, args[0], args[1])

    def _eval_binding_sample(self, node: BindingSampleAccess):
        """@Image(u, v) → sample, @Image(u, v, frame) → sample_frame"""
        image = self._eval(node.binding)
        args = [self._eval(a) for a in node.args]
        if len(args) == 3:
            return self.functions["sample_frame"](image, args[2], args[0], args[1])
        return self.functions["sample"](image, args[0], args[1])

    def _eval_function_call(self, node: FunctionCall) -> torch.Tensor | str:
        fn = self.functions.get(node.name)
        if fn is None:
            # Check user-defined functions
            user_fn = self._user_functions.get(node.name)
            if user_fn is not None:
                return self._call_user_function(user_fn, node)
            raise InterpreterError(
                f"Function '{node.name}' is not recognized",
                node.loc, source=self._source, code="E6050",
                hint=f"Check the spelling of '{node.name}'. See the TEX reference for available functions.",
            )

        # Fast paths for common arities (avoid list comprehension overhead)
        node_args = node.args
        nargs = len(node_args)
        try:
            if nargs == 1:
                result = fn(self._eval(node_args[0]))
            elif nargs == 2:
                result = fn(self._eval(node_args[0]), self._eval(node_args[1]))
            elif nargs == 3:
                result = fn(self._eval(node_args[0]), self._eval(node_args[1]),
                            self._eval(node_args[2]))
            else:
                result = fn(*[self._eval(arg) for arg in node_args])
        except Exception as e:
            raise InterpreterError(
                f"Function '{node.name}' encountered a problem: {e}",
                node.loc, source=self._source, code="E6050",
                hint="Check the number and types of arguments passed to this function.",
            )

        # Fast path: most stdlib functions return tensors
        if result.__class__ is torch.Tensor:
            return result
        # Slow path: str, list, or scalar
        if isinstance(result, (str, list)):
            return result
        return torch.scalar_tensor(float(result), dtype=self._dtype, device=self.device)

    def _eval_vec_constructor(self, node: VecConstructor) -> torch.Tensor:
        node_args = node.args
        n = node.size
        nargs = len(node_args)

        # Fast path for 1-arg: vec4(scalar) or vec4(vec4_val)
        if nargs == 1:
            val = self._eval(node_args[0])
            if val.__class__ is torch.Tensor:
                arg_type = self.type_map.get(id(node_args[0]))
                if arg_type is not None and arg_type.is_vector and arg_type.channels == n:
                    return val
                if val.dim() == 0:
                    return val.unsqueeze(-1).expand(*(() if not self.spatial_shape else self.spatial_shape), n)
                else:
                    return val.unsqueeze(-1).expand(*val.shape, n)
            return torch.scalar_tensor(float(val), dtype=self._dtype, device=self.device).unsqueeze(-1).expand(
                *(() if not self.spatial_shape else self.spatial_shape), n
            )

        # Fast path: N args for vecN — the overwhelmingly common case.
        # Check for all-scalar args without type_map lookups first:
        # if all args evaluate to tensors with the same spatial shape [B,H,W],
        # they can't be vectors (which would have shape [B,H,W,C]).
        if nargs == n:
            spatial = self.spatial_shape
            if spatial:
                # Evaluate all args and check for fast torch.stack
                args = [self._eval(arg) for arg in node_args]
                a0 = args[0]
                if a0.__class__ is torch.Tensor and a0.shape == spatial:
                    # Check all same shape (very common: vec4(r, g, b, 1.0))
                    all_same = True
                    for i in range(1, n):
                        if args[i].__class__ is not torch.Tensor or args[i].shape != spatial:
                            all_same = False
                            break
                    if all_same:
                        return torch.stack(args, dim=-1)

                # Mixed shapes — check if any are vectors via type_map
                all_scalar = True
                for i in range(n):
                    arg_type = self.type_map.get(id(node_args[i]))
                    if arg_type is not None and arg_type.is_vector:
                        all_scalar = False
                        break
                if all_scalar:
                    # Expand scalars to spatial shape, then stack.
                    # expand() is a zero-copy view — cheaper than
                    # torch.empty() + per-channel indexed assignment.
                    return torch.stack(
                        [c.expand(spatial) if c.shape != spatial else c
                         for c in args],
                        dim=-1,
                    )
            else:
                args = [self._eval(arg) for arg in node_args]
                all_scalar = True
                for i in range(n):
                    arg_type = self.type_map.get(id(node_args[i]))
                    if arg_type is not None and arg_type.is_vector:
                        all_scalar = False
                        break
                if all_scalar:
                    return torch.stack([_ensure_spatial(c, ()) for c in args], dim=-1)
        else:
            args = [self._eval(arg) for arg in node_args]

        # Flatten composite args: vec4(vec3, float) → [ch0, ch1, ch2, scalar]
        # Use the type_map from the type checker to distinguish vectors
        # from scalars (tensor shape alone is ambiguous — a spatial scalar
        # [B,H,W] where H or W is 3/4 would be misidentified).
        components: list[torch.Tensor] = []
        for i, arg in enumerate(args):
            arg_type = self.type_map.get(id(node.args[i]))
            if arg_type is not None and arg_type.is_vector:
                # Vector arg — split into per-channel components
                for ch in range(arg_type.channels):
                    components.append(arg[..., ch])
            else:
                components.append(arg)

        if len(components) != n:
            raise InterpreterError(
                f"vec{n}() needs {n} total components, but found {len(components)}",
                node.loc, source=self._source, code="E6060",
                hint=f"Provide exactly {n} scalar values, or combine vectors and scalars that add up to {n} components.",
            )

        # Allocate and fill channels
        max_shape = self._get_max_spatial_shape(components)
        if max_shape:
            result = torch.empty(*max_shape, n, dtype=self._dtype, device=self.device)
            for i, c in enumerate(components):
                result[..., i] = _ensure_spatial(c, max_shape)
            return result
        else:
            return torch.stack([_ensure_spatial(c, max_shape) for c in components], dim=-1)

    def _eval_mat_constructor(self, node: MatConstructor) -> torch.Tensor:
        n = node.size  # 3 or 4
        args = [self._eval(arg) for arg in node.args]

        if len(args) == 1:
            # Scaled identity: mat3(1.0) → identity * scalar
            val = args[0]
            eye = torch.eye(n, dtype=self._dtype, device=self.device)
            return eye * val
        elif len(args) == n * n:
            # Full specification: mat3(a,b,c,d,e,f,g,h,i) → 3×3 row-major
            max_shape = self._get_max_spatial_shape(args)
            expanded = [_ensure_spatial(a, max_shape) for a in args]
            flat = torch.stack(expanded, dim=-1)  # [..., n*n]
            return flat.reshape(flat.shape[:-1] + (n, n))
        else:
            raise InterpreterError(
                f"mat{n}() needs {n * n} arguments (full matrix) or 1 (scaled identity), but found {len(args)}",
                node.loc, source=self._source, code="E6060",
                hint=f"Use mat{n}(1.0) for an identity matrix, or provide all {n * n} elements.",
            )

    def _eval_cast(self, node: CastExpr) -> torch.Tensor | str:
        value = self._eval(node.expr)
        target = node.target_type
        if target == "float":
            # Fast path: already float32 — skip the .float() call entirely
            if value.__class__ is torch.Tensor and value.dtype == torch.float32:
                return value
            return value.float()
        if target == "int":
            return torch.floor(value)
        if target == "string":
            # Convert number to string
            if isinstance(value, torch.Tensor):
                v = value.item() if value.dim() == 0 else value.float().mean().item()
                return str(int(v)) if v == int(v) else str(v)
            return str(value)
        return value

    # -- Helpers --------------------------------------------------------

    def _default_value(self, t: TEXType) -> torch.Tensor | str:
        if t == TEXType.STRING:
            return ""
        elif t == TEXType.VEC3:
            if self.spatial_shape:
                B, H, W = self.spatial_shape
                return torch.zeros(B, H, W, 3, dtype=self._dtype, device=self.device)
            return torch.zeros(3, dtype=self._dtype, device=self.device)
        elif t == TEXType.VEC4:
            if self.spatial_shape:
                B, H, W = self.spatial_shape
                return torch.zeros(B, H, W, 4, dtype=self._dtype, device=self.device)
            return torch.zeros(4, dtype=self._dtype, device=self.device)
        elif t.is_matrix:
            n = t.mat_size
            return torch.zeros(n, n, dtype=self._dtype, device=self.device)
        else:
            return torch.scalar_tensor(0.0, dtype=self._dtype, device=self.device)

    def _get_max_spatial_shape(self, tensors: list[torch.Tensor]) -> tuple:
        """Find the broadcast-compatible spatial shape from a list of tensors."""
        if self.spatial_shape:
            return self.spatial_shape
        max_shape = ()
        for t in tensors:
            if isinstance(t, torch.Tensor) and t.dim() >= 3:
                shape = t.shape[:3]  # B, H, W
                if len(shape) > len(max_shape):
                    max_shape = shape
        return max_shape


# -- AST scanning for lazy builtins ------------------------------------

# Names that are built-in variables (not user-defined)
_BUILTIN_NAMES = frozenset({"ix", "iy", "u", "v", "iw", "ih", "px", "py", "fi", "fn", "PI", "TAU", "E", "ic"})


def _collect_identifiers(program: Program) -> frozenset[str]:
    """Collect all Identifier names referenced in a program (fast single-pass scan).

    Returns only names that match builtin variable names, for lazy construction.
    """
    found: set[str] = set()
    # Use an explicit stack to avoid recursion overhead
    stack: list[ASTNode] = list(program.statements)
    while stack:
        node = stack.pop()
        cls = type(node)

        if cls is Identifier:
            if node.name in _BUILTIN_NAMES:
                found.add(node.name)
            continue

        # Statements
        if cls is VarDecl:
            if node.initializer:
                stack.append(node.initializer)
        elif cls is Assignment:
            stack.append(node.target)
            stack.append(node.value)
        elif cls is IfElse:
            stack.append(node.condition)
            stack.extend(node.then_body)
            stack.extend(node.else_body)
        elif cls is ForLoop:
            stack.append(node.init)
            stack.append(node.condition)
            stack.append(node.update)
            stack.extend(node.body)
        elif cls is WhileLoop:
            stack.append(node.condition)
            stack.extend(node.body)
        elif cls is ExprStatement:
            stack.append(node.expr)
        elif cls is ArrayDecl:
            if node.initializer:
                stack.append(node.initializer)
        # Expressions
        elif cls is BinOp:
            stack.append(node.left)
            stack.append(node.right)
        elif cls is UnaryOp:
            stack.append(node.operand)
        elif cls is TernaryOp:
            stack.append(node.condition)
            stack.append(node.true_expr)
            stack.append(node.false_expr)
        elif cls is FunctionCall:
            stack.extend(node.args)
        elif cls is VecConstructor:
            stack.extend(node.args)
        elif cls is MatConstructor:
            stack.extend(node.args)
        elif cls is CastExpr:
            stack.append(node.expr)
        elif cls is ChannelAccess:
            stack.append(node.object)
        elif cls is ArrayIndexAccess:
            stack.append(node.array)
            stack.append(node.index)
        elif cls is ArrayLiteral:
            stack.extend(node.elements)
        elif cls is BindingIndexAccess:
            stack.extend(node.args)
        elif cls is BindingSampleAccess:
            stack.extend(node.args)
        elif cls is FunctionDef:
            stack.extend(node.body)
        elif cls is ReturnStmt:
            if node.value:
                stack.append(node.value)
        # NumberLiteral, StringLiteral, BindingRef, BreakStmt, ContinueStmt, ParamDecl — skip

    return frozenset(found)


# -- Module-level utility functions ------------------------------------

def _safe_array_index(idx: torch.Tensor, size: int) -> torch.Tensor:
    """Clamp a float index tensor to valid array bounds and convert to int64.

    Equivalent to ``torch.clamp(torch.floor(idx).long(), 0, size - 1)``.
    """
    return torch.clamp(torch.floor(idx).long(), 0, size - 1)


def _ensure_spatial(tensor: torch.Tensor, spatial_shape: tuple) -> torch.Tensor:
    """Expand a tensor to match a spatial shape [B, H, W] if needed."""
    if not spatial_shape:
        return tensor
    if tensor.dim() == 0:
        return tensor.expand(spatial_shape)
    if tensor.shape[:len(spatial_shape)] == spatial_shape:
        return tensor
    # Try broadcasting
    try:
        return tensor.expand(spatial_shape)
    except RuntimeError:
        return tensor


def _broadcast_pair(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Broadcast two tensors to be compatible for element-wise operations.

    Handles the key case: scalar [B,H,W] op with vector [B,H,W,C]
    by expanding the scalar with unsqueeze(-1).
    Also pads channel dimensions when both are vectors with different channel counts.
    """
    ad, bd = a.dim(), b.dim()
    if ad == bd:
        # Same rank — pad channel dim when both are vectors (e.g. vec2 [B,H,W,2] + vec3 [B,H,W,3])
        if ad >= 1 and a.shape[-1] != b.shape[-1]:
            ac, bc = a.shape[-1], b.shape[-1]
            if not (ac in _VEC_CHANNELS and bc in _VEC_CHANNELS):
                return a, b
            if ac < bc:
                a = torch.nn.functional.pad(a, (0, bc - ac))
            else:
                b = torch.nn.functional.pad(b, (0, ac - bc))
        return a, b

    # Pad trailing dims with single view+expand (no while loop)
    if ad > bd:
        b = b.view(*b.shape, *((1,) * (ad - bd))).expand_as(a)
        return a, b
    else:
        a = a.view(*a.shape, *((1,) * (bd - ad))).expand_as(b)
        return a, b


def _tensor_where(cond: torch.Tensor, then_val: torch.Tensor, else_val: torch.Tensor) -> torch.Tensor:
    """torch.where with broadcasting support for mixed scalar/vector cases."""
    then_val, else_val = _broadcast_pair(then_val, else_val)

    # Expand condition to match value shapes (single view instead of while loop)
    cd, td = cond.dim(), then_val.dim()
    if cd < td:
        cond = cond.view(*cond.shape, *((1,) * (td - cd)))
        try:
            cond = cond.expand_as(then_val)
        except RuntimeError:
            pass

    return torch.where(cond, then_val, else_val)
