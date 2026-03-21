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
    BreakStmt, ContinueStmt,
    BinOp, UnaryOp, TernaryOp, FunctionCall, Identifier, BindingRef,
    ChannelAccess, NumberLiteral, StringLiteral, VecConstructor, CastExpr, SourceLoc,
    ArrayDecl, ArrayIndexAccess, ArrayLiteral, MatConstructor, ParamDecl,
)
from ..tex_compiler.type_checker import TEXType, CHANNEL_MAP
from .stdlib import TEXStdlib, SAFE_EPSILON
# Codegen module available for future torch.compile integration but not used
# in the hot path — benchmarks show the interpreter dispatch is already fast
# enough that codegen overhead (constant creation, try/except) is a net loss.
# from .codegen import try_compile as _try_codegen, _CgBreak, _CgContinue

# Hard limit on for-loop iterations to prevent infinite loops
MAX_LOOP_ITERATIONS = 1024



class InterpreterError(Exception):
    def __init__(self, message: str, loc: SourceLoc = None):
        self.loc = loc
        prefix = f"[{loc}] " if loc else ""
        super().__init__(f"{prefix}{message}")


class _Break(Exception):
    """Internal signal for break statements inside for loops."""
    pass


class _Continue(Exception):
    """Internal signal for continue statements inside for loops."""
    pass


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
        self._literal_cache: dict[tuple[float, str], torch.Tensor] = {}  # (value, device) -> tensor
        self._inplace_ready: set[str] = set()  # vars that have been cloned for in-place ops

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
        # Skip inference_mode for pure string/scalar programs (no tensor bindings)
        has_tensors = any(isinstance(v, torch.Tensor) for v in bindings.values())
        if has_tensors:
            with torch.inference_mode():
                return self._execute_inner(program, bindings, type_map, device,
                                           latent_channel_count, output_names,
                                           precision, used_builtins=used_builtins)
        else:
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
            else:
                self.bindings[name] = torch.tensor(float(value), dtype=target_dtype, device=target_device)

        # Determine spatial context from image inputs
        self.spatial_shape = self._determine_spatial_shape()

        # Create built-in coordinate variables (lazy — only what the program uses)
        self._create_builtins(program, tile_offset=tile_offset, used_builtins=used_builtins)

        # Execute statements via tree-walking interpreter
        for stmt in program.statements:
            self._exec_stmt(stmt)

        # Convert outputs back to float32 if using reduced precision
        def _to_output(val):
            if self._dtype != torch.float32 and isinstance(val, torch.Tensor) and val.is_floating_point():
                return val.float()
            return val

        # Multi-output mode
        if output_names is not None:
            results = {}
            for name in output_names:
                if name in self.bindings:
                    results[name] = _to_output(self.bindings[name])
                else:
                    raise InterpreterError(
                        f"Output @{name} was not assigned by the TEX program"
                    )
            return results

        # Single-output backward compat
        if "OUT" not in self.bindings:
            raise InterpreterError(
                "TEX program must assign to at least one output "
                "(e.g. @OUT, or named outputs like @result)"
            )

        return _to_output(self.bindings["OUT"])

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

        if self.spatial_shape:
            B, H, W = self.spatial_shape

            # For tiled execution, use full image dimensions for coordinate normalization
            if tile_offset is not None:
                y_start, full_H, full_W = tile_offset
            else:
                y_start, full_H, full_W = 0, H, W

            # ix: pixel x-coordinate [0, W-1] — compact shape [1, 1, W]
            if "ix" in used:
                ix = torch.arange(W, dtype=self._dtype, device=self.device).view(1, 1, W)
                self.env["ix"] = ix

                if "u" in used:
                    self.env["u"] = (ix / max(full_W - 1, 1)).expand(B, H, W)
            elif "u" in used:
                ix = torch.arange(W, dtype=self._dtype, device=self.device).view(1, 1, W)
                self.env["u"] = (ix / max(full_W - 1, 1)).expand(B, H, W)

            # iy: pixel y-coordinate — offset by y_start for tiling
            if "iy" in used:
                iy = torch.arange(y_start, y_start + H, dtype=self._dtype, device=self.device).view(1, H, 1)
                self.env["iy"] = iy

                if "v" in used:
                    self.env["v"] = (iy / max(full_H - 1, 1)).expand(B, H, W)
            elif "v" in used:
                iy = torch.arange(y_start, y_start + H, dtype=self._dtype, device=self.device).view(1, H, 1)
                self.env["v"] = (iy / max(full_H - 1, 1)).expand(B, H, W)

            # iw, ih: FULL image dimensions (not tile dimensions)
            if "iw" in used:
                self.env["iw"] = torch.tensor(float(full_W), dtype=self._dtype, device=self.device)
            if "ih" in used:
                self.env["ih"] = torch.tensor(float(full_H), dtype=self._dtype, device=self.device)

            if "fi" in used:
                self.env["fi"] = torch.arange(B, dtype=self._dtype, device=self.device).view(B, 1, 1)
            if "fn" in used:
                self.env["fn"] = torch.tensor(float(B), dtype=self._dtype, device=self.device)
        else:
            # No spatial context — pure scalar mode (only create what's used)
            _scalar_defaults = {"ix": 0.0, "iy": 0.0, "u": 0.0, "v": 0.0,
                                "iw": 1.0, "ih": 1.0, "fi": 0.0, "fn": 1.0}
            for name, val in _scalar_defaults.items():
                if name in used:
                    self.env[name] = torch.tensor(val, device=self.device)

        if "PI" in used:
            self.env["PI"] = torch.tensor(math.pi, dtype=self._dtype, device=self.device)
        if "E" in used:
            self.env["E"] = torch.tensor(math.e, dtype=self._dtype, device=self.device)
        if "ic" in used:
            self.env["ic"] = torch.tensor(float(self.latent_channel_count), dtype=self._dtype, device=self.device)

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
        raise InterpreterError(f"Unknown statement: {type(node).__name__}", node.loc)

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
        is_vec = node.element_type_name in ("vec3", "vec4")
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
                    expanded = [e if isinstance(e, torch.Tensor) else torch.tensor(float(e), dtype=self._dtype, device=self.device) for e in elements]
                # For vec3/vec4 arrays, promote elements to consistent channel count
                if is_vec:
                    channels = 3 if node.element_type_name == "vec3" else 4
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
                channels = 3 if node.element_type_name == "vec3" else 4
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
            tensor = torch.tensor(float(tensor), dtype=self._dtype, device=self.device)
        if tensor.dim() >= 1 and tensor.shape[-1] in (3, 4):
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
    }

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
                # Pattern 1: x = x OP expr
                if isinstance(rhs.left, Identifier) and rhs.left.name == name:
                    current = self.env.get(name)
                    if current is not None and current.__class__ is torch.Tensor:
                        other_val = self._eval(rhs.right)
                        if other_val.__class__ is torch.Tensor and (
                            other_val.dim() == 0 or other_val.shape == current.shape
                        ):
                            # Clone on first in-place to establish ownership
                            # (prevents aliasing with literal cache or bindings)
                            if name not in self._inplace_ready:
                                current = current.clone()
                                self.env[name] = current
                                self._inplace_ready.add(name)
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
                            if name not in self._inplace_ready:
                                current = current.clone()
                                self.env[name] = current
                                self._inplace_ready.add(name)
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

        else:
            raise InterpreterError("Invalid assignment target", node.loc)

    def _exec_channel_assign(self, target: ChannelAccess, value: torch.Tensor):
        """Handle assignment to a channel: `@A.r = expr;` or `color.r = expr;`"""
        # Get the base tensor
        base = self._eval(target.object)
        channels = target.channels

        if len(channels) != 1:
            raise InterpreterError(
                f"Can only assign to single channels, not swizzle '.{channels}'",
                target.loc,
            )

        idx = CHANNEL_MAP.get(channels)
        if idx is None:
            raise InterpreterError(f"Invalid channel: '.{channels}'", target.loc)

        # Clone to avoid modifying the original in-place (breaks autograd)
        result = base.clone()
        if result.dim() >= 1 and result.shape[-1] > idx:
            result[..., idx] = _ensure_spatial(value, result.shape[:-1])
        else:
            raise InterpreterError(
                f"Cannot assign to channel {idx} of tensor with shape {result.shape}",
                target.loc,
            )

        # Write back to the correct location
        if isinstance(target.object, Identifier):
            self.env[target.object.name] = result
        elif isinstance(target.object, BindingRef):
            self.bindings[target.object.name] = result
        else:
            raise InterpreterError("Channel assignment requires a variable or binding target", target.loc)

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
            idx = torch.clamp(torch.floor(index).long(), 0, arr_size - 1)
            result = array.clone()

            if result.dim() == 2:
                # Non-spatial vector array [N, C]
                val_t = value if isinstance(value, torch.Tensor) else torch.tensor(float(value), dtype=self._dtype, device=self.device)
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
            idx = torch.clamp(torch.floor(index).long(), 0, arr_size - 1)
            result = array.clone()

            if result.dim() == 1:
                result[idx] = value if isinstance(value, torch.Tensor) else torch.tensor(float(value), dtype=self._dtype, device=self.device)
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
            raise InterpreterError("Array index assignment requires a variable target", target.loc)

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
        # Selective cloning: only snapshot variables that are assigned in branches
        modified_env, modified_bindings = self._collect_assigned_vars(node.then_body)
        if node.else_body:
            e2, b2 = self._collect_assigned_vars(node.else_body)
            modified_env |= e2
            modified_bindings |= b2

        # Snapshot only modified variables
        env_snapshot = {}
        for k in modified_env:
            v = self.env.get(k)
            if v is not None:
                env_snapshot[k] = v.clone() if isinstance(v, torch.Tensor) else v
        bindings_snapshot = {}
        for k in modified_bindings:
            v = self.bindings.get(k)
            if v is not None:
                bindings_snapshot[k] = v.clone() if isinstance(v, torch.Tensor) else v
        meta_snapshot = dict(self._array_meta)

        # Execute then-branch
        for stmt in node.then_body:
            self._exec_stmt(stmt)
        # Capture then-state for modified vars only
        then_env = {k: self.env.get(k) for k in modified_env}
        then_bindings = {k: self.bindings.get(k) for k in modified_bindings}
        then_meta = dict(self._array_meta)

        # Restore modified vars and execute else-branch
        for k, v in env_snapshot.items():
            self.env[k] = v
        for k, v in bindings_snapshot.items():
            self.bindings[k] = v
        self._array_meta = dict(meta_snapshot)

        if node.else_body:
            for stmt in node.else_body:
                self._exec_stmt(stmt)
            else_env = {k: self.env.get(k) for k in modified_env}
            else_bindings = {k: self.bindings.get(k) for k in modified_bindings}
        else:
            else_env = dict(env_snapshot)
            else_bindings = dict(bindings_snapshot)

        # Merge using torch.where (tensors) or scalar condition (strings)
        cond_bool = (cond > 0.5) if cond.is_floating_point() else cond.bool()
        # For string merging: use majority vote for spatial conditions
        cond_scalar_bool = cond.float().mean().item() > 0.5 if cond.dim() > 0 else cond.item() > 0.5

        # Merge only modified env variables
        for key in modified_env:
            then_val = then_env.get(key)
            else_val = else_env.get(key)
            if then_val is not None and else_val is not None:
                if isinstance(then_val, torch.Tensor) and isinstance(else_val, torch.Tensor):
                    self.env[key] = _tensor_where(cond_bool, then_val, else_val)
                elif isinstance(then_val, str) or isinstance(else_val, str):
                    self.env[key] = then_val if cond_scalar_bool else else_val
                else:
                    self.env[key] = then_val
            elif then_val is not None:
                self.env[key] = then_val
            elif else_val is not None:
                self.env[key] = else_val

        # Merge only modified bindings
        for key in modified_bindings:
            then_val = then_bindings.get(key)
            else_val = else_bindings.get(key)
            if then_val is not None and else_val is not None:
                if isinstance(then_val, torch.Tensor) and isinstance(else_val, torch.Tensor):
                    self.bindings[key] = _tensor_where(cond_bool, then_val, else_val)
                elif isinstance(then_val, str) or isinstance(else_val, str):
                    self.bindings[key] = then_val if cond_scalar_bool else else_val
                else:
                    self.bindings[key] = then_val
            elif then_val is not None:
                self.bindings[key] = then_val
            else:
                self.bindings[key] = else_val

        # Merge array metadata from then-branch (else-branch meta is already in self)
        self._array_meta.update(then_meta)

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
                    f"For loop exceeded maximum iteration limit ({MAX_LOOP_ITERATIONS}). "
                    f"Check your loop condition.",
                    node.loc,
                )

            device_str = self._device_str
            dtype = self._dtype
            for i in iter_range:
                # Set loop var directly as scalar tensor — no .item() needed
                key = (float(i), device_str, dtype)
                cached = self._literal_cache.get(key)
                if cached is not None:
                    self.env[loop_var] = cached
                else:
                    t = torch.tensor(float(i), dtype=dtype, device=self.device)
                    self._literal_cache[key] = t
                    self.env[loop_var] = t
                self._inplace_ready.discard(loop_var)

                try:
                    for stmt in node.body:
                        self._exec_stmt(stmt)
                except _Break:
                    break
                except _Continue:
                    pass  # Skip rest of body, proceed to next iteration
        else:
            # General case — execute init, evaluate condition each iteration
            self._exec_stmt(node.init)
            iteration = 0
            while iteration < MAX_LOOP_ITERATIONS:
                cond = self._eval(node.condition)

                if cond.dim() == 0:
                    if cond.item() <= 0.5:
                        break
                else:
                    if (cond > 0.5).float().sum().item() == 0:
                        break

                try:
                    for stmt in node.body:
                        self._exec_stmt(stmt)
                except _Break:
                    break
                except _Continue:
                    pass  # Skip rest of body, proceed to update
                self._exec_stmt(node.update)
                iteration += 1

            if iteration >= MAX_LOOP_ITERATIONS:
                raise InterpreterError(
                    f"For loop exceeded maximum iteration limit ({MAX_LOOP_ITERATIONS}). "
                    f"Check your loop condition.",
                    node.loc,
                )

    def _try_extract_static_range(self, node: ForLoop) -> tuple[str, range] | None:
        """
        Try to extract a fully static loop as a Python range().

        Matches pattern: for (int VAR = START; VAR < END; VAR = VAR + STEP)
        where START, END, STEP are all NumberLiterals (possibly after constant folding).

        Returns (loop_var_name, range_object) or None.
        """
        # Init: VarDecl with NumberLiteral initializer
        init = node.init
        if not isinstance(init, VarDecl) or init.initializer is None:
            return None
        if not isinstance(init.initializer, NumberLiteral):
            return None
        loop_var = init.name
        start = int(init.initializer.value)

        # Condition: VAR < END or VAR <= END
        cond = node.condition
        if not isinstance(cond, BinOp) or cond.op not in ("<", "<="):
            return None
        if not isinstance(cond.left, Identifier) or cond.left.name != loop_var:
            return None
        if not isinstance(cond.right, NumberLiteral):
            return None
        end = int(cond.right.value)
        if cond.op == "<=":
            end += 1

        # Update: VAR = VAR + STEP or VAR = VAR - STEP
        update = node.update
        if not isinstance(update, Assignment):
            return None
        if not isinstance(update.target, Identifier) or update.target.name != loop_var:
            return None
        if not isinstance(update.value, BinOp):
            return None
        upd = update.value
        if not isinstance(upd.left, Identifier) or upd.left.name != loop_var:
            return None
        if not isinstance(upd.right, NumberLiteral):
            return None
        if upd.op == "+":
            step = int(upd.right.value)
        elif upd.op == "-":
            step = -int(upd.right.value)
        else:
            return None

        if step == 0:
            return None

        try:
            r = range(start, end, step)
        except ValueError:
            return None

        return (loop_var, r)

    def _exec_while_loop(self, node: WhileLoop):
        """
        Execute a bounded while loop.

        Same safety guarantees as for loops: hard limit of MAX_LOOP_ITERATIONS.
        """
        iteration = 0
        while iteration < MAX_LOOP_ITERATIONS:
            cond = self._eval(node.condition)

            if cond.dim() == 0:
                if cond.item() <= 0.5:
                    break
            else:
                if (cond > 0.5).float().sum().item() == 0:
                    break

            try:
                for stmt in node.body:
                    self._exec_stmt(stmt)
            except _Break:
                break
            except _Continue:
                pass  # Skip rest of body, re-evaluate condition
            iteration += 1

        if iteration >= MAX_LOOP_ITERATIONS:
            raise InterpreterError(
                f"While loop exceeded maximum iteration limit ({MAX_LOOP_ITERATIONS}). "
                f"Check your loop condition.",
                node.loc,
            )

    # -- Expression evaluation ------------------------------------------

    def _eval(self, node: ASTNode) -> torch.Tensor | str:
        # Fast path: dispatch table lookup (O(1) instead of isinstance chain)
        handler = self._eval_dispatch.get(type(node))
        if handler is not None:
            return handler(node)
        raise InterpreterError(f"Unknown expression: {type(node).__name__}", node.loc)

    def _eval_number_literal(self, node: NumberLiteral) -> torch.Tensor:
        key = (node.value, self._device_str, self._dtype)
        cached = self._literal_cache.get(key)
        if cached is not None:
            return cached
        t = torch.tensor(node.value, dtype=self._dtype, device=self.device)
        self._literal_cache[key] = t
        return t

    def _eval_array_literal(self, node: ArrayLiteral) -> torch.Tensor:
        elements = [self._eval(elem) for elem in node.elements]
        if self.spatial_shape:
            expanded = [_ensure_spatial(e, self.spatial_shape) for e in elements]
        else:
            expanded = [e if isinstance(e, torch.Tensor) else torch.tensor(float(e), dtype=self._dtype, device=self.device) for e in elements]
        return torch.stack(expanded, dim=-1)

    def _eval_identifier(self, node: Identifier) -> torch.Tensor:
        val = self.env.get(node.name)
        if val is not None:
            return val
        raise InterpreterError(f"Undefined variable: '{node.name}'", node.loc)

    def _eval_binding(self, node: BindingRef) -> torch.Tensor:
        val = self.bindings.get(node.name)
        if val is not None:
            return val
        raise InterpreterError(f"Unbound input: '@{node.name}'", node.loc)

    def _eval_channel_access(self, node: ChannelAccess) -> torch.Tensor:
        base = self._eval(node.object)
        channels = node.channels

        if len(channels) == 1:
            idx = CHANNEL_MAP.get(channels)
            if idx is None:
                raise InterpreterError(f"Invalid channel: '.{channels}'", node.loc)
            if base.dim() >= 1 and base.shape[-1] > idx:
                return base[..., idx]
            raise InterpreterError(
                f"Cannot access channel '.{channels}' (index {idx}) on tensor with shape {base.shape}",
                node.loc,
            )

        # Multi-channel swizzle
        indices = [CHANNEL_MAP[ch] for ch in channels if ch in CHANNEL_MAP]
        if len(indices) != len(channels):
            raise InterpreterError(f"Invalid swizzle: '.{channels}'", node.loc)
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
            idx = torch.clamp(torch.floor(index).long(), 0, arr_size - 1)

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
        idx = torch.clamp(torch.floor(index).long(), 0, arr_size - 1)

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
            # Matrix operations: check types for * operator before broadcast
            if op == "*":
                lt = self.type_map.get(id(node.left))
                rt = self.type_map.get(id(node.right))
                if lt is not None and rt is not None:
                    if lt.is_matrix and rt.is_matrix:
                        return torch.matmul(left, right)
                    if lt.is_matrix and rt.is_vector:
                        return torch.matmul(left, right.unsqueeze(-1)).squeeze(-1)

            # Ensure compatible shapes (inline the common equal-dim case)
            if left.dim() != right.dim():
                left, right = _broadcast_pair(left, right)

            # Inline operator dispatch (avoids lambda call overhead)
            if op == "+":
                return left + right
            if op == "*":
                return left * right
            if op == "-":
                return left - right
            if op == "/":
                return left / (right + SAFE_EPSILON * (right == 0).float())
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
                return torch.fmod(left, right + SAFE_EPSILON * (right == 0).float())
            raise InterpreterError(f"Unknown operator: {op}", node.loc)

        # Slow path: string operations
        if isinstance(left, str) or isinstance(right, str):
            if isinstance(left, str) and isinstance(right, str):
                if op == "+":
                    return left + right
                elif op == "==":
                    return torch.tensor(1.0 if left == right else 0.0, dtype=self._dtype, device=self.device)
                elif op == "!=":
                    return torch.tensor(1.0 if left != right else 0.0, dtype=self._dtype, device=self.device)
                else:
                    raise InterpreterError(f"Operator '{op}' is not supported for strings", node.loc)
            raise InterpreterError(
                f"Cannot use operator '{op}' between string and numeric types", node.loc
            )

    def _eval_unary(self, node: UnaryOp) -> torch.Tensor:
        operand = self._eval(node.operand)
        if node.op == "-":
            return -operand
        elif node.op == "!":
            return (operand <= 0.5).float()
        raise InterpreterError(f"Unknown unary operator: {node.op}", node.loc)

    def _eval_ternary(self, node: TernaryOp) -> torch.Tensor | str:
        cond = self._eval(node.condition)
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

    def _eval_function_call(self, node: FunctionCall) -> torch.Tensor | str:
        fn = self.functions.get(node.name)
        if fn is None:
            raise InterpreterError(f"Unknown function: '{node.name}'", node.loc)

        args = [self._eval(arg) for arg in node.args]
        try:
            result = fn(*args)
        except Exception as e:
            raise InterpreterError(f"Error in function '{node.name}': {e}", node.loc)

        # Fast path: most stdlib functions return tensors
        if result.__class__ is torch.Tensor:
            return result
        # Slow path: str, list, or scalar
        if isinstance(result, (str, list)):
            return result
        return torch.tensor(float(result), dtype=self._dtype, device=self.device)

    def _eval_vec_constructor(self, node: VecConstructor) -> torch.Tensor:
        args = [self._eval(arg) for arg in node.args]
        n = node.size

        if len(args) == 1:
            val = args[0]
            if val.__class__ is torch.Tensor:
                # vec4(vec4_val) — identity / type cast
                arg_type = self.type_map.get(id(node.args[0]))
                if arg_type is not None and arg_type.is_vector and arg_type.channels == n:
                    return val
                if val.dim() == 0:
                    # Scalar tensor → broadcast to [..., N]
                    return val.unsqueeze(-1).expand(*(() if not self.spatial_shape else self.spatial_shape), n)
                else:
                    # Spatial scalar [B,H,W] → [B,H,W,N]
                    return val.unsqueeze(-1).expand(*val.shape, n)
            # Python scalar
            return torch.tensor(float(val), dtype=self._dtype, device=self.device).unsqueeze(-1).expand(
                *(() if not self.spatial_shape else self.spatial_shape), n
            )

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
                f"vec{n} expects {n} total components, got {len(components)}",
                node.loc,
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
                f"mat{n} expects {n * n} arguments or 1 (scaled identity), got {len(args)}",
                node.loc,
            )

    def _eval_cast(self, node: CastExpr) -> torch.Tensor | str:
        value = self._eval(node.expr)
        if node.target_type == "string":
            # Convert number to string
            if isinstance(value, torch.Tensor):
                v = value.item() if value.dim() == 0 else value.float().mean().item()
                return str(int(v)) if v == int(v) else str(v)
            return str(value)
        if node.target_type == "int":
            return torch.floor(value)
        elif node.target_type == "float":
            return value.float()
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
            return torch.tensor(0.0, dtype=self._dtype, device=self.device)

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
_BUILTIN_NAMES = frozenset({"ix", "iy", "u", "v", "iw", "ih", "fi", "fn", "PI", "E", "ic"})


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
        # NumberLiteral, StringLiteral, BindingRef, BreakStmt, ContinueStmt, ParamDecl — skip

    return frozenset(found)


# -- Module-level constants and utility functions ----------------------

# Operator dispatch table for _eval_binop (avoids if/elif chain)
_BINOP_TABLE = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / (b + SAFE_EPSILON * (b == 0).float()),
    "%": lambda a, b: torch.fmod(a, b + SAFE_EPSILON * (b == 0).float()),
    "==": lambda a, b: (a == b).float(),
    "!=": lambda a, b: (a != b).float(),
    "<": lambda a, b: (a < b).float(),
    ">": lambda a, b: (a > b).float(),
    "<=": lambda a, b: (a <= b).float(),
    ">=": lambda a, b: (a >= b).float(),
    "&&": lambda a, b: ((a > 0.5) & (b > 0.5)).float(),
    "||": lambda a, b: ((a > 0.5) | (b > 0.5)).float(),
}

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
    """
    ad, bd = a.dim(), b.dim()
    if ad == bd:
        return a, b

    # Fast path: 1 dimension difference (e.g. [B,H,W] vs [B,H,W,C])
    if ad - bd == 1:
        return a, b.unsqueeze(-1).expand_as(a)
    if bd - ad == 1:
        return a.unsqueeze(-1).expand_as(b), b

    # General case: multiple dimension difference
    if ad > bd:
        while b.dim() < ad:
            b = b.unsqueeze(-1)
        return a, b.expand_as(a)
    else:
        while a.dim() < bd:
            a = a.unsqueeze(-1)
        return a.expand_as(b), b


def _tensor_where(cond: torch.Tensor, then_val: torch.Tensor, else_val: torch.Tensor) -> torch.Tensor:
    """torch.where with broadcasting support for mixed scalar/vector cases."""
    then_val, else_val = _broadcast_pair(then_val, else_val)

    # Expand condition to match value shapes
    while cond.dim() < then_val.dim():
        cond = cond.unsqueeze(-1)

    try:
        cond = cond.expand_as(then_val)
    except RuntimeError:
        pass

    return torch.where(cond, then_val, else_val)
