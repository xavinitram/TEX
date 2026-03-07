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
    ASTNode, Program, VarDecl, Assignment, IfElse, ForLoop, ExprStatement,
    BinOp, UnaryOp, TernaryOp, FunctionCall, Identifier, BindingRef,
    ChannelAccess, NumberLiteral, StringLiteral, VecConstructor, CastExpr, SourceLoc,
    ArrayDecl, ArrayIndexAccess, ArrayLiteral, MatConstructor,
)
from ..tex_compiler.type_checker import TEXType, CHANNEL_MAP
from .stdlib import TEXStdlib

# Hard limit on for-loop iterations to prevent infinite loops
MAX_LOOP_ITERATIONS = 1024


class InterpreterError(Exception):
    def __init__(self, message: str, loc: SourceLoc = None):
        self.loc = loc
        prefix = f"[{loc}] " if loc else ""
        super().__init__(f"{prefix}{message}")


class Interpreter:
    """
    Evaluates a TEX AST against concrete tensor inputs.

    Usage:
        interp = Interpreter()
        result = interp.execute(ast, bindings, type_map)
    """

    def __init__(self):
        self.env: dict[str, torch.Tensor] = {}
        self.bindings: dict[str, torch.Tensor] = {}
        self.type_map: dict[int, TEXType] = {}
        self.functions: dict[str, callable] = TEXStdlib.get_functions()
        self.device: torch.device = torch.device("cpu")
        self.spatial_shape: tuple[int, int, int] | None = None  # (B, H, W)
        self._array_meta: dict[str, int] = {}  # var_name -> array size

    def execute(
        self,
        program: Program,
        bindings: dict[str, torch.Tensor | float | int],
        type_map: dict[int, TEXType],
        device: torch.device | str = "cpu",
        latent_channel_count: int = 0,
    ) -> torch.Tensor:
        """
        Execute a TEX program.

        Args:
            program: Parsed and type-checked AST
            bindings: @ binding values (name -> tensor or scalar)
            type_map: AST node id -> TEXType from type checker
            device: Target device

        Returns:
            The value of @OUT after execution
        """
        self.device = torch.device(device)
        self.type_map = type_map
        self.latent_channel_count = latent_channel_count
        self.env = {}
        self.bindings = {}
        self._array_meta = {}

        # Process bindings
        for name, value in bindings.items():
            if isinstance(value, str):
                self.bindings[name] = value  # strings pass through as Python str
            elif isinstance(value, torch.Tensor):
                self.bindings[name] = value.to(self.device)
            else:
                self.bindings[name] = torch.tensor(float(value), dtype=torch.float32, device=self.device)

        # Determine spatial context from image inputs
        self.spatial_shape = self._determine_spatial_shape()

        # Create built-in coordinate variables
        self._create_builtins()

        # Execute statements
        for stmt in program.statements:
            self._exec_stmt(stmt)

        # Return @OUT
        if "OUT" not in self.bindings:
            raise InterpreterError("TEX program must assign to @OUT")

        return self.bindings["OUT"]

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

    def _create_builtins(self):
        """Create built-in variables (ix, iy, u, v, iw, ih, PI, E)."""
        if self.spatial_shape:
            B, H, W = self.spatial_shape

            # ix: pixel x-coordinate [0, W-1]
            ix = torch.arange(W, dtype=torch.float32, device=self.device)
            ix = ix.view(1, 1, W).expand(B, H, W)
            self.env["ix"] = ix

            # iy: pixel y-coordinate [0, H-1]
            iy = torch.arange(H, dtype=torch.float32, device=self.device)
            iy = iy.view(1, H, 1).expand(B, H, W)
            self.env["iy"] = iy

            # u, v: normalized coordinates [0, 1]
            self.env["u"] = ix / max(W - 1, 1)
            self.env["v"] = iy / max(H - 1, 1)

            # iw, ih: dimensions
            self.env["iw"] = torch.tensor(float(W), dtype=torch.float32, device=self.device)
            self.env["ih"] = torch.tensor(float(H), dtype=torch.float32, device=self.device)

            # fi: frame/batch index [0, B-1]
            fi = torch.arange(B, dtype=torch.float32, device=self.device)
            fi = fi.view(B, 1, 1).expand(B, H, W)
            self.env["fi"] = fi

            # fn: total frame/batch count
            self.env["fn"] = torch.tensor(float(B), dtype=torch.float32, device=self.device)
        else:
            # No spatial context — pure scalar mode
            self.env["ix"] = torch.tensor(0.0, device=self.device)
            self.env["iy"] = torch.tensor(0.0, device=self.device)
            self.env["u"] = torch.tensor(0.0, device=self.device)
            self.env["v"] = torch.tensor(0.0, device=self.device)
            self.env["iw"] = torch.tensor(1.0, device=self.device)
            self.env["ih"] = torch.tensor(1.0, device=self.device)
            self.env["fi"] = torch.tensor(0.0, device=self.device)
            self.env["fn"] = torch.tensor(1.0, device=self.device)

        self.env["PI"] = torch.tensor(math.pi, dtype=torch.float32, device=self.device)
        self.env["E"] = torch.tensor(math.e, dtype=torch.float32, device=self.device)
        self.env["ic"] = torch.tensor(float(self.latent_channel_count), dtype=torch.float32, device=self.device)

    # -- Statement execution --------------------------------------------

    def _exec_stmt(self, node: ASTNode):
        if isinstance(node, VarDecl):
            self._exec_var_decl(node)
        elif isinstance(node, ArrayDecl):
            self._exec_array_decl(node)
        elif isinstance(node, Assignment):
            self._exec_assignment(node)
        elif isinstance(node, IfElse):
            self._exec_if_else(node)
        elif isinstance(node, ForLoop):
            self._exec_for_loop(node)
        elif isinstance(node, ExprStatement):
            self._eval(node.expr)
        else:
            raise InterpreterError(f"Unknown statement: {type(node).__name__}", node.loc)

    def _exec_var_decl(self, node: VarDecl):
        if node.initializer:
            value = self._eval(node.initializer)
        else:
            # Default initialize based on type
            declared = self.type_map.get(id(node), TEXType.FLOAT)
            value = self._default_value(declared)
        self.env[node.name] = value

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
                    expanded = [e if isinstance(e, torch.Tensor) else torch.tensor(float(e), dtype=torch.float32, device=self.device) for e in elements]
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
                    value = torch.zeros(B, H, W, size, channels, dtype=torch.float32, device=self.device)
                else:
                    value = torch.zeros(size, channels, dtype=torch.float32, device=self.device)
            else:
                if self.spatial_shape:
                    B, H, W = self.spatial_shape
                    value = torch.zeros(B, H, W, size, dtype=torch.float32, device=self.device)
                else:
                    value = torch.zeros(size, dtype=torch.float32, device=self.device)

        self.env[node.name] = value
        self._array_meta[node.name] = size

    def _promote_to_channels(self, tensor: torch.Tensor, channels: int) -> torch.Tensor:
        """Ensure a tensor has the correct number of channels for a vec array element."""
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(float(tensor), dtype=torch.float32, device=self.device)
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

    def _exec_assignment(self, node: Assignment):
        value = self._eval(node.value)
        target = node.target

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
                val_t = value if isinstance(value, torch.Tensor) else torch.tensor(float(value), dtype=torch.float32, device=self.device)
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
                result[idx] = value if isinstance(value, torch.Tensor) else torch.tensor(float(value), dtype=torch.float32, device=self.device)
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

    def _exec_if_else(self, node: IfElse):
        """
        Execute if/else by evaluating both branches and using torch.where.

        This is the vectorized approach: both branches execute, and the result
        is selected per-element by the condition mask.
        """
        cond = self._eval(node.condition)

        # Snapshot the environment before branches
        env_snapshot = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in self.env.items()}
        bindings_snapshot = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in self.bindings.items()}
        meta_snapshot = dict(self._array_meta)

        # Execute then-branch
        for stmt in node.then_body:
            self._exec_stmt(stmt)
        then_env = dict(self.env)
        then_bindings = dict(self.bindings)
        then_meta = dict(self._array_meta)

        # Restore and execute else-branch
        self.env = env_snapshot
        self.bindings = bindings_snapshot
        self._array_meta = dict(meta_snapshot)

        if node.else_body:
            # Re-snapshot for else (since we restored)
            env_snapshot2 = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in self.env.items()}
            bindings_snapshot2 = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in self.bindings.items()}

            for stmt in node.else_body:
                self._exec_stmt(stmt)
            else_env = dict(self.env)
            else_bindings = dict(self.bindings)
        else:
            else_env = dict(env_snapshot)
            else_bindings = dict(bindings_snapshot)

        # Merge using torch.where (tensors) or scalar condition (strings)
        cond_bool = (cond > 0.5) if cond.dtype == torch.float32 else cond.bool()
        # For string merging: use majority vote for spatial conditions
        cond_scalar_bool = cond.float().mean().item() > 0.5 if cond.dim() > 0 else cond.item() > 0.5

        # Merge env variables
        all_keys = set(then_env.keys()) | set(else_env.keys())
        for key in all_keys:
            then_val = then_env.get(key)
            else_val = else_env.get(key)
            if then_val is not None and else_val is not None:
                if isinstance(then_val, torch.Tensor) and isinstance(else_val, torch.Tensor):
                    self.env[key] = _tensor_where(cond_bool, then_val, else_val)
                elif isinstance(then_val, str) or isinstance(else_val, str):
                    # String merge: pick based on scalar condition
                    self.env[key] = then_val if cond_scalar_bool else else_val
                else:
                    self.env[key] = then_val
            elif then_val is not None:
                self.env[key] = then_val
            else:
                self.env[key] = else_val

        # Merge bindings
        all_bkeys = set(then_bindings.keys()) | set(else_bindings.keys())
        for key in all_bkeys:
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
        """
        # Execute initializer
        self._exec_stmt(node.init)

        iteration = 0
        while iteration < MAX_LOOP_ITERATIONS:
            # Evaluate condition
            cond = self._eval(node.condition)

            # Check if loop should continue
            # For scalar conditions, check the value directly
            if cond.dim() == 0:
                if cond.item() <= 0.5:
                    break
            else:
                # For spatial conditions, continue if ANY pixel still needs iteration
                # This is a simplification — the loop runs for the maximum count
                if (cond > 0.5).float().sum().item() == 0:
                    break

            # Execute body
            for stmt in node.body:
                self._exec_stmt(stmt)

            # Execute update
            self._exec_stmt(node.update)

            iteration += 1

        if iteration >= MAX_LOOP_ITERATIONS:
            raise InterpreterError(
                f"For loop exceeded maximum iteration limit ({MAX_LOOP_ITERATIONS}). "
                f"Check your loop condition.",
                node.loc,
            )

    # -- Expression evaluation ------------------------------------------

    def _eval(self, node: ASTNode) -> torch.Tensor | str:
        if isinstance(node, NumberLiteral):
            return torch.tensor(node.value, dtype=torch.float32, device=self.device)

        if isinstance(node, StringLiteral):
            return node.value

        if isinstance(node, Identifier):
            return self._eval_identifier(node)

        if isinstance(node, BindingRef):
            return self._eval_binding(node)

        if isinstance(node, ChannelAccess):
            return self._eval_channel_access(node)

        if isinstance(node, BinOp):
            return self._eval_binop(node)

        if isinstance(node, UnaryOp):
            return self._eval_unary(node)

        if isinstance(node, TernaryOp):
            return self._eval_ternary(node)

        if isinstance(node, FunctionCall):
            return self._eval_function_call(node)

        if isinstance(node, VecConstructor):
            return self._eval_vec_constructor(node)

        if isinstance(node, MatConstructor):
            return self._eval_mat_constructor(node)

        if isinstance(node, CastExpr):
            return self._eval_cast(node)

        if isinstance(node, ArrayIndexAccess):
            return self._eval_array_index(node)

        if isinstance(node, ArrayLiteral):
            # Evaluate elements and stack
            elements = [self._eval(elem) for elem in node.elements]
            if self.spatial_shape:
                expanded = [_ensure_spatial(e, self.spatial_shape) for e in elements]
            else:
                expanded = [e if isinstance(e, torch.Tensor) else torch.tensor(float(e), dtype=torch.float32, device=self.device) for e in elements]
            return torch.stack(expanded, dim=-1)

        raise InterpreterError(f"Unknown expression: {type(node).__name__}", node.loc)

    def _eval_identifier(self, node: Identifier) -> torch.Tensor:
        if node.name in self.env:
            return self.env[node.name]
        raise InterpreterError(f"Undefined variable: '{node.name}'", node.loc)

    def _eval_binding(self, node: BindingRef) -> torch.Tensor:
        if node.name in self.bindings:
            return self.bindings[node.name]
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

        # String operations
        if isinstance(left, str) or isinstance(right, str):
            if isinstance(left, str) and isinstance(right, str):
                if node.op == "+":
                    return left + right
                elif node.op == "==":
                    return torch.tensor(1.0 if left == right else 0.0, dtype=torch.float32, device=self.device)
                elif node.op == "!=":
                    return torch.tensor(1.0 if left != right else 0.0, dtype=torch.float32, device=self.device)
                else:
                    raise InterpreterError(f"Operator '{node.op}' is not supported for strings", node.loc)
            raise InterpreterError(
                f"Cannot use operator '{node.op}' between string and numeric types", node.loc
            )

        # Matrix operations: use type info to dispatch matmul vs element-wise
        lt = self.type_map.get(id(node.left))
        rt = self.type_map.get(id(node.right))
        if node.op == "*" and lt is not None and rt is not None:
            if lt.is_matrix and rt.is_matrix:
                return torch.matmul(left, right)
            if lt.is_matrix and rt.is_vector:
                return torch.matmul(left, right.unsqueeze(-1)).squeeze(-1)

        # Ensure compatible shapes
        left, right = _broadcast_pair(left, right)

        op = node.op
        if op == "+":
            return left + right
        elif op == "-":
            return left - right
        elif op == "*":
            return left * right
        elif op == "/":
            return left / (right + 1e-10 * (right == 0).float())
        elif op == "%":
            return torch.fmod(left, right + 1e-10 * (right == 0).float())
        elif op == "==":
            return (left == right).float()
        elif op == "!=":
            return (left != right).float()
        elif op == "<":
            return (left < right).float()
        elif op == ">":
            return (left > right).float()
        elif op == "<=":
            return (left <= right).float()
        elif op == ">=":
            return (left >= right).float()
        elif op == "&&":
            return ((left > 0.5) & (right > 0.5)).float()
        elif op == "||":
            return ((left > 0.5) | (right > 0.5)).float()
        else:
            raise InterpreterError(f"Unknown operator: {op}", node.loc)

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

        # String functions can return str, list functions can return list — pass through
        if isinstance(result, (str, list)):
            return result
        if not isinstance(result, torch.Tensor):
            result = torch.tensor(float(result), dtype=torch.float32, device=self.device)
        return result

    def _eval_vec_constructor(self, node: VecConstructor) -> torch.Tensor:
        args = [self._eval(arg) for arg in node.args]

        if len(args) == 1:
            # Broadcast: vec4(0.5) -> [0.5, 0.5, 0.5, 0.5]
            val = args[0]
            components = [val] * node.size
        elif len(args) == node.size:
            components = args
        else:
            raise InterpreterError(
                f"vec{node.size} expects {node.size} arguments or 1 (broadcast), got {len(args)}",
                node.loc,
            )

        # Ensure all components have compatible spatial shapes
        # Find the maximum spatial shape among components
        max_shape = self._get_max_spatial_shape(components)

        expanded = []
        for c in components:
            c_exp = _ensure_spatial(c, max_shape)
            expanded.append(c_exp)

        return torch.stack(expanded, dim=-1)

    def _eval_mat_constructor(self, node: MatConstructor) -> torch.Tensor:
        n = node.size  # 3 or 4
        args = [self._eval(arg) for arg in node.args]

        if len(args) == 1:
            # Scaled identity: mat3(1.0) → identity * scalar
            val = args[0]
            eye = torch.eye(n, dtype=torch.float32, device=self.device)
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
                return torch.zeros(B, H, W, 3, dtype=torch.float32, device=self.device)
            return torch.zeros(3, dtype=torch.float32, device=self.device)
        elif t == TEXType.VEC4:
            if self.spatial_shape:
                B, H, W = self.spatial_shape
                return torch.zeros(B, H, W, 4, dtype=torch.float32, device=self.device)
            return torch.zeros(4, dtype=torch.float32, device=self.device)
        elif t.is_matrix:
            n = t.mat_size
            return torch.zeros(n, n, dtype=torch.float32, device=self.device)
        else:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)

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


# -- Module-level utility functions ------------------------------------

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
    if a.dim() == b.dim():
        return a, b

    # One is vector (has channel dim), one is scalar
    if a.dim() > b.dim():
        # b is the scalar-like one, expand to match a
        while b.dim() < a.dim():
            b = b.unsqueeze(-1)
        return a, b.expand_as(a)
    else:
        while a.dim() < b.dim():
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
