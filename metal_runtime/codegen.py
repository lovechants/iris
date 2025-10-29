import ast
import inspect
from typing import Dict, List, Optional, Set, Callable
import textwrap


class MSLCodegenError(Exception):
    pass


class TypeInference:
    def __init__(self):
        self.var_types: Dict[str, str] = {}

    def infer_binop_type(self, left_type: str, right_type: str) -> str:
        if left_type == right_type:
            return right_type

        if "float" in left_type or "float" in right_type:
            return "float"

        return "int"

    def set_type(self, name: str, typ: str):
        self.var_types[name] = typ

    def get_type(self, name: str) -> str:
        return self.var_types.get(name, "float")


class MSLCodeGenerator(ast.NodeVisitor):
    def __init__(self, function_name: str, param_types: Dict[str, str]):
        self.function_name = function_name
        self.param_types = param_types
        self.type_inference = TypeInference()
        self.indent_level = 0
        self.code_lines: List[str] = []
        self.used_builtins: Set[str] = set()
        self.thread_id_var = None
        self.uses_3d_threads = None

    def indent(self) -> str:
        return "    " * self.indent_level

    def emit(self, line: str):
        self.code_lines.append(self.indent() + line)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> str:
        self.emit("#include <metal_stdlib>")
        self.emit("using namespace metal;")
        self.emit("")

        params = []
        for i, arg in enumerate(node.args.args):
            param_name = arg.arg
            if param_name in self.param_types:
                param_type = self.param_types[param_name]
                if param_type.startswith("device"):
                    params.append(f"{param_type} {param_name} [[buffer({i})]]")
                elif param_type.startswith("threadgroup"):
                    params.append(f"{param_type} {param_name} [[buffer({i})]]")
                else:
                    params.append(
                        f"constant {param_type}& {param_name} [[buffer({i})]]"
                    )
            else:
                raise MSLCodegenError(f"Unknown type for parameter: {param_name}")

        uses_3d = any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in ("thread_id_z", "thread_id_3d")
            for n in ast.walk(node)
        )
        uses_2d = any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in ("thread_id_x", "thread_id_y", "thread_id_2d")
            for n in ast.walk(node)
        )

        if uses_3d:
            params.append("uint3 tid3d [[thread_position_in_grid]]")
            self.uses_3d_threads = True
        elif uses_2d:
            params.append("uint2 tid2d [[thread_position_in_grid]]")
            self.uses_3d_threads = False
        else:
            params.append("uint tid [[thread_position_in_grid]]")
            self.uses_3d_threads = False

        for pname, ptype in self.param_types.items():
            base_type = ptype.split()[-1].replace("&", "").replace("*", "")
            if "uint" in base_type or "int" in base_type:
                self.type_inference.set_type(pname, "uint")

        self.emit(f"kernel void {self.function_name}(")
        self.indent_level += 1
        for j, param in enumerate(params):
            if j < len(params) - 1:
                self.emit(f"{param},")
            else:
                self.emit(param)
        self.indent_level -= 1
        self.emit(")")
        self.emit("{")

        self.indent_level += 1
        for stmt in node.body:
            self.visit(stmt)
        self.indent_level -= 1

        self.emit("}")
        return "\n".join(self.code_lines)

    def visit_Assign(self, node: ast.Assign):
        if len(node.targets) != 1:
            raise MSLCodegenError("Multiple assignment targets not supported")

        target = node.targets[0]
        value = self.visit(node.value)
        target_name = target.id if isinstance(target, ast.Name) else None

        if isinstance(target, ast.Name):
            if target_name in self.type_inference.var_types:
                self.emit(f"{target_name} = {value};")
                return

            if value == "tid" and self.thread_id_var is None:
                if target_name != "tid":
                    self.emit(f"uint {target_name} = tid;")
                return

            inferred_type = None
            if isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, float):
                    inferred_type = "float"
                elif isinstance(node.value.value, int):
                    inferred_type = "uint"
            else:
                rhs_vars = [n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)]
                rhs_types = [self.type_inference.get_type(v) for v in rhs_vars]

                if rhs_types:
                    if all(t in ("uint", "int") for t in rhs_types):
                        inferred_type = "uint"
                    elif any("float" in t for t in rhs_types):
                        inferred_type = "float"
                    else:
                        inferred_type = "float"
                else:
                    inferred_type = "float"

            # --- Force integer for index-like variables ---
            if target_name in ("x", "y", "z", "idx", "row", "col"):
                inferred_type = "uint"

            var_type = inferred_type or "float"
            self.type_inference.set_type(target_name, var_type)
            self.emit(f"{var_type} {target_name} = {value};")

        elif isinstance(target, ast.Subscript):
            target_code = self.visit(target)
            self.emit(f"{target_code} = {value};")
        else:
            raise MSLCodegenError(f"Unsupported assignment target: {type(target)}")

    def visit_If(self, node: ast.If):
        test = self.visit(node.test)
        self.emit(f"if ({test}) {{")
        self.indent_level += 1
        for stmt in node.body:
            self.visit(stmt)
        self.indent_level -= 1

        if node.orelse:
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                self.emit("} else if (")
                self.indent_level += 1
                elif_test = self.visit(node.orelse[0].test)
                self.indent_level -= 1
                self.code_lines[-1] = self.code_lines[-1].rstrip()
                self.code_lines[-1] += f"{elif_test}) {{"
                self.indent_level -= 1
                for stmt in node.orelse[0].body:
                    self.visit(stmt)
                self.indent_level += 1
                if node.orelse[0].orelse:
                    self.emit("} else {")
                    self.indent_level += 1
                    for stmt in node.orelse[0].orelse:
                        self.visit(stmt)
                    self.indent_level -= 1
            else:
                self.emit("} else {")
                self.indent_level += 1
                for stmt in node.orelse:
                    self.visit(stmt)
                self.indent_level -= 1

        self.emit("}")

    def visit_For(self, node: ast.For):
        if not isinstance(node.target, ast.Name):
            raise MSLCodegenError("For loop target must be a simple variable")
        if not isinstance(node.iter, ast.Call):
            raise MSLCodegenError("For loop iterator must be a range call")
        if not isinstance(node.iter.func, ast.Name) or node.iter.func.id != "range":
            raise MSLCodegenError("Only range() loops supported")

        var_name = node.target.id
        args = node.iter.args
        if len(args) == 1:
            start = "0"
            end = self.visit(args[0])
            step = "1"
        elif len(args) == 2:
            start = self.visit(args[0])
            end = self.visit(args[1])
            step = "1"
        elif len(args) == 3:
            start = self.visit(args[0])
            end = self.visit(args[1])
            step = self.visit(args[2])
        else:
            raise MSLCodegenError("range() must have 1-3 args")

        self.emit(
            f"for (uint {var_name} = {start}; {var_name} < {end}; {var_name} += {step}){{"
        )
        self.indent_level += 1
        for stmt in node.body:
            self.visit(stmt)
        self.indent_level -= 1
        self.emit("}")

    def visit_Expr(self, node: ast.Expr):
        self.emit(f"{self.visit(node.value)};")

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Attribute):
            attr = node.func.attr

            if attr == "thread_id":
                self.used_builtins.add("thread_id")
                return "tid"
            elif attr == "program_id":
                self.used_builtins.add("program_id")
                dim = self.visit(node.args[0]) if node.args else "0"
                if dim == "0":
                    return "tid"
                raise MSLCodegenError("Only 1D program_id supported now")

            elif attr == "thread_id_x":
                self.used_builtins.add("thread_id_x")
                self.type_inference.set_type("thread_id_x", "uint")
                return "tid3d.x" if self.uses_3d_threads else "tid2d.x"
            elif attr == "thread_id_y":
                self.used_builtins.add("thread_id_y")
                self.type_inference.set_type("thread_id_y", "uint")
                return "tid3d.y" if self.uses_3d_threads else "tid2d.y"
            elif attr == "thread_id_z":
                self.used_builtins.add("thread_id_z")
                self.type_inference.set_type("thread_id_z", "uint")
                return "tid3d.z"
            elif attr == "thread_id_2d":
                self.used_builtins.add("thread_id_2d")
                return "tid2d"
            elif attr == "thread_id_3d":
                self.used_builtins.add("thread_id_3d")
                return "tid3d"

        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name == "range":
                raise MSLCodegenError("range() should only appear in for loops")
            args = [self.visit(arg) for arg in node.args]
            return f"{func_name}({', '.join(args)})"

        raise MSLCodegenError(f"Unsupported call: {ast.dump(node)}")

    def visit_BinOp(self, node: ast.BinOp):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = self.visit(node.op)
        return f"({left} {op} {right})"
    
    # TODO: In the future this codegen needs to be much more robust to handle the leftside rightside infra 
    def visit_Compare(self, node: ast.Compare) -> str:
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise MSLCodegenError("Complex comparisons not yet supported")

        left = self.visit(node.left)
        op = self.visit(node.ops[0])
        right = self.visit(node.comparators[0])

        if isinstance(node.comparators[0], ast.Constant) and isinstance(node.comparators[0].value, (int, bool)):
            right = f"static_cast<float>({right})"
        
        return f"{left} {op} {right}"

    def visit_Name(self, node: ast.Name):
        return node.id

    def visit_UnaryOp(self, node: ast.UnaryOp):
        operand = self.visit(node.operand)
        op = self.visit(node.op)
        return f"{op}{operand}"

    def visit_BoolOp(self, node: ast.BoolOp):
        op = " && " if isinstance(node.op, ast.And) else " || "
        return op.join(self.visit(v) for v in node.values)

    def visit_Subscript(self, node: ast.Subscript):
        value = self.visit(node.value)
        index = self.visit(node.slice)
        return f"{value}[{index}]"

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, bool):
            return "true" if node.value else "false"
        return str(node.value)

    def visit_Add(self, node: ast.Add):
        return "+"

    def visit_Sub(self, node: ast.Sub):
        return "-"

    def visit_Mult(self, node: ast.Mult):
        return "*"

    def visit_Div(self, node: ast.Div):
        return "/"

    def visit_Mod(self, node: ast.Mod):
        return "%"

    def visit_Lt(self, node: ast.Lt):
        return "<"

    def visit_LtE(self, node: ast.LtE):
        return "<="

    def visit_Gt(self, node: ast.Gt):
        return ">"

    def visit_GtE(self, node: ast.GtE):
        return ">="

    def visit_Eq(self, node: ast.Eq):
        return "=="

    def visit_NotEq(self, node: ast.NotEq):
        return "!="

    def visit_USub(self, node: ast.USub):
        return "-"

    def visit_Not(self, node: ast.Not):
        return "!"

    def visit_FloorDiv(self, node: ast.FloorDiv):
        return "/"

    def visit_IfExp(self, node: ast.IfExp) -> str:
        """Handles inline ternary expressions: a if cond else b"""
        test = self.visit(node.test)
        body = self.visit(node.body)
        orelse = self.visit(node.orelse)
        return f"(({test}) ? ({body}) : ({orelse}))"

def generate_msl(
    func: Callable, function_name: str, param_types: Dict[str, str]
) -> str:
    source = inspect.getsource(func)
    source = textwrap.dedent(source)
    tree = ast.parse(source)

    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_def = node
            break
    if func_def is None:
        raise MSLCodegenError("Could not find function definition")

    if (
        func_def.body
        and isinstance(func_def.body[0], ast.Expr)
        and isinstance(func_def.body[0].value, ast.Constant)
        and isinstance(func_def.body[0].value.value, str)
    ):
        func_def.body.pop(0)

    codegen = MSLCodeGenerator(function_name, param_types)
    return codegen.visit_FunctionDef(func_def)
