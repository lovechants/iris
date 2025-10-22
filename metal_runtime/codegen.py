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

    def get_type(self, name:str) -> str:
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
                    params.append(f"constant {param_type}& {param_name} [[buffer({i})]]")
            else:
                raise MSLCodegenError(f"Unknown type for parameter: {param_name}")
        params.append("uint tid [[thread_position_in_grid]]")


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

        if isinstance(target, ast.Name):
            target_name = target.id
            if value == "tid" and self.thread_id_var is None:
                self.thread_id_var = target_name
                self.type_inference.set_type(name=target_name, typ="uint")
                self.emit(f"uint {target_name} = {value};")
            else:
                self.type_inference.set_type(name=target_name, typ="uint")
                self.emit(f"uint {target_name} = {value};")
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

        self.emit(f"for (uint {var_name} = {start}; {var_name} < {end}; {var_name} += {step}){{")
        self.indent_level += 1
        for stmt in node.body:
            self.visit(stmt)
        self.indent_level -= 1
        self.emit("}")

    def visit_Expr(self, node: ast.Expr):
        self.emit(f"{self.visit(node.value)};")

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "thread_id":
                self.used_builtins.add("thread_id")
                return "tid"
            elif node.func.attr == "program_id":
                self.used_builtins.add("program_id")
                dim = self.visit(node.args[0]) if node.args else "0"
                if dim == "0":
                    return "tid"
                else:
                    raise MSLCodegenError("Only 1D program_id supproted now")

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

    def visit_Compare(self, node: ast.Compare):
        left = self.visit(node.left)
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise MSLCodegenError("Complex comparisons not yet supported")

        op = self.visit(node.ops[0])
        right = self.visit(node.comparators[0])
        return f"{left} {op} {right}"

    def visit_Name(self, node: ast.Name):
        return node.id

    def visit_UnaryOp(self, node: ast.UnaryOp):
        operand = self.visit(node.operand)
        op = self.visit(node.op)
        return f"{op}{operand}"

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

def generate_msl(func: Callable, function_name: str, param_types: Dict[str, str]) -> str:
    source = inspect.getsource(func)
    source = textwrap.dedent(source) # Basically it counted the indentation from the source too so
    tree = ast.parse(source)
    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_def = node
            break
    if func_def is None:
        raise MSLCodegenError("Could not find function defintion")

    codegen = MSLCodeGenerator(function_name, param_types)
    return codegen.visit_FunctionDef(func_def)
