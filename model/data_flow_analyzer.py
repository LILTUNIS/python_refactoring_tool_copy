#data_flow_analyzer.py
import ast
import inspect
from typing import Dict, List, Any


def analyze_data_flow(tree: ast.AST, filename: str = "unknown_file.py") -> Dict[str, Dict]:
    """
    Analyze data flow within the AST.
    Returns a dictionary mapping file names to functions and their data flow details.
    """
    data_flow = {filename: {}}  # Wrap results under a filename

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            print(f"DEBUG: Processing function `{node.name}` in analyze_data_flow")

            if node.name not in data_flow[filename]:
                data_flow[filename][node.name] = {
                    "variables": {},
                    "dependencies": {
                        "reads": [],
                        "writes": [],
                        "returns": [],
                        "function_calls": [],
                        "control_flows": [],
                        "exception_handling": [],
                        "side_effects": [],
                        "input_output_relations": []
                    }
                }

            # Analyze variables and dependencies
            variables = {}
            dependencies = extract_dependencies(node)

            for child in ast.walk(node):
                # Track variable definitions and usage
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name):
                            variables[target.id] = {"defined": child.lineno, "used": []}
                elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                    if child.id in variables:
                        variables[child.id]["used"].append(child.lineno)

            data_flow[filename][node.name]["variables"] = variables
            data_flow[filename][node.name]["dependencies"] = dependencies

            print(f"DEBUG: Function `{node.name}` - Variables: {variables}")
            print(f"DEBUG: Function `{node.name}` - Dependencies: {dependencies}")

    print(f"DEBUG: Final Data Flow Analysis Results: {data_flow}")
    return data_flow

def _unparse_expr(node: ast.AST) -> str:
    """
    Safely unparse an AST node into a string.
    Fallback to str(node) if ast.unparse() is not available or fails.
    """
    if hasattr(ast, "unparse"):
        try:
            return ast.unparse(node)
        except Exception:
            return str(node)
    else:
        return str(node)

def extract_dependencies(node: ast.FunctionDef) -> Dict[str, List[Any]]:
    """
    Extract a list of dependencies for a function categorized as:
    - 'reads': Variables read within the function
    - 'writes': Variables assigned within the function
    - 'returns': Values or expressions returned by the function
    - 'function_calls': Other functions called within this function
    - 'control_flows': Control structures (if, for, while, try)
    - 'exception_handling': Exception handling (try, except, finally, raise)
    - 'side_effects': Global state changes or I/O operations
    - 'input_output_relations': How inputs are transformed to outputs
    """
    reads = set()
    writes = set()
    returns = set()
    function_calls = set()
    control_flows = set()
    exception_handling = set()
    side_effects = set()
    input_output_relations = []

    # We'll store BinOp nodes used directly in Return statements to avoid double counting
    skip_binops = set()

    for child in ast.walk(node):
        # 1. Track variable writes (assignments)
        if isinstance(child, ast.Assign):
            # e.g. s = x + y
            for t in child.targets:
                if isinstance(t, ast.Name):
                    writes.add(t.id)

            # If the assigned value is a BinOp, handle it here so we don't create a "temp"
            if isinstance(child.value, ast.BinOp):
                skip_binops.add(child.value)  # So we don't reprocess in the BinOp block

                left_str = _unparse_expr(child.value.left)
                right_str = _unparse_expr(child.value.right)
                op_type = type(child.value.op).__name__

                # If there's exactly one Name target, we can show a direct assignment
                if len(child.targets) == 1 and isinstance(child.targets[0], ast.Name):
                    target_name = child.targets[0].id
                    input_output_relations.append((f"{left_str} {op_type} {right_str}", target_name, op_type))
                else:
                    # If multiple targets or non-Name, fallback to "temp"
                    input_output_relations.append((f"{left_str} {op_type} {right_str}", "temp", op_type))

        # 2. Track variable reads (usages)
        elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            reads.add(child.id)

        # 3. Track return values and input-output relations
        elif isinstance(child, ast.Return):
            if child.value:
                return_expr = _unparse_expr(child.value)
                returns.add(return_expr)

                if isinstance(child.value, ast.BinOp):
                    skip_binops.add(child.value)
                    left_str = _unparse_expr(child.value.left)
                    right_str = _unparse_expr(child.value.right)
                    op_type = type(child.value.op).__name__
                    input_output_relations.append((f"{left_str} {op_type} {right_str}", "return", op_type))

        # 4. Track function calls
        elif isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                function_calls.add(child.func.id)
            elif isinstance(child.func, ast.Attribute):
                if isinstance(child.func.value, ast.Name):
                    full_call = f"{child.func.value.id}.{child.func.attr}"
                else:
                    full_call = child.func.attr
                function_calls.add(full_call)

        # 5. Track control flow structures
        elif isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
            parent = node.name
            flow_type = type(child).__name__
            control_flows.add(f"{parent}->{flow_type}")

        # 6. Track exception handling
        elif isinstance(child, (ast.ExceptHandler, ast.Raise)):
            exception_handling.add(type(child).__name__)

        # Special case for finally: It's part of ast.Try
        elif isinstance(child, ast.Try) and child.finalbody:
            exception_handling.add("Finally")

        # 7. Track I/O operations and global state changes
        elif (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr in ("write", "read", "append", "open", "print")
        ):
            side_effects.add("I/O Operation")

        # 8. Enhanced Input-Output Relations for Binary Operations
        elif isinstance(child, ast.BinOp):
            if child in skip_binops:
                continue

            left_str = _unparse_expr(child.left)
            right_str = _unparse_expr(child.right)
            op_type = type(child.op).__name__

            # Check context for better naming
            # 8.1 Condition Checks (e.g., n % 2 == 0)
            parent = getattr(child, 'parent', None)
            if isinstance(parent, ast.Compare) or isinstance(parent, ast.If) or isinstance(parent, ast.While):
                input_output_relations.append((f"{left_str} {op_type} {right_str}", "condition", op_type))

            # 8.2 Loop Range Calculations (e.g., n + 1 in range)
            elif isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name) and parent.func.id == "range":
                input_output_relations.append((f"{left_str} {op_type} {right_str}", "loop_range", op_type))

            # 8.3 Direct Assignments
            else:
                input_output_relations.append((f"{left_str} {op_type} {right_str}", "temp", op_type))


        # 9. Augmented Assignments (e.g., +=, -=, *=, etc.)
        elif isinstance(child, ast.AugAssign) and isinstance(child.target, ast.Name):
            target_str = child.target.id
            op_type = type(child.op).__name__
            value_str = _unparse_expr(child.value)
            input_output_relations.append((f"{target_str} {op_type} {value_str}", target_str, op_type))

    return {
        "reads": list(reads),
        "writes": list(writes),
        "returns": list(returns),
        "function_calls": list(function_calls),
        "control_flows": list(control_flows),
        "exception_handling": list(exception_handling),
        "side_effects": list(side_effects),
        "input_output_relations": input_output_relations
    }




def get_usage_info(func) -> Dict[str, Dict[str, Any]]:
    """
    Enhanced Usage Analyzer that:
      - Infers data types based on usage patterns and context
      - Supports int, float, str, list, dict, bool, Callable, and custom objects
      - Provides detailed usage context for each parameter
    """
    usage_info = {}

    # Get the source code of the function
    source = inspect.getsource(func)
    tree = ast.parse(source)

    # Get the function definition node
    func_def = next((node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)), None)
    if not func_def:
        return usage_info  # Return empty if no function found

    # Extract parameter names
    param_names = {arg.arg for arg in func_def.args.args}
    for name in param_names:
        usage_info[name] = {
            "type": "unknown",
            "iterated": False,
            "len_called": False,
            "arithmetic": False,
            "bool_check": False,
            "conditional_check": False,
            "argument_usage": False,
            "return_usage": False,
            "func_call": False,
            "method_call": False
        }

    # Analyze the AST of the function
    for node in ast.walk(func_def):
        # Check for iteration
        if isinstance(node, (ast.For, ast.While, ast.comprehension)):
            if isinstance(node.iter, ast.Name) and node.iter.id in param_names:
                usage_info[node.iter.id]["iterated"] = True
                usage_info[node.iter.id]["type"] = "list"

        # Check for len() calls
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "len":
            for arg in node.args:
                if isinstance(arg, ast.Name) and arg.id in param_names:
                    usage_info[arg.id]["len_called"] = True
                    usage_info[arg.id]["type"] = "list"

        # Check for arithmetic operations
        if isinstance(node, ast.BinOp):
            if isinstance(node.left, ast.Name) and node.left.id in param_names:
                usage_info[node.left.id]["arithmetic"] = True
                usage_info[node.left.id]["type"] = "int"
            if isinstance(node.right, ast.Name) and node.right.id in param_names:
                usage_info[node.right.id]["arithmetic"] = True
                usage_info[node.right.id]["type"] = "int"

        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            if node.target.id in param_names:
                usage_info[node.target.id]["arithmetic"] = True
                usage_info[node.target.id]["type"] = "int"

        # Check for boolean checks in if, while, and other conditional statements
        if isinstance(node, (ast.If, ast.While)) and isinstance(node.test, ast.Name):
            if node.test.id in param_names:
                usage_info[node.test.id]["bool_check"] = True
                usage_info[node.test.id]["type"] = "bool"

        # Check for conditional checks (e.g., x > 0, y == z)
        if isinstance(node, ast.Compare):
            for comparator in node.comparators:
                if isinstance(comparator, ast.Name) and comparator.id in param_names:
                    usage_info[comparator.id]["conditional_check"] = True
                    usage_info[comparator.id]["type"] = "int"
            if isinstance(node.left, ast.Name) and node.left.id in param_names:
                usage_info[node.left.id]["conditional_check"] = True
                usage_info[node.left.id]["type"] = "int"

        # Check if parameter is passed as an argument
        if isinstance(node, ast.Call):
            for arg in node.args:
                if isinstance(arg, ast.Name) and arg.id in param_names:
                    usage_info[arg.id]["argument_usage"] = True
                    usage_info[arg.id]["func_call"] = True
            for kw in node.keywords:
                if isinstance(kw.value, ast.Name) and kw.value.id in param_names:
                    usage_info[kw.value.id]["argument_usage"] = True
                    usage_info[kw.value.id]["func_call"] = True

        # Check for method calls (e.g., obj.method())
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id in param_names:
                usage_info[node.func.value.id]["method_call"] = True
                usage_info[node.func.value.id]["type"] = "object"

    # Contextual inference based on parameter name patterns
    for name, info in usage_info.items():
        if info["type"] == "unknown":
            # Infer from parameter name
            if any(keyword in name.lower() for keyword in ["list", "items", "values", "numbers"]):
                info["type"] = "list"
            elif any(keyword in name.lower() for keyword in ["dict", "mapping", "data"]):
                info["type"] = "dict"
            elif any(keyword in name.lower() for keyword in ["flag", "is_", "has_"]):
                info["type"] = "bool"
            elif any(keyword in name.lower() for keyword in ["callback", "func", "handler"]):
                info["type"] = "Callable"
            else:
                info["type"] = "int"  # Default to int if no pattern is matched

    return usage_info