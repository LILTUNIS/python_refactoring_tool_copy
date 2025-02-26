# data_flow_analyzer.py
"""
Analyzes data flow within Python ASTs, providing details on variable usage,
dependencies, control flows, exception handling, and side effects.

Enhancement:
- All entries in `input_output_relations` are stored as dictionaries, e.g.:
    {
      "var_name": "total",
      "desc": "Assign the result of 'int extra_computation'",
      "operation": "Assign",
      "context": "Single-target assignment"
    }
- This prevents unpacking errors (no more 3-element tuple).
- A helper function `create_human_readable_summary()` is included at the end
  to convert these dictionaries into a bullet-point style text.
"""

import ast
import logging
import inspect
from typing import Dict, List, Any

logging.basicConfig(level=logging.DEBUG)

def analyze_data_flow(tree: ast.AST, filename: str = "unknown_file.py") -> Dict[str, Dict]:
    """
    Analyzes data flow within the given AST (Abstract Syntax Tree).
    Returns a dictionary mapping `filename` to a dictionary of function-specific results.
    """
    print(f"Starting analyze_data_flow() for file: {filename}")
    print(f"AST Tree: {ast.dump(tree)}")

    data_flow = {filename: {}}

    # Traverse all nodes in the AST
    for node in ast.walk(tree):
        # Look for top-level function definitions
        if isinstance(node, ast.FunctionDef):
            print(f"Processing function `{node.name}` in analyze_data_flow")

            # Ensure we have an entry in data_flow for this function
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

            variables = {}
            dependencies = extract_dependencies(node)

            print(f"Collected Variables for `{node.name}`: {variables}")
            print(f"Collected Dependencies for `{node.name}`: {dependencies}")

            # Single pass to collect variable definitions and usage
            for child in ast.walk(node):
                # Variable definitions (Assign, AugAssign)
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name):
                            if target.id not in variables:
                                variables[target.id] = {
                                    "defined": child.lineno,
                                    "used": []
                                }
                elif isinstance(child, ast.AugAssign) and isinstance(child.target, ast.Name):
                    if child.target.id not in variables:
                        variables[child.target.id] = {
                            "defined": child.lineno,
                            "used": []
                        }

                # Variable usage (Name with Load ctx)
                elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                    if child.id in variables:
                        variables[child.id]["used"].append(child.lineno)

            data_flow[filename][node.name]["variables"] = variables
            data_flow[filename][node.name]["dependencies"] = dependencies

            print(f"Function `{node.name}` - Variables: {variables}")
            print(f"Function `{node.name}` - Dependencies: {dependencies}")

    print(f"Final Data Flow Analysis Results: {data_flow}")
    return data_flow


def _unparse_expr(node: ast.AST) -> str:
    """
    Safely unparse an AST node into a string.
    Fallback to str(node) if ast.unparse() is not available or fails.
    """
    print(f"\n[DEBUG] Starting _unparse_expr for Node: {type(node).__name__}")
    print(f"[DEBUG] AST Node Dump: {ast.dump(node)}")

    if hasattr(ast, "unparse"):
        print("[DEBUG] ast.unparse() is available.")
        try:
            unparsed_expr = ast.unparse(node)
            print(f"[DEBUG] Successfully Unparsed: {unparsed_expr}")
            return unparsed_expr
        except Exception as e:
            print(f"[ERROR] Exception in ast.unparse(): {e}")
            print("[DEBUG] Falling back to str(node).")
            return str(node)
    else:
        print("[WARNING] ast.unparse() is not available in this Python version.")
        print("[DEBUG] Falling back to str(node).")
        return str(node)


def extract_dependencies(node: ast.FunctionDef) -> Dict[str, List[Any]]:
    """
    Extracts dependencies and input-output relations for a single function definition.
    Now each entry in 'input_output_relations' is a dictionary, preventing unpacking errors.
    """
    reads = set()
    writes = set()
    returns = set()
    function_calls = set()
    control_flows = set()
    exception_handling = set()
    side_effects = set()
    input_output_relations = []

    skip_binops = set()
    custom_exceptions = set()

    print(f"\n[DEBUG] Starting extract_dependencies for function: {node.name}")
    print(f"[DEBUG] Function AST: {ast.dump(node)}")

    # 1. Detect custom exception classes in the function body
    for child in node.body:
        if isinstance(child, ast.ClassDef):
            for base in child.bases:
                if isinstance(base, ast.Name) and base.id == "Exception":
                    custom_exceptions.add(child.name)
                    exception_handling.add(f"Custom Exception: {child.name}")
    print(f"[DEBUG] Custom Exceptions Detected: {custom_exceptions}")

    for child in ast.walk(node):
        print(f"\n[DEBUG] Processing Node: {type(child).__name__} - {ast.dump(child)}")

        # A) Variable Writes (Assign)
        if isinstance(child, ast.Assign):
            print(f"[DEBUG] Assignment Detected: {ast.dump(child)}")
            for t in child.targets:
                if isinstance(t, ast.Name):
                    writes.add(t.id)
                    print(f"[DEBUG] Write Detected: {t.id}")

            # If the assigned value is a BinOp
            if isinstance(child.value, ast.BinOp):
                skip_binops.add(child.value)
                left_str = _unparse_expr(child.value.left)
                right_str = _unparse_expr(child.value.right)
                op_type = type(child.value.op).__name__

                if len(child.targets) == 1 and isinstance(child.targets[0], ast.Name):
                    target_name = child.targets[0].id
                    # Store as a dictionary
                    relation = {
                        "var_name": target_name,
                        "desc": f"Assign the result of '{left_str} {op_type} {right_str}'",
                        "operation": "Assign",
                        "context": "Single-target assignment"
                    }
                    input_output_relations.append(relation)
                else:
                    relation = {
                        "var_name": "temp",
                        "desc": f"Compute '{left_str} {op_type} {right_str}' as a temp (multi-assign)",
                        "operation": "MultiAssign",
                        "context": "Multiple or non-Name target"
                    }
                    input_output_relations.append(relation)

            print(f"[DEBUG] Input-Output Relations Updated: {input_output_relations}")

        # B) Variable Reads (Name)
        elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            reads.add(child.id)
            print(f"[DEBUG] Read Detected: {child.id}")

        # C) Return Statements
        elif isinstance(child, ast.Return):
            print(f"[DEBUG] Return Statement Detected: {ast.dump(child)}")
            if child.value:
                return_expr = _unparse_expr(child.value)
                returns.add(return_expr)

                if isinstance(child.value, ast.BinOp):
                    skip_binops.add(child.value)
                    left_str = _unparse_expr(child.value.left)
                    right_str = _unparse_expr(child.value.right)
                    binop_type = type(child.value.op).__name__
                    relation = {
                        "var_name": "return_value",
                        "desc": f"Return the result of '{left_str} {binop_type} {right_str}'",
                        "operation": "Return",
                        "context": "Return statement (BinOp)"
                    }
                    input_output_relations.append(relation)
                else:
                    relation = {
                        "var_name": "return_value",
                        "desc": f"Return the result of '{return_expr}'",
                        "operation": "Return",
                        "context": "Return statement"
                    }
                    input_output_relations.append(relation)

            print(f"[DEBUG] Returns Updated: {returns}")

        # D) Function Calls
        elif isinstance(child, ast.Call):
            print(f"[DEBUG] Function Call Detected: {ast.dump(child)}")
            if isinstance(child.func, ast.Name):
                function_calls.add(child.func.id)
                relation = {
                    "var_name": "function_call",
                    "desc": f"Call function '{child.func.id}'",
                    "operation": "Call",
                    "context": "Direct call"
                }
                input_output_relations.append(relation)
            elif isinstance(child.func, ast.Attribute):
                if isinstance(child.func.value, ast.Name):
                    full_call = f"{child.func.value.id}.{child.func.attr}"
                else:
                    full_call = child.func.attr
                function_calls.add(full_call)
                relation = {
                    "var_name": "function_call",
                    "desc": f"Call method '{full_call}'",
                    "operation": "Call",
                    "context": "Attribute-based call"
                }
                input_output_relations.append(relation)

            if (
                isinstance(child.func, ast.Attribute)
                and child.func.attr in ("write", "read", "append", "open", "print")
            ):
                side_effects.add("I/O Operation")

            print(f"[DEBUG] Function Calls Updated: {function_calls}")

        # E) Control Flow Structures
        elif isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
            flow_type = type(child).__name__
            control_flows.add(f"{node.name}->{flow_type}")
            print(f"[DEBUG] Control Flow Detected: {control_flows}")
            relation = {
                "var_name": "control_flow",
                "desc": f"Encountered control flow: {flow_type}",
                "operation": flow_type,
                "context": "ControlFlow"
            }
            input_output_relations.append(relation)

        elif isinstance(child, ast.Break):
            control_flows.add(f"{node.name}->Break at line {child.lineno}")
            print(f"[DEBUG] Break Statement Detected: {control_flows}")
            relation = {
                "var_name": "control_flow",
                "desc": f"Break statement at line {child.lineno}",
                "operation": "Break",
                "context": "ControlFlow"
            }
            input_output_relations.append(relation)

        elif isinstance(child, ast.Continue):
            control_flows.add(f"{node.name}->Continue at line {child.lineno}")
            print(f"[DEBUG] Continue Statement Detected: {control_flows}")
            relation = {
                "var_name": "control_flow",
                "desc": f"Continue statement at line {child.lineno}",
                "operation": "Continue",
                "context": "ControlFlow"
            }
            input_output_relations.append(relation)

    print(f"\n[DEBUG] Final Reads: {reads}")
    print(f"[DEBUG] Final Writes: {writes}")
    print(f"[DEBUG] Final Returns: {returns}")
    print(f"[DEBUG] Final Function Calls: {function_calls}")
    print(f"[DEBUG] Final Control Flows: {control_flows}")
    print(f"[DEBUG] Final Exception Handling: {exception_handling}")
    print(f"[DEBUG] Final Side Effects: {side_effects}")
    print(f"[DEBUG] Final Input-Output Relations: {input_output_relations}")

    return {
        "reads": sorted(reads),
        "writes": sorted(writes),
        "returns": sorted(returns),
        "function_calls": sorted(function_calls),
        "control_flows": sorted(control_flows),
        "exception_handling": sorted(exception_handling),
        "side_effects": sorted(side_effects),
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

    # Find the function definition node
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

    # Analyze usage within the function
    for node in ast.walk(func_def):
        # 1. Check for iteration
        if isinstance(node, (ast.For, ast.While, ast.comprehension)):
            if isinstance(node, ast.For):
                if isinstance(node.iter, ast.Name) and node.iter.id in param_names:
                    usage_info[node.iter.id]["iterated"] = True
                    usage_info[node.iter.id]["type"] = "list"

        # 2. Check for len() calls
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "len"
        ):
            for arg in node.args:
                if isinstance(arg, ast.Name) and arg.id in param_names:
                    usage_info[arg.id]["len_called"] = True
                    usage_info[arg.id]["type"] = "list"

        # 3. Arithmetic ops
        if isinstance(node, ast.BinOp):
            if isinstance(node.left, ast.Name) and node.left.id in param_names:
                usage_info[node.left.id]["arithmetic"] = True
                if usage_info[node.left.id]["type"] == "unknown":
                    usage_info[node.left.id]["type"] = "int"
            if isinstance(node.right, ast.Name) and node.right.id in param_names:
                usage_info[node.right.id]["arithmetic"] = True
                if usage_info[node.right.id]["type"] == "unknown":
                    usage_info[node.right.id]["type"] = "int"

        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name) and node.target.id in param_names:
                usage_info[node.target.id]["arithmetic"] = True
                if usage_info[node.target.id]["type"] == "unknown":
                    usage_info[node.target.id]["type"] = "int"

        # 4. Boolean checks in if/while condition
        if isinstance(node, (ast.If, ast.While)) and isinstance(node.test, ast.Name):
            if node.test.id in param_names:
                usage_info[node.test.id]["bool_check"] = True
                usage_info[node.test.id]["type"] = "bool"

        # 5. Conditional checks, e.g., x > 0
        if isinstance(node, ast.Compare):
            if isinstance(node.left, ast.Name) and node.left.id in param_names:
                usage_info[node.left.id]["conditional_check"] = True
                if usage_info[node.left.id]["type"] == "unknown":
                    usage_info[node.left.id]["type"] = "int"
            for cmp_ in node.comparators:
                if isinstance(cmp_, ast.Name) and cmp_.id in param_names:
                    usage_info[cmp_.id]["conditional_check"] = True
                    if usage_info[cmp_.id]["type"] == "unknown":
                        usage_info[cmp_.id]["type"] = "int"

        # 6. Parameter used as an argument to a function call
        if isinstance(node, ast.Call):
            for arg in node.args:
                if isinstance(arg, ast.Name) and arg.id in param_names:
                    usage_info[arg.id]["argument_usage"] = True
                    usage_info[arg.id]["func_call"] = True
            for kw in node.keywords:
                if isinstance(kw.value, ast.Name) and kw.value.id in param_names:
                    usage_info[kw.value.id]["argument_usage"] = True
                    usage_info[kw.value.id]["func_call"] = True

            # 7. Method calls on a parameter, e.g., param.method()
            if isinstance(node.func, ast.Attribute):
                if (isinstance(node.func.value, ast.Name)
                        and node.func.value.id in param_names):
                    usage_info[node.func.value.id]["method_call"] = True
                    usage_info[node.func.value.id]["type"] = "object"

    # Simple heuristics for naming patterns
    for name, info in usage_info.items():
        if info["type"] == "unknown":
            lower_name = name.lower()
            if any(k in lower_name for k in ["list", "items", "values", "numbers"]):
                info["type"] = "list"
            elif any(k in lower_name for k in ["dict", "mapping", "data"]):
                info["type"] = "dict"
            elif any(k in lower_name for k in ["flag", "is_", "has_"]):
                info["type"] = "bool"
            elif any(k in lower_name for k in ["callback", "func", "handler"]):
                info["type"] = "Callable"
            else:
                # Default to int if nothing matches
                info["type"] = "int"

    return usage_info


def create_human_readable_summary(input_output_relations: List[Dict[str, str]]) -> str:
    """
    Converts the list of dictionary-based relations into a bullet-point
    summary. Example:

    total is incremented by:
      - Converting `extra_computation` to an integer
      - sum of all positive numbers in `numbers`

    2. Loop Details:
      - Outer Loop: iterates over `numbers`
      - ...

    Adjust grouping/formatting logic here to match your exact readability needs.
    """
    from collections import defaultdict

    # Group relations by 'var_name'
    grouped = defaultdict(list)
    for rel in input_output_relations:
        grouped[rel["var_name"]].append(rel)

    lines = []

    # Example: For var_name == "total", we want "total is incremented by:"
    # Then bullet each "desc".
    for var_name, relations in grouped.items():
        if var_name == "total":
            lines.append(f"{var_name} is incremented by:")
            for r in relations:
                # Create a bullet from 'desc'
                lines.append(f"  - {r['desc']}")
        elif var_name == "return_value":
            lines.append("Return Value transformations:")
            for r in relations:
                lines.append(f"  - {r['desc']}")
        elif var_name == "function_call":
            lines.append("Function/Method Calls:")
            for r in relations:
                lines.append(f"  - {r['desc']}")
        elif var_name == "control_flow":
            lines.append("Control Flow Details:")
            for r in relations:
                lines.append(f"  - {r['desc']}")
        else:
            # A default grouping
            lines.append(f"{var_name} related operations:")
            for r in relations:
                lines.append(f"  - {r['desc']} ({r['operation']})")

    return "\n".join(lines)
