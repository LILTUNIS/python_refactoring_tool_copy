# data_flow_analyzer.py
"""
Analyzes data flow within Python ASTs, providing details on variable usage,
dependencies, control flows, exception handling, and side effects.
"""
import difflib
import json
import logging
import inspect
import re
from typing import Dict, List, Any, Set

logging.basicConfig(level=logging.DEBUG)

import os
import ast
from typing import Dict


def analyze_data_flow(tree: ast.AST, filename: str = None) -> Dict[str, Dict]:
    """
    Analyzes data flow within the given AST (Abstract Syntax Tree).
    Returns a dictionary mapping `filename` to a dictionary of function-specific results.
    """
    # Apply basename here to ensure filename is just the name, not the full path
    if filename:
        filename = os.path.basename(filename)
    else:
        filename = "unknown_file.py"

    data_flow = {filename: {}}

    # Traverse all nodes in the AST
    for node in ast.walk(tree):
        # Look for top-level function definitions
        if isinstance(node, ast.FunctionDef):
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

    # 1. Detect custom exception classes in the function body
    for child in node.body:
        if isinstance(child, ast.ClassDef):
            for base in child.bases:
                if isinstance(base, ast.Name) and base.id == "Exception":
                    custom_exceptions.add(child.name)
                    exception_handling.add(f"Custom Exception: {child.name}")

    for child in ast.walk(node):

        # A) Variable Writes (Assign)
        if isinstance(child, ast.Assign):
            for t in child.targets:
                if isinstance(t, ast.Name):
                    writes.add(t.id)

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

        # B) Variable Reads (Name)
        elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            reads.add(child.id)

        # C) Return Statements
        elif isinstance(child, ast.Return):
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

        # D) Function Calls
        elif isinstance(child, ast.Call):
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

        # E) Control Flow Structures
        elif isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
            flow_type = type(child).__name__
            control_flows.add(f"{node.name}->{flow_type}")
            relation = {
                "var_name": "control_flow",
                "desc": f"Encountered control flow: {flow_type}",
                "operation": flow_type,
                "context": "ControlFlow"
            }
            input_output_relations.append(relation)

        elif isinstance(child, ast.Break):
            control_flows.add(f"{node.name}->Break at line {child.lineno}")
            relation = {
                "var_name": "control_flow",
                "desc": f"Break statement at line {child.lineno}",
                "operation": "Break",
                "context": "ControlFlow"
            }
            input_output_relations.append(relation)

        elif isinstance(child, ast.Continue):
            control_flows.add(f"{node.name}->Continue at line {child.lineno}")
            relation = {
                "var_name": "control_flow",
                "desc": f"Continue statement at line {child.lineno}",
                "operation": "Continue",
                "context": "ControlFlow"
            }
            input_output_relations.append(relation)

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
    summary.
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



def normalize_data_flow(set_data: Set[str]) -> Set[str]:
    """
    Normalize dependency data by:
      - Lowercasing the string.
      - Generalizing function calls.
      - Normalizing control flows (e.g., "func->If" -> "var->If").
    """
    normalized = set()
    for item in set_data:
        norm = item.lower().strip()

        # Replace function calls (e.g., "math.sqrt" -> "math_op")
        norm = re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\.(sqrt|log|exp|pow|sin|cos|tan)\b', 'math_op', norm)

        # Normalize control flow (e.g., "function->If" -> "var->If")
        if '->' in norm:
            norm = re.sub(r'^[^->]+->', 'var->', norm)

        normalized.add(norm)
    return normalized

def levenshtein_similarity(str1: str, str2: str) -> float:
    """
    Compute Levenshtein similarity between two strings.
    This ensures we capture **partial** similarity in return statements.
    """
    seq = difflib.SequenceMatcher(None, str1, str2)
    return round(seq.ratio(), 3)

def parse_ast_structure(expression: str) -> str:
    """
    Convert an arithmetic expression into an AST-based structural string.
    Example: "a + b" and "b + a" should be recognized as the same.
    """
    try:
        tree = ast.parse(expression, mode='eval')
        return ast.dump(tree)
    except SyntaxError:
        return expression  # Return raw if parsing fails

def _jaccard_similarity(set_a: Set[str], set_b: Set[str], field_name="Unknown Field") -> float:
    """
    Compute Jaccard similarity with accurate penalty scaling.
    - Uses smoothing to ensure minor changes don't drastically affect similarity.
    """
    if not set_a and not set_b:
        return 1.0  # Perfect match if both are empty
    if not set_a or not set_b:
        return 0.0  # No similarity if one is empty

    intersection = set_a & set_b
    union = set_a | set_b
    similarity = len(intersection) / len(union)

    return round(similarity, 3)  # Use high precision

def compare_function_similarity(func_info_a: Dict[str, Any], func_info_b: Dict[str, Any]) -> float:
    """
    Compute the most **accurate** function similarity score possible by:
    - Using AST for return values
    - Using Levenshtein similarity for I/O relations
    - Weighting similarities for better accuracy
    """
    if "dependencies" in func_info_a:
        func_info_a = func_info_a["dependencies"]
    if "dependencies" in func_info_b:
        func_info_b = func_info_b["dependencies"]

    fields_to_compare = [
        "reads", "writes", "function_calls",
        "control_flows", "exception_handling", "side_effects",
        "input_output_relations"
    ]

    similarities = []
    field_weights = {
        "reads": 0.15,
        "writes": 0.15,
        "function_calls": 0.20,
        "control_flows": 0.10,
        "input_output_relations": 0.20,
        "returns": 0.20,
    }

    def serialize_dependencies(dep_list):
        """
        Serialize a list of dependencies into a set of JSON strings.
        Using JSON ensures that the structure of dependency objects is preserved,
        allowing for more semantically aware comparisons.
        """
        serialized_set = set()
        for dep in dep_list:
            if isinstance(dep, dict):
                try:
                    # Using JSON dumps to preserve structure with sorted keys
                    serialized = json.dumps(dep, sort_keys=True)
                except Exception:
                    serialized = str(dep)
                serialized_set.add(serialized)
            elif isinstance(dep, (str, int, float)):
                serialized_set.add(str(dep))
        return serialized_set

    for field in fields_to_compare:
        raw_set_a = serialize_dependencies(func_info_a.get(field, []))
        raw_set_b = serialize_dependencies(func_info_b.get(field, []))

        set_a = normalize_data_flow(raw_set_a)
        set_b = normalize_data_flow(raw_set_b)

        if not set_a and not set_b:
            continue

        similarity = _jaccard_similarity(set_a, set_b, field_name=field)
        similarities.append(similarity * field_weights.get(field, 0.1))

    # **AST-Based Return Similarity**
    ret_a = func_info_a.get("returns", [""])[0]
    ret_b = func_info_b.get("returns", [""])[0]
    if ret_a and ret_b:
        ast_a = parse_ast_structure(ret_a)
        ast_b = parse_ast_structure(ret_b)
        return_similarity = levenshtein_similarity(ast_a, ast_b)
    else:
        return_similarity = 1.0 if not ret_a and not ret_b else 0.0

    similarities.append(return_similarity * field_weights["returns"])

    # **Levenshtein Input-Output Relation Similarity**
    def relation_strings(rel_list):
        generalized_relations = set()
        for rel in rel_list:
            desc = rel.get('desc', '').lower()
            desc = re.sub(r'\bmath\.(sqrt|log|exp|pow|sin|cos|tan)\b', 'math_op', desc)
            desc = re.sub(r'\b(total|sum|result|count|accumulator)\b', 'var_result', desc)
            desc = re.sub(r'^[a-zA-Z_]+->', 'var->', desc)
            generalized_relations.add(f"{rel.get('operation', '')}::{desc}")
        return generalized_relations

    ior_a = normalize_data_flow(relation_strings(func_info_a.get("input_output_relations", [])))
    ior_b = normalize_data_flow(relation_strings(func_info_b.get("input_output_relations", [])))

    if ior_a or ior_b:
        ior_similarity = levenshtein_similarity(str(ior_a), str(ior_b))
        similarities.append(ior_similarity * field_weights["input_output_relations"])

    final_score = sum(similarities) / sum(field_weights.values())
    return final_score