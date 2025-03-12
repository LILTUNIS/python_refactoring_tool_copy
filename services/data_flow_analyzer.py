"""
Analyzes data flow within Python ASTs, providing details on variable usage,
dependencies, control flows, exception handling, and side effects.
"""

import ast
import difflib
import inspect
import logging
import os
import re
from typing import Any, Dict, List, Set, Union

from utils.ast_utils import normalize_ast_for_dataflow

logging.basicConfig(level=logging.DEBUG)


# --- Control Flow Simplification ---
def simplify_control_flow(node: ast.AST) -> ast.AST:
    """
    Simplify control flow constructs.
    (This stub can be extended to flatten nested if-statements, etc.)
    """
    return node


# --- Basic CFG Generation ---
def build_cfg(tree: ast.AST) -> Dict[str, Any]:
    """
    Build a Control Flow Graph (CFG) from the AST.
    (Stub implementation – a full CFG would traverse the AST and build nodes/edges.)
    """
    cfg = {"nodes": [], "edges": []}
    return cfg


# --- Main Analysis Function (Deep Mode Only) ---
def analyze_data_flow(tree_or_trees: Union[ast.AST, Dict[str, ast.AST]], filename: str = None) -> Dict[str, Any]:
    """
    Performs deep data flow analysis on a single or multiple ASTs.
    Deep mode applies full normalization, control flow simplification, and CFG generation.
    """

    def process_single_file(tree: ast.AST, filename: str) -> Dict[str, Any]:
        file_key = os.path.basename(filename) if filename else "unknown_file.py"
        data_flow = {file_key: {}}
        logging.debug(f"🔍 Processing AST for file: {file_key}")

        # Normalize AST (returns tree + reverse mapping)
        logging.debug("🔄 Normalizing AST for deep analysis")
        tree, reverse_mapping = normalize_ast_for_dataflow(tree)

        # Simplify control flow
        tree = simplify_control_flow(tree)

        # Build CFG (stub)
        cfg = build_cfg(tree)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                logging.debug(f"   📌 Found function: {func_name}")

                if func_name not in data_flow[file_key]:
                    data_flow[file_key][func_name] = {
                        "variables": {},
                        "dependencies": {
                            "reads": [], "writes": [], "returns": [],
                            "function_calls": [], "control_flows": [],
                            "exception_handling": [], "side_effects": [],
                            "input_output_relations": [],
                            "cfg": cfg,  # Attach CFG info for potential future use.
                        },
                        "reverse_mapping": reverse_mapping  # Store for UI
                    }

                # Extract dependencies with extra context.
                dependencies = extract_dependencies(node)
                if dependencies:
                    data_flow[file_key][func_name]["dependencies"] = dependencies
                else:
                    logging.warning(f"⚠️ Function {func_name} in {file_key} has NO dependencies!")
        return data_flow

    if isinstance(tree_or_trees, dict):
        all_data_flow_results = {}
        for file_path, tree in tree_or_trees.items():
            try:
                file_results = process_single_file(tree, file_path)
                all_data_flow_results.update(file_results)
            except Exception as e:
                logging.error(f"❌ [ERROR] Data flow analysis failed for {file_path}: {str(e)}", exc_info=True)
        return all_data_flow_results
    elif isinstance(tree_or_trees, ast.AST):
        return process_single_file(tree_or_trees, filename)
    else:
        raise ValueError("Invalid input: Expected an AST or a dictionary of ASTs.")


# --- Helper: Safe Unparsing ---
def _unparse_expr(node: ast.AST) -> str:
    """
    Safely unparse an AST node into a string.
    Fallback to str(node) if ast.unparse() is not available or fails.
    """
    if hasattr(ast, "unparse"):
        try:
            return ast.unparse(node)
        except Exception as e:
            logging.error(f"Unparsing failed: {e}", exc_info=True)
            return str(node)
    else:
        return str(node)


# --- Enhanced Dependency Extraction ---
def extract_dependencies(node: ast.FunctionDef) -> Dict[str, List[Any]]:
    """
    Extracts dependencies and input-output relations for a function.
    Adds extra metadata (e.g., line numbers) to enrich analysis.
    """
    reads = set()
    writes = set()
    returns = set()
    function_calls = set()
    control_flows = set()
    exception_handling = set()
    side_effects = set()
    input_output_relations = []

    # Detect custom exceptions
    for child in node.body:
        if isinstance(child, ast.ClassDef):
            for base in child.bases:
                if isinstance(base, ast.Name) and base.id == "Exception":
                    exception_handling.add(f"Custom Exception: {child.name} at line {child.lineno}")

    for child in ast.walk(node):
        try:
            if isinstance(child, ast.Assign):
                for t in child.targets:
                    if isinstance(t, ast.Name):
                        writes.add(t.id)
                if isinstance(child.value, ast.BinOp):
                    left_str = _unparse_expr(child.value.left)
                    right_str = _unparse_expr(child.value.right)
                    op_type = type(child.value.op).__name__
                    relation = {
                        "var_name": child.targets[0].id if (len(child.targets) == 1 and isinstance(child.targets[0], ast.Name)) else "temp",
                        "desc": f"Assign result of '{left_str} {op_type} {right_str}'",
                        "operation": "Assign",
                        "context": "Single-target" if (len(child.targets) == 1 and isinstance(child.targets[0], ast.Name)) else "Multi-assign",
                        "line": getattr(child, 'lineno', None)
                    }
                    input_output_relations.append(relation)

            elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                reads.add(child.id)

            elif isinstance(child, ast.Return):
                if child.value:
                    return_expr = _unparse_expr(child.value)
                    returns.add(return_expr)
                    if isinstance(child.value, ast.BinOp):
                        left_str = _unparse_expr(child.value.left)
                        right_str = _unparse_expr(child.value.right)
                        binop_type = type(child.value.op).__name__
                        relation = {
                            "var_name": "return_value",
                            "desc": f"Return result of '{left_str} {binop_type} {right_str}'",
                            "operation": "Return",
                            "context": "Return (BinOp)",
                            "line": getattr(child, 'lineno', None)
                        }
                        input_output_relations.append(relation)
                    else:
                        relation = {
                            "var_name": "return_value",
                            "desc": f"Return result of '{return_expr}'",
                            "operation": "Return",
                            "context": "Return",
                            "line": getattr(child, 'lineno', None)
                        }
                        input_output_relations.append(relation)

            elif isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    function_calls.add(child.func.id)
                    relation = {
                        "var_name": "function_call",
                        "desc": f"Call function '{child.func.id}'",
                        "operation": "Call",
                        "context": "Direct call",
                        "line": getattr(child, 'lineno', None)
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
                        "context": "Attribute call",
                        "line": getattr(child, 'lineno', None)
                    }
                    input_output_relations.append(relation)

                # I/O operations
                if (isinstance(child.func, ast.Attribute)
                    and child.func.attr in ("write", "read", "append", "open", "print")):
                    side_effects.add("I/O Operation")

            elif isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
                flow_type = type(child).__name__
                control_flows.add(f"{node.name}->{flow_type} at line {getattr(child, 'lineno', 'NA')}")
                relation = {
                    "var_name": "control_flow",
                    "desc": f"Encountered {flow_type} at line {getattr(child, 'lineno', 'NA')}",
                    "operation": flow_type,
                    "context": "ControlFlow",
                    "line": getattr(child, 'lineno', None)
                }
                input_output_relations.append(relation)

            elif isinstance(child, ast.Break):
                control_flows.add(f"{node.name}->Break at line {child.lineno}")
                relation = {
                    "var_name": "control_flow",
                    "desc": f"Break at line {child.lineno}",
                    "operation": "Break",
                    "context": "ControlFlow",
                    "line": child.lineno
                }
                input_output_relations.append(relation)

            elif isinstance(child, ast.Continue):
                control_flows.add(f"{node.name}->Continue at line {child.lineno}")
                relation = {
                    "var_name": "control_flow",
                    "desc": f"Continue at line {child.lineno}",
                    "operation": "Continue",
                    "context": "ControlFlow",
                    "line": child.lineno
                }
                input_output_relations.append(relation)

        except Exception as ex:
            logging.error(f"Error processing AST node: {ex}", exc_info=True)

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
    Enhanced Usage Analyzer that infers data types based on usage patterns.
    Supports int, float, str, list, dict, bool, Callable, and custom objects.
    """
    usage_info = {}
    source = inspect.getsource(func)
    tree = ast.parse(source)
    func_def = next((node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)), None)
    if not func_def:
        return usage_info

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

    for node in ast.walk(func_def):
        if isinstance(node, (ast.For, ast.While, ast.comprehension)):
            if isinstance(node, ast.For):
                if isinstance(node.iter, ast.Name) and node.iter.id in param_names:
                    usage_info[node.iter.id]["iterated"] = True
                    usage_info[node.iter.id]["type"] = "list"

        if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "len"):
            for arg in node.args:
                if isinstance(arg, ast.Name) and arg.id in param_names:
                    usage_info[arg.id]["len_called"] = True
                    usage_info[arg.id]["type"] = "list"

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

        if isinstance(node, (ast.If, ast.While)) and isinstance(node.test, ast.Name):
            if node.test.id in param_names:
                usage_info[node.test.id]["bool_check"] = True
                usage_info[node.test.id]["type"] = "bool"

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

        if isinstance(node, ast.Call):
            for arg in node.args:
                if isinstance(arg, ast.Name) and arg.id in param_names:
                    usage_info[arg.id]["argument_usage"] = True
                    usage_info[arg.id]["func_call"] = True
            for kw in node.keywords:
                if isinstance(kw.value, ast.Name) and kw.value.id in param_names:
                    usage_info[kw.value.id]["argument_usage"] = True
                    usage_info[kw.value.id]["func_call"] = True
            if isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id in param_names:
                    usage_info[node.func.value.id]["method_call"] = True
                    usage_info[node.func.value.id]["type"] = "object"

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
                info["type"] = "int"

    return usage_info


def create_human_readable_summary(input_output_relations: List[Dict[str, str]]) -> str:
    """
    Converts the list of relation dictionaries into a bullet-point summary.
    """
    from collections import defaultdict
    grouped = defaultdict(list)
    for rel in input_output_relations:
        grouped[rel["var_name"].lower()].append(rel)
    lines = []
    for var_name, relations in grouped.items():
        if var_name == "total":
            lines.append(f"{var_name} is incremented by:")
            for r in relations:
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
            lines.append(f"{var_name} related operations:")
            for r in relations:
                lines.append(f"  - {r['desc']} ({r['operation']})")
    return "\n".join(lines)


def normalize_data_flow(set_data: Set[str]) -> Set[str]:
    """
    Normalize dependency data by lowercasing, trimming, and generalizing common patterns.
    """
    normalized = set()
    for item in set_data:
        norm = item.lower().strip()
        # Unify common math calls
        norm = re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\.(sqrt|log|exp|pow|sin|cos|tan)\b', 'math_op', norm)
        # Unify control flow references
        if '->' in norm:
            norm = re.sub(r'^[^->]+->', 'var->', norm)
        normalized.add(norm)
    return normalized


def levenshtein_similarity(str1: str, str2: str) -> float:
    """
    Compute Levenshtein similarity between two strings.
    """
    seq = difflib.SequenceMatcher(None, str1, str2)
    return round(seq.ratio(), 3)


def parse_ast_structure(expression: str) -> str:
    """
    Convert an arithmetic expression into a normalized AST structure dump.
    This helps recognize semantically equivalent expressions.
    """
    try:
        tree = ast.parse(expression, mode='eval')
        normalized_tree, _ = normalize_ast_for_dataflow(tree)  # Unpack the tuple
        return ast.dump(normalized_tree)
    except SyntaxError:
        return expression


def _jaccard_similarity(set_a: Set[str], set_b: Set[str], field_name="Unknown Field") -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    similarity = (len(intersection) + 0.5) / (len(union) + 0.5)
    return round(similarity, 3)


already_compared = set()


def compare_function_similarity(func_info_a: Dict[str, Any], func_info_b: Dict[str, Any]) -> float:
    """
    Compute a similarity score between two functions using a graph-based approach.
    This method builds a dependency graph for each function based on its extracted dependencies,
    then computes similarity as a combination of Jaccard similarity over graph nodes and edges.

    The dependency tokens are normalized by lowercasing and trimming.
    """
    if not func_info_a or not func_info_b:
        logging.warning("Missing function info for similarity calculation.")
        return 0.0

    func_a_name = func_info_a.get("name") or "UNKNOWN_FUNC_A"
    func_b_name = func_info_b.get("name") or "UNKNOWN_FUNC_B"

    if func_a_name == "UNKNOWN_FUNC_A" or func_b_name == "UNKNOWN_FUNC_B":
        logging.error(f"Skipping similarity check for unknown functions: {func_a_name}, {func_b_name}")
        return 0.0

    if "dependencies" not in func_info_a or "dependencies" not in func_info_b:
        logging.warning(f"Missing dependencies for functions: {func_a_name}, {func_b_name}")
        return 0.0

    dependencies_a = func_info_a["dependencies"]
    dependencies_b = func_info_b["dependencies"]

    def build_dependency_graph(deps: Dict[str, Any]) -> (set, set):
        nodes = set()
        edges = set()
        # Process primary dependency fields
        for key in ["reads", "writes", "function_calls", "control_flows", "exception_handling", "side_effects"]:
            for item in deps.get(key, []):
                token = str(item).lower().strip()
                if token:
                    nodes.add(f"{key}:{token}")
        # Process input_output_relations: create nodes for var_name and operation, and an edge between them.
        for relation in deps.get("input_output_relations", []):
            var = str(relation.get("var_name", "")).lower().strip()
            op = str(relation.get("operation", "")).lower().strip()
            if var:
                nodes.add(f"relation:var:{var}")
            if op:
                nodes.add(f"relation:op:{op}")
            if var and op:
                edges.add((f"relation:var:{var}", f"relation:op:{op}"))
        return nodes, edges

    def jaccard_similarity(set_a: set, set_b: set) -> float:
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    nodes_a, edges_a = build_dependency_graph(dependencies_a)
    nodes_b, edges_b = build_dependency_graph(dependencies_b)

    node_sim = jaccard_similarity(nodes_a, nodes_b)
    edge_sim = jaccard_similarity(edges_a, edges_b)

    alpha = 0.5  # Weight for nodes vs. edges; can be tuned.
    final_score = alpha * node_sim + (1 - alpha) * edge_sim
    final_score = min(final_score, 1.0)

    logging.info(f"Graph-based similarity score = {final_score} for {tuple(sorted([func_a_name, func_b_name]))}")
    return final_score
