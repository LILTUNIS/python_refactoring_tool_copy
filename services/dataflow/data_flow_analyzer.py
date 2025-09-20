import ast
import difflib
import logging
import os
import inspect
from typing import Any, Dict, List, Set, Union

logging.basicConfig(level=logging.DEBUG)


# ------------------------------------------------------------------------
#  Control Flow Simplification (SCDetector & FA-AST)
# ------------------------------------------------------------------------
def simplify_control_flow(node: ast.AST) -> ast.AST:
    """
    Simplify control flow constructs to more easily embed control-flow
    edges within an AST representation, following the SCDetector & FA-AST
    concept of integrating control flow into AST nodes. (ASE2022_TreeCen)
    """
    return node


# ------------------------------------------------------------------------
#  Basic CFG Generation (DeepSim & FA-AST)
# ------------------------------------------------------------------------
def build_cfg(tree: ast.AST) -> Dict[str, Any]:
    """
    Build a Control Flow Graph (CFG) from the AST.
    As recommended by SCDetector & FA-AST, we embed edges into
    the AST to track branching. This CFG can later be combined
    with a DDG (DeepSim & FA-AST) to refine semantic similarity.
    """
    cfg = {
        "nodes": [],
        "edges": []
    }

    function_counter = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_counter += 1
            func_node_id = f"func_{function_counter}"
            cfg["nodes"].append({"id": func_node_id, "type": "Function", "name": node.name})

            block_counter = 0
            prev_block_id = None
            for stmt in node.body:
                block_counter += 1
                block_id = f"{func_node_id}_block_{block_counter}"
                cfg["nodes"].append({"id": block_id, "type": "BasicBlock", "stmt": _stmt_summary(stmt)})

                # Link the previous block to this block in a simple sequence
                if prev_block_id:
                    cfg["edges"].append((prev_block_id, block_id, "seq"))
                prev_block_id = block_id

                # If statement branching
                if isinstance(stmt, ast.If):
                    true_block_id = f"{block_id}_if_true"
                    cfg["nodes"].append({"id": true_block_id, "type": "BasicBlock", "stmt": "If-True-Branch"})
                    cfg["edges"].append((block_id, true_block_id, "if-true"))
                    for tstmt in stmt.body:
                        block_counter += 1
                        sub_block_id = f"{true_block_id}_sub_{block_counter}"
                        cfg["nodes"].append({"id": sub_block_id, "type": "BasicBlock", "stmt": _stmt_summary(tstmt)})
                        cfg["edges"].append((true_block_id, sub_block_id, "seq"))

                    false_block_id = f"{block_id}_if_false"
                    cfg["nodes"].append({"id": false_block_id, "type": "BasicBlock", "stmt": "If-False-Branch"})
                    cfg["edges"].append((block_id, false_block_id, "if-false"))
                    for fstmt in stmt.orelse:
                        block_counter += 1
                        sub_block_id = f"{false_block_id}_sub_{block_counter}"
                        cfg["nodes"].append({"id": sub_block_id, "type": "BasicBlock", "stmt": _stmt_summary(fstmt)})
                        cfg["edges"].append((false_block_id, sub_block_id, "seq"))

                elif isinstance(stmt, ast.While):
                    loop_block_id = f"{block_id}_while_body"
                    cfg["nodes"].append({"id": loop_block_id, "type": "BasicBlock", "stmt": "While-Body"})
                    cfg["edges"].append((block_id, loop_block_id, "while-true"))
                    # Potentially link back from last statement in the body, etc.

    return cfg


# ------------------------------------------------------------------------
#  Data Dependency Graph (DDG) Stub (DeepSim & FA-AST)
# ------------------------------------------------------------------------
def build_ddg(tree: ast.AST) -> Dict[str, Any]:
    """
    Build a Data Dependency Graph (DDG) from the AST.
    As per DeepSim & FA-AST, combining CFG and DDG refines semantic similarity.
    This stub simply tracks definitions/uses in each function.
    """
    ddg = {
        "nodes": [],
        "edges": []
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            # For each assignment, record a definition node
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign):
                    for tgt in sub.targets:
                        if isinstance(tgt, ast.Name):
                            def_id = f"{func_name}_{tgt.id}_def_{tgt.lineno}"
                            ddg["nodes"].append({
                                "id": def_id,
                                "var": tgt.id,
                                "type": "definition",
                                "scope": func_name
                            })
                elif isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                    use_id = f"{func_name}_{sub.id}_use_{sub.lineno}"
                    ddg["nodes"].append({
                        "id": use_id,
                        "var": sub.id,
                        "type": "use",
                        "scope": func_name
                    })
                    # A real approach would link the 'use' to the last 'def'
    return ddg


# ------------------------------------------------------------------------
#  Main Data Flow Analysis Function
# ------------------------------------------------------------------------
def analyze_data_flow(tree_or_trees: Union[ast.AST, Dict[str, ast.AST]], filename: str = None) -> Dict[str, Any]:
    """
    Performs deep data flow analysis on one or multiple ASTs.
    1) (Optional) Normalization
    2) Control flow simplification (SCDetector & FA-AST)
    3) CFG building
    4) DDG building
    5) Function-level dependency extraction
    """
    def process_single_file(tree: ast.AST, f_name: str) -> Dict[str, Any]:
        file_key = os.path.basename(f_name) if f_name else "unknown_file.py"
        data_flow = {file_key: {}}
        logging.debug(f"🔍 Processing AST for file: {file_key}")

        # Step 1: Optional normalization (skipped)
        # Step 2: Simplify control flow
        tree = simplify_control_flow(tree)

        # Step 3: Build CFG
        cfg = build_cfg(tree)

        # Step 4: Build DDG
        ddg = build_ddg(tree)

        # Step 5: Function-level dependency extraction
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
                            "cfg": cfg,
                            "ddg": ddg,
                        },
                        "reverse_mapping": {}
                    }

                deps = extract_dependencies(node)
                if deps:
                    data_flow[file_key][func_name]["dependencies"] = deps
                    data_flow[file_key][func_name]["dependencies"]["cfg"] = cfg
                    data_flow[file_key][func_name]["dependencies"]["ddg"] = ddg
                else:
                    logging.warning(f"⚠️ Function {func_name} in {file_key} has NO dependencies!")
        return data_flow

    if isinstance(tree_or_trees, dict):
        # Multiple ASTs
        all_data_flow_results = {}
        for file_path, ast_tree in tree_or_trees.items():
            try:
                single = process_single_file(ast_tree, file_path)
                all_data_flow_results.update(single)
            except Exception as e:
                logging.error(f"❌ [ERROR] Data flow analysis failed for {file_path}: {str(e)}", exc_info=True)
        return all_data_flow_results
    elif isinstance(tree_or_trees, ast.AST):
        # Single AST
        return process_single_file(tree_or_trees, filename)
    else:
        raise ValueError("Invalid input: Expected an AST or a dictionary of ASTs.")


# ------------------------------------------------------------------------
#  Enhanced Dependency Extraction
# ------------------------------------------------------------------------
def extract_dependencies(node: ast.FunctionDef) -> Dict[str, List[Any]]:
    """
    Extracts dependencies and input-output relations for a function:
    variable reads/writes, calls, returns, side effects, control flow, etc.
    """
    reads = set()
    writes = set()
    returns = set()
    function_calls = set()
    control_flows = set()
    exception_handling = set()
    side_effects = set()
    input_output_relations = []

    # Check for custom exceptions declared inside the function
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
                # Example: if the right side is a BinOp
                if isinstance(child.value, ast.BinOp):
                    left_str = _unparse_expr(child.value.left)
                    right_str = _unparse_expr(child.value.right)
                    op_type = type(child.value.op).__name__
                    var_name = "temp"
                    if len(child.targets) == 1 and isinstance(child.targets[0], ast.Name):
                        var_name = child.targets[0].id
                    relation = {
                        "var_name": var_name,
                        "desc": f"Assign result of '{left_str} {op_type} {right_str}'",
                        "operation": "Assign",
                        "context": "Single-target" if (len(child.targets) == 1) else "Multi-assign",
                        "line": getattr(child, 'lineno', None)
                    }
                    input_output_relations.append(relation)

            elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                reads.add(child.id)

            elif isinstance(child, ast.Return):
                if child.value:
                    returns.add(_unparse_expr(child.value))
                    # If returning a BinOp
                    if isinstance(child.value, ast.BinOp):
                        left_str = _unparse_expr(child.value.left)
                        right_str = _unparse_expr(child.value.right)
                        binop_type = type(child.value.op).__name__
                        rel = {
                            "var_name": "return_value",
                            "desc": f"Return result of '{left_str} {binop_type} {right_str}'",
                            "operation": "Return",
                            "context": "Return (BinOp)",
                            "line": getattr(child, 'lineno', None)
                        }
                        input_output_relations.append(rel)
                    else:
                        rel = {
                            "var_name": "return_value",
                            "desc": f"Return result of '{_unparse_expr(child.value)}'",
                            "operation": "Return",
                            "context": "Return",
                            "line": getattr(child, 'lineno', None)
                        }
                        input_output_relations.append(rel)

            elif isinstance(child, ast.Call):
                func_desc = ""
                if isinstance(child.func, ast.Name):
                    func_desc = child.func.id
                    function_calls.add(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    if isinstance(child.func.value, ast.Name):
                        func_desc = f"{child.func.value.id}.{child.func.attr}"
                    else:
                        func_desc = child.func.attr
                    function_calls.add(func_desc)

                relation = {
                    "var_name": "function_call",
                    "desc": f"Call function '{func_desc}'",
                    "operation": "Call",
                    "context": "Direct call" if isinstance(child.func, ast.Name) else "Attribute call",
                    "line": getattr(child, 'lineno', None)
                }
                input_output_relations.append(relation)

                # Check for I/O side effects
                if (isinstance(child.func, ast.Attribute)
                    and child.func.attr in ("write", "read", "append", "open", "print")):
                    side_effects.add("I/O Operation")

            elif isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
                flow_type = type(child).__name__
                control_flows.add(f"{node.name}->{flow_type} at line {getattr(child, 'lineno', 'NA')}")
                cflow_relation = {
                    "var_name": "control_flow",
                    "desc": f"Encountered {flow_type} at line {getattr(child, 'lineno', 'NA')}",
                    "operation": flow_type,
                    "context": "ControlFlow",
                    "line": getattr(child, 'lineno', None)
                }
                input_output_relations.append(cflow_relation)

            elif isinstance(child, ast.Break):
                control_flows.add(f"{node.name}->Break at line {child.lineno}")
                cflow_relation = {
                    "var_name": "control_flow",
                    "desc": f"Break at line {child.lineno}",
                    "operation": "Break",
                    "context": "ControlFlow",
                    "line": child.lineno
                }
                input_output_relations.append(cflow_relation)

            elif isinstance(child, ast.Continue):
                control_flows.add(f"{node.name}->Continue at line {child.lineno}")
                cflow_relation = {
                    "var_name": "control_flow",
                    "desc": f"Continue at line {child.lineno}",
                    "operation": "Continue",
                    "context": "ControlFlow",
                    "line": child.lineno
                }
                input_output_relations.append(cflow_relation)

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


# ------------------------------------------------------------------------
#  Helper: Safe Unparsing
# ------------------------------------------------------------------------
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


def _stmt_summary(stmt: ast.stmt) -> str:
    """
    Minimal text summary for a statement (used in CFG block labels).
    """
    if isinstance(stmt, ast.Assign):
        targets = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
        return f"Assign({','.join(targets)})"
    elif isinstance(stmt, ast.If):
        return "If(...)"
    elif isinstance(stmt, ast.While):
        return "While(...)"
    elif isinstance(stmt, ast.Expr):
        return "Expr(...)"
    else:
        return type(stmt).__name__


# ------------------------------------------------------------------------
#  REINTRODUCED: compare_function_similarity and helpers
# ------------------------------------------------------------------------
def levenshtein_similarity(str1: str, str2: str) -> float:
    """
    Compute Levenshtein-like similarity using difflib's ratio.
    """
    seq = difflib.SequenceMatcher(None, str1, str2)
    return round(seq.ratio(), 3)


def parse_ast_structure(expression: str) -> str:
    """
    Convert an expression string into a normalized AST structure dump.
    Used for advanced matching if needed.
    """
    try:
        tree = ast.parse(expression, mode='eval')
        return ast.dump(tree)
    except SyntaxError:
        return expression


def _jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """
    Basic Jaccard similarity for sets.
    Returns ratio of intersection to union.
    """
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return round(len(intersection) / len(union), 3)


def compare_function_similarity(func_info_a: Dict[str, Any], func_info_b: Dict[str, Any]) -> float:
    """
    Compute a data-flow-based similarity score between two function-level analyses
    by building mini dependency graphs and measuring overlap (Jaccard).
    """
    if not func_info_a or not func_info_b:
        logging.warning("Missing function info for similarity calculation.")
        return 0.0

    # Each function info should have a "dependencies" dict with "reads", "writes", etc.
    deps_a = func_info_a.get("dependencies", {})
    deps_b = func_info_b.get("dependencies", {})

    def build_dependency_graph(deps: Dict[str, Any]) -> (Set[str], Set[tuple]):
        """
        Build a minimal 'graph' from the function's dependencies:
          - nodes: read/written variables, function calls, control flows
          - edges: (var -> op) or (op -> var) pairs
        """
        nodes = set()
        edges = set()

        # Add simple nodes for reads, writes, calls, etc.
        for key in ["reads", "writes", "function_calls", "control_flows", "exception_handling", "side_effects"]:
            for item in deps.get(key, []):
                token = f"{key}:{item}"
                nodes.add(token.lower())

        # For input_output_relations, create edges from var_name => operation
        for relation in deps.get("input_output_relations", []):
            var_name = relation.get("var_name", "").lower().strip()
            operation = relation.get("operation", "").lower().strip()
            if var_name and operation:
                var_node = f"var:{var_name}"
                op_node = f"op:{operation}"
                nodes.add(var_node)
                nodes.add(op_node)
                edges.add((var_node, op_node))

        return nodes, edges

    def jaccard_similarity_for_graphs(nodes1, edges1, nodes2, edges2) -> float:
        """
        Weighted Jaccard on nodes & edges => final combined similarity.
        """
        node_sim = _jaccard_similarity(nodes1, nodes2)
        edge_sim = _jaccard_similarity(set(edges1), set(edges2))
        alpha = 0.6  # weighting
        return round(alpha * node_sim + (1 - alpha) * edge_sim, 3)

    # Build dependency graphs
    nodes_a, edges_a = build_dependency_graph(deps_a)
    nodes_b, edges_b = build_dependency_graph(deps_b)

    return jaccard_similarity_for_graphs(nodes_a, edges_a, nodes_b, edges_b)


# ------------------------------------------------------------------------
#  RESTORED: get_usage_info
# ------------------------------------------------------------------------
def get_usage_info(func) -> Dict[str, Dict[str, Any]]:
    """
    Enhanced Usage Analyzer that infers data types based on usage patterns.
    Supports int, float, str, list, dict, bool, Callable, and custom objects.
    """
    usage_info = {}
    try:
        # Grab source code from the function itself
        source = inspect.getsource(func)
        tree = ast.parse(source)
        func_def = next(
            (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)),
            None
        )
        if not func_def:
            return usage_info

        # Collect parameter names
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

        # Analyze AST to infer param usage
        for node in ast.walk(func_def):
            # For loops / comprehensions
            if isinstance(node, (ast.For, ast.comprehension)):
                if isinstance(node, ast.For) and isinstance(node.iter, ast.Name):
                    if node.iter.id in param_names:
                        usage_info[node.iter.id]["iterated"] = True
                        usage_info[node.iter.id]["type"] = "list"

            # Check for len(...) calls
            if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "len"):
                for arg in node.args:
                    if isinstance(arg, ast.Name) and arg.id in param_names:
                        usage_info[arg.id]["len_called"] = True
                        usage_info[arg.id]["type"] = "list"

            # Arithmetic usage
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

            # Boolean checks
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

            # Function call usage
            if isinstance(node, ast.Call):
                # For each argument or kwarg referencing param
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
                        # Type might become "object" unless we detect more detail

        # Heuristic naming fallback
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

    except Exception as ex:
        logging.error(f"get_usage_info failed: {ex}", exc_info=True)

    return usage_info
