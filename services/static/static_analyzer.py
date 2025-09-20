import ast
import logging
from typing import List, Dict, Any
import math

# Import the TSED ramp normalization function from the updated ast_utils
from services.static.ast_utils import ramp_tsed_distance

logging.basicConfig(level=logging.DEBUG)



#
# ------------------------------ AST → Tree Conversion -----------------------------
#
def ast_to_tree(node: ast.AST) -> tuple:
    """
    Converts a Python AST node into a (label, [children]) tuple.
    We embed info to achieve maximum structural differentiation:
      - Major node types (ClassDef, FunctionDef, For, If, Try, etc.)
      - Operator types (e.g. BinOp(Add), Compare(Eq), etc.)
      - Names, constants, attribute owners
      - Decorators, arguments, async markers, docstrings
      - Container literals (list, tuple, dict, etc.), comprehensions, slices, etc.

    If we find a node type not yet covered, we produce a fallback label = type(node).__name__.
    """
    if node is None:
        return ("<None>", [])

    def sub(n):
        return ast_to_tree(n)

    def sublist(lst):
        return [ast_to_tree(x) for x in lst]

    if isinstance(node, ast.Module):
        label = "Module"
        children = sublist(node.body)

    elif isinstance(node, ast.ClassDef):
        base_names = [base.id if isinstance(base, ast.Name) else ast_to_tree(base)[0]
                      for base in node.bases]
        deco_labels = [sub(d) for d in node.decorator_list]
        label = f"ClassDef({node.name}, bases={base_names})"
        children = deco_labels + sublist(node.body)

    elif isinstance(node, ast.FunctionDef):
        arg_list = [arg.arg for arg in node.args.args]
        is_async = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
        deco_labels = [sub(d) for d in node.decorator_list]
        label = f"{is_async}FunctionDef({node.name}, args={arg_list})"
        children = deco_labels + sublist(node.body)

    elif isinstance(node, ast.AsyncFunctionDef):
        arg_list = [arg.arg for arg in node.args.args]
        deco_labels = [sub(d) for d in node.decorator_list]
        label = f"async FunctionDef({node.name}, args={arg_list})"
        children = deco_labels + sublist(node.body)

    elif isinstance(node, ast.arg):
        label = f"arg({node.arg})"
        children = []

    elif isinstance(node, ast.Lambda):
        arg_list = [a.arg for a in node.args.args]
        label = f"Lambda(args={arg_list})"
        children = [sub(node.body)]

    elif isinstance(node, ast.Name):
        label = f"Name({node.id})"
        children = []

    elif isinstance(node, ast.Constant):
        label = f"Constant({repr(node.value)})"
        children = []

    elif isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name):
            label = f"Attribute({node.value.id}.{node.attr})"
        else:
            label = f"Attribute({node.attr})"
        children = [sub(node.value)]

    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            label = f"Call({node.func.id}, nargs={len(node.args)})"
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                label = f"Call({node.func.value.id}.{node.func.attr}, nargs={len(node.args)})"
            else:
                label = f"Call({node.func.attr}, nargs={len(node.args)})"
        else:
            label = f"Call(unknown, nargs={len(node.args)})"
        children = sublist(node.args) + sublist(node.keywords)

    elif isinstance(node, ast.keyword):
        label = f"kw({node.arg})"
        children = [sub(node.value)] if node.value else []

    elif isinstance(node, ast.BinOp):
        op_name = type(node.op).__name__
        label = f"BinOp({op_name})"
        children = [sub(node.left), sub(node.right)]

    elif isinstance(node, ast.UnaryOp):
        op_name = type(node.op).__name__
        label = f"UnaryOp({op_name})"
        children = [sub(node.operand)]

    elif isinstance(node, ast.BoolOp):
        op_name = type(node.op).__name__
        label = f"BoolOp({op_name})"
        children = sublist(node.values)

    elif isinstance(node, ast.Compare):
        op_names = [type(op).__name__ for op in node.ops]
        label = f"Compare({','.join(op_names)})"
        children = [sub(node.left)] + sublist(node.comparators)

    elif isinstance(node, ast.If):
        label = "If"
        children = [sub(node.test)] + sublist(node.body) + sublist(node.orelse)

    elif isinstance(node, ast.For):
        label = f"For(async={isinstance(node, ast.AsyncFor)})"
        children = [sub(node.target), sub(node.iter)] + sublist(node.body) + sublist(node.orelse)

    elif isinstance(node, ast.AsyncFor):
        label = "For(async=True)"
        children = [sub(node.target), sub(node.iter)] + sublist(node.body) + sublist(node.orelse)

    elif isinstance(node, ast.While):
        label = "While"
        children = [sub(node.test)] + sublist(node.body) + sublist(node.orelse)

    elif isinstance(node, ast.Try):
        label = "Try"
        children = sublist(node.body) + sublist(node.handlers) + sublist(node.orelse) + sublist(node.finalbody)

    elif isinstance(node, ast.ExceptHandler):
        if node.type:
            label = f"ExceptHandler({ast_to_tree(node.type)[0]})"
        else:
            label = "ExceptHandler()"
        children = sublist(node.body)

    elif isinstance(node, ast.With):
        label = "With"
        children = sublist(node.items) + sublist(node.body)

    elif isinstance(node, ast.AsyncWith):
        label = "AsyncWith"
        children = sublist(node.items) + sublist(node.body)

    elif isinstance(node, ast.withitem):
        label = "withitem"
        children = [sub(node.context_expr)]
        if node.optional_vars:
            children.append(sub(node.optional_vars))

    elif isinstance(node, ast.Raise):
        if node.exc:
            label = f"Raise({ast_to_tree(node.exc)[0]})"
            children = [sub(node.exc)]
            if node.cause:
                children.append(sub(node.cause))
        else:
            label = "Raise()"
            children = []

    elif isinstance(node, ast.Return):
        label = "Return"
        children = [sub(node.value)] if node.value else []

    elif isinstance(node, ast.Yield):
        label = "Yield"
        children = [sub(node.value)] if node.value else []

    elif isinstance(node, ast.YieldFrom):
        label = "YieldFrom"
        children = [sub(node.value)]

    elif isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
        label = f"Import({names})"
        children = []

    elif isinstance(node, ast.ImportFrom):
        names = [alias.name for alias in node.names]
        module = node.module if node.module else ""
        label = f"ImportFrom({module}, {names}, level={node.level})"
        children = []

    elif isinstance(node, ast.Subscript):
        label = "Subscript"
        children = [sub(node.value), sub(node.slice)]

    elif isinstance(node, ast.Slice):
        label = "Slice"
        children = []
        if node.lower:
            children.append(sub(node.lower))
        else:
            children.append(("<None>", []))
        if node.upper:
            children.append(sub(node.upper))
        else:
            children.append(("<None>", []))
        if node.step:
            children.append(sub(node.step))
        else:
            children.append(("<None>", []))

    elif isinstance(node, ast.ExtSlice):
        label = "ExtSlice"
        children = sublist(node.dims)

    elif isinstance(node, ast.Index):
        label = "Index"
        children = [sub(node.value)]

    elif isinstance(node, ast.List):
        label = "List"
        children = sublist(node.elts)

    elif isinstance(node, ast.Tuple):
        label = "Tuple"
        children = sublist(node.elts)

    elif isinstance(node, ast.Set):
        label = "Set"
        children = sublist(node.elts)

    elif isinstance(node, ast.Dict):
        label = "Dict"
        kv_pairs = []
        for k, v in zip(node.keys, node.values):
            if k is None:
                kv_pairs.append(("DictKey(None)", [sub(v)]))
            else:
                kv_pairs.append(("DictKeyVal", [sub(k), sub(v)]))
        children = kv_pairs

    else:
        label = type(node).__name__
        children = sublist(list(ast.iter_child_nodes(node)))

    return (label, children)

#
# ------------------------------ Utility Functions -----------------------------
#
def tree_size(T: tuple) -> int:
    """Count total nodes in the (label, children) tuple structure."""
    return 1 + sum(tree_size(child) for child in T[1])

def sequence_edit_distance(seq1: List[tuple], seq2: List[tuple], dist_fn) -> int:
    m, n = len(seq1), len(seq2)
    D = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        D[i][0] = i
    for j in range(n + 1):
        D[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = dist_fn(seq1[i - 1], seq2[j - 1])
            D[i][j] = min(
                D[i - 1][j] + 1,
                D[i][j - 1] + 1,
                D[i - 1][j - 1] + cost
            )
    return D[m][n]

def similar_labels(label1: str, label2: str) -> bool:
    """
    Returns True if the base types of the two labels are the same.
    For example, "Name(x)" and "Name(y)" are considered similar.
    """
    base1 = label1.split('(')[0]
    base2 = label2.split('(')[0]
    return base1 == base2

def zhang_shasha_distance(T1: tuple, T2: tuple) -> float:
    """
    Compute the tree edit distance using a modified Zhang–Shasha approach with weighted costs.
    """
    if not T1:
        return tree_size(T2)
    if not T2:
        return tree_size(T1)

    # Weighted substitution cost
    if T1[0] == T2[0]:
        cost_root = 0
    elif similar_labels(T1[0], T2[0]):
        cost_root = 0.3
    else:
        cost_root = 1

    cost_children = sequence_edit_distance(T1[1], T2[1], zhang_shasha_distance)
    return cost_root + cost_children

#
# ------------------------------ TSED Similarity Computation -----------------------------
#
def calculate_similarity(node1: ast.AST, node2: ast.AST) -> float:
    """
    Computes the TSED-based similarity between two AST nodes.
    Uses the Zhang–Shasha tree edit distance and applies a ramp normalization:
       TSED = max(1 - (edit_distance / max(number of nodes in either tree)), 0)
    """
    try:
        tree1 = ast_to_tree(node1)
        tree2 = ast_to_tree(node2)
        dist = zhang_shasha_distance(tree1, tree2)
        # Use the maximum node count from both trees for normalization
        max_nodes = max(tree_size(tree1), tree_size(tree2))
        similarity = ramp_tsed_distance(dist, max_nodes)
        return similarity
    except Exception as e:
        logging.error(f"Error in calculate_similarity: {e}", exc_info=True)
        return 0.0

def calculate_type1_similarity(node1: ast.AST, node2: ast.AST) -> float:
    """
    Type‐1 similarity by comparing raw AST dumps with a diff approach
    (ignores variable renaming but picks up structural changes).
    """
    try:
        from difflib import SequenceMatcher
        str1 = ast.dump(node1)
        str2 = ast.dump(node2)
        return SequenceMatcher(None, str1, str2).ratio()
    except Exception as e:
        logging.error(f"Error in calculate_type1_similarity: {e}")
        return 0.0

#
# ------------------------------ Additional Analysis Helpers -----------------------------
#
def calculate_complexity(node: ast.AST) -> int:
    """
    Cyclomatic complexity: +1 per function, +1 for each control statement, + additional for boolean ops
    """
    complexity = 1
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler, ast.Try, ast.With, ast.AsyncWith)):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += max(0, len(child.values) - 1)
    return complexity

def analyze_return_behavior(node: ast.FunctionDef) -> str:
    """
    Return "Always", "Never", or "Mixed" depending on whether every statement uses Return or none do, etc.
    """
    if not node.body:
        return "Never"

    def has_return(stmt):
        return any(isinstance(sub, ast.Return) for sub in ast.walk(stmt))

    returns = [has_return(stmt) for stmt in node.body]
    if all(returns):
        return "Always"
    elif not any(returns):
        return "Never"
    else:
        return "Mixed"

def calculate_lines_of_code(node: ast.AST, source_code: str) -> int:
    """
    Count lines of code used by the node's lineno..end_lineno range, excluding blank/comment lines.
    """
    if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
        lines = source_code.splitlines()[node.lineno - 1: node.end_lineno]
        meaningful = [ln for ln in lines if ln.strip() and not ln.strip().startswith("#")]
        return len(meaningful)
    return 0

def calculate_nesting_depth(node: ast.AST) -> int:
    """Compute maximum nesting depth of blocks (e.g., nested ifs/loops/try, etc.)."""
    def recurse(n: ast.AST, current: int) -> int:
        maxd = current
        for c in ast.iter_child_nodes(n):
            if isinstance(c, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.AsyncWith, ast.FunctionDef, ast.ClassDef)):
                maxd = max(maxd, recurse(c, current + 1))
            else:
                maxd = max(maxd, recurse(c, current))
        return maxd
    return recurse(node, 0)

#
# ------------------------------ Weighted + Log Normalization for Static Score -----------------------------
#

def compute_weighted_score(complexity: int, loc: int, params: int, nesting: int) -> float:
    """
    Weighted Summation:
      WeightedScore = 0.4*Complexity + 0.3*LOC + 0.1*Params + 0.2*NestingDepth
    """
    return (0.4 * complexity) + (0.3 * loc) + (0.1 * params) + (0.2 * nesting)

def log_normalize_score(weighted_score: float, max_weighted_score: float) -> float:
    """
    Logarithmic Normalization:
      score = ln(weighted_score + 1) / ln(max_weighted_score + 1)
    Ensures final score is in [0..1].
    """
    if max_weighted_score <= 0:
        return 0.0
    numerator = math.log1p(weighted_score)
    denominator = math.log1p(max_weighted_score)
    return numerator / denominator if denominator != 0 else 0.0

def compute_static_similarity(score1: float, score2: float) -> float:
    """
    Convert two [0..1] normalized scores into a similarity measure:
      StaticSim = 1 - |score1 - score2|
    """
    return 1.0 - abs(score1 - score2)

#
# ------------------------------ Main High‐Level Similar Node Finder + Static Score Calculation -----------------------------
#
def find_similar_nodes(
    tree: ast.AST,
    source_code: str,
    threshold: float,
    abs_file_path: str = "unknown_file.py"
) -> List[Dict[str, Any]]:
    """
    Walk the AST, gather all top‐level function definitions (including async),
    compare each pair with our high‐fidelity TSED similarity approach,
    and only keep pairs whose similarity is above the given threshold.

    Now also compute Weighted + Log-Normalized static scores for each function
    and store them in function metrics.
    """
    similar_nodes = []
    processed_pairs = set()

    # Gather candidate function nodes
    candidate_nodes = [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name and n.name.strip()
    ]
    candidate_nodes.sort(key=lambda n: (n.lineno, getattr(n, "end_lineno", n.lineno)))

    # 1. Precompute function metrics, including Weighted & Log-Normalized Score
    function_scores = {}
    weighted_scores = []
    for fn_node in candidate_nodes:
        name = fn_node.name.strip()
        if not name:
            continue

        # Basic metrics
        cplx = calculate_complexity(fn_node)
        loc = calculate_lines_of_code(fn_node, source_code)
        params = len(fn_node.args.args)
        nesting = calculate_nesting_depth(fn_node)

        # Weighted Summation
        w_score = compute_weighted_score(cplx, loc, params, nesting)
        weighted_scores.append(w_score)

        function_scores[name] = {
            "node": fn_node,
            "complexity": cplx,
            "loc": loc,
            "params": params,
            "nesting_depth": nesting,
            "weighted_score": w_score,  # Not yet normalized
        }

    # 2. Find max weighted score to do log normalization
    if weighted_scores:
        max_wscore = max(weighted_scores)
    else:
        max_wscore = 1.0

    # 3. Apply log normalization
    for name, info in function_scores.items():
        info["static_score"] = log_normalize_score(info["weighted_score"], max_wscore)

    # 4. Compare each pair for AST similarity
    for i in range(len(candidate_nodes)):
        for j in range(i + 1, len(candidate_nodes)):
            node1, node2 = candidate_nodes[i], candidate_nodes[j]
            name1, name2 = node1.name.strip(), node2.name.strip()
            if not (name1 and name2):
                continue
            pair_id = tuple(sorted([name1, name2]))
            if pair_id in processed_pairs:
                continue
            processed_pairs.add(pair_id)

            try:
                sim = calculate_similarity(node1, node2)
                type1_sim = calculate_type1_similarity(node1, node2)
            except Exception as e:
                logging.error(f"Error comparing {name1} vs {name2}: {e}", exc_info=True)
                continue

            if sim >= threshold:
                # Build function metrics
                node1_metrics = {
                    "name": name1,
                    "complexity": function_scores[name1]["complexity"],
                    "return_behavior": analyze_return_behavior(node1) if isinstance(node1, ast.FunctionDef) else "Mixed",
                    "loc": function_scores[name1]["loc"],
                    "parameter_count": function_scores[name1]["params"],
                    "nesting_depth": function_scores[name1]["nesting_depth"],
                    "start_line": node1.lineno,
                    "end_line": getattr(node1, "end_lineno", node1.lineno),
                    "file_path": abs_file_path,
                    "static_score": function_scores[name1]["static_score"],
                }
                node2_metrics = {
                    "name": name2,
                    "complexity": function_scores[name2]["complexity"],
                    "return_behavior": analyze_return_behavior(node2) if isinstance(node2, ast.FunctionDef) else "Mixed",
                    "loc": function_scores[name2]["loc"],
                    "parameter_count": function_scores[name2]["params"],
                    "nesting_depth": function_scores[name2]["nesting_depth"],
                    "start_line": node2.lineno,
                    "end_line": getattr(node2, "end_lineno", node2.lineno),
                    "file_path": abs_file_path,
                    "static_score": function_scores[name2]["static_score"],
                }

                result = {
                    "function1_metrics": node1_metrics,
                    "function2_metrics": node2_metrics,
                    "ast_similarity": sim,
                    "type1_similarity": type1_sim,
                }
                similar_nodes.append(result)
                logging.debug(f"   ✅ Accepted pair {name1} & {name2} with AST sim={sim:.4f}")

    return similar_nodes
