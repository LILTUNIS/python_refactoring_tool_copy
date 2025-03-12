import ast
import logging
from typing import List, Dict, Any
import math

# Set logging to INFO to reduce debug output
logging.basicConfig(level=logging.INFO)


#
# ------------------------------ AST → Tree Conversion with Caching -----------------------------
#
_AST_TO_TREE_CACHE: Dict[int, tuple] = {}  # Global cache for ast_to_tree to improve performance


def ast_to_tree(node: ast.AST) -> tuple:
    """
    Converts a Python AST node into a (label, [children]) tuple.
    Uses a global cache to store intermediate results for performance.
    """
    if node is None:
        return ("<None>", [])

    node_id = id(node)
    if node_id in _AST_TO_TREE_CACHE:
        return _AST_TO_TREE_CACHE[node_id]

    def sub(n):
        return ast_to_tree(n)  # Removed cache argument, using global cache

    def sublist(lst):
        return [ast_to_tree(x) for x in lst]  # Removed cache argument, using global cache

    if isinstance(node, ast.Module):
        label = "Module"
        children = sublist(node.body)

    elif isinstance(node, ast.ClassDef):
        base_names = tuple(base.id if isinstance(base, ast.Name) else ast_to_tree(base)[0] # Optimized base names to tuple
                          for base in node.bases)
        deco_labels = tuple(sub(d) for d in node.decorator_list) # Optimized deco_labels to tuple
        label = f"ClassDef({node.name}, bases={base_names})"
        children = deco_labels + tuple(sublist(node.body)) # Optimized children to tuple

    elif isinstance(node, ast.FunctionDef):
        arg_list = tuple(arg.arg for arg in node.args.args) # Optimized arg_list to tuple
        deco_labels = tuple(sub(d) for d in node.decorator_list) # Optimized deco_labels to tuple
        label = f"FunctionDef({node.name}, args={arg_list})"
        children = deco_labels + tuple(sublist(node.body)) # Optimized children to tuple

    elif isinstance(node, ast.AsyncFunctionDef):
        arg_list = tuple(arg.arg for arg in node.args.args) # Optimized arg_list to tuple
        deco_labels = tuple(sub(d) for d in node.decorator_list) # Optimized deco_labels to tuple
        label = f"async FunctionDef({node.name}, args={arg_list})"
        children = deco_labels + tuple(sublist(node.body)) # Optimized children to tuple

    elif isinstance(node, ast.arg):
        label = f"arg({node.arg})"
        children = () # Optimized to tuple

    elif isinstance(node, ast.Lambda):
        arg_list = tuple(a.arg for a in node.args.args) # Optimized arg_list to tuple
        label = f"Lambda(args={arg_list})"
        children = (sub(node.body),) # Optimized to tuple

    elif isinstance(node, ast.Name):
        label = f"Name({node.id})"
        children = () # Optimized to tuple

    elif isinstance(node, ast.Constant):
        label = f"Constant({repr(node.value)})"
        children = () # Optimized to tuple

    elif isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name):
            label = f"Attribute({node.value.id}.{node.attr})"
        else:
            label = f"Attribute({node.attr})"
        children = (sub(node.value),) # Optimized to tuple

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
        children = tuple(sublist(node.args) + sublist(node.keywords)) # Optimized children to tuple

    elif isinstance(node, ast.keyword):
        label = f"kw({node.arg})"
        children = (sub(node.value),) if node.value else () # Optimized to tuple

    elif isinstance(node, ast.BinOp):
        op_name = type(node.op).__name__
        label = f"BinOp({op_name})"
        children = (sub(node.left), sub(node.right)) # Optimized to tuple

    elif isinstance(node, ast.UnaryOp):
        op_name = type(node.op).__name__
        label = f"UnaryOp({op_name})"
        children = (sub(node.operand),) # Optimized to tuple

    elif isinstance(node, ast.BoolOp):
        op_name = type(node.op).__name__
        label = f"BoolOp({op_name})"
        children = tuple(sublist(node.values)) # Optimized children to tuple

    elif isinstance(node, ast.Compare):
        op_names = tuple(type(op).__name__ for op in node.ops) # Optimized op_names to tuple
        label = f"Compare({','.join(op_names)})"
        children = (sub(node.left),) + tuple(sublist(node.comparators)) # Optimized children to tuple

    elif isinstance(node, ast.If):
        label = "If"
        children = (sub(node.test),) + tuple(sublist(node.body) + sublist(node.orelse)) # Optimized children to tuple

    elif isinstance(node, ast.For):
        label = f"For(async={isinstance(node, ast.AsyncFor)})"
        children = (sub(node.target), sub(node.iter)) + tuple(sublist(node.body) + sublist(node.orelse)) # Optimized children to tuple

    elif isinstance(node, ast.AsyncFor):
        label = "For(async=True)"
        children = (sub(node.target), sub(node.iter)) + tuple(sublist(node.body) + sublist(node.orelse)) # Optimized children to tuple

    elif isinstance(node, ast.While):
        label = "While"
        children = (sub(node.test),) + tuple(sublist(node.body) + sublist(node.orelse)) # Optimized children to tuple

    elif isinstance(node, ast.Try):
        label = "Try"
        children = tuple(sublist(node.body) + sublist(node.handlers) + sublist(node.orelse) + sublist(node.finalbody)) # Optimized children to tuple

    elif isinstance(node, ast.ExceptHandler):
        if node.type:
            label = f"ExceptHandler({ast_to_tree(node.type)[0]})" # Removed cache argument, using global cache
        else:
            label = "ExceptHandler()"
        children = tuple(sublist(node.body)) # Optimized children to tuple

    elif isinstance(node, ast.With):
        label = "With"
        children = tuple(sublist(node.items) + sublist(node.body)) # Optimized children to tuple

    elif isinstance(node, ast.AsyncWith):
        label = "AsyncWith"
        children = tuple(sublist(node.items) + sublist(node.body)) # Optimized children to tuple

    elif isinstance(node, ast.withitem):
        label = "withitem"
        children = [sub(node.context_expr)] # Keep list for conditional append
        if node.optional_vars:
            children.append(sub(node.optional_vars))
        children = tuple(children) # Optimized to tuple

    elif isinstance(node, ast.Raise):
        children = [] # Keep list for conditional append
        if node.exc:
            label = f"Raise({ast_to_tree(node.exc)[0]})" # Removed cache argument, using global cache
            children.append(sub(node.exc))
            if node.cause:
                children.append(sub(node.cause))
        else:
            label = "Raise()"
        children = tuple(children) # Optimized to tuple

    elif isinstance(node, ast.Return):
        label = "Return"
        children = (sub(node.value),) if node.value else () # Optimized to tuple

    elif isinstance(node, ast.Yield):
        label = "Yield"
        children = (sub(node.value),) if node.value else () # Optimized to tuple

    elif isinstance(node, ast.YieldFrom):
        label = "YieldFrom"
        children = (sub(node.value),) # Optimized to tuple

    elif isinstance(node, ast.Import):
        names = tuple(alias.name for alias in node.names) # Optimized names to tuple
        label = f"Import({names})"
        children = () # Optimized to tuple

    elif isinstance(node, ast.ImportFrom):
        names = tuple(alias.name for alias in node.names) # Optimized names to tuple
        module = node.module if node.module else ""
        label = f"ImportFrom({module}, {names}, level={node.level})"
        children = () # Optimized to tuple

    elif isinstance(node, ast.Subscript):
        label = "Subscript"
        children = (sub(node.value), sub(node.slice)) # Optimized to tuple

    elif isinstance(node, ast.Slice):
        label = "Slice"
        children_list = [] # Keep list for conditional append
        children_list.append(sub(node.lower) if node.lower else ("<None>", []))
        children_list.append(sub(node.upper) if node.upper else ("<None>", []))
        children_list.append(sub(node.step) if node.step else ("<None>", []))
        children = tuple(children_list) # Optimized to tuple

    elif isinstance(node, ast.ExtSlice):
        label = "ExtSlice"
        children = tuple(sublist(node.dims)) # Optimized children to tuple

    elif isinstance(node, ast.Index):
        label = "Index"
        children = (sub(node.value),) # Optimized to tuple

    elif isinstance(node, ast.List):
        label = "List"
        children = tuple(sublist(node.elts)) # Optimized children to tuple

    elif isinstance(node, ast.Tuple):
        label = "Tuple"
        children = tuple(sublist(node.elts)) # Optimized children to tuple

    elif isinstance(node, ast.Set):
        label = "Set"
        children = tuple(sublist(node.elts)) # Optimized children to tuple

    elif isinstance(node, ast.Dict):
        label = "Dict"
        kv_pairs_list = [] # Keep list for conditional append
        for k, v in zip(node.keys, node.values):
            if k is None:
                kv_pairs_list.append(("DictKey(None)", (sub(v),))) # Optimized to tuple
            else:
                kv_pairs_list.append(("DictKeyVal", (sub(k), sub(v)))) # Optimized to tuple
        children = tuple(kv_pairs_list) # Optimized to tuple

    elif isinstance(node, ast.ListComp):
        label = "ListComp"
        children = (sub(node.elt),) + tuple(sublist(node.generators)) # Optimized children to tuple

    elif isinstance(node, ast.SetComp):
        label = "SetComp"
        children = (sub(node.elt),) + tuple(sublist(node.generators)) # Optimized children to tuple

    elif isinstance(node, ast.DictComp):
        label = "DictComp"
        children = (sub(node.key), sub(node.value)) + tuple(sublist(node.generators)) # Optimized children to tuple

    elif isinstance(node, ast.GeneratorExp):
        label = "GeneratorExp"
        children = (sub(node.elt),) + tuple(sublist(node.generators)) # Optimized children to tuple

    elif isinstance(node, ast.comprehension):
        label = "comprehension"
        children = (sub(node.target), sub(node.iter)) + tuple(sublist(node.ifs)) # Optimized children to tuple

    else:
        label = type(node).__name__
        children = tuple(sublist(list(ast.iter_child_nodes(node)))) # Optimized children to tuple

    result = (label, children)
    _AST_TO_TREE_CACHE[node_id] = result
    return result


#
# ------------------------------ Zhang–Shasha Similarity -----------------------------
#
def tree_size(T: tuple) -> int:
    """Count total nodes in the (label, children) tuple structure."""
    return 1 + sum(tree_size(child) for child in T[1])


_SEQUENCE_EDIT_DISTANCE_CACHE = {} # Global cache for sequence_edit_distance

def sequence_edit_distance(seq1: tuple, seq2: tuple, dist_fn) -> int: # Optimized seq1 and seq2 to tuple
    """Calculates the edit distance between two sequences of tree tuples, with caching."""
    cache_key = (id(seq1), id(seq2)) # Using ids of tuples as keys, assuming tuple immutability
    if cache_key in _SEQUENCE_EDIT_DISTANCE_CACHE:
        return _SEQUENCE_EDIT_DISTANCE_CACHE[cache_key]

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
    result = D[m][n]
    _SEQUENCE_EDIT_DISTANCE_CACHE[cache_key] = result
    return result


def similar_labels(label1: str, label2: str) -> bool:
    """
    Returns True if the base types of the two labels are the same.
    For example, "Name(x)" and "Name(y)" are considered similar.
    """
    base1_split = label1.split('(') # Split once
    base2_split = label2.split('(') # Split once
    base1 = base1_split[0]
    base2 = base2_split[0]
    return base1 == base2


_ZHANG_SHASHA_DISTANCE_CACHE = {} # Global cache for zhang_shasha_distance

def zhang_shasha_distance(T1: tuple, T2: tuple) -> float: # Optimized T1 and T2 to tuple
    """
    Compute the tree edit distance using a modified Zhang–Shasha approach with weighted costs, with caching.
    """
    cache_key = (id(T1), id(T2)) # Using ids of tuples as keys, assuming tuple immutability
    if cache_key in _ZHANG_SHASHA_DISTANCE_CACHE:
        return _ZHANG_SHASHA_DISTANCE_CACHE[cache_key]

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
    result = cost_root + cost_children
    _ZHANG_SHASHA_DISTANCE_CACHE[cache_key] = result
    return result


def calculate_similarity(node1: ast.AST, node2: ast.AST) -> float:
    """Calculates similarity between two AST nodes, clearing caches for each call."""
    _AST_TO_TREE_CACHE.clear() # Clear cache at the start of each similarity calculation
    _SEQUENCE_EDIT_DISTANCE_CACHE.clear()
    _ZHANG_SHASHA_DISTANCE_CACHE.clear()
    try:
        tree1 = ast_to_tree(node1)
        tree2 = ast_to_tree(node2)
        dist = zhang_shasha_distance(tree1, tree2)
        total_nodes = tree_size(tree1) + tree_size(tree2)
        if total_nodes == 0:
            return 1.0
        normalized_dist = dist / total_nodes
        similarity = 1.0 - normalized_dist
        return max(0.0, min(similarity, 1.0))
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
        meaningful = (ln for ln in lines if ln.strip() and not ln.strip().startswith("#")) # Use generator for efficiency
        return sum(1 for _ in meaningful) # Count using generator
    return 0


def calculate_nesting_depth(node: ast.AST) -> int:
    """Compute maximum nesting depth of blocks (e.g., nested ifs/loops/try, etc.)."""

    def recurse(n: ast.AST, current: int) -> int:
        maxd = current
        for c in ast.iter_child_nodes(n):
            if isinstance(c, (
            ast.If, ast.For, ast.While, ast.Try, ast.With, ast.AsyncWith, ast.FunctionDef, ast.ClassDef)):
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
    compare each pair with our high‐fidelity Zhang–Shasha approach,
    and only keep pairs whose similarity is above the given threshold.
    Also computes Weighted + Log-Normalized static scores for each function.
    Includes a preliminary filter based on static score differences.
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

        cplx = calculate_complexity(fn_node)
        loc = calculate_lines_of_code(fn_node, source_code)
        params = len(fn_node.args.args)
        nesting = calculate_nesting_depth(fn_node)
        w_score = compute_weighted_score(cplx, loc, params, nesting)
        weighted_scores.append(w_score)
        function_scores[name] = {
            "node": fn_node,
            "complexity": cplx,
            "loc": loc,
            "params": params,
            "nesting_depth": nesting,
            "weighted_score": w_score,
        }

    max_wscore = max(weighted_scores) if weighted_scores else 1.0

    for name, info in function_scores.items():
        info["static_score"] = log_normalize_score(info["weighted_score"], max_wscore)

    # Preliminary filter threshold for static score differences (configurable)
    static_score_diff_threshold = 0.2

    # 2. Compare each pair for AST similarity with a preliminary filter
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

            # Apply preliminary filter based on static scores
            score_diff = abs(function_scores[name1]["static_score"] - function_scores[name2]["static_score"])
            if score_diff > static_score_diff_threshold:
                continue  # Skip pairs that differ too much in static score

            try:
                sim = calculate_similarity(node1, node2)
                type1_sim = calculate_type1_similarity(node1, node2)
            except Exception as e:
                logging.error(f"Error comparing {name1} vs {name2}: {e}", exc_info=True)
                continue

            if sim >= threshold:
                node1_metrics = {
                    "name": name1,
                    "complexity": function_scores[name1]["complexity"],
                    "return_behavior": analyze_return_behavior(node1) if isinstance(node1,
                                                                                    ast.FunctionDef) else "Mixed",
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
                    "return_behavior": analyze_return_behavior(node2) if isinstance(node2,
                                                                                    ast.FunctionDef) else "Mixed",
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
                logging.info(f"Accepted pair {name1} & {name2} with AST similarity {sim:.4f}")
    return similar_nodes