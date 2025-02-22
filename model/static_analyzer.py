#static_analyzer.py
import ast
import logging
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Dict, Any

# Configure logging for better traceability
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


@dataclass
class SimilarityResult:
    """Data class to store similarity analysis results."""
    node1: ast.AST  # First node in the comparison
    node2: ast.AST  # Second node in the comparison
    similarity: float  # Similarity score between node1 and node2
    complexity: int  # Cyclomatic complexity of node1
    return_behavior: str  # Return behavior of node1 ("Always", "Never", "Mixed")
    loc: int  # Lines of code for node1
    parameter_count: int  # Number of parameters for node1
    nesting_depth: int  # Maximum nesting depth for node1
    ast_pattern: Dict  # Normalized AST structure for node1


def parse_files(files: List[str]) -> Dict[str, ast.AST]:
    """Parse multiple Python files into ASTs."""
    ast_trees = {}
    for file_path in files:
        try:
            with open(file_path, 'r', encoding="utf-8") as file:
                source_code = file.read()
            ast_trees[file_path] = ast.parse(source_code)
        except Exception as e:
            logging.error(f"Failed to parse {file_path}: {str(e)}")
    return ast_trees


def calculate_complexity(node: ast.AST) -> int:
    """Calculate cyclomatic complexity for an AST node."""
    complexity = 1
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += max(0, len(child.values) - 1)
    return complexity


def _normalize(n: ast.AST, cache: Dict[int, Any]) -> Any:
    """Helper function to normalize AST structures."""
    node_id = id(n)
    if node_id in cache:
        return cache[node_id]

    if isinstance(n, ast.FunctionDef):
        result = {
            "FunctionDef": {
                "name": n.name,
                "args": [arg.arg for arg in n.args.args],
                "body": [_normalize(stmt, cache) for stmt in n.body]
            }
        }
    elif isinstance(n, ast.If):
        result = {
            "If": {
                "test": _normalize(n.test, cache),
                "body": [_normalize(stmt, cache) for stmt in n.body],
                "orelse": [_normalize(stmt, cache) for stmt in n.orelse]
            }
        }
    elif isinstance(n, ast.For):
        result = {
            "For": {
                "target": _normalize(n.target, cache),
                "iter": _normalize(n.iter, cache),
                "body": [_normalize(stmt, cache) for stmt in n.body],
                "orelse": [_normalize(stmt, cache) for stmt in n.orelse]
            }
        }
    elif isinstance(n, ast.While):
        result = {
            "While": {
                "test": _normalize(n.test, cache),
                "body": [_normalize(stmt, cache) for stmt in n.body],
                "orelse": [_normalize(stmt, cache) for stmt in n.orelse]
            }
        }
    elif isinstance(n, ast.Return):
        result = {"Return": _normalize(n.value, cache) if n.value else None}
    elif isinstance(n, ast.BinOp):
        result = {
            "BinOp": {
                "op": type(n.op).__name__,
                "left": _normalize(n.left, cache),
                "right": _normalize(n.right, cache)
            }
        }
    elif isinstance(n, ast.Constant):
        result = {"Constant": n.value}
    elif isinstance(n, ast.Name):
        result = {"Name": n.id}
    else:
        result = str(type(n).__name__)

    cache[node_id] = result
    return result


def normalize_ast_structure(node: ast.AST) -> str:
    """Normalize AST structure to a consistent format."""
    return str(_normalize(node, {}))


def calculate_similarity(node1: ast.AST, node2: ast.AST) -> float:
    """Calculate similarity between two AST nodes."""
    try:
        normalized1 = normalize_ast_structure(node1)
        normalized2 = normalize_ast_structure(node2)
        return SequenceMatcher(None, normalized1, normalized2).ratio()
    except Exception as e:
        logging.error(f"Error in calculate_similarity: {e}")
        return 0.0


def analyze_return_behavior(node: ast.FunctionDef) -> str:
    """
    Analyze the return behavior of a function.

    Determines whether all paths have a return statement ("Always"),
    none do ("Never"), or if the behavior is inconsistent ("Mixed").

    Args:
        node: A FunctionDef AST node.

    Returns:
        A string: "Always", "Never", or "Mixed".
    """

    def has_explicit_return(n: ast.AST) -> bool:
        return any(isinstance(child, ast.Return) for child in ast.walk(n))

    returns = [has_explicit_return(stmt) for stmt in node.body]
    if returns and all(returns):
        return "Always"
    elif not any(returns):
        return "Never"
    else:
        return "Mixed"


def calculate_lines_of_code(node: ast.AST, source_code: str, debug: bool = False) -> int:
    """
    Calculate the number of executable lines of code for a node.

    Args:
        node: The AST node (typically a FunctionDef).
        source_code: The entire source code as a string.
        debug: If True, outputs debugging information.

    Returns:
        The number of executable (non-blank, non-comment) lines.
    """
    if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
        lines = source_code.splitlines()[node.lineno - 1:node.end_lineno]
        if debug:
            logging.debug(f"Lines for {getattr(node, 'name', 'node')}: {lines}")
        executable_lines = [line for line in lines if line.strip() and not line.strip().startswith("#")]
        if debug:
            logging.debug(f"Executable lines for {getattr(node, 'name', 'node')}: {len(executable_lines)}")
        return len(executable_lines)
    return 0


def calculate_nesting_depth(node: ast.AST) -> int:
    """
    Recursively calculate the maximum nesting depth of control flow structures in a node.

    Args:
        node: An AST node.

    Returns:
        The maximum nesting depth as an integer.
    """

    def helper(n: ast.AST, current_depth: int) -> int:
        max_depth = current_depth
        for child in ast.iter_child_nodes(n):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
                max_depth = max(max_depth, helper(child, current_depth + 1))
            else:
                max_depth = max(max_depth, helper(child, current_depth))
        return max_depth

    return helper(node, 0)


def find_similar_nodes(tree: ast.AST, source_code: str, threshold: float = 0.8) -> List[Dict]:
    """
    Find similar function nodes in an AST based on a similarity threshold.
    """
    similar_nodes = []
    processed_pairs = set()

    def count_parameters(node: ast.FunctionDef) -> int:
        return len(node.args.args)

    logging.debug(f"Total AST nodes: {len(list(ast.walk(tree)))}")
    candidate_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    logging.debug(f"Candidate Functions: {[node.name for node in candidate_nodes]}")
    candidate_nodes.sort(key=lambda n: (n.lineno, getattr(n, 'end_lineno', n.lineno)))

    for i in range(len(candidate_nodes)):
        for j in range(i + 1, len(candidate_nodes)):
            node1, node2 = candidate_nodes[i], candidate_nodes[j]
            pair_id = tuple(sorted([node1.name, node2.name]))

            if pair_id in processed_pairs:
                continue
            processed_pairs.add(pair_id)

            logging.debug(f"Comparing `{node1.name}` and `{node2.name}`")

            try:
                sim = calculate_similarity(node1, node2)
            except Exception as e:
                logging.error(f"Error calculating similarity for `{node1.name}` and `{node2.name}`: {e}")
                continue

            logging.debug(f"Similarity Score for `{node1.name}` vs `{node2.name}`: {sim:.2f}")

            if sim >= threshold:
                try:
                    node1_metrics = {
                        "name": node1.name,
                        "complexity": calculate_complexity(node1),
                        "return_behavior": analyze_return_behavior(node1),
                        "loc": calculate_lines_of_code(node1, source_code),
                        "parameter_count": count_parameters(node1),
                        "nesting_depth": calculate_nesting_depth(node1),
                        "ast_pattern": normalize_ast_structure(node1)
                    }
                    node2_metrics = {
                        "name": node2.name,
                        "complexity": calculate_complexity(node2),
                        "return_behavior": analyze_return_behavior(node2),
                        "loc": calculate_lines_of_code(node2, source_code),
                        "parameter_count": count_parameters(node2),
                        "nesting_depth": calculate_nesting_depth(node2),
                        "ast_pattern": normalize_ast_structure(node2)
                    }

                    # 🚀 **CRITICAL FIX**: Check if function metrics are missing before adding!
                    required_keys = ["name", "complexity", "return_behavior", "loc", "parameter_count", "nesting_depth", "ast_pattern"]
                    if not all(k in node1_metrics for k in required_keys) or not all(k in node2_metrics for k in required_keys):
                        logging.error(f"Skipping `{node1.name}` and `{node2.name}` due to missing required metrics.")
                        continue  # Skip this pair

                    logging.debug(f"✅ Storing function pair `{node1.name}` and `{node2.name}` with similarity: {sim:.2f}")

                    result = {
                        "function1_metrics": node1_metrics,
                        "function2_metrics": node2_metrics,
                        "similarity": sim
                    }
                    similar_nodes.append(result)

                except Exception as e:
                    logging.error(f"Error generating function metrics for `{node1.name}` or `{node2.name}`: {e}")

    logging.debug(f"Final similar_nodes content: {similar_nodes}")  # 🚀 FINAL DEBUGGING LOG
    return similar_nodes
