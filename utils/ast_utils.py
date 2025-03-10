import ast
import logging
import json

def normalize_structure(node: ast.AST, cache=None):
    """
    Recursively normalize an AST node into a dictionary representation.
    This version produces a canonical representation by replacing identifiers
    (such as function names, variable names, and attribute names) with generic placeholders.

    Improvements:
    - Differentiate constant types (string, number, boolean, None) instead of just "CONST".
    - Consider making the sorting of bodies and orelse blocks optional, or using a more AST-aware sorting if needed.
    - **[Improvement] Use a more robust way to sort AST node lists, potentially based on node types and attributes instead of just json.dumps.**
    """
    if cache is None:
        cache = {}
    node_id = id(node)
    if node_id in cache:
        return cache[node_id]

    if isinstance(node, ast.Module):
        result = {"Module": [normalize_structure(stmt, cache) for stmt in node.body]}
    elif isinstance(node, ast.FunctionDef):
        result = {
            "FunctionDef": {
                "name": "FUNC",  # Replace function name with generic placeholder
                "args": [normalize_structure(arg, cache) for arg in node.args.args],
                "body": [normalize_structure(stmt, cache) for stmt in node.body]
            }
        }
    elif isinstance(node, ast.AsyncFunctionDef):
        result = {
            "AsyncFunctionDef": {
                "name": "FUNC",
                "args": [normalize_structure(arg, cache) for arg in node.args.args],
                "body": [normalize_structure(stmt, cache) for stmt in node.body]
            }
        }
    elif isinstance(node, ast.arg):
        result = {"arg": "ARG"}  # Replace argument names with generic placeholder
    elif isinstance(node, ast.If):
        result = {
            "If": {
                "test.py": normalize_structure(node.test, cache),
                "body": sorted([normalize_structure(stmt, cache) for stmt in node.body], key=ast_node_key), # Use ast_node_key for sorting
                "orelse": sorted([normalize_structure(stmt, cache) for stmt in node.orelse], key=ast_node_key) # Use ast_node_key for sorting
            }
        }
    elif isinstance(node, ast.For):
        result = {
            "For": {
                "target": normalize_structure(node.target, cache),
                "iter": normalize_structure(node.iter, cache),
                "body": sorted([normalize_structure(stmt, cache) for stmt in node.body], key=ast_node_key), # Use ast_node_key for sorting
                "orelse": sorted([normalize_structure(stmt, cache) for stmt in node.orelse], key=ast_node_key) # Use ast_node_key for sorting
            }
        }
    elif isinstance(node, ast.While):
        result = {
            "While": {
                "test.py": normalize_structure(node.test, cache),
                "body": sorted([normalize_structure(stmt, cache) for stmt in node.body], key=ast_node_key), # Use ast_node_key for sorting
                "orelse": sorted([normalize_structure(stmt, cache) for stmt in node.orelse], key=ast_node_key) # Use ast_node_key for sorting
            }
        }
    elif isinstance(node, ast.Return):
        result = {"Return": normalize_structure(node.value, cache) if node.value else None}
    elif isinstance(node, ast.BinOp):
        # Canonicalize commutative operations by reordering operands
        left = normalize_structure(node.left, cache)
        right = normalize_structure(node.right, cache)
        if isinstance(node.op, (ast.Add, ast.Mult)):
            # Use ast_node_key for comparison instead of json.dumps
            if ast_node_key(left) > ast_node_key(right):
                left, right = right, left
        result = {
            "BinOp": {
                "op": type(node.op).__name__,
                "left": left,
                "right": right
            }
        }
    elif isinstance(node, ast.Assign):
        result = {
            "Assign": {
                "targets": sorted([normalize_structure(t, cache) for t in node.targets], key=ast_node_key), # Use ast_node_key for sorting
                "value": normalize_structure(node.value, cache)
            }
        }
    elif isinstance(node, ast.Call):
        result = {
            "Call": {
                "func": normalize_structure(node.func, cache),
                "args": [normalize_structure(arg, cache) for arg in node.args],
                "keywords": {kw.arg: normalize_structure(kw.value, cache) for kw in node.keywords if kw.arg is not None}
            }
        }
    elif isinstance(node, ast.ListComp):
        result = {
            "ListComp": {
                "elt": normalize_structure(node.elt, cache),
                "generators": [normalize_structure(gen, cache) for gen in node.generators]
            }
        }
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            result = {"Constant": "STR_CONST"}  # Differentiate string constants
        elif isinstance(node.value, (int, float)):
            result = {"Constant": "NUM_CONST"}  # Differentiate number constants
        elif isinstance(node.value, bool):
            result = {"Constant": "BOOL_CONST"} # Differentiate boolean constants
        elif node.value is None:
            result = {"Constant": "NONE_CONST"} # Differentiate None constant
        else:
            result = {"Constant": "CONST"}  # Generic constant if other type
    elif isinstance(node, ast.Name):
        result = {"Name": "VAR"}  # Replace all variable names with a generic placeholder
    elif isinstance(node, ast.Attribute):
        result = {
            "Attribute": {
                "value": normalize_structure(node.value, cache),
                "attr": "ATTR"  # Replace attribute names with a generic placeholder
            }
        }
    elif isinstance(node, ast.Compare):
        result = {
            "Compare": {
                "left": normalize_structure(node.left, cache),
                "ops": [type(op).__name__ for op in node.ops],
                "comparators": sorted([normalize_structure(comp, cache) for comp in node.comparators], key=ast_node_key) # Use ast_node_key for sorting
            }
        }
    elif isinstance(node, ast.BoolOp):
        result = {
            "BoolOp": {
                "op": type(node.op).__name__,
                "values": sorted([normalize_structure(val, cache) for val in node.values], key=ast_node_key) # Use ast_node_key for sorting
            }
        }
    elif isinstance(node, ast.ClassDef):
        result = {
            "ClassDef": {
                "name": "CLASS",
                "bases": sorted([normalize_structure(base, cache) for base in node.bases], key=ast_node_key), # Use ast_node_key for sorting
                "body": sorted([normalize_structure(stmt, cache) for stmt in node.body], key=ast_node_key) # Use ast_node_key for sorting
            }
        }
    elif isinstance(node, ast.Try):
        result = {
            "Try": {
                "body": sorted([normalize_structure(stmt, cache) for stmt in node.body], key=ast_node_key), # Use ast_node_key for sorting
                "handlers": sorted([normalize_structure(h, cache) for h in node.handlers], key=ast_node_key), # Use ast_node_key for sorting
                "orelse": sorted([normalize_structure(stmt, cache) for stmt in node.orelse], key=ast_node_key), # Use ast_node_key for sorting
                "finalbody": sorted([normalize_structure(stmt, cache) for stmt in node.finalbody], key=ast_node_key) # Use ast_node_key for sorting
            }
        }
    else:
        result = str(type(node).__name__)

    cache[node_id] = result
    return result

def ast_node_key(node_dict):
    """
    Key function for sorting AST node dictionaries.
    This provides a more AST-aware sorting than just json.dumps.
    It prioritizes node type and then recursively sorts based on the content.
    """
    if isinstance(node_dict, dict):
        if len(node_dict) == 1:
            node_type = list(node_dict.keys())[0]
            node_content = node_dict[node_type]
            if isinstance(node_content, list):
                return (node_type, tuple(ast_node_key(item) for item in node_content)) # Sort lists of nodes recursively
            elif isinstance(node_content, dict):
                return (node_type, tuple(sorted([(k, ast_node_key(v)) for k, v in node_content.items()]))) # Sort dict content recursively
            else:
                return (node_type, node_content) # Sort by node type and content
        else:
            # Handle cases where the dict is not a standard normalized node (if any) - rare case
            return tuple(sorted(node_dict.items()))
    elif isinstance(node_dict, list):
        return tuple(ast_node_key(item) for item in node_dict) # Sort lists recursively
    else:
        return node_dict # For basic types


class DataFlowNormalizer(ast.NodeTransformer):
    """
    AST transformer that renames variables and normalizes constants.
    This ensures that semantically equivalent code with different naming produces a similar AST.

    Improvements:
    - Consider normalizing function/class names as well for data flow if needed.
    """
    def __init__(self):
        super().__init__()
        self.name_mapping = {}      # Maps original names -> normalized names
        self.reverse_mapping = {}   # For UI or later reference
        self.counter = 0

    def visit_Name(self, node: ast.Name):
        if node.id not in self.name_mapping:
            self.counter += 1
            normalized_name = f"var_{self.counter}"
            self.name_mapping[node.id] = normalized_name
            self.reverse_mapping[normalized_name] = node.id
        node.id = self.name_mapping[node.id]
        return node

    def visit_Constant(self, node: ast.Constant):
        return ast.copy_location(ast.Constant(value="CONST"), node) # Keep as generic "CONST" for dataflow

    def visit_BinOp(self, node: ast.BinOp):
        self.generic_visit(node)
        if isinstance(node.op, (ast.Add, ast.Mult)):
            left_dump = normalize_structure(node.left) # Use normalized structure for comparison
            right_dump = normalize_structure(node.right) # Use normalized structure for comparison
            if ast_node_key(right_dump) < ast_node_key(left_dump): # Use ast_node_key for comparison
                node.left, node.right = node.right, node.left
        return node

def normalize_ast_for_dataflow(tree: ast.AST) -> (ast.AST, dict):
    """
    Normalize the AST for data flow analysis.
    Returns the normalized AST and a reverse mapping of renamed identifiers.
    """
    normalizer = DataFlowNormalizer()
    normalized_tree = normalizer.visit(tree)
    ast.fix_missing_locations(normalized_tree)
    return normalized_tree, normalizer.reverse_mapping

def normalized_ast_string(node: ast.AST) -> str:
    """
    Generate a normalized string representation of an AST node by
    first applying the DataFlowNormalizer and then using JSON canonicalization
    on the normalized structure.
    This representation is used across static analysis, data flow, and token-based clone detection.
    """
    norm_tree, _ = normalize_ast_for_dataflow(node)
    normalized_structure = normalize_structure(norm_tree)
    return json.dumps(normalized_structure, sort_keys=True)