import ast
import logging
import json
import math

# Set up logging to display debug-level messages.
logging.basicConfig(level=logging.DEBUG)


def normalize_structure(node: ast.AST, cache=None):
    """
    Recursively normalizes an AST node into a dictionary representation
    that is suitable for TSED (Tree-based Structural Edit Distance) analysis.

    This function abstracts away specifics by replacing identifiers (e.g., variable,
    function, and attribute names) with generic placeholders and classifies constants
    uniformly. For binary operations that are commutative (e.g., addition, multiplication),
    the operands are reordered to ensure a canonical form.

    Parameters:
        node (ast.AST): The AST node to be normalized.
        cache (dict, optional): Cache to store and reuse previously normalized nodes.

    Returns:
        dict: A dictionary representing the normalized structure of the AST node.
    """
    if cache is None:
        cache = {}
    node_id = id(node)
    # Return cached result if the node has been processed before.
    if node_id in cache:
        return cache[node_id]

    if isinstance(node, ast.Module):
        # Normalize the top-level module by processing each statement in its body.
        result = {"Module": [normalize_structure(stmt, cache) for stmt in node.body]}

    elif isinstance(node, ast.FunctionDef):
        # Normalize a function definition: replace its name with "FUNC" and normalize arguments and body.
        result = {
            "FunctionDef": {
                "name": "FUNC",
                "args": [normalize_structure(arg, cache) for arg in node.args.args],
                "body": [normalize_structure(stmt, cache) for stmt in node.body]
            }
        }

    elif isinstance(node, ast.AsyncFunctionDef):
        # Similar normalization strategy for async function definitions.
        result = {
            "AsyncFunctionDef": {
                "name": "FUNC",
                "args": [normalize_structure(arg, cache) for arg in node.args.args],
                "body": [normalize_structure(stmt, cache) for stmt in node.body]
            }
        }

    elif isinstance(node, ast.arg):
        # Replace argument names with a generic placeholder.
        result = {"arg": "ARG"}

    elif isinstance(node, ast.ClassDef):
        # Normalize class definitions: replace class names with "CLASS", process base classes, decorators, and body.
        base_names = [base.id if isinstance(base, ast.Name) else normalize_structure(base, cache) for base in
                      node.bases]
        deco_labels = [normalize_structure(d, cache) for d in node.decorator_list]
        result = {
            "ClassDef": {
                "name": "CLASS",
                "bases": base_names,
                "decorators": deco_labels,
                "body": [normalize_structure(stmt, cache) for stmt in node.body]
            }
        }

    elif isinstance(node, ast.Name):
        # Replace all variable names with a generic placeholder.
        result = {"Name": "VAR"}

    elif isinstance(node, ast.Constant):
        # Normalize constants uniformly, classifying them by type.
        if isinstance(node.value, str):
            result = {"Constant": "STR_CONST"}
        elif isinstance(node.value, (int, float)):
            result = {"Constant": "NUM_CONST"}
        elif isinstance(node.value, bool):
            result = {"Constant": "BOOL_CONST"}
        elif node.value is None:
            result = {"Constant": "NONE_CONST"}
        else:
            result = {"Constant": "CONST"}

    elif isinstance(node, ast.Attribute):
        # Normalize attribute access: e.g. x.foo becomes a generic representation.
        if isinstance(node.value, ast.Name):
            result = {
                "Attribute": {
                    "value": "VAR",
                    "attr": "ATTR"
                }
            }
        else:
            result = {
                "Attribute": {
                    "value": normalize_structure(node.value, cache),
                    "attr": "ATTR"
                }
            }

    elif isinstance(node, ast.Call):
        # Normalize function calls by processing the function, its positional arguments, and keywords.
        func_label = normalize_structure(node.func, cache)
        arg_list = [normalize_structure(a, cache) for a in node.args]
        kw_map = {kw.arg: normalize_structure(kw.value, cache) for kw in node.keywords if kw.arg}
        result = {
            "Call": {
                "func": func_label,
                "args": arg_list,
                "keywords": kw_map
            }
        }

    elif isinstance(node, ast.Assign):
        # Normalize assignment statements by processing targets and value.
        targets_norm = [normalize_structure(t, cache) for t in node.targets]
        val_norm = normalize_structure(node.value, cache)
        result = {
            "Assign": {
                "targets": targets_norm,
                "value": val_norm
            }
        }

    elif isinstance(node, ast.Return):
        # Normalize return statements; if there is no value, return None.
        result = {
            "Return": normalize_structure(node.value, cache) if node.value else None
        }

    elif isinstance(node, ast.BinOp):
        # Normalize binary operations. For commutative operations, sort operands to a canonical order.
        left = normalize_structure(node.left, cache)
        right = normalize_structure(node.right, cache)
        op_name = type(node.op).__name__

        if op_name in ("Add", "Mult"):
            # Compare keys to decide whether to swap operands.
            if ast_node_key(left) > ast_node_key(right):
                left, right = right, left

        result = {
            "BinOp": {
                "op": op_name,
                "left": left,
                "right": right
            }
        }

    elif isinstance(node, ast.BoolOp):
        # Normalize boolean operations and sort values for consistency.
        op_name = type(node.op).__name__
        values = [normalize_structure(v, cache) for v in node.values]
        values.sort(key=ast_node_key)
        result = {
            "BoolOp": {
                "op": op_name,
                "values": values
            }
        }

    elif isinstance(node, ast.Compare):
        # Normalize comparison operations by processing the left-hand side and comparators.
        left = normalize_structure(node.left, cache)
        comps = [normalize_structure(c, cache) for c in node.comparators]
        ops = [type(o).__name__ for o in node.ops]
        comps.sort(key=ast_node_key)
        result = {
            "Compare": {
                "left": left,
                "ops": ops,
                "comparators": comps
            }
        }

    elif isinstance(node, ast.If):
        # Normalize if statements by processing test condition, body, and else clause.
        test_norm = normalize_structure(node.test, cache)
        body_norm = [normalize_structure(st, cache) for st in node.body]
        orelse_norm = [normalize_structure(st, cache) for st in node.orelse]
        # Sort body and orelse for canonical ordering, if desired.
        result = {
            "If": {
                "test": test_norm,
                "body": sorted(body_norm, key=ast_node_key),
                "orelse": sorted(orelse_norm, key=ast_node_key)
            }
        }

    elif isinstance(node, ast.For):
        # Normalize for-loops by processing the target, iterable, body, and else clause.
        target_norm = normalize_structure(node.target, cache)
        iter_norm = normalize_structure(node.iter, cache)
        body_norm = [normalize_structure(st, cache) for st in node.body]
        orelse_norm = [normalize_structure(st, cache) for st in node.orelse]
        result = {
            "For": {
                "target": target_norm,
                "iter": iter_norm,
                "body": sorted(body_norm, key=ast_node_key),
                "orelse": sorted(orelse_norm, key=ast_node_key)
            }
        }

    elif isinstance(node, ast.While):
        # Normalize while-loops by processing the test, body, and else clause.
        test_norm = normalize_structure(node.test, cache)
        body_norm = [normalize_structure(st, cache) for st in node.body]
        orelse_norm = [normalize_structure(st, cache) for st in node.orelse]
        result = {
            "While": {
                "test": test_norm,
                "body": sorted(body_norm, key=ast_node_key),
                "orelse": sorted(orelse_norm, key=ast_node_key)
            }
        }

    elif isinstance(node, ast.Try):
        # Normalize try-except-finally constructs by processing the body, exception handlers, else, and finally clauses.
        body_norm = [normalize_structure(st, cache) for st in node.body]
        handlers_norm = [normalize_structure(h, cache) for h in node.handlers]
        orelse_norm = [normalize_structure(st, cache) for st in node.orelse]
        final_norm = [normalize_structure(st, cache) for st in node.finalbody]
        result = {
            "Try": {
                "body": sorted(body_norm, key=ast_node_key),
                "handlers": sorted(handlers_norm, key=ast_node_key),
                "orelse": sorted(orelse_norm, key=ast_node_key),
                "finalbody": sorted(final_norm, key=ast_node_key)
            }
        }

    elif isinstance(node, ast.ExceptHandler):
        # Normalize exception handlers by processing the exception type and body.
        handler_type = normalize_structure(node.type, cache) if node.type else "None"
        body_norm = [normalize_structure(st, cache) for st in node.body]
        result = {
            "ExceptHandler": {
                "type": handler_type,
                "body": sorted(body_norm, key=ast_node_key)
            }
        }

    elif isinstance(node, ast.With):
        # Normalize with statements by processing context managers and the body.
        items_norm = [normalize_structure(i, cache) for i in node.items]
        body_norm = [normalize_structure(st, cache) for st in node.body]
        result = {
            "With": {
                "items": items_norm,
                "body": sorted(body_norm, key=ast_node_key)
            }
        }

    elif isinstance(node, ast.AsyncWith):
        # Normalize async with statements similarly to synchronous ones.
        items_norm = [normalize_structure(i, cache) for i in node.items]
        body_norm = [normalize_structure(st, cache) for st in node.body]
        result = {
            "AsyncWith": {
                "items": items_norm,
                "body": sorted(body_norm, key=ast_node_key)
            }
        }

    elif isinstance(node, ast.withitem):
        # Normalize individual context items used in with statements.
        context_expr = normalize_structure(node.context_expr, cache)
        optional_vars = normalize_structure(node.optional_vars, cache) if node.optional_vars else None
        result = {
            "withitem": {
                "context_expr": context_expr,
                "optional_vars": optional_vars
            }
        }

    elif isinstance(node, ast.Raise):
        # Normalize raise statements by processing the exception and its cause.
        exc_norm = normalize_structure(node.exc, cache) if node.exc else None
        cause_norm = normalize_structure(node.cause, cache) if node.cause else None
        result = {
            "Raise": {
                "exc": exc_norm,
                "cause": cause_norm
            }
        }

    elif isinstance(node, ast.Import):
        # Normalize import statements by listing the imported module names.
        names = [alias.name for alias in node.names]
        result = {"Import": names}

    elif isinstance(node, ast.ImportFrom):
        # Normalize "from ... import ..." statements.
        names = [alias.name for alias in node.names]
        module_name = node.module if node.module else ""
        result = {
            "ImportFrom": {
                "module": module_name,
                "names": names,
                "level": node.level
            }
        }

    elif isinstance(node, ast.List):
        # Normalize list literals by processing each element.
        elts_norm = [normalize_structure(e, cache) for e in node.elts]
        result = {"List": elts_norm}

    elif isinstance(node, ast.Tuple):
        # Normalize tuple literals similarly.
        elts_norm = [normalize_structure(e, cache) for e in node.elts]
        result = {"Tuple": elts_norm}

    elif isinstance(node, ast.Set):
        # Normalize set literals and sort elements for a canonical order.
        elts_norm = [normalize_structure(e, cache) for e in node.elts]
        elts_norm.sort(key=ast_node_key)
        result = {"Set": elts_norm}

    elif isinstance(node, ast.Dict):
        # Normalize dictionaries by processing key-value pairs.
        pairs = []
        for k, v in zip(node.keys, node.values):
            key_norm = normalize_structure(k, cache) if k else "None"
            val_norm = normalize_structure(v, cache)
            pairs.append({"key": key_norm, "value": val_norm})
        # Sort dictionary pairs for consistent ordering.
        result = {"Dict": sorted(pairs, key=lambda x: json.dumps(x, sort_keys=True))}

    # ... [remaining node types, comprehensions, etc. can be handled similarly] ...
    else:
        # Fallback for any unhandled node type: iterate over its children.
        fallback_label = type(node).__name__
        children = list(ast.iter_child_nodes(node))
        result = {
            fallback_label: [normalize_structure(ch, cache) for ch in children]
        }

    # Cache the result to avoid reprocessing the same node.
    cache[node_id] = result
    return result


def ast_node_key(node_dict):
    """
    Provides a key function for sorting normalized AST node dictionaries.

    This function ensures a consistent ordering of children in commutative operations,
    which is essential for TSED-based comparisons.

    Parameters:
        node_dict: The normalized AST node (as a dict or list).

    Returns:
        A tuple or value that can be used to sort AST nodes in a canonical order.
    """
    if isinstance(node_dict, dict):
        if len(node_dict) == 1:
            node_type = list(node_dict.keys())[0]
            content = node_dict[node_type]
            if isinstance(content, list):
                return (node_type, tuple(sorted((ast_node_key(c) for c in content))))
            elif isinstance(content, dict):
                return (node_type, tuple(sorted([(k, ast_node_key(v)) for k, v in content.items()])))
            else:
                return (node_type, content)
        else:
            return tuple(sorted(node_dict.items()))
    elif isinstance(node_dict, list):
        return tuple(sorted((ast_node_key(c) for c in node_dict)))
    else:
        return node_dict


class DataFlowNormalizer(ast.NodeTransformer):
    """
    AST transformer to normalize variables and constants for data flow analysis.

    This transformer renames variables consistently, replacing them with
    placeholders (e.g., "var_1", "var_2", ...), and normalizes constants to a
    single generic value ("CONST"). This approach is used to reduce irrelevant
    differences when comparing ASTs.
    """

    def __init__(self):
        super().__init__()
        self.name_mapping = {}  # Map original names to normalized names.
        self.reverse_mapping = {}  # Reverse mapping for potential debugging or reconstruction.
        self.counter = 0  # Counter to generate unique placeholder names.

    def visit_Name(self, node: ast.Name):
        # For each variable, assign a unique normalized name.
        if node.id not in self.name_mapping:
            self.counter += 1
            normalized_name = f"var_{self.counter}"
            self.name_mapping[node.id] = normalized_name
            self.reverse_mapping[normalized_name] = node.id
        node.id = self.name_mapping[node.id]
        return node

    def visit_Constant(self, node: ast.Constant):
        # Uniformly transform all constants to "CONST" to abstract away literal values.
        return ast.copy_location(ast.Constant(value="CONST"), node)

    def visit_BinOp(self, node: ast.BinOp):
        # Process binary operations: visit children and reorder commutative operations.
        self.generic_visit(node)
        if isinstance(node.op, (ast.Add, ast.Mult)):
            left_dump = normalize_structure(node.left)
            right_dump = normalize_structure(node.right)
            if ast_node_key(right_dump) < ast_node_key(left_dump):
                node.left, node.right = node.right, node.left
        return node


def normalize_ast_for_dataflow(tree: ast.AST) -> (ast.AST, dict):
    """
    Normalizes an entire AST for data-flow analysis or TSED-based comparisons.

    This includes renaming variables, generalizing constants, and reordering
    commutative binary operations. Returns the transformed AST along with a
    reverse mapping from normalized identifiers to their original names.

    Parameters:
        tree (ast.AST): The input abstract syntax tree.

    Returns:
        tuple: A tuple containing the transformed AST and the reverse mapping.
    """
    normalizer = DataFlowNormalizer()
    transformed_tree = normalizer.visit(tree)
    ast.fix_missing_locations(transformed_tree)
    return transformed_tree, normalizer.reverse_mapping


def normalized_ast_string(node: ast.AST) -> str:
    """
    Converts an AST node to a TSED-friendly canonical JSON string.

    The function first normalizes the AST structure and then dumps it as a JSON
    string with sorted keys, ensuring consistent representation for comparison,
    hashing, or storage.

    Parameters:
        node (ast.AST): The AST node to convert.

    Returns:
        str: A canonical JSON string representation of the normalized AST.
    """
    norm = normalize_structure(node)
    return json.dumps(norm, sort_keys=True)


def ramp_tsed_distance(tree_edit_dist: float, max_nodes: int) -> float:
    """
    Applies TSED's ramp normalization to compute a similarity score.

    The ramp function computes a value based on the ratio of the tree edit distance
    to the maximum number of nodes. The score is clamped between 0 and 1.

    Parameters:
        tree_edit_dist (float): The computed tree edit distance.
        max_nodes (int): The maximum number of nodes in the AST.

    Returns:
        float: The normalized TSED similarity score.
    """
    if max_nodes <= 0:
        return 0.0
    return max(1.0 - (tree_edit_dist / max_nodes), 0.0)
