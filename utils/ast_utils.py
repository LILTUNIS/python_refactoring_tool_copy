# ast_utils.py
import ast
import json
import logging

def normalize_structure(node: ast.AST, cache=None):
    """
    Recursively converts a Python AST node into a (potentially) minimal
    dictionary representation that preserves most details, including:
      - Actual variable names
      - Actual constant values
      - Original statement ordering
      - Real function/class names

    This means code fragments that differ in variable naming or constants
    will produce significantly different structures, lowering similarity.
    """
    if cache is None:
        cache = {}

    node_id = id(node)
    if node_id in cache:
        return cache[node_id]

    if isinstance(node, ast.Module):
        # Keep the module body in original order
        result = {
            "Module": [normalize_structure(stmt, cache) for stmt in node.body]
        }

    elif isinstance(node, ast.FunctionDef):
        # Preserve the real function name, arg names, and body order
        result = {
            "FunctionDef": {
                "name": node.name,
                "args": [normalize_structure(arg, cache) for arg in node.args.args],
                "body": [normalize_structure(stmt, cache) for stmt in node.body]
            }
        }

    elif isinstance(node, ast.AsyncFunctionDef):
        result = {
            "AsyncFunctionDef": {
                "name": node.name,
                "args": [normalize_structure(arg, cache) for arg in node.args.args],
                "body": [normalize_structure(stmt, cache) for stmt in node.body]
            }
        }

    elif isinstance(node, ast.arg):
        # Keep the real argument name
        result = {"arg": node.arg}

    elif isinstance(node, ast.ClassDef):
        # Keep the real class name and original body order
        result = {
            "ClassDef": {
                "name": node.name,
                "bases": [normalize_structure(base, cache) for base in node.bases],
                "body": [normalize_structure(stmt, cache) for stmt in node.body]
            }
        }

    elif isinstance(node, ast.If):
        # Keep the original order of body and orelse
        result = {
            "If": {
                "test": normalize_structure(node.test, cache),
                "body": [normalize_structure(stmt, cache) for stmt in node.body],
                "orelse": [normalize_structure(stmt, cache) for stmt in node.orelse]
            }
        }

    elif isinstance(node, ast.For):
        result = {
            "For": {
                "target": normalize_structure(node.target, cache),
                "iter": normalize_structure(node.iter, cache),
                "body": [normalize_structure(stmt, cache) for stmt in node.body],
                "orelse": [normalize_structure(stmt, cache) for stmt in node.orelse]
            }
        }

    elif isinstance(node, ast.While):
        result = {
            "While": {
                "test": normalize_structure(node.test, cache),
                "body": [normalize_structure(stmt, cache) for stmt in node.body],
                "orelse": [normalize_structure(stmt, cache) for stmt in node.orelse]
            }
        }

    elif isinstance(node, ast.Try):
        result = {
            "Try": {
                "body": [normalize_structure(stmt, cache) for stmt in node.body],
                "handlers": [normalize_structure(h, cache) for h in node.handlers],
                "orelse": [normalize_structure(stmt, cache) for stmt in node.orelse],
                "finalbody": [normalize_structure(stmt, cache) for stmt in node.finalbody]
            }
        }

    elif isinstance(node, ast.ExceptHandler):
        # Keep the exception name if present
        exctype = normalize_structure(node.type, cache) if node.type else None
        result = {
            "ExceptHandler": {
                "type": exctype,
                "name": node.name,
                "body": [normalize_structure(stmt, cache) for stmt in node.body]
            }
        }

    elif isinstance(node, ast.Return):
        result = {"Return": normalize_structure(node.value, cache) if node.value else None}

    elif isinstance(node, ast.BinOp):
        # Keep left and right in original order, do not reorder for commutative ops
        result = {
            "BinOp": {
                "op": type(node.op).__name__,
                "left": normalize_structure(node.left, cache),
                "right": normalize_structure(node.right, cache)
            }
        }

    elif isinstance(node, ast.UnaryOp):
        result = {
            "UnaryOp": {
                "op": type(node.op).__name__,
                "operand": normalize_structure(node.operand, cache)
            }
        }

    elif isinstance(node, ast.BoolOp):
        # Keep values in original order
        result = {
            "BoolOp": {
                "op": type(node.op).__name__,
                "values": [normalize_structure(val, cache) for val in node.values]
            }
        }

    elif isinstance(node, ast.Compare):
        # Keep comparators in original order
        result = {
            "Compare": {
                "left": normalize_structure(node.left, cache),
                "ops": [type(op).__name__ for op in node.ops],
                "comparators": [normalize_structure(comp, cache) for comp in node.comparators]
            }
        }

    elif isinstance(node, ast.Assign):
        result = {
            "Assign": {
                "targets": [normalize_structure(t, cache) for t in node.targets],
                "value": normalize_structure(node.value, cache)
            }
        }

    elif isinstance(node, ast.AugAssign):
        result = {
            "AugAssign": {
                "target": normalize_structure(node.target, cache),
                "op": type(node.op).__name__,
                "value": normalize_structure(node.value, cache)
            }
        }

    elif isinstance(node, ast.Call):
        result = {
            "Call": {
                "func": normalize_structure(node.func, cache),
                "args": [normalize_structure(arg, cache) for arg in node.args],
                "keywords": {
                    kw.arg: normalize_structure(kw.value, cache) for kw in node.keywords if kw.arg is not None
                }
            }
        }

    elif isinstance(node, ast.Attribute):
        result = {
            "Attribute": {
                "value": normalize_structure(node.value, cache),
                "attr": node.attr
            }
        }

    elif isinstance(node, ast.Name):
        # Preserve the actual variable name
        result = {"Name": node.id}

    elif isinstance(node, ast.Constant):
        # Preserve the actual constant value
        # (Wrap it so JSON can handle unprintable or tricky values)
        result = {"Constant": repr(node.value)}

    elif isinstance(node, ast.List):
        result = {
            "List": [normalize_structure(elt, cache) for elt in node.elts]
        }

    elif isinstance(node, ast.Tuple):
        result = {
            "Tuple": [normalize_structure(elt, cache) for elt in node.elts]
        }

    elif isinstance(node, ast.Dict):
        result = {
            "Dict": {
                "keys": [normalize_structure(k, cache) for k in node.keys],
                "values": [normalize_structure(v, cache) for v in node.values]
            }
        }

    elif isinstance(node, ast.ListComp):
        result = {
            "ListComp": {
                "elt": normalize_structure(node.elt, cache),
                "generators": [normalize_structure(gen, cache) for gen in node.generators]
            }
        }

    elif isinstance(node, ast.comprehension):
        result = {
            "comprehension": {
                "target": normalize_structure(node.target, cache),
                "iter": normalize_structure(node.iter, cache),
                "ifs": [normalize_structure(i, cache) for i in node.ifs],
                "is_async": node.is_async
            }
        }

    # ... add other node types as needed, e.g. DictComp, Set, etc. ...

    else:
        # Fallback for unhandled node types
        result = str(type(node).__name__)

    cache[node_id] = result
    return result


# Since we want minimal normalization, let's disable the DataFlowNormalizer by default
# or remove it entirely. Here, we simply define it but do nothing.

class DataFlowNormalizer(ast.NodeTransformer):
    """
    A no-op normalizer for data flow. If you want to rename variables or unify constants,
    you can reintroduce that logic. Right now, it just returns the node unchanged.
    """
    def visit(self, node):
        return self.generic_visit(node)


def normalize_ast_for_dataflow(tree: ast.AST):
    """
    If you wanted to rename variables or unify constants, you could do it here.
    Currently, we do nothing and just return the original tree + an empty mapping.
    """
    normalizer = DataFlowNormalizer()
    normalized_tree = normalizer.visit(tree)
    ast.fix_missing_locations(normalized_tree)
    reverse_mapping = {}  # Since we aren't renaming anything
    return normalized_tree, reverse_mapping


def normalized_ast_string(node: ast.AST) -> str:
    """
    Generate a string representation of the AST by first applying (empty) dataflow
    normalization and then converting to our dictionary structure.

    This version is very sensitive to changes in variable names, constants,
    statement ordering, etc.
    """
    norm_tree, _ = normalize_ast_for_dataflow(node)
    struct = normalize_structure(norm_tree)
    return json.dumps(struct, sort_keys=False)
