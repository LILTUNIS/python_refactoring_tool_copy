import ast
from typing import Dict, List

def analyze_data_flow(tree: ast.AST) -> Dict[str, Dict]:
    """
    Analyze data flow within the AST.
    Returns a dictionary mapping function names to their data flow details.
    """
    data_flow = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            print(f"DEBUG: Processing function `{node.name}` in analyze_data_flow")

            # 🚀 Fix: Use function name instead of filename
            if node.name not in data_flow:
                data_flow[node.name] = {"variables": {}, "dependencies": {"reads": [], "writes": []}}

            variables = {}
            for child in ast.walk(node):
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name):
                            variables[target.id] = {"defined": child.lineno, "used": []}
                elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                    if child.id in variables:
                        variables[child.id]["used"].append(child.lineno)

            dependencies = extract_dependencies(node)
            data_flow[node.name]["variables"] = variables
            data_flow[node.name]["dependencies"] = dependencies

            print(f"DEBUG: Function `{node.name}` - Variables: {variables}")
            print(f"DEBUG: Function `{node.name}` - Dependencies: {dependencies}")

    print(f"DEBUG: Final Data Flow Analysis Results: {data_flow}")
    return data_flow



def extract_dependencies(node: ast.FunctionDef) -> Dict[str, List[str]]:
    """
    Extract a list of dependencies for a function categorized as 'reads' and 'writes'.
    """
    reads = set()
    writes = set()

    for child in ast.walk(node):
        if isinstance(child, ast.Assign):
            for t in child.targets:
                if isinstance(t, ast.Name):
                    writes.add(t.id)
                elif isinstance(t, ast.Constant) and isinstance(t.value, (int, float)):  # 🚨 Potential Issue!
                    print(f"WARNING: Detected unexpected numeric value in writes: {t.value}")
        elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            reads.add(child.id)

    # 🚨 Ensure only valid names are stored
    reads = {r for r in reads if isinstance(r, str)}
    writes = {w for w in writes if isinstance(w, str)}

    return {"reads": list(reads), "writes": list(writes)}
