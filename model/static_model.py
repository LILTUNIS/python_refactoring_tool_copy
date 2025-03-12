from dataclasses import dataclass
import ast
from typing import Dict

@dataclass
class static_model:
    """
    Holds data produced by the static analyzer about two code nodes.
    """
    node1: ast.AST
    node2: ast.AST
    similarity: float
    complexity: int
    return_behavior: str  # e.g. "Always", "Never", or "Mixed"
    loc: int
    parameter_count: int
    nesting_depth: int
    ast_pattern: Dict
