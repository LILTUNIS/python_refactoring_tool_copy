# refactoring_sug.py

from typing import List, Dict, Any
from model.data_flow_analyzer import compare_function_similarity

class RefactoringSug:
    """
    Step 1: Generate textual refactoring suggestions based on
    multi-metric analysis (token, AST, data flow, etc.).
    """

    @staticmethod
    def generate_suggestions(
        clones: List[Dict[str, Any]],
        data_flow_results: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process 'merged_clones' to produce textual suggestions.

        Returns a list of dictionaries, each with:
          - "func1": str
          - "func2": str
          - "refactoring_type": e.g. "Function Extraction"
          - "suggested_change": plain text
          - "refactor_needed": bool (True if we actually propose a refactoring)
        """
        suggestions = []

        for clone in clones:
            func1 = clone.get("func1", "").strip()
            func2 = clone.get("func2", "").strip()
            if not func1 or not func2:
                continue

            token_similarity = clone.get("token_similarity", 0.0)
            ast_similarity   = clone.get("ast_similarity", 0.0)
            # We rely on your existing method to get data flow similarity from the data_flow dict
            # but in 'merged_clones' we might already have "dataflow_similarity":
            data_flow_score  = clone.get("dataflow_similarity", 0.0)

            suggestion = {
                "func1": func1,
                "func2": func2,
                "refactoring_type": "None",
                "suggested_change": "No refactoring needed.",
                "refactor_needed": False
            }

            # Simple logic (similar to your existing approach):
            if token_similarity >= 0.85 and data_flow_score >= 0.85:
                suggestion["refactoring_type"] = "Function Extraction"
                suggestion["suggested_change"] = (
                    f"Extract a new function for shared logic in `{func1}` and `{func2}` "
                    f"(token={token_similarity:.2f}, data_flow={data_flow_score:.2f})."
                )
                suggestion["refactor_needed"] = True

            elif ast_similarity >= 0.80 and data_flow_score >= 0.80:
                suggestion["refactoring_type"] = "Merge Similar Functions"
                suggestion["suggested_change"] = (
                    f"Consider merging `{func1}` and `{func2}` (AST={ast_similarity:.2f}, DF={data_flow_score:.2f})."
                )
                suggestion["refactor_needed"] = True

            elif (token_similarity >= 0.70 and ast_similarity >= 0.70 and data_flow_score >= 0.70):
                suggestion["refactoring_type"] = "Parameterize Functions"
                suggestion["suggested_change"] = (
                    f"Convert `{func1}` and `{func2}` into a single parameterized function "
                    f"(token={token_similarity:.2f}, AST={ast_similarity:.2f}, DF={data_flow_score:.2f})."
                )
                suggestion["refactor_needed"] = True

            suggestions.append(suggestion)

        return suggestions
