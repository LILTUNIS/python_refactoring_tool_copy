from typing import List, Dict, Any

from model.data_flow_analyzer import compare_function_similarity


class RefactoringSug:  #
    """
    Generates automated refactoring suggestions based on detected token & AST clones.
    """

    def generate_suggestions(clones: List[Dict[str, Any]], data_flow_results: Dict[str, Dict[str, Any]]) -> List[
        Dict[str, str]]:
        """
        Process clone detection results and suggest refactoring, integrating data flow analysis.
        """
        suggestions = []

        for clone in clones:
            func1, func2 = clone["func1"], clone["func2"]
            token_similarity, ast_similarity = clone["token_similarity"], clone["ast_similarity"]
            func1_code, func2_code = clone["func1_code"], clone["func2_code"]

            # Default suggestion
            suggestion = {
                "func1": func1,
                "func2": func2,
                "refactoring_type": "None",
                "suggested_change": "No refactoring needed.",
            }

            # Extract data flow information
            func1_flow = data_flow_results.get(func1, {})
            func2_flow = data_flow_results.get(func2, {})

            # Compute data flow similarity
            data_flow_score = compare_function_similarity(func1_flow, func2_flow)

            # Print debug info
            print(f"[DEBUG] {func1} vs {func2} | Data Flow Score: {data_flow_score:.2f}")

            # 1️⃣ **Exact or Near-Exact Duplicate Code** (Token Similarity > 85%)
            if token_similarity >= 0.85 and data_flow_score > 0.85:
                suggestion["refactoring_type"] = "Function Extraction"
                suggestion["suggested_change"] = (
                    f"Extract a new function for shared logic in `{func1}` and `{func2}`, as they have high code and data flow similarity."
                )

            # 2️⃣ **Structurally Similar Functions** (AST Similarity > 80% and Data Flow > 80%)
            elif ast_similarity >= 0.80 and data_flow_score > 0.80:
                suggestion["refactoring_type"] = "Merge Similar Functions"
                suggestion["suggested_change"] = (
                    f"Consider merging `{func1}` and `{func2}` into a single function, as their logic and data flow are highly aligned."
                )

            # 3️⃣ **Parameterizable Functions** (Token Similarity ≥ 70%, AST Similarity ≥ 70%, and Data Flow > 70%)
            elif token_similarity >= 0.70 and ast_similarity >= 0.70 and data_flow_score > 0.70:
                suggestion["refactoring_type"] = "Function Parameterization"
                suggestion["suggested_change"] = (
                    f"Convert `{func1}` and `{func2}` into a single function with parameters, as they differ only in constants or minor logic, and share similar data flow."
                )

            # 4️⃣ **Different Data Flow but Similar Code** (AST Similarity > 80% but Data Flow < 50%)
            elif ast_similarity >= 0.80 and data_flow_score < 0.50:
                suggestion["refactoring_type"] = "Review Before Merging"
                suggestion["suggested_change"] = (
                    f"Although `{func1}` and `{func2}` look structurally similar, their data flow is different. Review dependencies before merging."
                )

            # Append the suggestion
            suggestions.append(suggestion)

        return suggestions