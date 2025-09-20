# refactoring_module.py

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from model.refactoring_plan_model import RefactoringPlan

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
#                         RefactoringSug Component
# ---------------------------------------------------------------------------

class RefactoringSug:
    """
    Generate textual refactoring suggestions based on
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
            data_flow_score  = clone.get("dataflow_similarity", 0.0)

            suggestion = {
                "func1": func1,
                "func2": func2,
                "refactoring_type": "None",
                "suggested_change": "No refactoring needed.",
                "refactor_needed": False
            }

            # Generate suggestion based on similarity thresholds.
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

# ---------------------------------------------------------------------------
#                       RefactoringPlanner Component
# ---------------------------------------------------------------------------

class RefactoringPlanner:
    """
    Convert textual suggestions and line-range data
    into actionable RefactoringPlan objects.
    """

    def __init__(self):
        # Configuration or thresholds can be set here if needed.
        pass

    def create_plans_from_suggestions(
            self,
            merged_clones: List[Dict[str, Any]],
            suggestions: List[Dict[str, Any]]
    ) -> List[RefactoringPlan]:
        """
        For each suggestion that requires refactoring, find
        the line-range data in 'merged_clones' for func1/func2
        and build a corresponding RefactoringPlan.

        Returns a list of RefactoringPlan objects.
        """
        plans = []

        # Build a lookup from (func1, func2) to the merged clone entry
        clone_map = {}
        for mc in merged_clones:
            f1 = mc["func1"]
            f2 = mc["func2"]
            key = tuple(sorted([f1, f2]))
            clone_map[key] = mc

        for sugg in suggestions:
            if not sugg.get("refactor_needed"):
                continue

            f1 = sugg["func1"]
            f2 = sugg["func2"]
            pair_key = tuple(sorted([f1, f2]))

            if pair_key not in clone_map:
                logger.warning(f"No line-range data found for {pair_key}")
                continue

            mc_entry = clone_map[pair_key]
            f1_info = mc_entry["function1_metrics"]
            f2_info = mc_entry["function2_metrics"]

            file_path = f1_info.get("file_path", "")
            start_line = f1_info.get("start_line", 0)
            end_line = f1_info.get("end_line", 0)

            # Map the suggestion type into a plan_type recognized by the refactoring engine.
            plan_type = self._map_refactoring_type(sugg["refactoring_type"])

            if plan_type == "none":
                continue

            # Gather extra information including code from func2.
            extra = {
                "suggested_text": sugg["suggested_change"],
                "func2": f2,
                "func2_file_path": f2_info.get("file_path", ""),
                "func2_start_line": str(f2_info.get("start_line", 0)),
                "func2_end_line": str(f2_info.get("end_line", 0)),
                "func2_code": self._read_function_code(
                    f2_info.get("file_path", ""),
                    f2_info.get("start_line", 0),
                    f2_info.get("end_line", 0)
                )
            }

            # For extraction plans, set a new function name.
            if plan_type == "extract_method":
                new_name = f"extracted_{f1}_common"
                extra["new_func_name"] = new_name

            plan = RefactoringPlan(
                plan_type=plan_type,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                extra_info=extra
            )
            plans.append(plan)
            logger.debug(f"Created plan: {plan}")

        return plans

    def _read_function_code(self, file_path: str, start_line: int, end_line: int) -> str:
        """
        Extract code from a file between start_line and end_line.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            # Adjust for 1-based line numbering.
            return "".join(lines[start_line - 1:end_line])
        except Exception as e:
            logger.error(f"Failed to read function code from {file_path}: {e}")
            return ""

    def _map_refactoring_type(self, sug_type: str) -> str:
        """
        Map a suggestion's type to a refactoring plan type.
        """
        lower = sug_type.lower()
        if "extraction" in lower:
            return "extract_method"
        elif "merge" in lower:
            return "ai_merge_functions"
        elif "parameter" in lower:
            return "parameterize"
        else:
            return "none"
