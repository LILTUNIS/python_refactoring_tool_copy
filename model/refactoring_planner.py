# refactoring_planner.py

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@dataclass
class RefactoringPlan:
    """
    A single refactoring action, ready for Rope to execute.
    """
    plan_type: str         # e.g. "extract_method", "merge_functions", "parameterize"
    file_path: str
    start_line: int
    end_line: int
    extra_info: Dict[str, Optional[str]] = None


class RefactoringPlanner:
    """
    Step 2: Convert textual suggestions + line-range data
    into actionable RefactoringPlan objects.
    """

    def __init__(self):
        # We can define thresholds or config here if needed
        pass

    def create_plans_from_suggestions(
        self,
        merged_clones: List[Dict[str, any]],
        suggestions: List[Dict[str, any]]
    ) -> List[RefactoringPlan]:
        """
        For each suggestion that says "refactor_needed=True", find
        the line-range data in 'merged_clones' for func1/func2
        and build a corresponding RefactoringPlan.

        Returns a list of RefactoringPlan objects.
        """
        plans = []

        # Build a quick lookup from (func1, func2) -> the merged_clones entry
        # so we can retrieve file_path, start_line, end_line, etc.
        clone_map = {}
        for mc in merged_clones:
            f1 = mc["func1"]
            f2 = mc["func2"]
            key = tuple(sorted([f1, f2]))
            clone_map[key] = mc

        for sugg in suggestions:
            if not sugg.get("refactor_needed"):
                # Skip "No refactoring needed."
                continue

            f1 = sugg["func1"]
            f2 = sugg["func2"]
            pair_key = tuple(sorted([f1, f2]))

            # Retrieve the merged clone data for line info
            if pair_key not in clone_map:
                logger.warning(f"No line-range data found for {pair_key}")
                continue

            mc_entry = clone_map[pair_key]
            # We'll pick function1's lines as the region to transform
            # (In practice, you might unify lines from both)
            f1_info = mc_entry["function1_metrics"]
            file_path = f1_info.get("file_path", "")
            start_line = f1_info.get("start_line", 0)
            end_line   = f1_info.get("end_line", 0)

            # Convert the textual type from suggestions to a plan_type
            # e.g. "Function Extraction" -> "extract_method"
            plan_type = self._map_refactoring_type(sugg["refactoring_type"])

            if plan_type == "none":
                # skip
                continue

            # Build the plan
            extra = {
                "suggested_text": sugg["suggested_change"],  # Keep the plain text around
                "func2": f2,
            }
            # If they want a new function name, we can guess:
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

    def _map_refactoring_type(self, sug_type: str) -> str:
        """
        Map a plain-English type from suggestions into
        the Rope-friendly plan_type name.
        """
        lower = sug_type.lower()
        if "extraction" in lower:
            return "extract_method"
        elif "merge" in lower:
            return "merge_functions"
        elif "parameter" in lower:
            return "parameterize"
        else:
            return "none"
