# rope_refactor_engine.py

import os
import logging
from typing import List
from rope.base.project import Project
from rope.refactor.extract import ExtractMethod
# from rope.refactor.inline import Inline
# from rope.refactor.rename import Rename

from .refactoring_planner import RefactoringPlan

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class RopeRefactorEngine:
    """
    Step 3: Take RefactoringPlan objects and use Rope to
    transform the code automatically.
    """

    def __init__(self, project_path: str):
        self.project_path = os.path.abspath(project_path)
        if not os.path.isdir(self.project_path):
            raise ValueError(f"Invalid project path: {self.project_path}")
        self.project = Project(self.project_path)
        logger.debug(f"RopeRefactorEngine initialized with project path: {self.project_path}")

    def apply_plans(self, plans: List[RefactoringPlan]) -> None:
        """
        Apply a batch of plans in sequence. If one fails, log an error
        and move on to the next.
        """
        for plan in plans:
            try:
                self.apply_refactor_plan(plan)
            except Exception as e:
                logger.error(f"Failed to apply plan {plan}: {e}", exc_info=True)

    def apply_refactor_plan(self, plan: RefactoringPlan) -> None:
        logger.debug(f"Applying plan: {plan}")
        plan_type = plan.plan_type.lower()

        if plan_type == "extract_method":
            self._extract_method(plan)
        elif plan_type == "merge_functions":
            self._merge_functions(plan)
        elif plan_type == "parameterize":
            self._parameterize_function(plan)
        else:
            logger.warning(f"Unknown plan type '{plan.plan_type}'. Skipping.")

    def _extract_method(self, plan: RefactoringPlan) -> None:
        resource = self._get_resource(plan.file_path)
        source_code = resource.read()
        start_offset = self._compute_offset(source_code, plan.start_line)
        end_offset   = self._compute_offset(source_code, plan.end_line)

        new_func_name = plan.extra_info.get("new_func_name", "extracted_func")
        logger.info(f"Extracting lines {plan.start_line}-{plan.end_line} -> {new_func_name}")

        extractor = ExtractMethod(resource, start_offset, end_offset, new_func_name)
        changes = extractor.get_changes()
        self.project.do(changes)
        logger.info(f"ExtractMethod applied successfully for {plan.file_path}")

    def _merge_functions(self, plan: RefactoringPlan) -> None:
        logger.info(f"Merging not fully implemented. Strategy: {plan.extra_info.get('strategy')}")
        # Placeholder. In real usage, you'd do a multi-step approach:
        # 1) Extract common code from function1
        # 2) Possibly inline function2 or rename it
        # 3) ...
        pass

    def _parameterize_function(self, plan: RefactoringPlan) -> None:
        logger.info(f"Parameterizing function(s). Strategy: {plan.extra_info.get('param_strategy')}")
        # Placeholder for parameterization logic
        pass

    def _get_resource(self, file_path: str):
        rel_path = os.path.relpath(file_path, self.project_path)
        return self.project.get_file(rel_path)

    def _compute_offset(self, source_code: str, line_number: int) -> int:
        if line_number <= 1:
            return 0
        lines = source_code.splitlines(True)
        offset = 0
        for i in range(line_number - 1):
            offset += len(lines[i])
        return offset
