# rope_refactor_engine.py

import os
import logging
from typing import List
from rope.base.project import Project
from rope.refactor.extract import ExtractMethod

from services.ai_merge import ai_merge_functions
# Import your AI-based merge function
# Make sure 'ai_merge.py' is accessible via the 'model' or appropriate path

from services.refactoring_planner import RefactoringPlan

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class RopeRefactorEngine:
    """
    A class that applies structured refactoring plans using Rope or AI-based code merges.

    Supported plan types:
      - "extract_method": standard Rope ExtractMethod
      - "ai_merge_functions": merges two functions with AI21, storing the merged code for user preview
      - "parameterize": placeholder for future parameterization logic
    """

    def __init__(self, project_path: str):
        self.project_path = os.path.abspath(project_path)
        if not os.path.isdir(self.project_path):
            raise ValueError(f"Invalid project path: {self.project_path}")
        try:
            self.project = Project(self.project_path)
            logger.debug(f"RopeRefactorEngine initialized with project path: {self.project_path}")
        except Exception as e:
            logger.error(f"Failed to initialize Rope project: {e}", exc_info=True)
            raise

    def apply_plans(self, plans: List[RefactoringPlan]) -> None:
        """
        Applies multiple refactoring plans in sequence.
        If one plan fails, logs an error and continues with the others.
        """
        for plan in plans:
            try:
                self.apply_refactor_plan(plan)
            except Exception as e:
                logger.error(f"Failed to apply plan {plan}: {e}", exc_info=True)

    def apply_refactor_plan(self, plan: RefactoringPlan) -> None:
        """
        Applies a single refactoring plan.
        If it's AI-based merging, we generate code but do not commit changes automatically
        (the user must preview & approve).
        """
        logger.debug(f"Applying plan: {plan}")
        plan_type = plan.plan_type.lower()

        if plan_type == "extract_method":
            self._extract_method(plan)
        elif plan_type == "ai_merge_functions":
            # AI-based merging: produce the merged code but don't commit automatically
            self._ai_merge_functions(plan)
        elif plan_type == "parameterize":
            self._parameterize_function(plan)
        else:
            logger.warning(f"Unknown plan type '{plan.plan_type}'. Skipping.")

    # ----------------------------------------------------------------------
    #                        EXTRACT METHOD
    # ----------------------------------------------------------------------
    def _extract_method(self, plan: RefactoringPlan) -> None:
        try:
            resource = self._get_resource(plan.file_path)
            source_code = resource.read()
            logger.debug(f"Successfully read source code from {plan.file_path}")
        except Exception as e:
            logger.error(f"Error reading source code from {plan.file_path}: {e}", exc_info=True)
            return

        try:
            start_offset = self._compute_offset(source_code, plan.start_line)
            end_offset   = self._compute_offset(source_code, plan.end_line)
            logger.debug(f"Computed offsets: start={start_offset}, end={end_offset}")
        except Exception as e:
            logger.error(f"Error computing offsets: {e}", exc_info=True)
            return

        new_func_name = plan.extra_info.get("new_func_name", "extracted_func")
        logger.info(f"Extracting lines {plan.start_line}-{plan.end_line} -> {new_func_name}")

        try:
            extractor = ExtractMethod(resource, start_offset, end_offset, new_func_name)
            changes = extractor.get_changes()
            self.project.do(changes)
            logger.info(f"ExtractMethod applied successfully for {plan.file_path}")
        except Exception as e:
            logger.error(f"Error during ExtractMethod for {plan.file_path}: {e}", exc_info=True)

    # ----------------------------------------------------------------------
    #                        AI MERGE FUNCTIONS
    # ----------------------------------------------------------------------
    def _ai_merge_functions(self, plan: RefactoringPlan) -> None:
        logger.info(f"AI-based merge requested: {plan}")

        # 1. Read function A's code from the selected lines
        try:
            func_a_code = self._get_function_code(plan.file_path, plan.start_line, plan.end_line)
            logger.debug(f"Function A code extracted (first 30 chars): {func_a_code[:30]}...")
        except Exception as e:
            logger.error(f"Error reading function A code: {e}", exc_info=True)
            plan.extra_info["ai_generated_code"] = f"Error reading function A code: {str(e)}"
            return

        # 2. Retrieve function B's code from extra_info
        func_b_code = plan.extra_info.get("func2_code", "")
        if not func_b_code.strip():
            logger.warning("func2_code is missing or empty; cannot merge.")
            plan.extra_info["ai_generated_code"] = "Error: No code for second function."
            return
        else:
            logger.debug(f"Function B code extracted (first 30 chars): {func_b_code[:30]}...")

        # 3. Call AI to produce the merged function, catching any exceptions.
        new_func_name = plan.extra_info.get("new_func_name", "merged_function")
        try:
            logger.debug(f"Merging with AI21: A='{func_a_code[:30]}...', B='{func_b_code[:30]}...'")
            ai_merged_code = ai_merge_functions(func_a_code, func_b_code, new_func_name)
            logger.debug("AI merge function returned merged code.")
        except Exception as e:
            logger.error("AI merge failed", exc_info=True)
            plan.extra_info["ai_generated_code"] = f"Error during AI merge: {str(e)}"
            return

        # 4. Store AI-generated code for user preview
        plan.extra_info["ai_generated_code"] = ai_merged_code
        logger.info("AI-based merge code stored in plan.extra_info['ai_generated_code']")

    # ----------------------------------------------------------------------
    #                        PARAMETERIZE (Placeholder)
    # ----------------------------------------------------------------------
    def _parameterize_function(self, plan: RefactoringPlan) -> None:
        logger.info(f"Parameterizing function(s). Strategy: {plan.extra_info.get('param_strategy')}")
        # Placeholder for parameterization logic
        pass

    # ----------------------------------------------------------------------
    #                        HELPER METHODS
    # ----------------------------------------------------------------------
    def _get_function_code(self, file_path: str, start_line: int, end_line: int) -> str:
        """
        Extracts the function code from 'start_line' to 'end_line' in the file.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            snippet = lines[start_line - 1:end_line]
            code = "".join(snippet)
            logger.debug(f"Extracted code from {file_path} lines {start_line}-{end_line}.")
            return code
        except Exception as e:
            logger.error(f"Failed to extract function code from {file_path}: {e}", exc_info=True)
            raise

    def _get_resource(self, file_path: str):
        """
        Converts an absolute file path into a Rope resource object.
        """
        try:
            rel_path = os.path.relpath(file_path, self.project_path)
            resource = self.project.get_file(rel_path)
            logger.debug(f"Converted {file_path} to Rope resource with relative path {rel_path}.")
            return resource
        except Exception as e:
            logger.error(f"Failed to get resource for {file_path}: {e}", exc_info=True)
            raise

    def _compute_offset(self, source_code: str, line_number: int) -> int:
        """
        Convert a 1-based line number into a byte offset for Rope.
        """
        try:
            if line_number <= 1:
                return 0
            lines = source_code.splitlines(True)
            offset = sum(len(lines[i]) for i in range(line_number - 1))
            logger.debug(f"Computed offset for line {line_number}: {offset}")
            return offset
        except Exception as e:
            logger.error(f"Failed to compute offset for line {line_number}: {e}", exc_info=True)
            raise

    def preview_refactor_plan(self, plan: RefactoringPlan) -> str:
        """
        Generates a preview (diff text) for the given refactoring plan.
        For AI-based merges, it triggers the AI merge process and returns the merged code.
        For other plan types, it returns a placeholder message.
        """
        plan_type = plan.plan_type.lower()

        try:
            if plan_type == "ai_merge_functions":
                # Call the AI merge function to generate merged code
                self._ai_merge_functions(plan)
                diff_text = plan.extra_info.get("ai_generated_code", "")
                if diff_text.strip():
                    logger.debug("Preview generated for AI merge.")
                    return diff_text
                else:
                    logger.debug("AI-based merge preview not available (empty result).")
                    return "AI-based merge preview not available."
            elif plan_type == "extract_method":
                # Optionally, implement a diff preview for extract method using Rope's diff tools
                return "Preview for 'extract_method' is not implemented yet."
            elif plan_type == "parameterize":
                return "Preview for 'parameterize' is not implemented yet."
            else:
                return f"No preview available for plan type: {plan.plan_type}"
        except Exception as e:
            logger.error(f"Error generating preview for plan {plan}: {e}", exc_info=True)
            return f"Error generating preview: {str(e)}"
