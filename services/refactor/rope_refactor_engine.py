import os
import logging
from typing import List
# Import Rope’s project and refactoring modules
from rope.base.project import Project
from rope.refactor.extract import ExtractMethod

# Import the AI-based merge function from our services.
from services.refactor.ai_merge import ai_merge_functions
# Import the refactoring plan model used to describe changes.
from services.refactor.refactoring_planner import RefactoringPlan

# Set up a logger for this module.
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class RopeRefactorEngine:
    """
    Applies structured refactoring plans using either standard Rope techniques
    or AI-based code merges.

    Supported plan types include:
      - "extract_method": Uses Rope's ExtractMethod refactoring.
      - "ai_merge_functions": Leverages an AI merge function to combine two functions.
      - "parameterize": A placeholder for future parameterization logic.
    """

    def __init__(self, project_path: str):
        # Convert project path to an absolute path.
        self.project_path = os.path.abspath(project_path)
        # Ensure that the project directory exists.
        if not os.path.isdir(self.project_path):
            raise ValueError(f"Invalid project path: {self.project_path}")
        try:
            # Initialize a Rope project for refactoring operations.
            self.project = Project(self.project_path)
            logger.debug(f"RopeRefactorEngine initialized with project path: {self.project_path}")
        except Exception as e:
            logger.error(f"Failed to initialize Rope project: {e}", exc_info=True)
            raise

    def apply_plans(self, plans: List[RefactoringPlan]) -> None:
        """
        Iterates through a list of refactoring plans and applies each one.

        If a particular plan fails, it logs the error and continues with the remaining plans.
        """
        for plan in plans:
            try:
                self.apply_refactor_plan(plan)
            except Exception as e:
                logger.error(f"Failed to apply plan {plan}: {e}", exc_info=True)

    def apply_refactor_plan(self, plan: RefactoringPlan) -> None:
        """
        Applies a single refactoring plan based on its type.

        For AI-based merges, the merged code is generated and stored for user preview,
        rather than being applied automatically.
        """
        logger.debug(f"Applying plan: {plan}")
        plan_type = plan.plan_type.lower()

        if plan_type == "extract_method":
            self._extract_method(plan)
        elif plan_type == "ai_merge_functions":
            # For AI-based merging, generate the merged code without committing changes.
            self._ai_merge_functions(plan)
        elif plan_type == "parameterize":
            self._parameterize_function(plan)
        else:
            logger.warning(f"Unknown plan type '{plan.plan_type}'. Skipping.")

    # ----------------------------------------------------------------------
    #                        EXTRACT METHOD
    # ----------------------------------------------------------------------
    def _extract_method(self, plan: RefactoringPlan) -> None:
        """
        Performs an 'extract method' refactoring using Rope.

        This process:
          1. Reads the source code from the file.
          2. Computes byte offsets corresponding to the specified line numbers.
          3. Uses Rope's ExtractMethod to extract the selected code block.
          4. Applies the changes to the Rope project.
        """
        try:
            resource = self._get_resource(plan.file_path)
            source_code = resource.read()
            logger.debug(f"Successfully read source code from {plan.file_path}")
        except Exception as e:
            logger.error(f"Error reading source code from {plan.file_path}: {e}", exc_info=True)
            return

        try:
            # Compute the start and end offsets for the selected code block.
            start_offset = self._compute_offset(source_code, plan.start_line)
            end_offset = self._compute_offset(source_code, plan.end_line)
            logger.debug(f"Computed offsets: start={start_offset}, end={end_offset}")
        except Exception as e:
            logger.error(f"Error computing offsets: {e}", exc_info=True)
            return

        # Retrieve the new function name from the plan; default to 'extracted_func' if not provided.
        new_func_name = plan.extra_info.get("new_func_name", "extracted_func")
        logger.info(f"Extracting lines {plan.start_line}-{plan.end_line} -> {new_func_name}")

        try:
            # Initialize Rope's ExtractMethod refactoring.
            extractor = ExtractMethod(resource, start_offset, end_offset, new_func_name)
            changes = extractor.get_changes()
            # Apply the changes to the project.
            self.project.do(changes)
            logger.info(f"ExtractMethod applied successfully for {plan.file_path}")
        except Exception as e:
            logger.error(f"Error during ExtractMethod for {plan.file_path}: {e}", exc_info=True)

    # ----------------------------------------------------------------------
    #                        AI MERGE FUNCTIONS
    # ----------------------------------------------------------------------
    def _ai_merge_functions(self, plan: RefactoringPlan) -> None:
        """
        Merges two functions using an AI-based merge process.

        Process:
          1. Extract function A's code from the given file and line range.
          2. Retrieve function B's code from plan.extra_info.
          3. Call the AI merge service to merge both functions.
          4. Store the AI-generated merged code in plan.extra_info for user review.
        """
        logger.info(f"AI-based merge requested: {plan}")

        # 1. Extract function A's code.
        try:
            func_a_code = self._get_function_code(plan.file_path, plan.start_line, plan.end_line)
            logger.debug(f"Function A code extracted (first 30 chars): {func_a_code[:30]}...")
        except Exception as e:
            logger.error(f"Error reading function A code: {e}", exc_info=True)
            plan.extra_info["ai_generated_code"] = f"Error reading function A code: {str(e)}"
            return

        # 2. Retrieve function B's code from the extra_info field.
        func_b_code = plan.extra_info.get("func2_code", "")
        if not func_b_code.strip():
            logger.warning("func2_code is missing or empty; cannot merge.")
            plan.extra_info["ai_generated_code"] = "Error: No code for second function."
            return
        else:
            logger.debug(f"Function B code extracted (first 30 chars): {func_b_code[:30]}...")

        # 3. Call the AI merge function, passing both function codes and a new function name.
        new_func_name = plan.extra_info.get("new_func_name", "merged_function")
        try:
            logger.debug(f"Merging with AI21: A='{func_a_code[:30]}...', B='{func_b_code[:30]}...'")
            ai_merged_code = ai_merge_functions(func_a_code, func_b_code, new_func_name)
            logger.debug("AI merge function returned merged code.")
        except Exception as e:
            logger.error("AI merge failed", exc_info=True)
            plan.extra_info["ai_generated_code"] = f"Error during AI merge: {str(e)}"
            return

        # 4. Store the AI-generated merged code for user preview.
        plan.extra_info["ai_generated_code"] = ai_merged_code
        logger.info("AI-based merge code stored in plan.extra_info['ai_generated_code']")

    # ----------------------------------------------------------------------
    #                        PARAMETERIZE (Placeholder)
    # ----------------------------------------------------------------------
    def _parameterize_function(self, plan: RefactoringPlan) -> None:
        """
        Placeholder for future parameterization refactoring logic.

        This method would modify a function to make hard-coded values into parameters.
        """
        logger.info(f"Parameterizing function(s). Strategy: {plan.extra_info.get('param_strategy')}")
        # TODO: Implement parameterization logic.
        pass

    # ----------------------------------------------------------------------
    #                        HELPER METHODS
    # ----------------------------------------------------------------------
    def _get_function_code(self, file_path: str, start_line: int, end_line: int) -> str:
        """
        Extracts a snippet of code from a file, given start and end line numbers.

        Parameters:
            file_path: Path to the source file.
            start_line: 1-based start line number.
            end_line: 1-based end line number.

        Returns:
            The code snippet as a single string.
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

        This is required for Rope to perform refactoring actions on a file.
        """
        try:
            # Calculate the relative path from the project root.
            rel_path = os.path.relpath(file_path, self.project_path)
            resource = self.project.get_file(rel_path)
            logger.debug(f"Converted {file_path} to Rope resource with relative path {rel_path}.")
            return resource
        except Exception as e:
            logger.error(f"Failed to get resource for {file_path}: {e}", exc_info=True)
            raise

    def _compute_offset(self, source_code: str, line_number: int) -> int:
        """
        Converts a 1-based line number into a byte offset for Rope refactoring.

        This allows Rope to correctly locate the starting position for code extraction.
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

        For AI-based merges, it triggers the merge process and returns the generated code.
        For other plan types, it returns a placeholder message indicating that preview is not implemented.
        """
        plan_type = plan.plan_type.lower()

        try:
            if plan_type == "ai_merge_functions":
                # Generate merged code via the AI merge process.
                self._ai_merge_functions(plan)
                diff_text = plan.extra_info.get("ai_generated_code", "")
                if diff_text.strip():
                    logger.debug("Preview generated for AI merge.")
                    return diff_text
                else:
                    logger.debug("AI-based merge preview not available (empty result).")
                    return "AI-based merge preview not available."
            elif plan_type == "extract_method":
                # Future implementation: return a diff preview for extract method refactoring.
                return "Preview for 'extract_method' is not implemented yet."
            elif plan_type == "parameterize":
                return "Preview for 'parameterize' is not implemented yet."
            else:
                return f"No preview available for plan type: {plan.plan_type}"
        except Exception as e:
            logger.error(f"Error generating preview for plan {plan}: {e}", exc_info=True)
            return f"Error generating preview: {str(e)}"
