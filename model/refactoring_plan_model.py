from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class RefactoringPlan:
    """
    A single refactoring action, ready for Rope to execute.
    """
    plan_type: str  # e.g. "extract_method", "ai_merge_functions", "parameterize"
    file_path: str
    start_line: int
    end_line: int
    extra_info: Dict[str, Optional[str]] = None