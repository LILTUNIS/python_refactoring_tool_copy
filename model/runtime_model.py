from dataclasses import field, dataclass
from typing import List, Dict, Any

@dataclass
class RuntimeMetrics:
    """
    Stores runtime metrics for each function.
    """
    execution_time: float = 0.0
    execution_times: List[float] = field(default_factory=list)
    call_count: int = 0
    error_count: int = 0
    avg_time: float = 0.0
    memory_usages: List[float] = field(default_factory=list)
    avg_memory_kb: float = 0.0
    coverage_data: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    test_results: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    # Additional counters
    function_calls: int = 0
    branch_executions: int = 0
    # Tracks input / output patterns
    input_patterns: Dict[str, int] = field(default_factory=dict)
    output_patterns: Dict[str, int] = field(default_factory=dict)

    def update(self, elapsed: float, input_data: Dict[str, Any], output_data: Any) -> None:
        """
        Update aggregated metrics on each function call (timing, input, output).
        """
        self.call_count += 1
        self.execution_time += elapsed
        self.execution_times.append(elapsed)
        self.avg_time = self.execution_time / self.call_count

        input_key = str(input_data)
        self.input_patterns[input_key] = self.input_patterns.get(input_key, 0) + 1

        output_key = str(output_data)
        self.output_patterns[output_key] = self.output_patterns.get(output_key, 0) + 1

    def update_memory(self, mem_usage: float) -> None:
        """
        Update memory usage metrics (KB).
        """
        self.memory_usages.append(mem_usage)
        if self.memory_usages:
            self.avg_memory_kb = sum(self.memory_usages) / len(self.memory_usages)
