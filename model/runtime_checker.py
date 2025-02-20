#runtime_checker.py
import functools
import inspect
import logging
import random
import time
import statistics
import tracemalloc
from dataclasses import dataclass, field
from inspect import Parameter
from typing import Any, Dict, List, Tuple, Callable


# Example: Set to INFO. DEBUG messages are disabled to reduce overhead.
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

@dataclass
class RuntimeMetrics:
    execution_time: float = 0.0
    call_count: int = 0
    avg_time: float = 0.0
    input_patterns: Dict[str, int] = field(default_factory=dict)
    output_patterns: Dict[str, int] = field(default_factory=dict)
    input_args: List[dict] = field(default_factory=list)
    memory_usage: float = 0.0
    total_memory: float = 0.0
    peak_memory: float = 0.0

    test_results: Dict[int, Dict[str, float]] = field(default_factory=dict)

    def update(self, elapsed: float, input_data: Dict[str, Any], output_data: Any) -> None:
        self.call_count += 1
        self.execution_time += elapsed
        self.avg_time = self.execution_time / self.call_count

        input_key = str(input_data)
        self.input_patterns[input_key] = self.input_patterns.get(input_key, 0) + 1

        output_key = str(output_data)
        self.output_patterns[output_key] = self.output_patterns.get(output_key, 0) + 1


class RuntimeAnalyzer:
    def __init__(self) -> None:
        """
        Initialize the RuntimeAnalyzer.
        We seed the RNG for reproducibility in dummy argument generation.
        """
        random.seed(12345)  # Ensures consistent random values across runs
        self.metrics: Dict[str, RuntimeMetrics] = {}

    def get_metrics(self) -> Dict[str, RuntimeMetrics]:
        return self.metrics

    def runtime_check(self, func: Callable) -> Callable:
        """
        Decorator to measure execution time and update the corresponding RuntimeMetrics for each call.
        """
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            elapsed = end - start

            # Create a dictionary for input data to record patterns
            input_data = {'args': args, 'kwargs': kwargs}

            # Ensure metrics exist for the function
            func_name = func.__name__
            if func_name not in self.metrics:
                self.metrics[func_name] = RuntimeMetrics()

            # Update the metrics with the elapsed time and the inputs/outputs
            self.metrics[func_name].update(elapsed, input_data, result)
            return result

        return wrapper

    def generate_dummy_arguments(self, func: Callable, list_size: int = 100) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Generate dummy arguments with a configurable list size for the function signature.
        """
        def generate_test_list(size=list_size):
            return [random.randint(-100, 100) for _ in range(size)]

        signature = inspect.signature(func)
        args, kwargs = [], {}

        for name, param in signature.parameters.items():
            if param.default is not Parameter.empty:
                kwargs[name] = param.default
            elif param.annotation == int:
                args.append(42)
            elif param.annotation == float:
                args.append(3.14159)
            elif param.annotation == str:
                args.append("test_string")
            elif getattr(param.annotation, '__origin__', None) == list:
                args.append(generate_test_list())
            elif getattr(param.annotation, '__origin__', None) == dict:
                kwargs[name] = {f"key_{i}": i for i in range(10)}
            elif param.kind == Parameter.VAR_POSITIONAL:
                args.extend(generate_test_list())
            elif param.kind == Parameter.VAR_KEYWORD:
                kwargs.update({f"key_{i}": i for i in range(10)})
            else:
                # Default to a list if no specific annotation is found
                args.append(generate_test_list())

        return args, kwargs

    def _run_cpu_time_test(self,
                           func: Callable,
                           iterations: int,
                           args: List[Any],
                           kwargs: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
        """
        Measure CPU usage and wall-clock time with:
         - A warm-up phase to reduce cold-start overhead.
         - Individual iteration timing to gather statistics.
        Returns:
          (avg_cpu_time_per_call, total_wall_time, cpu_usage_percent, avg_wall_per_call, stdev_wall_per_call)
        """

        # Warm-up runs (discard any timing during warm-up)
        for _ in range(5):
            func(*args, **kwargs)

        # Capture CPU and wall time at the start of the real test
        start_cpu_ns = time.process_time_ns()
        start_wall_ns = time.perf_counter_ns()

        # Measure per-call wall time for detailed statistics
        iteration_wall_times = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            result = func(*args, **kwargs)
            t1 = time.perf_counter_ns()
            # Prevent "optimization" by referencing the result in some trivial way
            _ = hash(result)
            iteration_wall_times.append((t1 - t0) / 1e9)  # convert ns to seconds

        end_wall_ns = time.perf_counter_ns()
        end_cpu_ns = time.process_time_ns()

        # Calculate totals
        total_cpu_time = (end_cpu_ns - start_cpu_ns) / 1e9
        total_wall_time_measured = (end_wall_ns - start_wall_ns) / 1e9

        # Derived metrics
        avg_cpu_time_per_call = total_cpu_time / iterations if iterations > 0 else 0.0
        cpu_usage_percent = (
            (total_cpu_time / total_wall_time_measured) * 100
            if total_wall_time_measured > 0 else 0.0
        )

        # Basic statistics on wall time per call
        if iteration_wall_times:
            avg_wall_per_call = statistics.mean(iteration_wall_times)
            stdev_wall_per_call = (
                statistics.pstdev(iteration_wall_times)
                if len(iteration_wall_times) > 1 else 0.0
            )
        else:
            avg_wall_per_call = 0.0
            stdev_wall_per_call = 0.0

        return (avg_cpu_time_per_call,
                total_wall_time_measured,
                cpu_usage_percent,
                avg_wall_per_call,
                stdev_wall_per_call)

    def _run_memory_test(self,
                         func: Callable,
                         iterations: int,
                         args: List[Any],
                         kwargs: Dict[str, Any]) -> Tuple[float, float]:
        """
        Single-snapshot approach:
         - Start tracemalloc once before all iterations.
         - Stop after all iterations.
         - Compare final snapshot to the initial one.
        Returns:
          (avg_mem_kb, peak_mem_kb)
        """
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

        for _ in range(iterations):
            func(*args, **kwargs)

        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Compare memory usage
        stats = snapshot_after.compare_to(snapshot_before, 'lineno')
        # Sum absolute differences
        mem_diff = sum(abs(stat.size_diff) for stat in stats)
        peak_diff = max(abs(stat.size_diff) for stat in stats) if stats else 0

        # Convert bytes to KB; average per iteration
        avg_mem_kb = (mem_diff / iterations) / 1024.0 if iterations else 0.0
        peak_mem_kb = peak_diff / 1024.0

        return avg_mem_kb, peak_mem_kb


    def apply_runtime_checks(self,
                            similar_nodes: List[Dict],
                            global_namespace: Dict[str, Any],
                            num_tests: int,
                            dummy_list_size: int = 100) -> None:
        """
        Apply runtime checks using robust timing and memory profiling.
        """
        for result in similar_nodes:
            func1_metrics = result.get("function1_metrics")
            func2_metrics = result.get("function2_metrics")

            if not func1_metrics or not func2_metrics:
                continue

            for func_metrics in [func1_metrics, func2_metrics]:
                func_name = func_metrics.get("name")
                if func_name and func_name in global_namespace:
                    original_func = global_namespace[func_name]
                    # Decorate for real-time metrics on each call
                    decorated_func = self.runtime_check(original_func)
                    global_namespace[func_name] = decorated_func

                    # Generate dummy arguments
                    args, kwargs = self.generate_dummy_arguments(original_func, list_size=dummy_list_size)

                    # Initialize metrics if missing
                    if func_name not in self.metrics:
                        self.metrics[func_name] = RuntimeMetrics()

                    # Run CPU timing tests
                    (avg_cpu_time, total_wall_time, cpu_usage_percent,
                     avg_wall_per_call, stdev_wall_per_call) = self._run_cpu_time_test(
                        decorated_func, num_tests, args, kwargs
                    )

                    # Run memory tests
                    avg_mem_kb, peak_mem_kb = self._run_memory_test(
                        decorated_func, num_tests, args, kwargs
                    )

                    # Store consolidated results
                    self.metrics[func_name].test_results[num_tests] = {
                        "avg_cpu_time": avg_cpu_time,
                        "total_wall_time": total_wall_time,
                        "cpu_usage_percent": cpu_usage_percent,
                        "avg_mem_kb": avg_mem_kb,
                        "peak_mem_kb": peak_mem_kb,
                        "avg_wall_per_call": avg_wall_per_call,
                        "stdev_wall_per_call": stdev_wall_per_call,
                    }

                    logging.info(
                        f"[{func_name} | {num_tests} calls] "
                        f"CPU/call={avg_cpu_time:.3e}s, wall_total={total_wall_time:.3e}s, "
                        f"CPU%={cpu_usage_percent:.2f}%, mem_avg={avg_mem_kb:.2f}KB, "
                        f"mem_peak={peak_mem_kb:.2f}KB, wall_avg/call={avg_wall_per_call:.3e}s, "
                        f"wall_stdev={stdev_wall_per_call:.3e}s"
                    )

