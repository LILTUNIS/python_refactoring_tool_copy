from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QTextEdit, QGroupBox, QTreeWidget, QTreeWidgetItem
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QTextCursor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def create_treewidget(columns, headings, width=120, parent=None):
    """
    Helper to create a QTreeWidget with given columns and headings.
    """
    tree = QTreeWidget(parent)
    tree.setColumnCount(len(columns))
    tree.setHeaderLabels(headings)
    for col in range(len(columns)):
        tree.setColumnWidth(col, width)
    return tree


class SummaryTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Dictionary to hold key metric labels.
        self.summary_labels = {}
        self.refactor_text = None
        self.refactor_button = None
        self.chart_canvas = None  # Placeholder for the matplotlib canvas.
        self.top_complex_text = None
        self.slowest_text = None
        self.duplication_text = None

        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout(self)

        # Key Metrics Group
        metrics_group = QGroupBox("Key Metrics")
        metrics_layout = QGridLayout(metrics_group)
        metrics = [
            "Total Functions",
            "Total Static Pairs",
            "Avg Complexity",
            "Avg LOC",
            "Overall Similarity",
            "Total Duplicate Pairs"
        ]
        for i, key in enumerate(metrics):
            label = QLabel(f"{key}: N/A")
            metrics_layout.addWidget(label, i // 2, i % 2)
            self.summary_labels[key] = label
        main_layout.addWidget(metrics_group)

        # Complexity Distribution Chart Group
        chart_group = QGroupBox("Complexity Distribution")
        self.chart_layout = QVBoxLayout(chart_group)
        main_layout.addWidget(chart_group)

        # Refactoring Suggestions Group
        refactor_group = QGroupBox("🚀 Refactoring Suggestions")
        refactor_layout = QVBoxLayout(refactor_group)
        self.refactor_text = QTextEdit()
        self.refactor_text.setReadOnly(True)
        refactor_layout.addWidget(self.refactor_text)
        self.refactor_button = QPushButton("🛠 Refactor Code")
        self.refactor_button.clicked.connect(self.trigger_refactoring)
        refactor_layout.addWidget(self.refactor_button)
        main_layout.addWidget(refactor_group)

        # Additional Dashboard Sections

        # Most Complex Functions
        complex_group = QGroupBox("📍 Most Complex Functions")
        complex_layout = QVBoxLayout(complex_group)
        self.top_complex_text = QTextEdit()
        self.top_complex_text.setReadOnly(True)
        complex_layout.addWidget(self.top_complex_text)
        main_layout.addWidget(complex_group)

        # Slowest Functions
        slowest_group = QGroupBox("⏳ Slowest Functions")
        slowest_layout = QVBoxLayout(slowest_group)
        self.slowest_text = QTextEdit()
        self.slowest_text.setReadOnly(True)
        slowest_layout.addWidget(self.slowest_text)
        main_layout.addWidget(slowest_group)

        # Potential Code Duplication
        duplication_group = QGroupBox("🔍 Potential Code Duplication")
        duplication_layout = QVBoxLayout(duplication_group)
        self.duplication_text = QTextEdit()
        self.duplication_text.setReadOnly(True)
        duplication_layout.addWidget(self.duplication_text)
        main_layout.addWidget(duplication_group)

    def trigger_refactoring(self):
        """Simulate an automated refactoring process."""
        self.refactor_text.append("\n🔄 Generating refactoring suggestions...\n")
        dummy_suggestions = [
            "🔧 Extract common logic from function_a and function_b.",
            "🔧 Parameterize repeated hard-coded values in function_c.",
            "🔧 Restructure nested if-statements in function_d to improve readability."
        ]
        self.show_refactoring_suggestions(dummy_suggestions)

    def show_refactoring_suggestions(self, suggestions: list):
        self.refactor_text.clear()
        if not suggestions:
            self.refactor_text.append("✅ No refactoring suggestions.\n")
            return
        for s in suggestions:
            self.refactor_text.append(f"• {s}")

    def show_top_complex_functions(self, data):
        self.top_complex_text.clear()
        if not data:
            self.top_complex_text.append("No data.\n")
            return
        for func_name, comp in data:
            self.top_complex_text.append(f"{func_name} => Complexity: {comp}")

    def show_slowest_functions(self, data):
        self.slowest_text.clear()
        if not data:
            self.slowest_text.append("No data.\n")
            return
        for func_name, wall in data:
            self.slowest_text.append(f"{func_name} => Total Wall Time: {wall:.4f} s")

    def show_duplicate_pairs(self, data):
        self.duplication_text.clear()
        if not data:
            self.duplication_text.append("No duplicates found.\n")
            return
        for f1, f2, sim in data:
            self.duplication_text.append(f"{f1} & {f2} => Similarity: {sim:.2f}")

    def update_summary(self, results: dict):
        # You can update charts or store summary data here.
        pass

    def update_key_metrics(self, metrics: dict):
        total_funcs = metrics.get("total_functions", 0)
        self.summary_labels["Total Functions"].setText(f"Total Functions: {total_funcs}")
        total_pairs = metrics.get("total_static_pairs", 0)
        self.summary_labels["Total Static Pairs"].setText(f"Total Static Pairs: {total_pairs}")
        avg_complex = metrics.get("avg_complexity", 0.0)
        self.summary_labels["Avg Complexity"].setText(f"Avg Complexity: {avg_complex:.2f}")
        avg_loc = metrics.get("avg_loc", 0.0)
        self.summary_labels["Avg LOC"].setText(f"Avg LOC: {avg_loc:.1f}")
        overall_sim = metrics.get("overall_similarity", 0.0)
        self.summary_labels["Overall Similarity"].setText(f"Overall Similarity: {overall_sim:.2f}")
        total_dup = metrics.get("total_duplicate_pairs", 0)
        self.summary_labels["Total Duplicate Pairs"].setText(f"Total Duplicate Pairs: {total_dup}")

    def plot_complexity_chart(self, complexities):
        # Clear any existing widgets in the chart layout.
        for i in reversed(range(self.chart_layout.count())):
            widget = self.chart_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
        if not complexities:
            self.chart_layout.addWidget(QLabel("No complexity data available."))
            return
        bins = range(0, int(max(complexities)) + 2)
        fig = Figure(figsize=(5, 3), dpi=100)
        ax = fig.add_subplot(111)
        ax.hist(complexities, bins=bins, color="#17a2b8", edgecolor="black")
        ax.set_title("Complexity Distribution")
        ax.set_xlabel("Complexity")
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        canvas = FigureCanvas(fig)
        self.chart_layout.addWidget(canvas)


class StaticTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.static_tree = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        cols = ("Function", "Complexity", "LOC", "Params", "Nesting")
        heads = ["Function", "Complexity", "LOC", "Params", "Nesting Depth"]
        self.static_tree = create_treewidget(cols, heads, width=120)
        layout.addWidget(self.static_tree)

    def update_static(self, static_results):
        self.static_tree.clear()
        for r in static_results:
            f1 = r.get("function1_metrics", {})
            f2 = r.get("function2_metrics", {})
            # Insert first function's metrics.
            item1 = QTreeWidgetItem([
                f1.get("name", ""),
                str(f1.get("complexity", "")),
                str(f1.get("loc", "")),
                str(f1.get("parameter_count", "")),
                str(f1.get("nesting_depth", ""))
            ])
            self.static_tree.addTopLevelItem(item1)
            # Insert second function's metrics.
            item2 = QTreeWidgetItem([
                f2.get("name", ""),
                str(f2.get("complexity", "")),
                str(f2.get("loc", "")),
                str(f2.get("parameter_count", "")),
                str(f2.get("nesting_depth", ""))
            ])
            self.static_tree.addTopLevelItem(item2)


class CloneTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.clone_tree = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        cols = ("Pair", "Token Sim", "AST Sim")
        heads = ["Function Pair", "Token Similarity", "AST Similarity"]
        self.clone_tree = create_treewidget(cols, heads, width=150)
        layout.addWidget(self.clone_tree)

    def update_clone(self, clone_results):
        self.clone_tree.clear()
        for res in clone_results:
            f1 = res.get("func1") or res.get("function1_metrics", {}).get("name", "")
            f2 = res.get("func2") or res.get("function2_metrics", {}).get("name", "")
            token_sim = res.get("token_similarity", 0)
            ast_sim = res.get("ast_similarity", res.get("similarity", 0))
            pair = f"{f1} & {f2}"
            item = QTreeWidgetItem([
                pair,
                f"{token_sim:.2f}",
                f"{ast_sim:.2f}"
            ])
            self.clone_tree.addTopLevelItem(item)


class RuntimeTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.runtime_tree = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        cols = (
            "Function", "Tests", "AvgCpuTime", "TotalWallTime", "CpuUsage",
            "AvgMem", "PeakMem", "AvgWallPerCall", "StdDevWallPerCall"
        )
        heads = [
            "Function Name", "Tests Run", "Avg CPU Time (s)", "Total Wall Time (s)",
            "CPU Usage (%)", "Avg Mem (KB)", "Peak Mem (KB)",
            "Avg Wall/Call (s)", "StdDev Wall/Call (s)"
        ]
        self.runtime_tree = create_treewidget(cols, heads, width=130)
        layout.addWidget(self.runtime_tree)

    def update_runtime(self, runtime):
        self.runtime_tree.clear()
        funcs = runtime.get("functions", [])
        for f in funcs:
            func_name = f.get("func_name", "Unknown Function")
            for iters, data in f.get("test_results", {}).items():
                avg_cpu = float(data.get("avg_cpu_time", 0.0))
                total_wall = float(data.get("total_wall_time", 0.0))
                cpu_usage = float(data.get("cpu_usage_percent", 0.0))
                avg_mem = float(data.get("avg_mem_kb", 0.0))
                peak_mem = float(data.get("peak_mem_kb", 0.0))
                avg_wall_call = float(data.get("avg_wall_per_call", 0.0))
                stdev_wall_call = float(data.get("stdev_wall_per_call", 0.0))
                item = QTreeWidgetItem([
                    func_name,
                    str(iters),
                    f"{avg_cpu:.8f}",
                    f"{total_wall:.8f}",
                    f"{cpu_usage:.2f}",
                    f"{avg_mem:.6f}",
                    f"{peak_mem:.6f}",
                    f"{avg_wall_call:.8f}",
                    f"{stdev_wall_call:.8f}"
                ])
                self.runtime_tree.addTopLevelItem(item)


class DataFlowTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.dataflow_text = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        self.dataflow_text = QTextEdit()
        self.dataflow_text.setReadOnly(True)
        layout.addWidget(self.dataflow_text)

    def update_dataflow(self, data_flow):
        self.dataflow_text.clear()
        if not data_flow:
            self.dataflow_text.append("No data flow analysis available.\n")
            return
        for file, funcs in data_flow.items():
            self.dataflow_text.append(f"File: {file}")
            for func, details in funcs.items():
                self.dataflow_text.append(f"  Function: {func}")
                self.dataflow_text.append("    Variables:")
                for var, info in details.get("variables", {}).items():
                    defined = info.get("defined", "")
                    used = info.get("used", "")
                    self.dataflow_text.append(f"      {var}: Defined at line {defined}, Used at lines {used}")
                deps = details.get("dependencies", {})
                reads = deps.get("reads", [])
                writes = deps.get("writes", [])
                self.dataflow_text.append("    Dependencies:")
                self.dataflow_text.append(f"      Reads: {', '.join(reads) if reads else 'None'}")
                self.dataflow_text.append(f"      Writes: {', '.join(writes) if writes else 'None'}")
                self.dataflow_text.append("")  # Extra newline for spacing.
        self.dataflow_text.moveCursor(QTextCursor.End)
