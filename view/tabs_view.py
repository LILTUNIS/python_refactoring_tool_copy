import json
import logging
import re

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QTextEdit, QGroupBox, QTreeWidget, QTreeWidgetItem, QApplication, QTabWidget
)
from PyQt5.QtGui import QTextCursor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from model.rope_refactor_engine import RopeRefactorEngine


def denormalize_text(text: str, reverse_mapping: dict) -> str:
    """
    Replaces all occurrences of var_N in 'text' with the original name
    using reverse_mapping.
    """
    pattern = r"(var_\d+)"  # Matches var_ followed by digits

    def replace_var(match):
        var_name = match.group(1)  # e.g. var_1
        return reverse_mapping.get(var_name, var_name)  # Replace with original name

    return re.sub(pattern, replace_var, text)
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
            "Total Avg Memory",
            "Peak Memory",
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
            # Only display if function name is non-empty.
            if func_name and func_name.strip():
                self.top_complex_text.append(f"{func_name} => Complexity: {comp}")

    def show_slowest_functions(self, data):
        self.slowest_text.clear()
        if not data:
            self.slowest_text.append("No data.\n")
            return
        for func_name, wall in data:
            if func_name and func_name.strip():
                self.slowest_text.append(f"{func_name} => Total Wall Time: {wall:.4f} s")

    def show_duplicate_pairs(self, data):
        self.duplication_text.clear()
        if not data:
            self.duplication_text.append("No duplicates found.\n")
            return
        for f1, f2, sim in data:
            if f1 and f2 and f1.strip() and f2.strip():
                self.duplication_text.append(f"{f1} & {f2} => Similarity: {sim:.2f}")


    def update_key_metrics(self, metrics: dict):
        total_funcs = metrics.get("total_functions", 0)
        self.summary_labels["Total Functions"].setText(f"Total Functions: {total_funcs}")
        avg_complex = metrics.get("avg_complexity", 0.0)
        self.summary_labels["Avg Complexity"].setText(f"Avg Complexity: {avg_complex:.2f}")
        avg_loc = metrics.get("avg_loc", 0.0)
        self.summary_labels["Avg LOC"].setText(f"Avg LOC: {avg_loc:.1f}")
        overall_sim = metrics.get("overall_similarity", 0.0)
        self.summary_labels["Overall Similarity"].setText(f"Overall Similarity: {overall_sim:.2f}")
        total_dup = metrics.get("total_duplicate_pairs", 0)
        self.summary_labels["Total Duplicate Pairs"].setText(f"Total Duplicate Pairs: {total_dup}")
        total_avg_mem = metrics.get("total_avg_memory", 0.0)
        peak_mem = metrics.get("peak_memory", 0.0)
        self.summary_labels["Total Avg Memory"].setText(f"Total Avg Memory: {total_avg_mem:.2f} KB")
        self.summary_labels["Peak Memory"].setText(f"Peak Memory: {peak_mem:.2f} KB")

    def plot_complexity_chart(self, complexities):
        # Clear existing widgets in the chart layout.
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

        # Updated columns to include "Static Score"
        cols = ("Function", "Complexity", "LOC", "Params", "Nesting", "Static Score")
        heads = ["Function", "Complexity", "LOC", "Params", "Nesting Depth", "Static Score"]

        self.static_tree = create_treewidget(cols, heads, width=120)
        layout.addWidget(self.static_tree)

    def update_static(self, static_results):
        """
        Populate the static_tree with data from static_results,
        which contains pairs of functions with metrics like complexity, loc, etc.
        """
        self.static_tree.clear()

        # Dictionary to store unique function info by name
        unique_funcs = {}

        for pair in static_results:
            for key in ["function1_metrics", "function2_metrics"]:
                fn_info = pair.get(key, {})
                name = fn_info.get("name", "").strip()
                if name:
                    # If not already stored, save this function's metrics
                    if name not in unique_funcs:
                        unique_funcs[name] = fn_info

        # Now add one row per unique function
        for name, info in unique_funcs.items():
            complexity = info.get("complexity", "")
            loc = info.get("loc", "")
            params = info.get("parameter_count", "")
            nesting = info.get("nesting_depth", "")
            # 'static_score' is the log-normalized value in [0..1]
            static_score = info.get("static_score", 0.0)

            # Format static_score to 3 decimals for readability
            static_score_str = f"{static_score:.3f}"

            item = QTreeWidgetItem([
                name,
                str(complexity),
                str(loc),
                str(params),
                str(nesting),
                static_score_str,
            ])
            self.static_tree.addTopLevelItem(item)

# --- In your CloneTab class ---
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem, QApplication, QHeaderView
)
from PyQt5.QtCore import Qt

class CloneTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.clone_tree = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        # Updated columns to include "Static Sim"
        self.columns = (
            "Pair",
            "Token Sim",
            "AST Sim",
            "Data Flow Sim",
            "Runtime Sim",
            "Static Sim",
            "Overall",
            "Exact Clone",
            "Refactoring Suggestion"
        )
        self.heads = [
            "Function Pair",
            "Token Similarity",
            "AST Similarity",
            "Data Flow Similarity",
            "Runtime Similarity",
            "Static Similarity",
            "Overall Similarity",
            "Exact Clone",
            "Refactoring Suggestion"
        ]

        self.clone_tree = QTreeWidget()
        self.clone_tree.setColumnCount(len(self.columns))
        self.clone_tree.setHeaderLabels(self.heads)
        self.clone_tree.setAlternatingRowColors(True)
        self.clone_tree.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.clone_tree.setSortingEnabled(True)

        layout.addWidget(self.clone_tree)

    def update_clone(self, clone_results):
        """
        Populate the clone_tree with data from clone_results,
        which now includes 'static_similarity'.
        """
        self.clone_tree.clear()

        if not clone_results:
            print("DEBUG: No clone results found.")
            return

        for res in clone_results:
            f1 = res.get("func1", "").strip()
            f2 = res.get("func2", "").strip()
            if not f1 or not f2:
                continue

            pair = f"{f1} & {f2}"

            token_sim = res.get("token_similarity", 0.0)
            ast_sim   = res.get("ast_similarity", 0.0)
            df_sim    = res.get("dataflow_similarity", 0.0)
            rt_sim    = res.get("runtime_similarity", 0.0)
            static_sim = res.get("static_similarity", 0.0)

            # (Optional) If you want to compute a new overall with static included:
            # overall = (token_sim + ast_sim + df_sim + rt_sim + static_sim) / 5
            # Or keep your existing approach.

            overall_str = "N/A"
            # If you already compute an "Overall Similarity" somewhere else, read it here
            # overall = res.get("new_overall", None)
            # if overall is not None:
            #     overall_str = f"{overall:.2f}"

            # For demonstration, let's do a naive average:
            # (You can refine weighting as you like)
            if None not in (token_sim, ast_sim, df_sim, rt_sim, static_sim):
                naive_avg = (token_sim + ast_sim + df_sim + rt_sim + static_sim) / 5
                overall_str = f"{naive_avg:.2f}"

            # Check if type1_similarity == 1.0 for "Exact Clone"
            type1_sim = res.get("type1_similarity", 0.0)
            exact_clone = "Yes" if abs(type1_sim - 1.0) < 1e-9 else "No"

            refactoring_suggestion = "N/A"
            # (Optional) You can fill in actual suggestions if your code provides them.

            row_item = QTreeWidgetItem([
                pair,
                f"{token_sim:.2f}",
                f"{ast_sim:.2f}",
                f"{df_sim:.2f}",
                f"{rt_sim:.2f}",
                f"{static_sim:.2f}",
                overall_str,
                exact_clone,
                refactoring_suggestion
            ])
            self.clone_tree.addTopLevelItem(row_item)

        # Resize columns to fit contents
        for col in range(self.clone_tree.columnCount()):
            self.clone_tree.resizeColumnToContents(col)

        QApplication.processEvents()


from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem, QHeaderView, QTabWidget, QScrollArea
)
from PyQt5.QtCore import Qt


class RuntimeTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.runtime_tree_summary = None
        self.runtime_tree_details = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        # Create a tab widget inside "Runtime"
        self.tabs = QTabWidget(self)

        # Summary Tab
        self.summary_tab = QWidget()
        summary_layout = QVBoxLayout(self.summary_tab)

        # Columns for Summary Table
        summary_cols = ("Function", "Tests", "Avg CPU Time", "Total Wall Time",
                        "Avg Mem", "Function Calls", "Branch Executions")
        summary_heads = ["Function Name", "Tests Run", "Avg CPU Time (s)", "Total Wall Time (s)",
                         "Avg Mem (KB)", "Function Calls", "Branch Executions"]

        self.runtime_tree_summary = self.create_treewidget(summary_cols, summary_heads)
        summary_layout.addWidget(self.runtime_tree_summary)

        # Execution Details Tab
        self.details_tab = QWidget()
        details_layout = QVBoxLayout(self.details_tab)

        # Columns for Execution Details Table
        details_cols = ("Function", "Branch Execution Frequency", "Coverage Map")
        details_heads = ["Function Name", "Branch Execution Frequency", "Coverage Map"]

        self.runtime_tree_details = self.create_treewidget(details_cols, details_heads)
        details_layout.addWidget(self.runtime_tree_details)

        # Add tabs to the QTabWidget
        self.tabs.addTab(self.summary_tab, "Summary")
        self.tabs.addTab(self.details_tab, "Branch Execution Details")

        layout.addWidget(self.tabs)

    def create_treewidget(self, cols, headers):
        """ Helper function to create a QTreeWidget with auto-resizing and scroll. """
        tree = QTreeWidget()
        tree.setColumnCount(len(cols))
        tree.setHeaderLabels(headers)
        tree.setAlternatingRowColors(True)
        tree.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # ✅ Allow horizontal scrolling

        # ✅ Configure column resizing:
        header = tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Auto-fit "Function Name"
        header.setSectionResizeMode(1, QHeaderView.Stretch)           # Stretch second column
        header.setSectionResizeMode(QHeaderView.ResizeToContents)     # Default for all others

        return tree

    def update_runtime(self, runtime):
        """
        Reads runtime metrics and updates the tables.
        """
        # Clear previous data
        self.runtime_tree_summary.clear()
        self.runtime_tree_details.clear()

        funcs = runtime.get("functions", [])

        for f in funcs:
            func_name = f.get("func_name", "").strip()
            if not func_name:
                continue  # Skip functions with missing names

            test_results = f.get("test_results", {})

            for iters, data in test_results.items():
                runtime_data = data.get("runtime", {})

                # Extract runtime summary data
                avg_cpu = float(runtime_data.get("avg_cpu_time", 0.0))
                total_wall = float(runtime_data.get("total_wall_time", 0.0))
                avg_mem = float(runtime_data.get("avg_memory_kb", 0.0))
                function_calls = runtime_data.get("function_calls", 0)
                branch_executions = runtime_data.get("branch_executions", 0)

                # Add to the Summary Table
                summary_item = QTreeWidgetItem([
                    func_name,
                    str(iters),
                    f"{avg_cpu:.8f}",
                    f"{total_wall:.8f}",
                    f"{avg_mem:.6f}",
                    str(function_calls),
                    str(branch_executions)
                ])
                self.runtime_tree_summary.addTopLevelItem(summary_item)

                # Extract Execution Details Data
                branch_exec_freq = runtime_data.get("branch_execution_frequency", {})

                # ✅ New Formatting: Show branches in multiple lines
                formatted_branch_exec = "\n".join(
                    f"{start} → {end} ({count}x)" for (start, end), count in sorted(
                        branch_exec_freq.items(), key=lambda x: x[1], reverse=True
                    )
                ) if branch_exec_freq else "N/A"

                coverage_map = runtime_data.get("executed_lines", {})
                formatted_coverage = "\n".join(
                    f"{file}: {sorted(lines)}" for file, lines in coverage_map.items()
                )

                # Add to the Execution Details Table
                details_item = QTreeWidgetItem([
                    func_name,
                    formatted_branch_exec,
                    formatted_coverage
                ])

                # ✅ Add tooltip to show full details on hover
                details_item.setToolTip(1, formatted_branch_exec)
                details_item.setToolTip(2, formatted_coverage)

                self.runtime_tree_details.addTopLevelItem(details_item)

        # Update UI
        self.runtime_tree_summary.viewport().update()
        self.runtime_tree_details.viewport().update()



from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem, QHeaderView
)
# ... rest of your imports

class DataFlowTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.dataflow_tree = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        self.dataflow_tree = QTreeWidget()
        self.dataflow_tree.setHeaderLabels(["Element", "Details"])

        # ▼ Configure the header so the first column resizes to fit its contents,
        #   and the second column stretches to fill the remaining space.
        header = self.dataflow_tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # auto-resize left column
        header.setSectionResizeMode(1, QHeaderView.Stretch)           # stretch right column

        layout.addWidget(self.dataflow_tree)
        self.setLayout(layout)

    def update_dataflow(self, data_flow):
        self.dataflow_tree.clear()

        if not data_flow:
            no_data_item = QTreeWidgetItem(["No data flow analysis available.", ""])
            self.dataflow_tree.addTopLevelItem(no_data_item)
            return

        for file, funcs in data_flow.items():
            file_item = QTreeWidgetItem([f"📂 File", file])
            self.dataflow_tree.addTopLevelItem(file_item)

            for func, details in funcs.items():
                func_item = QTreeWidgetItem(["🔹 Function", func])
                file_item.addChild(func_item)

                reverse_mapping = details.get("reverse_mapping", {})
                def denormalize(var_name: str) -> str:
                    return reverse_mapping.get(var_name, var_name)

                deps = details.get("dependencies", {})

                for dep_key, emoji in [
                    ("reads", "📥 Reads"),
                    ("writes", "✏️ Writes"),
                    ("returns", "🔄 Returns"),
                    ("function_calls", "📞 Function Calls"),
                    ("control_flows", "🔀 Control Flows"),
                    ("exception_handling", "⚠️ Exception Handling"),
                    ("side_effects", "💥 Side Effects")
                ]:
                    dep_values = deps.get(dep_key, [])
                    if dep_values:
                        dep_values_denorm = []
                        for fc in dep_values:
                            if "." in fc:
                                left, right = fc.split(".", 1)
                                left = denormalize(left)
                                fc = left + "." + right
                            else:
                                fc = denormalize(fc)
                            dep_values_denorm.append(fc)

                        dep_item = QTreeWidgetItem([emoji, ", ".join(dep_values_denorm)])
                        func_item.addChild(dep_item)

                io_relations = deps.get("input_output_relations", [])
                if io_relations:
                    io_item = QTreeWidgetItem(["🔄 Input-Output Relations", ""])
                    func_item.addChild(io_item)
                    for relation in io_relations:
                        var_name = denormalize_text(relation.get("var_name", "??"), reverse_mapping)
                        desc = denormalize_text(relation.get("desc", ""), reverse_mapping)
                        op_type = relation.get("operation", "")
                        relation_item = QTreeWidgetItem([f"{var_name} ({op_type})", desc])
                        io_item.addChild(relation_item)

        self.dataflow_tree.expandAll()  # auto-expand all items
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem,
    QPushButton, QHBoxLayout, QHeaderView, QMessageBox
)
from PyQt5.QtCore import Qt

# tabs_view.py (RefactorTab class)




class RefactorTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.refactor_tree = None
        self.apply_button = None
        self.engine = None  # We'll init RopeRefactorEngine later
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout(self)

        # QTreeWidget: (checkbox) plan_type, file_path, lines, extra
        self.refactor_tree = QTreeWidget()
        self.refactor_tree.setColumnCount(4)
        self.refactor_tree.setHeaderLabels(["Select", "Plan Type", "Location", "Details"])
        self.refactor_tree.setAlternatingRowColors(True)
        self.refactor_tree.setRootIsDecorated(False)
        self.refactor_tree.setSelectionMode(QTreeWidget.NoSelection)

        header = self.refactor_tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Stretch)

        main_layout.addWidget(self.refactor_tree)

        # "Apply Refactor" button
        button_layout = QHBoxLayout()
        self.apply_button = QPushButton("Apply Refactor")
        self.apply_button.clicked.connect(self.on_apply_refactor)
        button_layout.addWidget(self.apply_button)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def set_project_path(self, project_path: str):
        """
        Optionally call this if you want to init the RopeRefactorEngine
        once the user chooses a folder.
        """
        self.engine = RopeRefactorEngine(project_path)

    def update_refactor_plans(self, plans):
        """
        Display a list of RefactoringPlan objects in the tree.
        Each plan has plan_type, file_path, start_line, end_line, etc.
        """
        self.refactor_tree.clear()

        if not plans:
            no_item = QTreeWidgetItem(["", "No plans", "", ""])
            self.refactor_tree.addTopLevelItem(no_item)
            return

        for plan in plans:
            # E.g. "extract_method" => line range => new_func_name
            plan_type = plan.plan_type
            location_str = f"{plan.file_path}:{plan.start_line}-{plan.end_line}"
            extra = plan.extra_info.get("suggested_text", "")

            row_item = QTreeWidgetItem(["", plan_type, location_str, extra])
            row_item.setCheckState(0, Qt.Unchecked)
            # store the plan object so we can retrieve it on "Apply"
            row_item.setData(0, Qt.UserRole, plan)
            self.refactor_tree.addTopLevelItem(row_item)

    def on_apply_refactor(self):
        """
        Gather checked items => call RopeRefactorEngine
        """
        if not self.engine:
            QMessageBox.warning(self, "No Engine", "RopeRefactorEngine not initialized.")
            return

        checked_plans = []
        for i in range(self.refactor_tree.topLevelItemCount()):
            item = self.refactor_tree.topLevelItem(i)
            if item.checkState(0) == Qt.Checked:
                plan = item.data(0, Qt.UserRole)  # Retrieve the stored plan
                checked_plans.append(plan)

        if not checked_plans:
            QMessageBox.information(self, "No Selection", "No refactoring plans selected.")
            return

        # Apply them
        self.engine.apply_plans(checked_plans)
        QMessageBox.information(self, "Refactoring Complete", f"Applied {len(checked_plans)} plan(s).")
