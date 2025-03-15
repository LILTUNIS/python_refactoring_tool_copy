import re
from typing import List

# Import required PyQt5 modules for creating GUI components.
from PyQt5.QtWidgets import (
    QWidget, QGridLayout, QLabel, QGroupBox, QLineEdit
)
# Import matplotlib components to integrate plots within the Qt application.
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def denormalize_text(text: str, reverse_mapping: dict) -> str:
    """
    Replace all occurrences of normalized variable names (e.g., "var_1")
    in the given text with their original names provided in reverse_mapping.

    Parameters:
        text (str): The input text containing normalized variable names.
        reverse_mapping (dict): A dictionary mapping normalized names (e.g., "var_1")
                                to their original variable names.

    Returns:
        str: The text after replacing normalized names with original names.
    """
    # Define a regex pattern that matches 'var_' followed by one or more digits.
    pattern = r"(var_\d+)"

    def replace_var(match):
        # Extract the normalized variable name from the regex match.
        var_name = match.group(1)  # e.g. "var_1"
        # Return the original name if found in reverse_mapping; otherwise, return the original normalized name.
        return reverse_mapping.get(var_name, var_name)

    # Substitute all matching occurrences in the text.
    return re.sub(pattern, replace_var, text)


def create_treewidget(columns, headings, width=120, parent=None):
    """
    Helper function to create and configure a QTreeWidget.

    Parameters:
        columns: A tuple/list defining column identifiers.
        headings: A list of header titles for the tree widget.
        width (int, optional): Default width for each column.
        parent: The parent widget, if any.

    Returns:
        QTreeWidget: A configured tree widget with specified columns and headings.
    """
    from PyQt5.QtWidgets import QTreeWidget  # Imported here to avoid circular import issues.
    tree = QTreeWidget(parent)
    tree.setColumnCount(len(columns))
    tree.setHeaderLabels(headings)
    # Set each column to the specified width.
    for col in range(len(columns)):
        tree.setColumnWidth(col, width)
    return tree


class SummaryTab(QWidget):
    """
    A QWidget representing the Summary Tab of the analysis dashboard.

    This tab displays key metrics, a complexity distribution chart,
    refactoring suggestions, and additional sections like the most complex
    and slowest functions along with potential code duplications.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # Initialize member variables to hold references to GUI elements.
        self.summary_labels = {}  # Dictionary to store key metric labels.
        self.refactor_text = None
        self.refactor_button = None
        self.chart_canvas = None  # Placeholder for the matplotlib canvas.
        self.top_complex_text = None
        self.slowest_text = None
        self.duplication_text = None

        self.initUI()

    def initUI(self):
        """
        Set up the user interface by creating and arranging all necessary widgets.
        """
        from PyQt5.QtWidgets import QVBoxLayout, QTextEdit, QPushButton  # Local import for clarity
        main_layout = QVBoxLayout(self)

        # --- Key Metrics Section ---
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
        # Create a QLabel for each metric and add it to the grid layout.
        for i, key in enumerate(metrics):
            label = QLabel(f"{key}: N/A")
            metrics_layout.addWidget(label, i // 2, i % 2)
            self.summary_labels[key] = label
        main_layout.addWidget(metrics_group)

        # --- Complexity Distribution Chart Section ---
        chart_group = QGroupBox("Complexity Distribution")
        self.chart_layout = QVBoxLayout(chart_group)
        main_layout.addWidget(chart_group)

        # --- Refactoring Suggestions Section ---
        refactor_group = QGroupBox("🚀 Refactoring Suggestions")
        refactor_layout = QVBoxLayout(refactor_group)
        self.refactor_text = QTextEdit()
        self.refactor_text.setReadOnly(True)
        refactor_layout.addWidget(self.refactor_text)
        self.refactor_button = QPushButton("🛠 Refactor Code")
        # Connect the button click to trigger the refactoring process.
        self.refactor_button.clicked.connect(self.trigger_refactoring)
        refactor_layout.addWidget(self.refactor_button)
        main_layout.addWidget(refactor_group)

        # --- Additional Dashboard Sections ---
        # Section for displaying the most complex functions.
        complex_group = QGroupBox("📍 Most Complex Functions")
        complex_layout = QVBoxLayout(complex_group)
        self.top_complex_text = QTextEdit()
        self.top_complex_text.setReadOnly(True)
        complex_layout.addWidget(self.top_complex_text)
        main_layout.addWidget(complex_group)

        # Section for displaying the slowest functions.
        slowest_group = QGroupBox("⏳ Slowest Functions")
        slowest_layout = QVBoxLayout(slowest_group)
        self.slowest_text = QTextEdit()
        self.slowest_text.setReadOnly(True)
        slowest_layout.addWidget(self.slowest_text)
        main_layout.addWidget(slowest_group)

        # Section for displaying potential code duplication issues.
        duplication_group = QGroupBox("🔍 Potential Code Duplication")
        duplication_layout = QVBoxLayout(duplication_group)
        self.duplication_text = QTextEdit()
        self.duplication_text.setReadOnly(True)
        duplication_layout.addWidget(self.duplication_text)
        main_layout.addWidget(duplication_group)

    def trigger_refactoring(self):
        """
        Simulate an automated refactoring process by generating dummy suggestions.

        In a real-world scenario, this would trigger analysis and produce
        actionable refactoring recommendations.
        """
        self.refactor_text.append("\n🔄 Generating refactoring suggestions...\n")
        dummy_suggestions = [
            "🔧 Extract common logic from function_a and function_b.",
            "🔧 Parameterize repeated hard-coded values in function_c.",
            "🔧 Restructure nested if-statements in function_d to improve readability."
        ]
        self.show_refactoring_suggestions(dummy_suggestions)

    def show_refactoring_suggestions(self, suggestions: list):
        """
        Update the refactoring suggestions text area with the provided suggestions.

        Parameters:
            suggestions (list): List of refactoring suggestion strings.
        """
        self.refactor_text.clear()
        if not suggestions:
            self.refactor_text.append("✅ No refactoring suggestions.\n")
            return
        # Append each suggestion as a bullet point.
        for s in suggestions:
            self.refactor_text.append(f"• {s}")

    def show_top_complex_functions(self, data):
        """
        Display a list of the most complex functions along with their complexity scores.

        Parameters:
            data (iterable): An iterable of tuples (function name, complexity score).
        """
        self.top_complex_text.clear()
        if not data:
            self.top_complex_text.append("No data.\n")
            return
        for func_name, comp in data:
            if func_name and func_name.strip():
                self.top_complex_text.append(f"{func_name} => Complexity: {comp}")

    def show_slowest_functions(self, data):
        """
        Display a list of the slowest functions along with their wall time measurements.

        Parameters:
            data (iterable): An iterable of tuples (function name, wall time).
        """
        self.slowest_text.clear()
        if not data:
            self.slowest_text.append("No data.\n")
            return
        for func_name, wall in data:
            if func_name and func_name.strip():
                self.slowest_text.append(f"{func_name} => Total Wall Time: {wall:.4f} s")

    def show_duplicate_pairs(self, data):
        """
        Display pairs of functions that have high similarity, indicating potential code duplication.

        Parameters:
            data (iterable): An iterable of tuples (function1, function2, similarity score).
        """
        self.duplication_text.clear()
        if not data:
            self.duplication_text.append("No duplicates found.\n")
            return
        for f1, f2, sim in data:
            if f1 and f2 and f1.strip() and f2.strip():
                self.duplication_text.append(f"{f1} & {f2} => Similarity: {sim:.2f}")

    def update_key_metrics(self, metrics: dict):
        """
        Update the key metric labels in the UI with the latest computed values.

        Parameters:
            metrics (dict): Dictionary containing various computed metrics.
        """
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
        """
        Plot a histogram chart to display the distribution of complexity metrics.

        Parameters:
            complexities (list): List of complexity values.
        """
        from PyQt5.QtWidgets import QLabel  # Import QLabel locally
        # Clear any existing widgets from the chart layout.
        for i in reversed(range(self.chart_layout.count())):
            widget = self.chart_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
        if not complexities:
            self.chart_layout.addWidget(QLabel("No complexity data available."))
            return
        # Define bins based on the maximum complexity value.
        bins = range(0, int(max(complexities)) + 2)
        # Create a matplotlib figure and axis.
        fig = Figure(figsize=(5, 3), dpi=100)
        ax = fig.add_subplot(111)
        # Plot histogram with defined bins, a set edge color, and a default fill color.
        ax.hist(complexities, bins=bins, color="#17a2b8", edgecolor="black")
        ax.set_title("Complexity Distribution")
        ax.set_xlabel("Complexity")
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        # Embed the matplotlib figure into the PyQt5 widget using FigureCanvas.
        canvas = FigureCanvas(fig)
        self.chart_layout.addWidget(canvas)


class StaticTab(QWidget):
    """
    Tab for displaying static analysis results.

    This tab presents data about individual functions, including complexity,
    lines of code, parameter count, nesting depth, and a normalized static score.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.static_tree = None
        self.initUI()

    def initUI(self):
        """
        Initialize the static analysis tab UI by setting up the tree widget.
        """
        from PyQt5.QtWidgets import QVBoxLayout  # Local import for clarity
        layout = QVBoxLayout(self)

        # Define column identifiers and headings.
        cols = ("Function", "Complexity", "LOC", "Params", "Nesting", "Static Score")
        heads = ["Function", "Complexity", "LOC", "Params", "Nesting Depth", "Static Score"]

        self.static_tree = create_treewidget(cols, heads, width=120)
        layout.addWidget(self.static_tree)

    def update_static(self, static_results):
        """
        Populate the static analysis tree with results.

        Parameters:
            static_results (list): A list containing pairs of function metrics.
        """
        from PyQt5.QtWidgets import QTreeWidgetItem  # Local import
        self.static_tree.clear()

        # Use a dictionary to store unique function information to avoid duplicates.
        unique_funcs = {}

        for pair in static_results:
            for key in ["function1_metrics", "function2_metrics"]:
                fn_info = pair.get(key, {})
                name = fn_info.get("name", "").strip()
                if name:
                    if name not in unique_funcs:
                        unique_funcs[name] = fn_info

        # Add a single row per unique function.
        for name, info in unique_funcs.items():
            complexity = info.get("complexity", "")
            loc = info.get("loc", "")
            params = info.get("parameter_count", "")
            nesting = info.get("nesting_depth", "")
            # 'static_score' is a log-normalized value between 0 and 1.
            static_score = info.get("static_score", 0.0)
            static_score_str = f"{static_score:.3f}"  # Format score for readability.

            item = QTreeWidgetItem([
                name,
                str(complexity),
                str(loc),
                str(params),
                str(nesting),
                static_score_str,
            ])
            self.static_tree.addTopLevelItem(item)


# --- CloneTab: Handles detection and display of code clones ---
from PyQt5.QtWidgets import QWidget


class CloneTab(QWidget):
    """
    Tab for displaying code clone analysis.

    This tab includes a search feature and a tree widget to list clone pairs
    along with similarity metrics and potential refactoring suggestions.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.clone_tree = None
        self.search_box = None
        self.all_clone_results = []  # Store all clone results for later filtering.
        self.initUI()

    def initUI(self):
        """
        Set up the CloneTab UI with search capabilities and a tree widget.
        """
        from PyQt5.QtWidgets import QVBoxLayout  # Local import for clarity
        layout = QVBoxLayout(self)

        # Create a label and a search box for filtering function pairs.
        search_label = QLabel("🔍 Search Function Pairs:")
        layout.addWidget(search_label)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Enter function name to search...")
        self.search_box.textChanged.connect(self.filter_clone_results)
        layout.addWidget(self.search_box)

        # Define the columns and headings for the clone analysis tree.
        self.columns = (
            "Pair",
            "Token Sim",
            "AST Sim",
            "Data Flow Sim",
            "Runtime Sim",
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
            "Overall Similarity",
            "Exact Clone",
            "Refactoring Suggestion"
        ]

        from PyQt5.QtWidgets import QTreeWidget  # Local import for clarity
        self.clone_tree = QTreeWidget()
        self.clone_tree.setColumnCount(len(self.columns))
        self.clone_tree.setHeaderLabels(self.heads)
        self.clone_tree.setAlternatingRowColors(True)
        self.clone_tree.setSortingEnabled(True)
        layout.addWidget(self.clone_tree)

    def update_clone(self, clone_results):
        """
        Populate the clone_tree with code clone data.

        Parameters:
            clone_results (list): A list of dictionaries with clone analysis data.
        """
        from PyQt5.QtWidgets import QTreeWidgetItem  # Local import for clarity
        self.clone_tree.clear()
        self.all_clone_results = clone_results  # Save results for filtering.

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
            ast_sim = res.get("ast_similarity", 0.0)
            df_sim = res.get("dataflow_similarity", 0.0)
            rt_sim = res.get("runtime_similarity", 0.0)

            overall_str = "N/A"
            # Calculate naive average if all values are present.
            if None not in (token_sim, ast_sim, df_sim, rt_sim,):
                naive_avg = (token_sim + ast_sim + df_sim + rt_sim) / 4
                overall_str = f"{naive_avg:.2f}"

            # Determine if this clone pair is an exact clone.
            type1_sim = res.get("type1_similarity", 0.0)
            exact_clone = "Yes" if abs(type1_sim - 1.0) < 1e-9 else "No"

            refactoring_suggestion = "N/A"

            row_item = QTreeWidgetItem([
                pair,
                f"{token_sim:.2f}",
                f"{ast_sim:.2f}",
                f"{df_sim:.2f}",
                f"{rt_sim:.2f}",
                overall_str,
                exact_clone,
                refactoring_suggestion
            ])
            self.clone_tree.addTopLevelItem(row_item)

        self.clone_tree.resizeColumnToContents(0)

    def filter_clone_results(self):
        """
        Filter the displayed clone pairs based on the text entered in the search box.
        """
        from PyQt5.QtWidgets import QTreeWidgetItem  # Local import for clarity
        search_text = self.search_box.text().strip().lower()
        self.clone_tree.clear()

        # Iterate over stored clone results and display only matching pairs.
        for res in self.all_clone_results:
            f1 = res.get("func1", "").strip().lower()
            f2 = res.get("func2", "").strip().lower()
            pair = f"{f1} & {f2}"

            # Check if the search text is contained in either function name or the pair string.
            if search_text in f1 or search_text in f2 or search_text in pair:
                token_sim = res.get("token_similarity", 0.0)
                ast_sim = res.get("ast_similarity", 0.0)
                df_sim = res.get("dataflow_similarity", 0.0)
                rt_sim = res.get("runtime_similarity", 0.0)
                static_sim = res.get("static_similarity", 0.0)

                overall_str = "N/A"
                if None not in (token_sim, ast_sim, df_sim, rt_sim, static_sim):
                    naive_avg = (token_sim + ast_sim + df_sim + rt_sim + static_sim) / 5
                    overall_str = f"{naive_avg:.2f}"

                type1_sim = res.get("type1_similarity", 0.0)
                exact_clone = "Yes" if abs(type1_sim - 1.0) < 1e-9 else "No"

                refactoring_suggestion = "N/A"

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

        self.clone_tree.resizeColumnToContents(0)


from PyQt5.QtWidgets import (
    QWidget, QTabWidget
)
from PyQt5.QtCore import Qt


class RuntimeTab(QWidget):
    """
    Tab for displaying runtime analysis results.

    This tab contains two sub-tabs: one for summary metrics and another for detailed branch execution analysis.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.runtime_tree_summary = None
        self.runtime_tree_details = None
        self.initUI()

    def initUI(self):
        """
        Initialize the RuntimeTab by creating a QTabWidget containing summary and detail views.
        """
        from PyQt5.QtWidgets import QVBoxLayout  # Local import for clarity
        layout = QVBoxLayout(self)

        # Create the main tab widget for runtime data.
        self.tabs = QTabWidget(self)

        # --- Summary Tab ---
        self.summary_tab = QWidget()
        summary_layout = QVBoxLayout(self.summary_tab)

        # Define columns and headers for the summary table.
        summary_cols = ("Function", "Tests", "Avg CPU Time", "Total Wall Time",
                        "Avg Mem", "Function Calls", "Branch Executions")
        summary_heads = ["Function Name", "Tests Run", "Avg CPU Time (s)", "Total Wall Time (s)",
                         "Avg Mem (KB)", "Function Calls", "Branch Executions"]

        self.runtime_tree_summary = self.create_treewidget(summary_cols, summary_heads)
        summary_layout.addWidget(self.runtime_tree_summary)

        # --- Execution Details Tab ---
        self.details_tab = QWidget()
        details_layout = QVBoxLayout(self.details_tab)

        # Define columns and headers for the execution details table.
        details_cols = ("Function", "Branch Execution Frequency", "Coverage Map")
        details_heads = ["Function Name", "Branch Execution Frequency", "Coverage Map"]

        self.runtime_tree_details = self.create_treewidget(details_cols, details_heads)
        details_layout.addWidget(self.runtime_tree_details)

        # Add both tabs to the main QTabWidget.
        self.tabs.addTab(self.summary_tab, "Summary")
        self.tabs.addTab(self.details_tab, "Branch Execution Details")

        layout.addWidget(self.tabs)

    def create_treewidget(self, cols, headers):
        """
        Helper function to create a QTreeWidget with specified columns and headers.
        This widget supports auto-resizing and scrolling.

        Parameters:
            cols: Tuple of column identifiers.
            headers: List of header titles.

        Returns:
            QTreeWidget: Configured tree widget instance.
        """
        from PyQt5.QtWidgets import QTreeWidget, QHeaderView  # Local imports
        tree = QTreeWidget()
        tree.setColumnCount(len(cols))
        tree.setHeaderLabels(headers)
        tree.setAlternatingRowColors(True)
        tree.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Configure header section resizing for optimal display.
        header = tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Auto-fit first column
        header.setSectionResizeMode(1, QHeaderView.Stretch)  # Stretch second column
        header.setSectionResizeMode(QHeaderView.ResizeToContents)  # Default resizing for other columns

        return tree

    def update_runtime(self, runtime):
        """
        Update runtime analysis tables with new data.

        Parameters:
            runtime (dict): Dictionary containing runtime metrics for functions.
        """
        from PyQt5.QtWidgets import QTreeWidgetItem  # Local import for clarity
        # Clear previous content.
        self.runtime_tree_summary.clear()
        self.runtime_tree_details.clear()

        funcs = runtime.get("functions", [])

        # Iterate through each function's runtime data.
        for f in funcs:
            func_name = f.get("func_name", "").strip()
            if not func_name:
                continue  # Skip functions without a valid name

            test_results = f.get("test_results", {})

            for iters, data in test_results.items():
                runtime_data = data.get("runtime", {})

                # Extract summary metrics.
                avg_cpu = float(runtime_data.get("avg_cpu_time", 0.0))
                total_wall = float(runtime_data.get("total_wall_time", 0.0))
                avg_mem = float(runtime_data.get("avg_memory_kb", 0.0))
                function_calls = runtime_data.get("function_calls", 0)
                branch_executions = runtime_data.get("branch_executions", 0)

                # Create a summary row item with formatted runtime data.
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

                # --- Execution Details Data ---
                branch_exec_freq = runtime_data.get("branch_execution_frequency", {})

                # Format branch execution frequency: display each branch on a new line.
                formatted_branch_exec = "\n".join(
                    f"{start} → {end} ({count}x)" for (start, end), count in sorted(
                        branch_exec_freq.items(), key=lambda x: x[1], reverse=True
                    )
                ) if branch_exec_freq else "N/A"

                coverage_map = runtime_data.get("executed_lines", {})
                formatted_coverage = "\n".join(
                    f"{file}: {sorted(lines)}" for file, lines in coverage_map.items()
                )

                # Create a details row item.
                details_item = QTreeWidgetItem([
                    func_name,
                    formatted_branch_exec,
                    formatted_coverage
                ])
                # Add tooltips to show full details on hover.
                details_item.setToolTip(1, formatted_branch_exec)
                details_item.setToolTip(2, formatted_coverage)

                self.runtime_tree_details.addTopLevelItem(details_item)

        # Refresh the UI components.
        self.runtime_tree_summary.viewport().update()
        self.runtime_tree_details.viewport().update()


class DataFlowTab(QWidget):
    """
    Tab for displaying data flow analysis.

    This tab uses a tree widget to list files and their functions, along with
    data flow details such as variable reads, writes, and function calls.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dataflow_tree = None
        self.initUI()

    def initUI(self):
        """
        Initialize the DataFlowTab UI with a tree widget.
        """
        from PyQt5.QtWidgets import QVBoxLayout  # Local import for clarity
        layout = QVBoxLayout(self)

        self.dataflow_tree = QTreeWidget()
        self.dataflow_tree.setHeaderLabels(["Element", "Details"])

        # Configure header resizing: left column auto-resizes; right column stretches.
        header = self.dataflow_tree.header()
        from PyQt5.QtWidgets import QHeaderView
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)

        layout.addWidget(self.dataflow_tree)
        self.setLayout(layout)

    def update_dataflow(self, data_flow):
        """
        Populate the data flow tree with analysis results.

        Parameters:
            data_flow (dict): Dictionary mapping files to functions and their data flow details.
        """
        from PyQt5.QtWidgets import QTreeWidgetItem  # Local import for clarity
        self.dataflow_tree.clear()

        if not data_flow:
            no_data_item = QTreeWidgetItem(["No data flow analysis available.", ""])
            self.dataflow_tree.addTopLevelItem(no_data_item)
            return

        # Iterate over each file and its associated functions.
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

                # Process various dependency types (reads, writes, etc.) with corresponding emojis.
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
                            # Handle qualified names (e.g., module.function).
                            if "." in fc:
                                left, right = fc.split(".", 1)
                                left = denormalize(left)
                                fc = left + "." + right
                            else:
                                fc = denormalize(fc)
                            dep_values_denorm.append(fc)

                        dep_item = QTreeWidgetItem([emoji, ", ".join(dep_values_denorm)])
                        func_item.addChild(dep_item)

                # Handle input-output relations separately.
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

        # Expand all tree nodes to reveal the full structure.
        self.dataflow_tree.expandAll()


# --- RefactorTab: Handles display and application of refactoring plans ---
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem, QHeaderView,
    QPushButton, QHBoxLayout, QMessageBox, QDialog, QTextEdit
)
from PyQt5.QtCore import Qt


class RefactorTab(QWidget):
    """
    Tab for managing and applying refactoring plans.

    This tab allows users to preview and apply code refactoring suggestions
    generated by the analysis controller.
    """

    def __init__(self, parent=None, controller=None):
        """
        Initialize the RefactorTab.

        Parameters:
            parent: Parent widget.
            controller: An instance of CodeAnalysisController responsible for processing refactor plans.
        """
        super().__init__(parent)
        self.controller = controller  # Controller to handle refactoring logic.
        self.refactor_tree = None  # Tree widget to display refactor plans.
        self.preview_button = None
        self.apply_button = None

        # The refactor tree will store plan objects/dictionaries in its UserRole.
        self.initUI()

    def set_controller(self, controller):
        """
        Set or update the controller after the tab has been instantiated.

        Parameters:
            controller: An instance of CodeAnalysisController.
        """
        self.controller = controller

    def initUI(self):
        """
        Initialize the UI components for the RefactorTab.
        """
        main_layout = QVBoxLayout(self)

        # Create the QTreeWidget for listing refactoring plans.
        self.refactor_tree = QTreeWidget()
        self.refactor_tree.setColumnCount(4)
        self.refactor_tree.setHeaderLabels(["Select", "Plan Type", "Location", "Details"])
        self.refactor_tree.setAlternatingRowColors(True)
        self.refactor_tree.setRootIsDecorated(False)
        self.refactor_tree.setSelectionMode(QTreeWidget.NoSelection)

        # Configure header resizing for each column.
        header = self.refactor_tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Stretch)

        main_layout.addWidget(self.refactor_tree)

        # Create a horizontal layout for the action buttons.
        button_layout = QHBoxLayout()
        self.preview_button = QPushButton("Preview")
        # Connect the preview button to display a preview of changes.
        self.preview_button.clicked.connect(self.on_preview_changes)
        button_layout.addWidget(self.preview_button)

        self.apply_button = QPushButton("Apply Refactor")
        # Connect the apply button to trigger the refactoring process.
        self.apply_button.clicked.connect(self.on_apply_refactor)
        button_layout.addWidget(self.apply_button)

        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

    def update_refactor_plans(self, plans):
        """
        Display a list of refactoring plans in the tree widget.

        Parameters:
            plans (list): A list of plan objects/dictionaries from the controller.
        """
        from PyQt5.QtWidgets import QTreeWidgetItem  # Local import for clarity
        self.refactor_tree.clear()

        if not plans:
            no_item = QTreeWidgetItem(["", "No plans", "", ""])
            self.refactor_tree.addTopLevelItem(no_item)
            return

        for plan in plans:
            # Retrieve plan attributes from either object properties or dictionary keys.
            plan_type = getattr(plan, "plan_type", None) or plan.get("plan_type", "N/A")
            file_path = getattr(plan, "file_path", None) or plan.get("file_path", "")
            start_line = getattr(plan, "start_line", None) or plan.get("start_line", 0)
            end_line = getattr(plan, "end_line", None) or plan.get("end_line", 0)
            location_str = f"{file_path}:{start_line}-{end_line}"

            extra_info = getattr(plan, "extra_info", None) or plan.get("extra_info", {})
            details = extra_info.get("suggested_text", "")

            row_item = QTreeWidgetItem(["", plan_type, location_str, details])
            # Add a checkable item for user selection.
            row_item.setCheckState(0, Qt.Unchecked)
            # Store the complete plan object in the UserRole for later retrieval.
            row_item.setData(0, Qt.UserRole, plan)
            self.refactor_tree.addTopLevelItem(row_item)

    def on_preview_changes(self):
        """
        Handle the preview action for refactoring changes.

        This method retrieves selected refactoring plans and requests a preview
        from the controller. It then displays the preview diff in a dialog.
        """
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton  # Local imports
        checked_plans = self._get_checked_plans()
        if not checked_plans:
            QMessageBox.information(self, "No Selection", "No refactoring plans selected.")
            return

        if not self.controller:
            QMessageBox.warning(self, "No Controller", "No CodeAnalysisController is set.")
            return

        # For demonstration, only preview the first selected plan.
        plan = checked_plans[0]

        # Call the controller's preview method if available.
        if hasattr(self.controller, "preview_refactor_plan"):
            diff_text = self.controller.preview_refactor_plan(plan)
        else:
            diff_text = "No preview method implemented in the controller."

        # Display the preview diff in a modal dialog.
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Preview: {getattr(plan, 'plan_type', 'N/A')}")
        layout = QVBoxLayout(dialog)
        diff_edit = QTextEdit()
        diff_edit.setReadOnly(True)
        diff_edit.setPlainText(diff_text)
        layout.addWidget(diff_edit)

        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.close)
        layout.addWidget(close_button)

        dialog.setLayout(layout)
        dialog.resize(800, 600)
        dialog.exec_()

    def on_apply_refactor(self):
        """
        Handle the application of selected refactoring plans.

        This method gathers checked plans and delegates the refactoring process
        to the controller.
        """
        checked_plans = self._get_checked_plans()
        if not checked_plans:
            QMessageBox.information(self, "No Selection", "No refactoring plans selected.")
            return

        if not self.controller:
            QMessageBox.warning(self, "No Controller", "No CodeAnalysisController is set.")
            return

        # Call the controller's method to apply refactoring.
        self.controller.apply_refactoring(checked_plans)
        QMessageBox.information(self, "Refactoring Complete", f"Applied {len(checked_plans)} plan(s).")

    def _get_checked_plans(self):
        """
        Helper method to retrieve refactoring plans that have been selected (checked) by the user.

        Returns:
            list: A list of plan objects/dictionaries corresponding to checked items.
        """
        checked_plans = []
        for i in range(self.refactor_tree.topLevelItemCount()):
            item = self.refactor_tree.topLevelItem(i)
            if item.checkState(0) == Qt.Checked:
                plan = item.data(0, Qt.UserRole)
                if plan:
                    checked_plans.append(plan)
        return checked_plans
