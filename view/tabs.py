# tabs.py
import tkinter as tk
from ttkbootstrap import Style, ttk
from ttkbootstrap.constants import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def create_treeview(parent, columns, headings, width=120):
    """Helper to create a Treeview + scrollbar."""
    tv = ttk.Treeview(parent, columns=columns, show="headings", selectmode="browse")
    for col, head in zip(columns, headings):
        tv.heading(col, text=head)
        tv.column(col, anchor="center", width=width)
    tv.pack(fill="both", expand=True, padx=10, pady=10)
    vsb = ttk.Scrollbar(parent, orient="vertical", command=tv.yview)
    tv.configure(yscrollcommand=vsb.set)
    vsb.pack(side="right", fill="y")
    return tv


class SummaryTab:
    def __init__(self, parent):
        self.tab_frame = ttk.Frame(parent)
        self.parent = parent

        # Key metrics
        self.summary_labels = {}
        self.refactor_text = None
        self.refactor_button = None
        self.chart_frame = None

        self._create_summary_widgets()

    def _create_summary_widgets(self):
        metrics_frame = ttk.LabelFrame(self.tab_frame, text="Key Metrics", padding=10)
        metrics_frame.pack(fill="x", padx=10, pady=10)

        # Example of creating summary labels
        for i, key in enumerate([
            "Total Static Pairs", "Avg Complexity", "Avg LOC", "Overall Similarity",
            "Total Runtime Functions", "Avg Execution Time", "Total Memory", "Peak Memory"
        ]):
            lbl = ttk.Label(metrics_frame, text=f"{key}: N/A", font=("Helvetica", 10))
            lbl.grid(row=i // 2, column=i % 2, padx=10, pady=5, sticky="w")
            self.summary_labels[key] = lbl

        # Complexity Chart Section
        self.chart_frame = ttk.LabelFrame(self.tab_frame, text="Complexity Distribution", padding=10)
        self.chart_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Refactoring suggestions
        self._create_refactor_section()

    def _create_refactor_section(self):
        refactor_frame = ttk.LabelFrame(self.tab_frame, text="🚀 Refactoring Suggestions", padding=10)
        refactor_frame.pack(fill="x", padx=10, pady=10)

        self.refactor_text = tk.Text(refactor_frame, height=6, wrap="word", font=("Consolas", 10), bg="#f8f9fa")
        self.refactor_text.pack(fill="both", expand=True, padx=10, pady=5)

        vsb = ttk.Scrollbar(refactor_frame, orient="vertical", command=self.refactor_text.yview)
        self.refactor_text.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")

        self.refactor_button = ttk.Button(refactor_frame, text="🛠 Refactor Code", bootstyle=SUCCESS)
        self.refactor_button.pack(pady=5)

    def update_summary(self, results: dict):
        """
        Put your logic for updating these summary labels and
        refactoring suggestions here.
        """
        # Example updates:
        # self.summary_labels["Total Static Pairs"].config(text="some updated text")
        pass

    def plot_complexity_chart(self, complexities):
        # Example if you want to embed a chart
        for w in self.chart_frame.winfo_children():
            w.destroy()
        if not complexities:
            ttk.Label(self.chart_frame, text="No complexity data available.").pack()
            return
        bins = range(0, int(max(complexities)) + 2)
        fig = Figure(figsize=(5, 3), dpi=100)
        ax = fig.add_subplot(111)
        ax.hist(complexities, bins=bins, color="#17a2b8", edgecolor="black")
        ax.set_title("Complexity Distribution")
        ax.set_xlabel("Complexity")
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


class StaticTab:
    def __init__(self, parent):
        self.tab_frame = ttk.Frame(parent)
        self.static_tree = None
        self._create_static_widgets()

    def _create_static_widgets(self):
        cols = ("Function", "Complexity", "LOC", "Params", "Nesting")
        heads = ["Function", "Complexity", "LOC", "Params", "Nesting Depth"]
        self.static_tree = create_treeview(self.tab_frame, cols, heads)

    def update_static(self, static_results):
        for item in self.static_tree.get_children():
            self.static_tree.delete(item)
        for r in static_results:
            f1, f2 = r.get("function1_metrics", {}), r.get("function2_metrics", {})
            self.static_tree.insert("", "end", values=(
                f1.get("name", ""), f1.get("complexity", ""), f1.get("loc", ""),
                f1.get("parameter_count", ""), f1.get("nesting_depth", "")
            ))
            self.static_tree.insert("", "end", values=(
                f2.get("name", ""), f2.get("complexity", ""), f2.get("loc", ""),
                f2.get("parameter_count", ""), f2.get("nesting_depth", "")
            ))


class CloneTab:
    def __init__(self, parent):
        self.tab_frame = ttk.Frame(parent)
        self.clone_tree = None
        self._create_clone_widgets()

    def _create_clone_widgets(self):
        cols = ("Pair", "Token Sim", "AST Sim")
        heads = ["Function Pair", "Token Similarity", "AST Similarity"]
        self.clone_tree = create_treeview(self.tab_frame, cols, heads, width=150)

    def update_clone(self, clone_results):
        for item in self.clone_tree.get_children():
            self.clone_tree.delete(item)
        for res in clone_results:
            f1 = res.get("func1") or res.get("function1_metrics", {}).get("name", "")
            f2 = res.get("func2") or res.get("function2_metrics", {}).get("name", "")
            token_sim = res.get("token_similarity", 0)
            ast_sim = res.get("ast_similarity", res.get("similarity", 0))
            pair = f"{f1} & {f2}"
            self.clone_tree.insert("", "end", values=(pair, f"{token_sim:.2f}", f"{ast_sim:.2f}",))


class RuntimeTab:
    def __init__(self, parent):
        self.tab_frame = ttk.Frame(parent)
        self.runtime_tree = None
        self._create_runtime_widgets()

    def _create_runtime_widgets(self):
        cols = (
            "Function", "Tests", "AvgCpuTime", "TotalWallTime", "CpuUsage",
            "AvgMem", "PeakMem", "AvgWallPerCall", "StdDevWallPerCall"
        )
        heads = [
            "Function Name", "Tests Run", "Avg CPU Time (s)", "Total Wall Time (s)",
            "CPU Usage (%)", "Avg Mem (KB)", "Peak Mem (KB)", "Avg Wall/Call (s)", "StdDev Wall/Call (s)"
        ]
        self.runtime_tree = create_treeview(self.tab_frame, cols, heads, width=130)

    def update_runtime(self, runtime):
        for item in self.runtime_tree.get_children():
            self.runtime_tree.delete(item)
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
                self.runtime_tree.insert("", "end", values=(
                    func_name, iters, f"{avg_cpu:.8f}", f"{total_wall:.8f}",
                    f"{cpu_usage:.2f}", f"{avg_mem:.6f}", f"{peak_mem:.6f}",
                    f"{avg_wall_call:.8f}", f"{stdev_wall_call:.8f}"
                ))


class DataFlowTab:
    def __init__(self, parent):
        self.tab_frame = ttk.Frame(parent)
        self.dataflow_text = None
        self._create_dataflow_widgets()

    def _create_dataflow_widgets(self):
        self.dataflow_text = tk.Text(self.tab_frame, wrap="word", font=("Consolas", 10), bg="#f8f9fa")
        self.dataflow_text.pack(fill="both", expand=True, padx=10, pady=10)
        vsb = ttk.Scrollbar(self.tab_frame, orient="vertical", command=self.dataflow_text.yview)
        self.dataflow_text.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")

    def update_dataflow(self, data_flow):
        self.dataflow_text.delete("1.0", tk.END)
        if not data_flow:
            self.dataflow_text.insert(tk.END, "No data flow analysis available.\n")
            return
        for file, funcs in data_flow.items():
            self.dataflow_text.insert(tk.END, f"File: {file}\n", "header")
            for func, details in funcs.items():
                self.dataflow_text.insert(tk.END, f"  Function: {func}\n", "info")
                self.dataflow_text.insert(tk.END, "    Variables:\n", "info")
                for var, info in details.get("variables", {}).items():
                    self.dataflow_text.insert(
                        tk.END,
                        f"      {var}: Defined at line {info.get('defined', '')}, "
                        f"Used at lines {info.get('used', '')}\n"
                    )
                deps = details.get("dependencies", {})
                reads, writes = deps.get("reads", []), deps.get("writes", [])
                self.dataflow_text.insert(tk.END, "    Dependencies:\n", "info")
                self.dataflow_text.insert(tk.END, f"      Reads: {', '.join(reads) if reads else 'None'}\n")
                self.dataflow_text.insert(tk.END, f"      Writes: {', '.join(writes) if writes else 'None'}\n\n")
        self.dataflow_text.see(tk.END)
