import logging
import tkinter as tk
from tkinter import filedialog, messagebox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from ttkbootstrap import Style, ttk
from ttkbootstrap.constants import *


class ApplicationView:
    """
    Advanced GUI application for analyzing code similarity using static, token-based,
    and runtime analysis techniques. Provides interactive file selection, threshold adjustment,
    a summary dashboard with charts, and detailed analysis results in separate tabs.
    """
    def __init__(self, controller):
        self.controller = controller
        self.root = tk.Tk()
        self.style = Style(theme="cosmo")  # Use a Bootstrap-like theme
        self.root.title("Code Similarity Analyzer")
        self.root.geometry("1000x900")
        self.root.resizable(False, False)

        # Variables for user selection
        self.file_path = None
        self.folder_path = None
        self.threshold_value = 0.8

        # Dictionary to store analysis results for export, etc.
        self.analysis_results = {}

        self.create_widgets()

    def create_widgets(self):
        """Set up a professional, space-efficient UI layout."""
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        # --- Header Frame: File Selection & Controls ---
        header_frame = ttk.Frame(main)
        header_frame.pack(fill="x", pady=5)

        # Title
        ttk.Label(
            header_frame, text="Select a Python file or folder to analyze:", font=("Helvetica", 14, "bold")
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=5, pady=5)

        # --- File Selection Buttons ---
        file_buttons_frame = ttk.Frame(header_frame)
        file_buttons_frame.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="w")

        ttk.Button(file_buttons_frame, text="Browse Python File", command=self.browse_file, bootstyle=PRIMARY).pack(
            side="left", padx=5
        )
        ttk.Button(file_buttons_frame, text="Browse Project Folder", command=self.browse_folder,
                   bootstyle=PRIMARY).pack(
            side="left", padx=5
        )

        # --- Threshold Slider ---
        threshold_frame = ttk.Frame(header_frame)
        threshold_frame.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky="w")

        ttk.Label(threshold_frame, text="Set Similarity Threshold (0.5 - 1.0):", font=("Helvetica", 12)).pack(
            side="left", padx=5)
        self.threshold_slider = ttk.Scale(
            threshold_frame,
            from_=0.5,
            to=1.0,
            orient="horizontal",
            length=200,
            value=self.threshold_value,
            bootstyle=PRIMARY,
            command=lambda v: self.threshold_label.config(text=f"Threshold: {float(v):.2f}"),
        )
        self.threshold_slider.pack(side="left", padx=5)
        self.threshold_label = ttk.Label(threshold_frame, text=f"Threshold: {self.threshold_value:.2f}",
                                         font=("Helvetica", 10))
        self.threshold_label.pack(side="left", padx=5)

        # --- Number of Test Cases ---
        test_cases_frame = ttk.Frame(header_frame)
        test_cases_frame.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky="w")

        ttk.Label(test_cases_frame, text="Number of Test Cases:", font=("Helvetica", 12)).pack(side="left", padx=5)
        self.test_cases_entry = ttk.Entry(test_cases_frame, width=10)
        self.test_cases_entry.pack(side="left", padx=5)
        self.test_cases_entry.insert(0, "10")  # Default value

        # --- Progress Bar ---
        self.progress = ttk.Progressbar(header_frame, orient="horizontal", length=400, mode="indeterminate")
        self.progress.grid(row=4, column=0, columnspan=3, padx=5, pady=10, sticky="w")
        self.progress.stop()

        # --- Control Buttons ---
        buttons_frame = ttk.Frame(header_frame)
        buttons_frame.grid(row=5, column=0, columnspan=3, padx=5, pady=5, sticky="w")

        ttk.Button(buttons_frame, text="Start Analysis", command=self.on_start_analysis, bootstyle=SUCCESS).pack(
            side="left", padx=5
        )
        ttk.Button(buttons_frame, text="Restart", command=self.on_restart, bootstyle=DANGER).pack(side="left", padx=5)
        ttk.Button(buttons_frame, text="Export Report", command=self.export_report, bootstyle=INFO).pack(side="left",
                                                                                                         padx=5)

        # --- Notebook for Detailed Results ---
        self.notebook = ttk.Notebook(main)
        self.notebook.pack(fill="both", expand=True, pady=10)

        # Summary Tab
        self.summary_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.summary_tab, text="Summary")

        # Static Analysis Tab (Treeview)
        self.static_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.static_tab, text="Static Analysis")

        # Clone Detection Tab (Treeview)
        self.clone_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.clone_tab, text="Clone Detection")

        # Runtime Analysis Tab (Treeview)
        self.runtime_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.runtime_tab, text="Runtime")

        # Data Flow Tab (Text)
        self.dataflow_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.dataflow_tab, text="Data Flow")

        # Insights Tab (Text)
        self.insights_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.insights_tab, text="Insights")

        # Create widgets for each tab
        self.create_summary_widgets()
        self.create_static_widgets()
        self.create_clone_widgets()
        self.create_runtime_widgets()
        self.create_dataflow_widgets()
        self.create_insights_widgets()

    def create_summary_widgets(self):
        """Set up the summary dashboard, including key metrics and an embedded chart."""
        # Clear summary tab
        for widget in self.summary_tab.winfo_children():
            widget.destroy()

        # Create a frame for key metrics
        metrics_frame = ttk.LabelFrame(self.summary_tab, text="Key Metrics", padding=10)
        metrics_frame.pack(fill="x", padx=10, pady=10)

        self.summary_labels = {}
        for i, key in enumerate(["Total Static Pairs", "Avg Complexity", "Avg LOC", "Overall Similarity",
                                   "Total Runtime Functions", "Avg Execution Time", "Total Memory", "Peak Memory"]):
            lbl = ttk.Label(metrics_frame, text=f"{key}: N/A", font=("Helvetica", 10))
            lbl.grid(row=i // 2, column=i % 2, padx=10, pady=5, sticky="w")
            self.summary_labels[key] = lbl

        # Add a matplotlib chart for complexity distribution
        chart_frame = ttk.LabelFrame(self.summary_tab, text="Complexity Distribution", padding=10)
        chart_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.chart_frame = chart_frame

    def create_static_widgets(self):
        """Set up a Treeview widget for static analysis results."""
        self.static_tree = ttk.Treeview(self.static_tab, columns=("Function", "Complexity", "LOC", "Params", "Nesting"),
                                        show="headings", selectmode="browse")
        for col, heading in zip(self.static_tree["columns"],
                                ["Function", "Complexity", "LOC", "Params", "Nesting Depth"]):
            self.static_tree.heading(col, text=heading)
            self.static_tree.column(col, anchor="center", width=120)
        self.static_tree.pack(fill="both", expand=True, padx=10, pady=10)

        # Add a vertical scrollbar
        vsb = ttk.Scrollbar(self.static_tab, orient="vertical", command=self.static_tree.yview)
        self.static_tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")

    def create_clone_widgets(self):
        """Set up a Treeview widget for clone detection results."""
        self.clone_tree = ttk.Treeview(self.clone_tab, columns=("Pair", "Token Sim", "AST Sim", "Recommendation"),
                                       show="headings", selectmode="browse")
        for col, heading in zip(self.clone_tree["columns"],
                                ["Function Pair", "Token Similarity", "AST Similarity", "Recommendation"]):
            self.clone_tree.heading(col, text=heading)
            self.clone_tree.column(col, anchor="center", width=150)
        self.clone_tree.pack(fill="both", expand=True, padx=10, pady=10)
        vsb = ttk.Scrollbar(self.clone_tab, orient="vertical", command=self.clone_tree.yview)
        self.clone_tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")

    def create_runtime_widgets(self):
        """Set up a Treeview widget for runtime analysis results."""
        self.runtime_tree = ttk.Treeview(self.runtime_tab,
                                         columns=(
                                             "Function", "Tests", "Total Time", "Avg Time", "Memory", "Peak Memory", "CPU Usage",

                                         ),
                                         show="headings", selectmode="browse")
        for col, heading in zip(self.runtime_tree["columns"],
                                ["Function Name", "Tests Run", "Total Time (s)", "Avg Time (s)", "Memory (KB)", "Peak Memory (KB)",
                                 "CPU Usage (%)"]):
            self.runtime_tree.heading(col, text=heading)
            self.runtime_tree.column(col, anchor="center", width=130)
        self.runtime_tree.pack(fill="both", expand=True, padx=10, pady=10)
        vsb = ttk.Scrollbar(self.runtime_tab, orient="vertical", command=self.runtime_tree.yview)
        self.runtime_tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")

    def create_dataflow_widgets(self):
        """Set up a Text widget for data flow analysis details."""
        self.dataflow_text = tk.Text(self.dataflow_tab, wrap="word", font=("Consolas", 10), bg="#f8f9fa")
        self.dataflow_text.pack(fill="both", expand=True, padx=10, pady=10)
        vsb = ttk.Scrollbar(self.dataflow_tab, orient="vertical", command=self.dataflow_text.yview)
        self.dataflow_text.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")

    def create_insights_widgets(self):
        """Set up a Text widget for insights/cross-validation details."""
        self.insights_text = tk.Text(self.insights_tab, wrap="word", font=("Consolas", 10), bg="#f8f9fa")
        self.insights_text.pack(fill="both", expand=True, padx=10, pady=10)
        vsb = ttk.Scrollbar(self.insights_tab, orient="vertical", command=self.insights_text.yview)
        self.insights_text.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")

    def browse_file(self):
        """Open file dialog to select a Python file."""
        path = filedialog.askopenfilename(filetypes=[("Python Files", "*.py")])
        if path:
            self.file_path = path
            self.folder_path = None
            message = f"Selected file: {path}"
            messagebox.showinfo("File Selected", message)

    def browse_folder(self):
        """Open directory dialog to select a project folder."""
        path = filedialog.askdirectory()
        if path:
            self.folder_path = path
            self.file_path = None
            message = f"Selected folder: {path}"
            messagebox.showinfo("Folder Selected", message)

    def on_start_analysis(self):
        """Run the analysis and update all tabs accordingly."""
        if not (self.file_path or self.folder_path):
            messagebox.showerror("Error", "No file or folder selected!")
            return

        try:
            threshold = float(self.threshold_slider.get())
            num_tests = int(self.test_cases_entry.get())  # Get user input for number of test cases

            if num_tests <= 0:
                raise ValueError("Number of test cases must be greater than zero.")

            analysis_path = self.file_path if self.file_path else self.folder_path
            self.progress.start()

            # ✅ Pass num_tests to start_analysis
            results = self.controller.start_analysis(analysis_path, threshold, num_tests)

            # Save results for export
            self.analysis_results = results

            # Update Summary Dashboard
            self.update_summary(results)
            # Update Static Analysis tab
            self.update_static(results.get("static", []))
            # Update Clone Detection tab
            self.update_clone(results.get("token_clones", []))
            # Update Runtime tab
            self.update_runtime(results.get("runtime", {}))
            # Update Data Flow tab
            self.update_dataflow(results.get("data_flow", {}))
            # Update Insights tab
            self.update_insights(results.get("insights", []))

        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid positive integer for test cases.")
        except Exception as e:
            messagebox.showerror("Analysis Error", str(e))
        finally:
            self.progress.stop()

    def update_summary(self, results):
        """Update the summary dashboard with key metrics and update the embedded chart."""
        # Compute metrics from static analysis results
        static_results = results.get("static", [])
        total_static = len(static_results)
        complexities = []
        locs = []
        similarities = []
        for result in static_results:
            f1 = result.get("function1_metrics", {})
            f2 = result.get("function2_metrics", {})
            try:
                complexities.append(float(f1.get("complexity", 0)))
                complexities.append(float(f2.get("complexity", 0)))
                locs.append(float(f1.get("loc", 0)))
                locs.append(float(f2.get("loc", 0)))
            except Exception:
                pass
            similarities.append(result.get("similarity", 0))
        avg_complexity = sum(complexities) / len(complexities) if complexities else 0
        avg_loc = sum(locs) / len(locs) if locs else 0
        overall_similarity = sum(similarities) / len(similarities) if similarities else 0

        # Compute metrics from runtime analysis
        runtime = results.get("runtime", {})
        runtime_funcs = runtime.get("functions", [])
        total_runtime = len(runtime_funcs)
        avg_exec_time = sum([f.get("execution_time", 0) for f in runtime_funcs]) / total_runtime if total_runtime else 0
        total_memory = runtime.get("total_memory_usage", 0)
        peak_memory = runtime.get("peak_memory", 0)

        # Update label texts in summary dashboard
        self.summary_labels["Total Static Pairs"].config(text=f"Total Static Pairs: {total_static}")
        self.summary_labels["Avg Complexity"].config(text=f"Avg Complexity: {avg_complexity:.2f}")
        self.summary_labels["Avg LOC"].config(text=f"Avg LOC per function: {avg_loc:.1f}")
        self.summary_labels["Overall Similarity"].config(text=f"Overall Static Similarity: {overall_similarity:.2f}")
        self.summary_labels["Total Runtime Functions"].config(text=f"Total Runtime Functions: {total_runtime}")
        self.summary_labels["Avg Execution Time"].config(text=f"Avg Execution Time: {avg_exec_time:.6f} s")
        self.summary_labels["Total Memory"].config(text=f"Total Memory: {total_memory:.2f} KB")
        self.summary_labels["Peak Memory"].config(text=f"Peak Memory: {peak_memory:.2f} KB")

        # Plot a chart for complexity distribution
        self.plot_complexity_chart(complexities)

    def plot_complexity_chart(self, complexities):
        """Embed a matplotlib bar chart of complexity distribution in the summary tab."""
        # Clear previous chart
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        if not complexities:
            ttk.Label(self.chart_frame, text="No complexity data available.", font=("Helvetica", 10)).pack()
            return

        # Create bins for complexity distribution
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

    def update_static(self, static_results):
        """Populate the Static Analysis Treeview."""
        for item in self.static_tree.get_children():
            self.static_tree.delete(item)
        for result in static_results:
            f1 = result.get("function1_metrics", {})
            f2 = result.get("function2_metrics", {})
            similarity = result.get("similarity", 0)
            # Insert two rows per result
            self.static_tree.insert("", "end", values=(f1.get("name", ""),
                                                        f1.get("complexity", ""),
                                                        f1.get("loc", ""),
                                                        f1.get("parameter_count", ""),
                                                        f1.get("nesting_depth", "")))
            self.static_tree.insert("", "end", values=(f2.get("name", ""),
                                                        f2.get("complexity", ""),
                                                        f2.get("loc", ""),
                                                        f2.get("parameter_count", ""),
                                                        f2.get("nesting_depth", "")))
            # Optionally, insert a separator row or set a tag for grouping

    def update_clone(self, clone_results):
        """Populate the Clone Detection Treeview."""
        for item in self.clone_tree.get_children():
            self.clone_tree.delete(item)
        for res in clone_results:
            f1 = res.get("func1") or res.get("function1_metrics", {}).get("name", "")
            f2 = res.get("func2") or res.get("function2_metrics", {}).get("name", "")
            token_sim = res.get("token_similarity", 0)
            ast_sim = res.get("ast_similarity", res.get("similarity", 0))
            # Recommendation based on similarities:
            if ast_sim > 0.85 and token_sim > 0.85:
                rec = "✅ Strong candidate for refactoring"
            elif ast_sim > 0.85:
                rec = "🔹 Similar by AST"
            elif token_sim > 0.85:
                rec = "🔹 Similar by Token"
            else:
                rec = "⚠️ Not strong"
            pair = f"{f1} & {f2}"
            self.clone_tree.insert("", "end", values=(pair, f"{token_sim:.2f}", f"{ast_sim:.2f}", rec))

    import logging

    def update_runtime(self, runtime):
        """Populate the Runtime Analysis Treeview with one row per iteration count."""
        logging.debug("Called update_runtime function.")

        # Ensure the treeview is cleared before inserting new data
        for item in self.runtime_tree.get_children():
            self.runtime_tree.delete(item)

        # Retrieve function analysis results
        functions = runtime.get("functions", [])
        logging.debug(f"Extracted {len(functions)} functions from runtime data.")

        for f in functions:
            func_name = f.get("func_name", "Unknown Function")
            test_results = f.get("test_results", {})

            logging.debug(f"Processing function: {func_name}, Test results: {test_results}")

            for iter_count, data in test_results.items():
                avg_time = data.get("avg_cpu_time", 0)  # Correct key for CPU time
                total_time = data.get("total_wall_time", 0)  # Correct key for total execution time
                total_mem = data.get("total_memory_kb", 0)  # Correct key for total memory usage
                peak_mem = data.get("peak_memory_kb", 0)  # Correct key for peak memory
                cpu_usage_percent = data.get("cpu_usage_percent", 0)  # CPU usage percentage

                logging.debug(
                    f"[{func_name}] Iteration {iter_count}: "
                    f"Total Time: {total_time}, Avg Time: {avg_time}, "
                    f"Total Mem: {total_mem}, Peak Mem: {peak_mem}, CPU Usage: {cpu_usage_percent}"
                )

                # Insert row into Treeview with reordered columns
                self.runtime_tree.insert("", "end", values=(
                    func_name,  # Column 1: Function Name
                    iter_count,  # Column 2: Tests Run
                    total_time,  # Column 3: Total Time
                    avg_time,  # Column 4: Avg Time
                    total_mem,  # Column 5: Total Memory
                    peak_mem,  # Column 6: Peak Memory
                    cpu_usage_percent  # Column 7: CPU Usage
                ))

        logging.debug("Finished populating runtime analysis treeview.")

        logging.debug("Finished populating runtime analysis treeview.")

    def update_dataflow(self, data_flow):
        """Populate the Data Flow Text widget."""
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
                    self.dataflow_text.insert(tk.END, f"      {var}: Defined at line {info.get('defined', '')}, Used at lines {info.get('used', '')}\n")
                deps = details.get("dependencies", {})
                reads = deps.get("reads", [])
                writes = deps.get("writes", [])
                self.dataflow_text.insert(tk.END, f"    Dependencies:\n", "info")
                self.dataflow_text.insert(tk.END, f"      Reads: {', '.join(reads) if reads else 'None'}\n")
                self.dataflow_text.insert(tk.END, f"      Writes: {', '.join(writes) if writes else 'None'}\n")
                self.dataflow_text.insert(tk.END, "\n")
        self.dataflow_text.see(tk.END)

    def update_insights(self, insights):
        """Populate the Insights Text widget."""
        self.insights_text.delete("1.0", tk.END)
        if not insights:
            self.insights_text.insert(tk.END, "No insights available.\n")
            return
        for line in insights:
            self.insights_text.insert(tk.END, line + "\n\n")
        self.insights_text.see(tk.END)

    def export_report(self):
        """Export the complete analysis report (all tabs) to a text file."""
        report_lines = []
        report_lines.append("=== Summary ===")
        for key, lbl in self.summary_labels.items():
            report_lines.append(lbl.cget("text"))
        report_lines.append("\n=== Static Analysis ===")
        for child in self.static_tree.get_children():
            report_lines.append(" | ".join(self.static_tree.item(child)["values"]))
        report_lines.append("\n=== Clone Detection ===")
        for child in self.clone_tree.get_children():
            report_lines.append(" | ".join(self.clone_tree.item(child)["values"]))
        report_lines.append("\n=== Runtime Analysis ===")
        for child in self.runtime_tree.get_children():
            report_lines.append(" | ".join(str(v) for v in self.runtime_tree.item(child)["values"]))
        report_lines.append("\n=== Data Flow Analysis ===")
        report_lines.append(self.dataflow_text.get("1.0", tk.END))
        report_lines.append("\n=== Insights ===")
        report_lines.append(self.insights_text.get("1.0", tk.END))
        report = "\n".join(report_lines)

        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                 filetypes=[("Text files", "*.txt")],
                                                 title="Save Report As")
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(report)
                messagebox.showinfo("Export Successful", f"Report exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Export Error", str(e))

    def on_restart(self):
        """Reset application state to allow new analysis."""
        self.file_path = None
        self.folder_path = None
        # Clear header messages
        self.threshold_slider.set(0.8)
        self.threshold_label.config(text="Threshold: 0.80")
        # Clear all tabs
        for tab in [self.summary_tab, self.static_tab, self.clone_tab, self.runtime_tab, self.dataflow_tab, self.insights_tab]:
            for widget in tab.winfo_children():
                widget.destroy()
        # Recreate tabs
        self.create_summary_widgets()
        self.create_static_widgets()
        self.create_clone_widgets()
        self.create_runtime_widgets()
        self.create_dataflow_widgets()
        self.create_insights_widgets()
        messagebox.showinfo("Reset", "Application reset. Please select a new file to analyze.")

    def start(self):
        """Start the Tkinter main event loop."""
        self.root.mainloop()
