# mainView.py
import tkinter as tk
from tkinter import filedialog, messagebox
from ttkbootstrap import Style, ttk
from ttkbootstrap.constants import *

from view.tabs import (
    SummaryTab,
    StaticTab,
    CloneTab,
    RuntimeTab,
    DataFlowTab
)


class ApplicationView:
    """GUI for analyzing code similarity and presenting results."""

    def __init__(self, controller):
        self.controller = controller
        self.root = tk.Tk()
        self.style = Style(theme="flatly")
        self.root.title("Code Similarity Analyzer")
        self.root.geometry("1100x800")
        self.root.resizable(True, True)

        self.file_path = None
        self.folder_path = None
        self.threshold_value = 0.8
        self.analysis_results = {}

        # For the test cases
        self.test_cases_entry = None
        self.threshold_slider = None
        self.threshold_label = None
        self.progress = None

        # Tabs
        self.summary_tab = None
        self.static_tab = None
        self.clone_tab = None
        self.runtime_tab = None
        self.dataflow_tab = None
        self.notebook = None

        self.create_widgets()

    def create_widgets(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        # Header controls
        self.create_header_widgets(main)

        # Notebook
        self.notebook = ttk.Notebook(main)
        self.notebook.pack(fill="both", expand=True, pady=10)

        # Instantiate tab classes
        self.summary_tab = SummaryTab(self.notebook)
        self.static_tab = StaticTab(self.notebook)
        self.clone_tab = CloneTab(self.notebook)
        self.runtime_tab = RuntimeTab(self.notebook)
        self.dataflow_tab = DataFlowTab(self.notebook)

        # Add each tab class’s frame to the Notebook
        self.notebook.add(self.summary_tab.tab_frame, text="Summary")
        self.notebook.add(self.static_tab.tab_frame, text="Static Analysis")
        self.notebook.add(self.clone_tab.tab_frame, text="Clone Detection")
        self.notebook.add(self.runtime_tab.tab_frame, text="Runtime")
        self.notebook.add(self.dataflow_tab.tab_frame, text="Data Flow")

    def create_header_widgets(self, parent):
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill="x", pady=5)

        ttk.Label(header_frame, text="Select a Python File or Folder:", font=("Helvetica", 14, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", padx=10, pady=5
        )
        ttk.Button(header_frame, text="Browse File", command=self.browse_file, bootstyle=PRIMARY).grid(
            row=1, column=0, padx=10, pady=5, sticky="w"
        )
        ttk.Button(header_frame, text="Browse Folder", command=self.browse_folder, bootstyle=PRIMARY).grid(
            row=1, column=1, padx=10, pady=5, sticky="w"
        )

        ttk.Label(header_frame, text="Threshold:", font=("Helvetica", 12)).grid(
            row=2, column=0, padx=10, pady=5, sticky="w"
        )
        self.threshold_slider = ttk.Scale(
            header_frame, from_=0.5, to=1.0, orient="horizontal", length=200,
            value=self.threshold_value, bootstyle=PRIMARY,
            command=lambda v: self.threshold_label.config(text=f"{float(v):.2f}")
        )
        self.threshold_slider.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.threshold_label = ttk.Label(header_frame, text=f"{self.threshold_value:.2f}", font=("Helvetica", 10))
        self.threshold_label.grid(row=2, column=2, padx=5, pady=5, sticky="w")

        ttk.Label(header_frame, text="Test Cases:", font=("Helvetica", 12)).grid(
            row=3, column=0, padx=10, pady=5, sticky="w"
        )
        self.test_cases_entry = ttk.Entry(header_frame, width=10)
        self.test_cases_entry.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        self.test_cases_entry.insert(0, "10")

        self.progress = ttk.Progressbar(header_frame, orient="horizontal", length=400, mode="indeterminate")
        self.progress.grid(row=4, column=0, columnspan=3, padx=10, pady=10, sticky="w")

        ttk.Button(header_frame, text="Start Analysis", command=self.on_start_analysis, bootstyle=SUCCESS).grid(
            row=5, column=0, padx=10, pady=5, sticky="w"
        )
        ttk.Button(header_frame, text="Restart", command=self.on_restart, bootstyle=DANGER).grid(
            row=5, column=1, padx=10, pady=5, sticky="w"
        )
        ttk.Button(header_frame, text="Export Report", command=self.export_report, bootstyle=INFO).grid(
            row=5, column=2, padx=10, pady=5, sticky="w"
        )

    def browse_file(self):
        path = filedialog.askopenfilename(filetypes=[("Python Files", "*.py")])
        if path:
            self.file_path, self.folder_path = path, None
            messagebox.showinfo("File Selected", f"Selected file: {path}")

    def browse_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.folder_path, self.file_path = path, None
            messagebox.showinfo("Folder Selected", f"Selected folder: {path}")

    def on_start_analysis(self):
        if not (self.file_path or self.folder_path):
            messagebox.showerror("Error", "No file or folder selected!")
            return
        try:
            threshold = float(self.threshold_slider.get())
            num_tests = int(self.test_cases_entry.get())
            if num_tests <= 0:
                raise ValueError("Number of test cases must be > 0.")
            analysis_path = self.file_path if self.file_path else self.folder_path

            self.progress.start()
            results = self.controller.start_analysis(analysis_path, threshold, num_tests)
            self.analysis_results = results

            # --- Update each tab ---
            # 1. Summary
            self.summary_tab.update_summary(results)
            # 2. Static
            self.static_tab.update_static(results.get("static", []))
            # 3. Clones
            self.clone_tab.update_clone(results.get("token_clones", []))
            # 4. Runtime
            self.runtime_tab.update_runtime(results.get("runtime", {}))
            # 5. Data Flow
            self.dataflow_tab.update_dataflow(results.get("data_flow", {}))

        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid positive integer for test cases.")
        except Exception as e:
            messagebox.showerror("Analysis Error", str(e))
        finally:
            self.progress.stop()

    def on_restart(self):
        """Reset application state to allow new analysis."""
        self.file_path = None
        self.folder_path = None
        self.threshold_slider.set(0.8)
        self.threshold_label.config(text=f"{0.8:.2f}")

        # Clear all tab data
        self.static_tab.update_static([])
        self.clone_tab.update_clone([])
        self.runtime_tab.update_runtime({})
        self.dataflow_tab.update_dataflow({})
        self.summary_tab.update_summary({})

        messagebox.showinfo("Reset", "Application reset. Please select a new file/folder to analyze.")

    def export_report(self):
        # Just a skeleton for exporting. You can fill in the needed data from each tab.
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
            title="Save Report As"
        )
        if not file_path:
            return

        try:
            lines = ["=== Summary ==="]
            # Grab data from summary tab’s labels or text
            for k, lbl in self.summary_tab.summary_labels.items():
                lines.append(lbl.cget("text"))

            lines.append("\n=== Static Analysis ===")
            # Similarly, read data from self.static_tab’s tree
            for child in self.static_tab.static_tree.get_children():
                lines.append(" | ".join(map(str, self.static_tab.static_tree.item(child)["values"])))

            # And so on for clone, runtime, data flow...
            report = "\n".join(lines)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(report)
            messagebox.showinfo("Export Successful", f"Report exported to {file_path}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def start(self):
        self.root.mainloop()
