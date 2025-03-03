from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QGridLayout, QFileDialog, QMessageBox, QTabWidget, QSlider, QLineEdit,
    QProgressBar, QStyleFactory, QScrollArea, QFrame
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon

# Assuming these tab classes are available and have been converted to PyQt5.
# They should expose similar update_* methods as in your original code.
from view.tabs_view import SummaryTab, StaticTab, CloneTab, RuntimeTab, DataFlowTab


class ApplicationView(QMainWindow):
    """GUI for analyzing code similarity and presenting results using PyQt5."""

    def __init__(self, controller):
        super().__init__()
        self.controller = controller

        self.file_path = None
        self.folder_path = None
        self.threshold_value = 0.8
        self.analysis_results = {}

        self.initUI()
        self.apply_modern_style()

    def initUI(self):
        self.setWindowTitle("Code Similarity Analyzer")
        self.resize(1200, 1000)

        # Create a scroll area that will allow the content to be scrolled if it's too long.
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.setCentralWidget(scroll_area)

        # Create a central widget and assign it to the scroll area.
        central_widget = QWidget()
        scroll_area.setWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        self.create_header_widgets(main_layout)

        self.notebook = QTabWidget()
        main_layout.addWidget(self.notebook)

        self.summary_tab = SummaryTab()
        self.static_tab = StaticTab()
        self.clone_tab = CloneTab()
        self.runtime_tab = RuntimeTab()
        self.dataflow_tab = DataFlowTab()

        self.notebook.addTab(self.summary_tab, "Summary")
        self.notebook.addTab(self.static_tab, "Static Analysis")
        self.notebook.addTab(self.clone_tab, "Similarity")
        self.notebook.addTab(self.runtime_tab, "Runtime")
        self.notebook.addTab(self.dataflow_tab, "Data Flow")

    def create_header_widgets(self, layout):
        header_widget = QWidget()
        header_layout = QGridLayout(header_widget)

        header_label = QLabel("Code Similarity Analysis Tool")
        header_label.setStyleSheet("font-size: 22px; font-weight: 600; color: #3498db;")
        header_layout.addWidget(header_label, 0, 0, 1, 4)

        btn_browse_file = QPushButton("Browse File")
        btn_browse_file.clicked.connect(self.browse_file)
        header_layout.addWidget(btn_browse_file, 1, 0)

        btn_browse_folder = QPushButton("Browse Folder")
        btn_browse_folder.clicked.connect(self.browse_folder)
        header_layout.addWidget(btn_browse_folder, 1, 1)

        threshold_label = QLabel("Similarity Threshold:")
        threshold_label.setStyleSheet("font-size: 14px; font-weight: 500;")
        header_layout.addWidget(threshold_label, 2, 0)

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(50)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(int(self.threshold_value * 100))
        self.threshold_slider.setTickInterval(1)
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)
        header_layout.addWidget(self.threshold_slider, 2, 1)

        self.threshold_value_label = QLabel(f"{self.threshold_value:.2f}")
        header_layout.addWidget(self.threshold_value_label, 2, 2)

        test_cases_label = QLabel("Number of Tests:")
        test_cases_label.setStyleSheet("font-size: 14px; font-weight: 500;")
        header_layout.addWidget(test_cases_label, 3, 0)

        self.test_cases_entry = QLineEdit()
        self.test_cases_entry.setFixedWidth(60)
        self.test_cases_entry.setText("10")
        header_layout.addWidget(self.test_cases_entry, 3, 1)

        self.progress = QProgressBar()
        self.progress.setMinimum(0)
        self.progress.setMaximum(0)
        self.progress.setVisible(False)
        header_layout.addWidget(self.progress, 4, 0, 1, 4)

        btn_start = QPushButton("Start Analysis")
        btn_start.clicked.connect(self.on_start_analysis)
        header_layout.addWidget(btn_start, 5, 0)

        btn_restart = QPushButton("Reset")
        btn_restart.clicked.connect(self.on_restart)
        header_layout.addWidget(btn_restart, 5, 1)

        btn_export = QPushButton("Export Report")
        btn_export.clicked.connect(self.export_report)
        header_layout.addWidget(btn_export, 5, 2)

        layout.addWidget(header_widget)

    def apply_modern_style(self):
        QApplication.setStyle(QStyleFactory.create("Fusion"))

        # Modern grey palette with blue accent
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor("#e0e0e0"))           # Light grey background
        palette.setColor(QPalette.WindowText, QColor("#2e2e2e"))         # Dark grey text
        palette.setColor(QPalette.Base, QColor("#ffffff"))               # White for text entry background
        palette.setColor(QPalette.AlternateBase, QColor("#e0e0e0"))
        palette.setColor(QPalette.ToolTipBase, QColor("#ffffff"))
        palette.setColor(QPalette.ToolTipText, QColor("#2e2e2e"))
        palette.setColor(QPalette.Text, QColor("#2e2e2e"))
        palette.setColor(QPalette.Button, QColor("#d0d0d0"))             # Medium grey buttons
        palette.setColor(QPalette.ButtonText, QColor("#2e2e2e"))
        palette.setColor(QPalette.BrightText, QColor("#ff0000"))
        palette.setColor(QPalette.Link, QColor("#3498db"))
        palette.setColor(QPalette.Highlight, QColor("#3498db"))
        palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
        QApplication.instance().setPalette(palette)

        # Set a modern system font
        font = QFont("Segoe UI", 10)
        QApplication.instance().setFont(font)

        # Update QPushButton style with a flat modern look and blue accent
        for btn in self.findChildren(QPushButton):
            btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #3498db;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 8px 16px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                }
                QPushButton:pressed {
                    background-color: #2471a3;
                }
                """
            )

        # Update QLineEdit style with a clean white background
        self.test_cases_entry.setStyleSheet(
            """
            QLineEdit {
                background-color: #ffffff;
                color: #2e2e2e;
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 5px;
            }
            QLineEdit:focus {
                border: 1px solid #3498db;
            }
            """
        )

        # Update QSlider style with a clean, flat design using lighter greys
        self.threshold_slider.setStyleSheet(
            """
            QSlider::groove:horizontal {
                border: 1px solid #aaa;
                height: 8px;
                background: #ccc;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                border: 1px solid #3498db;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            """
        )

        # Update QProgressBar style with soft grey tones
        self.progress.setStyleSheet(
            """
            QProgressBar {
                border: 2px solid #ccc;
                border-radius: 5px;
                text-align: center;
                background-color: #f0f0f0;
                color: #2e2e2e;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                width: 20px;
                border-radius: 5px;
            }
            """
        )

        # Update QTabWidget style for a modern grey look with blue accent on hover/selected
        self.notebook.setStyleSheet(
            """
            QTabBar::tab {
                background: #d0d0d0;
                color: #2e2e2e;
                border: 1px solid #aaa;
                border-bottom: none;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected, QTabBar::tab:hover {
                background: #3498db;
                color: white;
            }
            QTabWidget::pane {
                border: 1px solid #aaa;
                top: -1px;
                background: #f0f0f0;
            }
            """
        )

    def update_threshold_label(self, value):
        # Convert slider value (50-100) back to a float (0.5-1.0)
        threshold = value / 100.0
        self.threshold_value_label.setText(f"{threshold:.2f}")

    def browse_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Python File", "", "Python Files (*.py)"
        )
        if file_name:
            self.file_path = file_name
            self.folder_path = None
            QMessageBox.information(self, "File Selected", f"Selected file: {file_name}")

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.folder_path = folder
            self.file_path = None
            QMessageBox.information(self, "Folder Selected", f"Selected folder: {folder}")

    def on_start_analysis(self):
        if not (self.file_path or self.folder_path):
            QMessageBox.critical(self, "Error", "No file or folder selected!")
            return

        try:
            threshold = self.threshold_slider.value() / 100.0

            # Get and clean the input for test cases
            num_tests_text = self.test_cases_entry.text().strip()

            # Enhanced validation for positive integer
            if not num_tests_text.isdigit():
                raise ValueError("Test cases input is not a valid positive integer.")

            num_tests = int(num_tests_text)

            # Check if the number is positive
            if num_tests <= 0:
                raise ValueError("Number of test cases must be greater than 0.")

            analysis_path = self.file_path if self.file_path else self.folder_path

            # Show progress bar during analysis
            self.progress.setVisible(True)
            QApplication.processEvents()  # Force update of UI

            # Call the controller's analysis method
            results = self.controller.start_analysis(analysis_path, threshold, num_tests)

            self.analysis_results = results

            # Update each tab with its corresponding results
            self.summary_tab.update_summary(results)
            self.static_tab.update_static(results.get("static", []))
            self.clone_tab.update_clone(results.get("token_clones", []))
            self.runtime_tab.update_runtime(results.get("runtime", {}))
            self.dataflow_tab.update_dataflow(results.get("data_flow", {}))

            insights = results.get("analysis_insights", {})
            key_metrics = insights.get("key_metrics", {})
            self.summary_tab.update_key_metrics(key_metrics)
            self.summary_tab.show_top_complex_functions(insights.get("top_complex_functions", []))
            self.summary_tab.show_slowest_functions(insights.get("slowest_functions", []))
            self.summary_tab.show_duplicate_pairs(insights.get("top_duplicate_pairs", []))
            self.summary_tab.show_refactoring_suggestions(insights.get("refactoring_suggestions", []))

        except ValueError as ve:
            QMessageBox.critical(self, "Input Error", str(ve))

        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", str(e))

        finally:
            self.progress.setVisible(False)

    def on_restart(self):
        """Reset application state to allow new analysis."""
        self.file_path = None
        self.folder_path = None
        self.threshold_slider.setValue(80)
        self.threshold_value_label.setText(f"{0.8:.2f}")

        # Clear all tab data (assuming each tab has an update method to clear its content)
        self.static_tab.update_static([])
        self.clone_tab.update_clone([])
        self.runtime_tab.update_runtime({})
        self.dataflow_tab.update_dataflow({})
        self.summary_tab.update_summary({})

        QMessageBox.information(self, "Reset", "Application reset. Please select a new file/folder to analyze.")

    def export_report(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Report As", "", "Text Files (*.txt)"
        )
        if not file_path:
            return

        try:
            lines = ["=== Summary ==="]
            # Assuming summary_tab has a dictionary of QLabel objects named summary_labels
            if hasattr(self.summary_tab, 'summary_labels'):
                for lbl in self.summary_tab.summary_labels.values():
                    lines.append(lbl.text())

            lines.append("\n=== Static Analysis ===")
            # If static_tab uses a QTreeWidget called static_tree:
            if hasattr(self.static_tab, 'static_tree'):
                tree = self.static_tab.static_tree
                for i in range(tree.topLevelItemCount()):
                    item = tree.topLevelItem(i)
                    values = [item.text(col) for col in range(item.columnCount())]
                    lines.append(" | ".join(values))

            # Continue for other tabs as needed...
            report = "\n".join(lines)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(report)
            QMessageBox.information(self, "Export Successful", f"Report exported to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def start(self):
        self.show()


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for a modern look
    view = ApplicationView(controller=None)  # Replace with your actual controller
    view.start()
    sys.exit(app.exec_())
