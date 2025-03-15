import csv
import importlib
import os
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QGridLayout, QFileDialog, QMessageBox, QTabWidget, QSlider, QLineEdit,
    QProgressBar, QStyleFactory, QScrollArea
)

from controller.code_analysis_controller import CodeAnalysisController
# Import individual tab classes from the view package for a modular design
from view.tabs_view import (
    StaticTab,
    CloneTab,
    RuntimeTab,
    DataFlowTab,
    RefactorTab
)


def import_user_script(file_path):
    """
    Dynamically imports a user-selected Python file as a module.

    This function uses importlib to load a module from a given file path, allowing
    for runtime integration of user code.
    """
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    user_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = user_module
    spec.loader.exec_module(user_module)
    return user_module


class ApplicationView(QMainWindow):
    """
    Main application window for the Code Similarity Analyzer.

    This class sets up the PyQt5-based GUI, managing user interactions,
    dynamic loading of analysis modules, and display of various analysis results.
    """

    def __init__(self, controller):
        """
        Initialize the application view with its controller and default settings.
        """
        super().__init__()
        self.controller = controller

        # Paths for selected file or folder and threshold value for analysis
        self.file_path = None
        self.folder_path = None
        self.threshold_value = 0.0  # Default threshold value (scaled 0.0-1.0)
        self.analysis_results = {}

        # Initialize UI components and apply modern styling
        self.initUI()
        self.apply_modern_style()

    def initUI(self):
        """
        Set up the main window, including scroll area, header widgets, and tabs.
        """
        self.setWindowTitle("Code Similarity Analyzer")
        self.resize(1200, 1000)

        # Create a scrollable area to accommodate long content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.setCentralWidget(scroll_area)

        # Set up the central widget and its layout
        central_widget = QWidget()
        scroll_area.setWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create header controls (file selection, threshold, etc.)
        self.create_header_widgets(main_layout)

        # Initialize tabbed interface for displaying different analysis results.
        # The first tab will now be Similarity.
        self.notebook = QTabWidget()
        main_layout.addWidget(self.notebook)

        # Create each analysis tab instance
        self.clone_tab = CloneTab()
        self.static_tab = StaticTab()
        self.runtime_tab = RuntimeTab()
        self.dataflow_tab = DataFlowTab()
        self.refactor_tab = RefactorTab()

        # Add tabs to the QTabWidget. The Similarity tab (clone_tab) is now first.
        self.notebook.addTab(self.clone_tab, "Similarity")
        self.notebook.addTab(self.static_tab, "Static Analysis")
        self.notebook.addTab(self.runtime_tab, "Runtime")
        self.notebook.addTab(self.dataflow_tab, "Data Flow")
        self.notebook.addTab(self.refactor_tab, "Refactor")

    def create_header_widgets(self, layout):
        """
        Create header widgets for file/folder selection, threshold slider, test cases entry,
        progress bar, and action buttons.
        """
        header_widget = QWidget()
        header_layout = QGridLayout(header_widget)

        # Main header title
        header_label = QLabel("Code Similarity Analysis Tool")
        header_label.setStyleSheet("font-size: 22px; font-weight: 600; color: #3498db;")
        header_layout.addWidget(header_label, 0, 0, 1, 4)

        # Button to browse and select a Python file
        btn_browse_file = QPushButton("Browse File")
        btn_browse_file.clicked.connect(self.browse_file)
        header_layout.addWidget(btn_browse_file, 1, 0)

        # Button to browse and select a folder
        btn_browse_folder = QPushButton("Browse Folder")
        btn_browse_folder.clicked.connect(self.browse_folder)
        header_layout.addWidget(btn_browse_folder, 1, 1)

        # Label and slider for setting the similarity threshold
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

        # Label and entry for the number of tests to run
        test_cases_label = QLabel("Number of Tests:")
        test_cases_label.setStyleSheet("font-size: 14px; font-weight: 500;")
        header_layout.addWidget(test_cases_label, 3, 0)

        self.test_cases_entry = QLineEdit()
        self.test_cases_entry.setFixedWidth(60)
        self.test_cases_entry.setText("10")
        header_layout.addWidget(self.test_cases_entry, 3, 1)

        # Progress bar to indicate ongoing analysis operations
        self.progress = QProgressBar()
        self.progress.setMinimum(0)
        self.progress.setMaximum(0)
        self.progress.setVisible(False)
        header_layout.addWidget(self.progress, 4, 0, 1, 4)

        # Action buttons for starting analysis, resetting, and exporting report
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
        """
        Apply a modern visual style using Fusion style, custom palette, and font settings.
        """
        QApplication.setStyle(QStyleFactory.create("Fusion"))

        # Define a modern grey palette with blue accent colors
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor("#e0e0e0"))  # Light grey background
        palette.setColor(QPalette.WindowText, QColor("#2e2e2e"))
        palette.setColor(QPalette.Base, QColor("#ffffff"))
        palette.setColor(QPalette.AlternateBase, QColor("#e0e0e0"))
        palette.setColor(QPalette.ToolTipBase, QColor("#ffffff"))
        palette.setColor(QPalette.ToolTipText, QColor("#2e2e2e"))
        palette.setColor(QPalette.Text, QColor("#2e2e2e"))
        palette.setColor(QPalette.Button, QColor("#d0d0d0"))
        palette.setColor(QPalette.ButtonText, QColor("#2e2e2e"))
        palette.setColor(QPalette.BrightText, QColor("#ff0000"))
        palette.setColor(QPalette.Link, QColor("#3498db"))
        palette.setColor(QPalette.Highlight, QColor("#3498db"))
        palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
        QApplication.instance().setPalette(palette)

        # Set a modern system font
        font = QFont("Segoe UI", 10)
        QApplication.instance().setFont(font)

        # Customize the style for QPushButton elements
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

        # Customize QLineEdit style for test cases input
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

        # Customize QSlider style for the similarity threshold control
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

        # Customize QProgressBar style for visual consistency
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

        # Customize QTabWidget style to match the modern design
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
        """
        Update the displayed similarity threshold label when the slider value changes.

        The slider value is converted from an integer (50-100) to a float (0.50-1.0).
        """
        threshold = value / 100.0
        self.threshold_value_label.setText(f"{threshold:.2f}")

    def browse_file(self):
        """
        Open a file dialog to allow the user to select a single Python file.
        """
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Python File", "", "Python Files (*.py)"
        )
        if file_name:
            self.file_path = file_name
            self.folder_path = None
            QMessageBox.information(self, "File Selected", f"Selected file: {file_name}")

    def browse_folder(self):
        """
        Open a folder dialog to allow the user to select a directory for analysis.
        """
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.folder_path = folder
            self.file_path = None
            QMessageBox.information(self, "Folder Selected", f"Selected folder: {folder}")

    def on_start_analysis(self):
        """
        Validate inputs and initiate the code analysis process.

        This method performs the following steps:
          - Checks for selected file or folder.
          - Validates the threshold and number of tests.
          - Imports a single file if selected.
          - Sets the project path for analysis.
          - Displays the progress bar during analysis.
          - Updates all analysis tabs with the results.
        """
        if not (self.file_path or self.folder_path):
            QMessageBox.critical(self, "Error", "No file or folder selected!")
            return

        try:
            # Convert slider value to a float threshold
            threshold = self.threshold_slider.value() / 100.0

            # Validate the number of tests input
            num_tests_text = self.test_cases_entry.text().strip()
            if not num_tests_text.isdigit():
                raise ValueError("Test cases input is not a valid positive integer.")
            num_tests = int(num_tests_text)
            if num_tests <= 0:
                raise ValueError("Number of tests must be > 0.")

            analysis_path = self.file_path if self.file_path else self.folder_path

            # If a single file is selected, dynamically import it for analysis
            if self.file_path:
                import_user_script(self.file_path)

            # Set the project path based on the user's selection
            if self.folder_path:
                self.controller.set_project_path(self.folder_path)
            elif self.file_path:
                project_root = os.path.dirname(self.file_path)
                self.controller.set_project_path(project_root)

            # Display the progress bar while analysis is running
            self.progress.setVisible(True)
            QApplication.processEvents()

            # Execute analysis and store results
            results = self.controller.start_analysis(analysis_path, threshold, num_tests)
            self.analysis_results = results

            # Update individual tabs with corresponding analysis results
            self.static_tab.update_static(results.get("static", []))
            self.clone_tab.update_clone(results.get("merged_clones", []))
            self.runtime_tab.update_runtime(results.get("runtime", {}))
            self.dataflow_tab.update_dataflow(results.get("data_flow", {}))
            self.refactor_tab.update_refactor_plans(results.get("refactoring_plans", []))

        except ValueError as ve:
            QMessageBox.critical(self, "Input Error", str(ve))
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", str(e))
        finally:
            self.progress.setVisible(False)

    def on_restart(self):
        """
        Reset the application state to allow a new analysis session.

        This includes clearing selected file/folder, resetting slider values,
        and clearing data in all analysis tabs.
        """
        self.file_path = None
        self.folder_path = None
        self.threshold_slider.setValue(80)
        self.threshold_value_label.setText(f"{0.0:.2f}")

        # Clear all tabs' data
        self.static_tab.update_static([])
        self.clone_tab.update_clone([])
        self.runtime_tab.update_runtime({})
        self.dataflow_tab.update_dataflow({})
        self.refactor_tab.update_refactor_plans([])

        QMessageBox.information(self, "Reset", "Application reset. Please select a new file/folder to analyze.")

    def export_report(self):
        """
        Export the analysis results into a CSV report.

        The report includes sections for static analysis, clone similarity analysis, and runtime metrics.
        """
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Report As", "", "CSV Files (*.csv)"
        )
        if not file_path:
            return

        try:
            with open(file_path, mode="w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)

                # Write static analysis details
                writer.writerow(["=== Static Analysis ==="])
                if hasattr(self.static_tab, 'static_tree'):
                    tree = self.static_tab.static_tree
                    headers = [tree.headerItem().text(col) for col in range(tree.columnCount())]
                    writer.writerow(headers)
                    for i in range(tree.topLevelItemCount()):
                        item = tree.topLevelItem(i)
                        row = [item.text(col) for col in range(tree.columnCount())]
                        writer.writerow(row)

                # Write clone similarity analysis details
                writer.writerow([])
                writer.writerow(["=== Clone Similarity Analysis ==="])
                if hasattr(self.clone_tab, 'clone_tree'):
                    tree = self.clone_tab.clone_tree
                    headers = [tree.headerItem().text(col) for col in range(tree.columnCount())]
                    writer.writerow(headers)
                    for i in range(tree.topLevelItemCount()):
                        item = tree.topLevelItem(i)
                        row = [item.text(col) for col in range(tree.columnCount())]
                        writer.writerow(row)

                # Write runtime metrics summary
                writer.writerow([])
                writer.writerow(["=== Runtime Metrics ==="])
                if hasattr(self.runtime_tab, 'runtime_tree_summary'):
                    tree = self.runtime_tab.runtime_tree_summary
                    headers = [tree.headerItem().text(col) for col in range(tree.columnCount())]
                    writer.writerow(headers)
                    for i in range(tree.topLevelItemCount()):
                        item = tree.topLevelItem(i)
                        row = [item.text(col) for col in range(tree.columnCount())]
                        writer.writerow(row)

            QMessageBox.information(self, "Export Successful", f"CSV report exported to {file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def start(self):
        """
        Display the main application window.
        """
        self.show()


if __name__ == "__main__":
    # Initialize the QApplication and set the Fusion style for modern look
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Initialize the controller and the main application view
    controller = CodeAnalysisController()
    view = ApplicationView(controller)
    view.start()

    # Start the event loop
    sys.exit(app.exec_())
