from PyQt5.QtWidgets import QApplication
from view.mainView import ApplicationView
from controller.code_analysis_controller import CodeAnalysisController

def main():
    import sys
    app = QApplication(sys.argv)  # Initialize here
    controller = CodeAnalysisController()
    view = ApplicationView(controller)
    view.start()
    sys.exit(app.exec_())  # Properly close the app

if __name__ == "__main__":
    main()
