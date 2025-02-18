from view.application_view  import ApplicationView
from controller.code_analysis_controller  import CodeAnalysisController

def main():
    controller = CodeAnalysisController()
    app = ApplicationView(controller)
    app.start()

if __name__ == "__main__":
    main()
