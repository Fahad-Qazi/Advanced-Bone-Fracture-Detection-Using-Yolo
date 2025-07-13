import os
import streamlit as st

class Sidebar:
    def __init__(self, title_img='images/medical.jpg') -> None:
        """Initialize the sidebar with a title image and configurations."""
        self.title_img = title_img
        self.model_name = None
        self.confidence_threshold = None
        self.models = {
            'YOLOv8n': r"C:\Users\Admin\OneDrive\Desktop\Fahad_Qazi_Project\01_project\new_combined_trainings\New\Yolo_V8\Yolo_V8n_improved\best (1).pt",
            'YOLOv8s': r"C:\Users\Admin\OneDrive\Desktop\Fahad_Qazi_Project\01_project\new_combined_trainings\New\Yolo_V8\Yolo_V8s\best (1).pt",
            'YOLOv9t': r"C:\Users\Admin\OneDrive\Desktop\Fahad_Qazi_Project\01_project\new_combined_trainings\New\Yolo_V9\Yolo_V9t\best (1).pt",
            'YOLOv10n': r"C:\Users\Admin\OneDrive\Desktop\Fahad_Qazi_Project\01_project\new_combined_trainings\New\Yolo_V10\Yolo_V10n\best (1).pt",
            'YOLOv10s': r"C:\Users\Admin\OneDrive\Desktop\Fahad_Qazi_Project\01_project\new_combined_trainings\New\Yolo_V10\Yolo_V10s\best (1).pt"
        }

        # Check if model paths exist
        for model_name, model_path in self.models.items():
            if not os.path.exists(model_path):
                st.error(f"Model file not found for {model_name}: {model_path}")
                st.stop()  # Stop execution if any model file is missing

        self._title_image()
        self._model_selection()
        self._confidence_threshold()

    def _title_image(self):
        """Display the title image in the sidebar."""
        st.sidebar.image(self.title_img, caption="Medical Detection")

    def _model_selection(self):
        """Allow the user to select a YOLO model."""
        st.sidebar.markdown('## Select Your Model')
        model_options = list(self.models.keys())
        selected_model = st.sidebar.selectbox(
            'Which model would you like to choose?',
            options=model_options,
            index=0,
            key="model_selection"  # Add a unique key
        )
        self.model_name = self.models[selected_model]  # Store model path, not name

    def _confidence_threshold(self):
        """Set the confidence threshold for predictions."""
        st.sidebar.markdown('## Set a Confidence Threshold')
        self.confidence_threshold = st.sidebar.slider(
            "Confidence Threshold", 0.00, 1.00, 0.5, 0.01,
            key="confidence_threshold"  # Add a unique key
        )














































































































