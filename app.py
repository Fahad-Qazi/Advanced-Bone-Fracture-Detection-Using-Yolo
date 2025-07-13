import streamlit as st
from ultralytics import YOLO
import PIL
from PIL import Image
import torch
import numpy as np
import cv2
import random
import os
import warnings
import time

from sidebar import Sidebar

# Hide deprecation warnings
warnings.filterwarnings("ignore")

# Set a fixed random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Sidebar
sb = Sidebar()
title_image = sb.title_img
model = sb.model_name
conf_threshold = sb.confidence_threshold

# Load the YOLO model
yolo_detection_model = YOLO(model)

# Define color codes
colors = {
    'fractured_bone': (255, 0, 0)  # Red for fracture
}

# Main Title
st.title("Advance Bone Fracture Detection Using Yolov8")
st.write("Upload an X-ray image to detect fractures using a YOLOv8 model!")

# Custom CSS
st.markdown("""
<style>
    .result-container {
        font-size: 20px;
        padding: 10px;
        border-radius: 5px;
        margin-top: 20px;
        text-align: center;
    }
    .fracture {
        color: red;
    }
    .no-fracture {
        color: green;
    }
</style>
""", unsafe_allow_html=True)

# Upload button
if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def set_clicked():
    st.session_state.clicked = True

st.button('Upload Image', on_click=set_clicked)

# Annotate image (only draw boxes for fractured_bone)
def annotate_image(image, results):
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = yolo_detection_model.names[class_id].lower()

            # Only draw for fractured_bone
            if class_name != "fractured_bone":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            color = colors.get(class_name, (0, 0, 255))

            # Draw bounding box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {confidence:.2f}"
            font_scale = 0.5
            font_thickness = 1
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            cv2.rectangle(image, (x1, y1), (x1 + text_size[0], y1 - text_size[1] - 10), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    return image

if st.session_state.clicked:
    image = st.file_uploader("", type=["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp", "gif"])

    if image is not None:
        st.write("You selected the file:", image.name)
        uploaded_image = np.array(PIL.Image.open(image).convert("RGB"))

        col1, col2 = st.columns(2)

        with col1:
            st.image(image=uploaded_image, caption="Uploaded Image", use_column_width=True)

            if st.button("Run Detection"):
                with st.spinner("Running..."):
                    start_time = time.time()

                    res = yolo_detection_model.predict(uploaded_image, conf=conf_threshold, augment=True, max_det=2)
                    boxes = res[0].boxes
                    detection_time = time.time() - start_time

                    # Prepare results for plotting
                    fracture_detected = False
                    fracture_count = 0

                    # Filter boxes to keep only fractured_bone
                    filtered_boxes = []
                    for box in boxes:
                        class_id = int(box.cls[0])
                        class_name = yolo_detection_model.names[class_id].lower()
                        if class_name == "fractured_bone":
                            fracture_detected = True
                            fracture_count += 1
                            filtered_boxes.append(box)

                    # Draw boxes only for fractured_bone
                    annotated_image = uploaded_image.copy()
                    res[0].boxes = filtered_boxes
                    res_plotted = res[0].plot()[:, :, ::-1]  # Convert to RGB

                    with col2:
                        st.image(res_plotted, caption="Detected Image", use_column_width=True)

                        st.sidebar.markdown(
                            f"""
                            <div style="
                                background-color: #d4edda; 
                                color: #155724; 
                                padding: 10px; 
                                border-radius: 8px;
                                text-align: center;
                                font-weight: bold;">
                                Detection Time: {detection_time:.2f} seconds
                            </div>
                            """, unsafe_allow_html=True
                        )

                        # Show results
                        if fracture_detected:
                            message = f"<b>Fracture Detected</b><br>Number of fractures detected: {fracture_count}"
                            st.markdown(f'<div class="result-container fracture">{message}</div>', unsafe_allow_html=True)
                        else:
                            message = "<b>No Fracture Detected</b>"
                            st.markdown(f'<div class="result-container no-fracture">{message}</div>', unsafe_allow_html=True)
else:
    st.write("Please upload an image to test.")
