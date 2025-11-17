🚶 Real-Time People Detector (OpenCV HOG)

This is a Streamlit web application that performs real-time pedestrian detection using the classical computer vision approach of HOG (Histogram of Oriented Gradients) combined with a pre-trained SVM (Support Vector Machine) classifier, both provided by the OpenCV library.

This app is designed to be fully functional and demonstrates core object detection concepts without needing large, external YOLO weight files.

Features

Real-Time Detection: Runs the HOG + SVM algorithm to identify human figures in uploaded images.

Bounding Boxes: Draws red bounding boxes around detected objects.

Confidence Score: Displays the detection confidence (weight) for each bounding box.

Performance: The detector is fast, relying on efficient OpenCV implementation and caching via Streamlit.

Setup and Installation

1. Prerequisites

Ensure you have Python (3.7+) installed.

2. Install Dependencies

Install all necessary libraries using the provided requirements.txt file:

pip install -r requirements.txt


3. Run the Application

Execute the Streamlit script from your terminal:

streamlit run yolo_detector_real.py


The application will automatically open in your default web browser.