import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import tempfile
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)


# ==============================================================================
# 1. OPTIMIZED YOLO FOR CPU
# ==============================================================================

class OptimizedYOLODetector:
    def __init__(self):
        self.net = None
        self.classes = []
        self.output_layers = []
        self.load_model()

    def load_model(self):
        """Load optimized YOLO model for CPU"""
        try:
            # Use YOLOv3-tiny for much faster CPU performance
            weights_url = "https://pjreddie.com/media/files/yolov3-tiny.weights"
            config_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg"
            names_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

            temp_dir = tempfile.gettempdir()
            weights_path = os.path.join(temp_dir, "yolov3-tiny.weights")
            config_path = os.path.join(temp_dir, "yolov3-tiny.cfg")
            names_path = os.path.join(temp_dir, "coco.names")

            # Download files with progress
            if not os.path.exists(weights_path):
                with st.spinner("Downloading YOLOv3-tiny weights (much faster on CPU)..."):
                    self.download_file(weights_url, weights_path)

            if not os.path.exists(config_path):
                self.download_file(config_url, config_path)

            if not os.path.exists(names_path):
                self.download_file(names_url, names_path)

            # Load network
            self.net = cv2.dnn.readNet(weights_path, config_path)

            # Force CPU usage
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            # Load class names
            with open(names_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]

            # Get output layers
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

            st.success("✅ YOLOv3-tiny loaded (Optimized for CPU)")
            return True

        except Exception as e:
            st.error(f"❌ Failed to load YOLO model: {e}")
            # Fallback to even lighter model
            return self.load_fallback_model()

    def load_fallback_model(self):
        """Ultra-lightweight fallback using OpenCV's DNN face detector"""
        try:
            st.info("Loading ultra-lightweight detector...")
            # This is a very small model that should always work
            proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
            model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

            temp_dir = tempfile.gettempdir()
            proto_path = os.path.join(temp_dir, "deploy.prototxt")
            model_path = os.path.join(temp_dir, "res10_300x300_ssd_iter_140000.caffemodel")

            if not os.path.exists(proto_path):
                self.download_file(proto_url, proto_path)
            if not os.path.exists(model_path):
                with st.spinner("Downloading lightweight model..."):
                    self.download_file(model_url, model_path)

            self.net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
            self.classes = ['person']  # Simplified for fallback
            st.warning("⚠️ Using lightweight detector (limited to clear frontal faces)")
            return True

        except Exception as e:
            st.error(f"❌ All models failed to load: {e}")
            return False

    def download_file(self, url, filename):
        """Download file from URL"""
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pass  # Silent download for Streamlit Cloud

        return True

    def detect_people(self, image, confidence_threshold=0.5, nms_threshold=0.4):
        """Detect people in image using optimized YOLO"""
        try:
            height, width = image.shape[:2]

            # Use smaller input size for faster processing on CPU
            blob = cv2.dnn.blobFromImage(
                image,
                1 / 255.0,
                (320, 320),  # Smaller than standard 416 for speed
                swapRB=True,
                crop=False
            )

            self.net.setInput(blob)
            outputs = self.net.forward(self.output_layers)

            boxes = []
            confidences = []
            class_ids = []

            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    # Filter for person class (class_id = 0)
                    if class_id == 0 and confidence > confidence_threshold:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Apply NMS
            indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

            results = []
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    confidence = confidences[i]
                    results.append({
                        'box': (x, y, w, h),
                        'confidence': confidence,
                        'class': 'person'
                    })

            return results

        except Exception as e:
            logging.error(f"Detection failed: {e}")
            return []


# ==============================================================================
# 2. LIGHTWEIGHT DETECTION ALTERNATIVE
# ==============================================================================

def detect_people_haar(image):
    """Ultra-fast Haar cascade for person detection (fallback)"""
    try:
        # Load Haar cascade for full body detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_fullbody.xml'
        if not os.path.exists(cascade_path):
            # Download if not available
            cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_fullbody.xml"
            response = requests.get(cascade_url)
            cascade_path = os.path.join(tempfile.gettempdir(), "haarcascade_fullbody.xml")
            with open(cascade_path, 'wb') as f:
                f.write(response.content)

        cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect bodies
        bodies = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30)
        )

        results = []
        for (x, y, w, h) in bodies:
            results.append({
                'box': (x, y, w, h),
                'confidence': 0.7,  # Estimated confidence for Haar
                'class': 'person'
            })

        return results

    except Exception as e:
        logging.error(f"Haar cascade failed: {e}")
        return []


# ==============================================================================
# 3. STREAMLIT APPLICATION (OPTIMIZED FOR STREAMLIT CLOUD)
# ==============================================================================

@st.cache_resource
def load_detector():
    """Load detector with caching"""
    return OptimizedYOLODetector()


def preprocess_image(image_bytes, max_dimension=800):
    """Optimized image preprocessing for Streamlit Cloud"""
    try:
        img_pil = Image.open(BytesIO(image_bytes)).convert("RGB")
        width, height = img_pil.size

        # More aggressive resizing for CPU performance
        if width > max_dimension or height > max_dimension:
            if width > height:
                new_width = max_dimension
                new_height = int(height * max_dimension / width)
            else:
                new_height = max_dimension
                new_width = int(width * max_dimension / height)

            img_pil = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)

        img_np = np.array(img_pil)
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), img_pil.size

    except Exception as e:
        logging.error(f"Image preprocessing failed: {e}")
        return None, None


def draw_detections(image, detections):
    """Draw bounding boxes and labels"""
    result_image = image.copy()

    for detection in detections:
        x, y, w, h = detection['box']
        confidence = detection['confidence']

        # Ensure coordinates are within bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)

        # Draw bounding box
        color = (0, 255, 0)  # Green
        cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)

        # Draw label
        label = f"Person: {confidence:.2f}"
        cv2.putText(result_image, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return result_image


def main():
    st.set_page_config(
        page_title="Fast Person Detector",
        layout="wide",
        page_icon="🚶"
    )

    st.title("🚶 Fast Person Detector (CPU Optimized)")
    st.markdown("""
    **Optimized for Streamlit Cloud** - Uses YOLOv3-tiny for fast CPU processing
    """)

    # Initialize detector
    detector = load_detector()

    if detector.net is None:
        st.error("❌ Detector initialization failed. Using basic fallback.")

    # Sidebar with optimized settings
    st.sidebar.header("⚡ CPU-Optimized Settings")

    detection_method = st.sidebar.selectbox(
        "Detection Method",
        ["YOLOv3-tiny (Recommended)", "Haar Cascade (Fastest)"],
        help="YOLO is more accurate, Haar is faster"
    )

    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.4,  # Lower default for better detection
        step=0.1,
        help="Lower = more detections, Higher = fewer but more confident"
    )

    max_image_size = st.sidebar.selectbox(
        "Max Image Size",
        options=[400, 600, 800],
        index=1,
        help="Smaller images process faster"
    )

    st.sidebar.header("📤 Image Input")
    uploaded_file = st.sidebar.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        help="For fastest results, use images < 1MB"
    )

    # Performance tips
    with st.sidebar.expander("💡 Performance Tips"):
        st.markdown("""
        - **Small images** process faster (600px recommended)
        - **JPG format** loads quicker than PNG
        - **Clear, well-lit** images work best
        - **Haar Cascade** for maximum speed
        - **Lower confidence** for more detections
        """)

    # Main content
    if uploaded_file is None:
        st.info("👈 Upload an image to start detection")

        # Performance comparison
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Processing Speed", "Fast")
        with col2:
            st.metric("Accuracy", "Good")
        with col3:
            st.metric("CPU Usage", "Optimized")

        return

    # File info
    file_size = len(uploaded_file.getvalue()) / 1024
    if file_size > 1000:
        st.warning(f"⚠️ Large image ({file_size:.0f} KB). Processing may be slow.")

    # Process image
    with st.spinner("🔍 Detecting people..."):
        # Preprocess
        image_cv, original_size = preprocess_image(
            uploaded_file.getvalue(),
            max_dimension=max_image_size
        )

        if image_cv is None:
            st.error("❌ Failed to process image")
            return

        # Choose detection method
        if detection_method == "YOLOv3-tiny (Recommended)" and detector.net is not None:
            detections = detector.detect_people(
                image_cv,
                confidence_threshold=confidence_threshold
            )
            method_used = "YOLOv3-tiny"
        else:
            detections = detect_people_haar(image_cv)
            method_used = "Haar Cascade"

        # Draw results
        result_image = draw_detections(image_cv, detections)

    # Display results
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Detection Results")

        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        st.image(result_image_rgb, use_container_width=True,
                 caption=f"{method_used} - {len(detections)} people detected")

    with col2:
        st.subheader("📈 Detection Report")

        if len(detections) > 0:
            st.success(f"✅ **{len(detections)}** person(s) detected")

            for i, detection in enumerate(detections, 1):
                confidence = detection['confidence']
                st.write(f"**Person {i}:** {confidence:.1%} confidence")

            # Performance stats
            st.metric("Detection Method", method_used)
            if detections:
                avg_conf = np.mean([d['confidence'] for d in detections])
                st.metric("Average Confidence", f"{avg_conf:.1%}")

        else:
            st.warning("❌ No people detected")
            st.info("""
            Try:
            - Lower confidence threshold
            - Different detection method
            - Clearer image quality
            """)

    # Quick actions
    if len(detections) > 0:
        st.download_button(
            label="💾 Download Result",
            data=cv2.imencode('.jpg', result_image)[1].tobytes(),
            file_name="detection_result.jpg",
            mime="image/jpeg"
        )


if __name__ == "__main__":
    main()