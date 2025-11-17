import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO


# ==============================================================================
# 1. REAL PERSON DETECTION (using OpenCV HOG Detector)
# ==============================================================================

@st.cache_resource
def load_detector():
    """Loads the pre-trained HOG descriptor and SVM detector for pedestrian detection."""
    try:
        # Create the HOG descriptor/person detector
        hog = cv2.HOGDescriptor()
        # Set the SVM detector to the default one trained for people
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        return hog
    except Exception as e:
        st.error(f"Error loading HOG detector: {e}")
        return None


@st.cache_data
# FIX: Added underscore '_' to hog_detector to prevent Streamlit from hashing the unhashable object.
def detect_and_draw(uploaded_file, _hog_detector):
    """Detects people in the image and draws bounding boxes."""
    if uploaded_file is None or _hog_detector is None:
        return None, []

    # 1. Load and prepare image (PIL to OpenCV)
    image_bytes = uploaded_file.read()
    img_pil = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Convert PIL to numpy array (BGR format for OpenCV)
    img_cv = np.array(img_pil)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # ** ROBUSTNESS FIX 2: Ensure correct data type (np.uint8) **
    # This addresses potential C++ memory/type issues.
    if img_cv.dtype != np.uint8:
        img_cv = img_cv.astype(np.uint8)

    # 2. Convert to Grayscale for detection
    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # 3. Resize image for faster detection (standard practice for HOG)
    scale_percent = 50
    width = int(gray_img.shape[1] * scale_percent / 100)
    height = int(gray_img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_img = cv2.resize(gray_img, dim, interpolation=cv2.INTER_AREA)

    # 4. Perform detection (using multiple scales/windows)
    # Use the resized grayscale image for detection
    (rects, weights) = _hog_detector.detectMultiScale(
        resized_img,
        winStride=(4, 4),
        padding=(8, 8),
        scale=1.05
    )

    detected_people = []

    # 5. Process Detections
    scaling_factor = 100 / scale_percent

    for i, rect in enumerate(rects):
        x, y, w, h = rect

        # Scale coordinates back up to original image size
        x_orig = int(x * scaling_factor)
        y_orig = int(y * scaling_factor)
        w_orig = int(w * scaling_factor)
        h_orig = int(h * scaling_factor)

        confidence = weights[i][0]

        # Draw rectangle (Color: Red, Thickness: 3) on the original COLOR image (img_cv)
        color = (0, 0, 255)
        cv2.rectangle(img_cv, (x_orig, y_orig), (x_orig + w_orig, y_orig + h_orig), color, 3)

        # Add label
        text = f"Person: {confidence:.2f}"
        cv2.putText(img_cv, text, (x_orig, y_orig - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        detected_people.append(f"Person (Confidence: {confidence:.2f})")

    return img_cv, detected_people


# ==============================================================================
# 2. STREAMLIT APPLICATION LAYOUT
# ==============================================================================

def main():
    st.set_page_config(page_title="OpenCV People Detector", layout="wide")
    st.title("🚶 Real-Time People Detector (OpenCV HOG)")
    st.markdown("This application uses OpenCV's pre-trained HOG + SVM algorithm for fast pedestrian detection.")

    # Load detector (cached in st.cache_resource)
    hog_detector = load_detector()

    if hog_detector is None:
        st.stop()

    st.sidebar.header("Image Input")
    uploaded_file = st.sidebar.file_uploader(
        "Upload an image (best results with clear human figures)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        st.info("Upload an image in the sidebar to run the detection model.")
        return

    # --- Run Detection ---
    with st.spinner("Running HOG + SVM Person Detection..."):
        # CALL FIX: Passing hog_detector to the cached function
        processed_image_cv, detected_objects = detect_and_draw(uploaded_file, hog_detector)

    # --- Display Results ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Detection Results (People in Red Boxes)")
        processed_image_rgb = cv2.cvtColor(processed_image_cv, cv2.COLOR_BGR2RGB)
        st.image(processed_image_rgb, use_container_width=True, caption="HOG Detected Persons")

    with col2:
        st.subheader("Detection Report")

        num_detections = len(detected_objects)
        if num_detections > 0:
            st.success(f"Found {num_detections} {'person' if num_detections == 1 else 'people'}.")
            for item in detected_objects:
                st.markdown(f"- **{item}**")
        else:
            st.info("No people detected with sufficient confidence.")

        st.markdown("---")
        st.subheader("Technical Note")
        st.markdown(
            """
            The **HOG + SVM** model is highly specific to pedestrian detection and is optimized for speed over the complex features of modern deep learning models.
            """
        )


if __name__ == "__main__":
    main()