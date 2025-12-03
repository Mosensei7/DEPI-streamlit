import streamlit as st
import os
import mlflow
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
st.set_page_config(
    page_title="Autonomous Vehicle Object Detection",
    page_icon="🚗",
    layout="wide"
)

# Constants
MODEL_PATH = os.getenv("MODEL_PATH", "yolov11s_final.pt")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
EXPERIMENT_NAME = "YOLOv11s_Autonomous_Driving_OD_Predictions"

# --- MLflow Setup ---
def setup_mlflow():
    if not MLFLOW_TRACKING_URI:
        st.warning("⚠️ MLFLOW_TRACKING_URI not set. Logging disabled.")
        return False
    
    try:
        # Set credentials for Databricks
        if DATABRICKS_TOKEN:
            os.environ['DATABRICKS_TOKEN'] = DATABRICKS_TOKEN
        if DATABRICKS_HOST:
            os.environ['DATABRICKS_HOST'] = DATABRICKS_HOST
            
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            mlflow.create_experiment(EXPERIMENT_NAME)
        mlflow.set_experiment(EXPERIMENT_NAME)
        return True
    except Exception as e:
        st.error(f"❌ MLflow setup failed: {e}")
        return False

# --- Model Loading ---
@st.cache_resource
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# --- Main App ---
def main():
    st.title("🚗 Autonomous Vehicle Object Detection")
    st.markdown("---")

    # Sidebar
    st.sidebar.header("Configuration")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    iou_threshold = st.sidebar.slider("IOU Threshold", 0.0, 1.0, 0.45, 0.05)
    
    # MLflow Status
    mlflow_active = setup_mlflow()
    if mlflow_active:
        st.sidebar.success(f"✅ MLflow Connected: {EXPERIMENT_NAME}")
    else:
        st.sidebar.warning("⚠️ MLflow Not Connected")

    # Load Model
    model = load_model(MODEL_PATH)
    if model is None:
        st.stop()

    # File Upload
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)

        # Run Inference
        if st.button("Detect Objects", type="primary"):
            with st.spinner("Running detection..."):
                try:
                    # Convert to numpy for YOLO
                    image_np = np.array(image)
                    
                    # Run prediction
                    start_time = datetime.now()
                    results = model.predict(
                        image_np,
                        conf=confidence_threshold,
                        iou=iou_threshold,
                        imgsz=640,
                        verbose=False
                    )
                    end_time = datetime.now()
                    inference_time = (end_time - start_time).total_seconds()

                    # Process results
                    result = results[0]
                    annotated_image = result.plot()
                    
                    # Display result
                    with col2:
                        st.subheader("Detected Objects")
                        st.image(annotated_image, channels="BGR", use_column_width=True)

                    # Metrics & Logging
                    predictions = []
                    class_counts = {}
                    
                    if result.boxes:
                        for box in result.boxes:
                            cls_id = int(box.cls[0])
                            cls_name = model.names[cls_id]
                            conf = float(box.conf[0])
                            
                            predictions.append({
                                "class": cls_name,
                                "confidence": conf,
                                "bbox": box.xyxy[0].tolist()
                            })
                            
                            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

                    # Display Stats
                    st.markdown("### 📊 Detection Statistics")
                    stat_col1, stat_col2, stat_col3 = st.columns(3)
                    stat_col1.metric("Objects Detected", len(predictions))
                    stat_col2.metric("Inference Time", f"{inference_time*1000:.2f} ms")
                    stat_col3.metric("Classes Found", len(class_counts))

                    # Detailed list
                    if predictions:
                        with st.expander("See detailed detections"):
                            st.json(predictions)

                    # Log to MLflow
                    if mlflow_active:
                        with mlflow.start_run(run_name=f"streamlit_pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                            # Log Params
                            mlflow.log_param("confidence_threshold", confidence_threshold)
                            mlflow.log_param("iou_threshold", iou_threshold)
                            mlflow.log_param("image_size", f"{image.width}x{image.height}")
                            mlflow.log_param("model_path", MODEL_PATH)

                            # Log Metrics
                            mlflow.log_metric("num_detections", len(predictions))
                            mlflow.log_metric("inference_time_ms", inference_time * 1000)
                            
                            for cls, count in class_counts.items():
                                mlflow.log_metric(f"count_{cls}", count)

                            # Log Image (Optional - can be heavy)
                            # mlflow.log_image(image, "original_image.jpg")
                            # mlflow.log_image(Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)), "annotated_image.jpg")

                            # Tags
                            mlflow.set_tag("source", "streamlit_app")
                            mlflow.set_tag("model", "yolov11s")

                        st.toast("✅ Logged to MLflow", icon="📝")

                except Exception as e:
                    st.error(f"Error during inference: {e}")

if __name__ == "__main__":
    main()
