# Anomaly Detection in Video Surveillance

This project aims to detect anomalies in video surveillance footage using the YOLOv8 object detection model.  It provides tools for training the model, making predictions on images and videos, and a simple GUI for interacting with the model.

## Project Structure

*   **`README.md`**: This file, providing an overview of the project.
*   **`check.py`**: A simple script to check the installed YOLOv5 version (or YOLOv7, if you have it installed).  This script is primarily for verifying your YOLO installation.
*   **`gui.py`**:  A Tkinter-based GUI application that allows users to:
    *   Load images.
    *   Run object detection on the loaded images using a trained YOLOv8 model.
    *   Display the original and processed images.
*   **`predict.py`**:  A script to run object detection on a single image using a trained YOLOv8 model.  It displays the result using OpenCV.
*   **`train.py`**:  A script to train a YOLOv8 model using a specified dataset.  It includes WandB integration for experiment tracking (requires WandB API key).
*   **`video_predict.py`**:  A script to perform object detection on a video file using a trained YOLOv8 model.  It saves the processed video with bounding boxes to an output file.

## Dependencies

*   **Python 3.x**
*   **`torch`**: PyTorch for tensor operations.
*   **`yolov5` or `ultralytics`**:  YOLOv5 or YOLOv8 library for object detection.  The code uses `ultralytics` (YOLOv8) primarily, but `check.py` attempts to import `yolov5`.  Install with `pip install ultralytics`.
*   **`opencv-python`**: OpenCV for image and video processing. Install with `pip install opencv-python`.
*   **`Pillow`**: Python Imaging Library for image manipulation. Install with `pip install Pillow`.
*   **`tkinter`**:  Python's standard GUI library (usually included with Python).
*   **`ttkthemes`**:  For themed Tkinter widgets in the GUI. Install with `pip install ttkthemes`.
*   **`wandb`**: Weights & Biases for experiment tracking (optional, used in `train.py`). Install with `pip install wandb`.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Install the required packages:**

    ```bash
    pip install -r requirements.txt  # Create this file with all the dependencies
    ```
    Alternatively, install the dependencies individually:

    ```bash
    pip install torch ultralytics opencv-python Pillow tkinter ttkthemes wandb
    ```

## Usage

### 1. Training the Model

1.  **Prepare your dataset:**  Your dataset should be in a format compatible with YOLOv8 (e.g., the YOLO format with a `data.yaml` file).  The `data.yaml` file specifies the paths to your training and validation images and labels, as well as the number of classes and their names.  The paths used in the provided `train.py` are hardcoded and will need to be changed to your specific dataset location.

2.  **Configure `train.py`:**
    *   Replace `"C:\check\yolov8n.pt"` with the correct path to a pre-trained YOLOv8 model (e.g., `yolov8s.pt`, `yolov8m.pt`).  If you don't have a local copy, YOLOv8 will download it.
    *   Replace `"C:\check\data.yaml"` with the actual path to your `data.yaml` file.
    *   **WandB API Key:**  Make sure you have a WandB account and API key.  Replace `'6fb6b5f03ad0218018783fa7bf99b0afd457b9e3'` with your actual WandB API key.
    *   Adjust the `epochs` and `imgsz` parameters as needed.  Since the script is using `device='cpu'`, training will be slow. Consider using a GPU if available.

3.  **Run the training script:**

    ```bash
    python train.py
    ```

    The trained model will be saved in the `runs/detect/train/weights/best.pt` directory (or a similar directory, depending on the run number).

### 2. Making Predictions on Images

1.  **Configure `predict.py`:**
    *   Update `img_pth` with the path to the image you want to analyze.
    *   Update the model path to the location of your trained model (e.g., `"runs/detect/train/weights/best.pt"`).

2.  **Run the prediction script:**

    ```bash
    python predict.py
    ```

    This will display the image with bounding boxes around detected objects.

### 3. Making Predictions on Videos

1.  **Configure `video_predict.py`:**
    *   Update the model path to the location of your trained model (e.g., `"runs/detect/train/weights/best.pt"`).
    *   Update `video_path` with the path to your video file.

2.  **Run the video prediction script:**

    ```bash
    python video_predict.py
    ```

    This will process the video, display the results in a window, and save the output video to `output_video.avi`.

### 4. Using the GUI

1.  **Configure `gui.py`:**
    *   Update the model path within the `ObjectDetectionApp` class constructor to the location of your trained model (e.g., `"runs/detect/train/weights/best.pt"`).

2.  **Run the GUI script:**

    ```bash
    python gui.py
    ```

    This will launch the GUI application, allowing you to load images and run object detection.

## Important Notes

*   **Model Paths:** Ensure that the paths to your trained models are correct in all scripts (`gui.py`, `predict.py`, `video_predict.py`).
*   **Dataset Format:**  The training script expects your dataset to be in a YOLO-compatible format.
*   **Device:** The `train.py` script is set to use the CPU (`device='cpu'`). Training on a CPU can be very slow. If you have a GPU, change this to `device='cuda'` for significantly faster training.  Make sure you have the appropriate CUDA drivers and PyTorch version installed for GPU support.
*   **WandB Integration:** The `train.py` script uses WandB for experiment tracking. You'll need a WandB account and API key to use this feature.
*   **Output Video Codec:** The `video_predict.py` script uses the `'XVID'` codec for the output video.  You may need to install this codec on your system if you encounter issues.  Consider using a different codec if necessary.
*   **Anomaly Detection Logic:** This project provides the foundation for object detection. The actual anomaly detection logic (e.g., identifying unusual object combinations, behaviors, or locations) needs to be implemented based on your specific requirements.  This project provides the bounding box detections, and you'll need to write code to interpret those detections in the context of anomaly detection.
*   **GUI Customization:** The GUI is a basic example. You can customize it further to add more features, such as adjusting detection thresholds, displaying confidence scores, and implementing anomaly detection logic.
*   **requirements.txt:**  It's best practice to create a `requirements.txt` file to ensure consistent dependency management. You can generate this file using `pip freeze > requirements.txt`.
