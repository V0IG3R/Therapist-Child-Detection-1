# Therapist-Child Classification with YOLOv8 and DeepSORT

This project implements a real-time object detection and tracking system using the YOLOv8 model for object detection and DeepSORT for tracking. The code is designed to process video input, detect people, classify them and track the detected objects across frames.


### Prerequisites

Before running the code, ensure you have the following dependencies installed:

-   Python 3.7 or higher
-   OpenCV (`cv2`)
-   TensorFlow
-   Keras
-   Ultralytics (`YOLOv8`)
-   Torch
-   CUDA (if using a GPU)
-   DeepSORT
  
You can install the necessary libraries using pip:
`pip install -r requirements.txt`

You can download the rest of the files from the link in additionals.txt.

## Files and Folders

-   `weights/yolov8n.pt`: YOLOv8 model weights.
-   `adchcheck.h5`: Pre-trained age classification model. 
-   `deep_sort/deep/checkpoint/ckpt.t7`: DeepSORT checkpoint for tracking.



## How to Run

1.  **Set Up the Environment:** Make sure you have all dependencies installed. Create a folder called `testvids/` and store the test videos in there.
    
2.  **Run the Script:** Execute the script in your Python environment:
`python track_count_persons.py`

3. **Output:** The processed video will be saved in the `output/` directory as `out4.mp4`.


## Detailed Code Explanation

### Object Detection with YOLOv8

The script uses the YOLOv8 model to detect objects within each frame of the video. YOLOv8 is a state-of-the-art object detection model capable of detecting multiple classes of objects. In this script, the focus is on detecting people.

### Object Tracking with DeepSORT

DeepSORT is used to track the detected objects across video frames. Each detected object is assigned a unique track ID, which is maintained throughout the video. This allows the system to recognize when the same person is detected across different frames.

### Age Classification

A pre-trained Keras model (`adchcheck.h5`) is used to classify the age of detected persons into two categories: "Child" or "Therapist". The detected person’s face is cropped and passed through this model, and the result is displayed on the video.

### Video Processing

The video is processed frame by frame:

1.  **Object Detection:** YOLOv8 detects objects in the current frame.
2.  **Tracking:** Detected objects are tracked using DeepSORT.
3.  **Age Classification:** For each detected person, the age classification model is applied.
4.  **Annotation:** Each frame is annotated with the class name, track ID, and age classification.
5.  **Output:** The annotated frame is written to the output video file.

### Customization

-   **Confidence Threshold:** The `conf=0.8` parameter can be adjusted to change the confidence threshold for object detection.
-   **Tracking Parameters:** DeepSORT tracking parameters like `max_age` can be modified to better suit your use case.
-   **Model Paths:** Ensure the paths to the model weights and video files are correct.

### Error Handling

The script includes error handling to catch and manage errors related to OpenCV operations and tensor operations that might arise during video processing. This ensures that the script continues execution even if an error occurs.

## Additional Notes
### Challenges Faced:
1. The bounding boxes overlapped a lot leading to often switch of identities. But overall accuracy was higher in this case.
2. The computational power needed and time taken were extremely higher in this case.

### Possible Solutions:
1. Use of FairMOT to detect and identify persons. While in theory, it has a greater accuracy, I am yet to implement it but am unable to because of lower computational power.
2. Approach 2 involves use of OpenVINO toolkit and models which offered comparatively less accuracy but were extremely fast.
