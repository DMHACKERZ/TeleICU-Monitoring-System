# TeleICU-Monitoring-System
# 🔍 Project Overview
This project implements an innovative monitoring system for TeleICU patients using video processing and deep learning techniques. The system aims to reduce the burden on remote healthcare professionals by allowing them to monitor multiple patients simultaneously.

# 📋 Key Features

**Multi-source Video Input**: The system can process video from various sources including local files, YouTube links, and live CCTV feeds.

**Patient Detection and Tracking**: Utilizes the YOLOv8 object detection model to identify and track patients, doctors, nurses, monitors and others in the video feed.

**Movement Analysis**: Implements advanced motion detection algorithms to analyze patient movements, including:
 *  Overall motion detection using Structural Similarity Index (SSIM) method for measuring the similarity between two images.
 *  Specific movement detection for head, hands, legs, and chest (for breathlessness)

**Fall Detection**: Incorporates a separate YOLOv8 model trained specifically for detecting fall incidents.

**Real-time Alerting**: Generates visual alerts on the video feed when critical events (like falls) are detected.

**Movement Logging**: Records detected movements with timestamps in a log file for later analysis.

**Video Output**: Processes and saves the annotated video feed with detected objects, movements, and alerts.

# 🛠️ Technical Approach

**Video Processing**

* OpenCV (cv2) is used for video capture, processing, and output.
* The system supports multiple video sources, including local files, YouTube links (using yt-dlp), and live CCTV feeds.

**Object Detection**

YOLOv8 models are employed for object detection:

1. A general model ('person_labelling.pt') for detecting patients, doctors, nurses, monitor and others.
2. A specialized model ('fall_detection.pt') for fall detection.

**Movement Analysis**

* Optical flow (cv2.calcOpticalFlowFarneback) is used to detect and analyze specific movements.

* The patient's body is divided into regions (head, hands, legs, chest) for targeted movement analysis.

* SSIM (Structural Similarity Index) is used to detect overall motion between frames.

**Fall Detection**

A separate thread continuously analyzes frames for potential fall incidents.
When a fall is detected, an alert is displayed on the video feed for a set duration.

**Real-time Processing**

The system processes video frames in real-time, performing object detection, movement analysis, and fall detection on each frame.
Multithreading is used to ensure smooth performance, especially for the continuous fall detection.

# 🪄 Unique Features of Our Project

1. **Multi-Source Video Input**
     * Supports video input from local files, URLs (e.g., YouTube), and live CCTV feeds.
       
2. **Person Detection and Identification**
      * Detects various individuals in the ICU, including doctors, nurses, and patients using the YOLOv8 model (person_labelling.pt).
      * Identifies when a patient is without a doctor or nurse and changes the bounding box color accordingly:
          * Green: Indicates presence of a doctor or nurse.
          * Yellow: Indicates no significant motion detected and no doctor/nurse present.
          * Blue: Indicates significant motion detected without specific movements.
          * Red: Indicates significant movement or specific detected movements (e.g., head, hand, leg, breathlessness).

3. **Specific Movement Detection**
      * Detects and analyzes specific patient movements such as head, hand, leg movements, and signs of breathlessness using optical flow and structural similarity metrics.

4. **Fall Detection and Alert System**
      * Uses a secondary YOLOv8 model (fall_detection.pt) for detecting falls and other dangerous movements.
      * Generates real-time alerts with visual indicators and log entries for critical detections.
      * Displays a red border around the video frame and an alert message for a specified duration when a fall is detected.

5. **Real-Time Video Processing and Output**
      * Processes video in real-time, applying detections and annotations dynamically.
      * Saves processed video with annotations for later review and analysis.

6. **Movement Logging**
      * Logs detected movements with timestamps for comprehensive documentation and future analysis.
      * Updates movement logs in real-time to ensure accuracy.

# ⚙️ Implementation Details

**Language: Python**

**Main Libraries**:

* OpenCV (cv2) for video processing
* Ultralytics YOLO for object detection and fall detection
* NumPy for numerical operations
* scikit-image for image processing algorithms
* PyTorch for deep learning model inference
* Video Handling: yt-dlp for downloading YouTube videos
* Multithreading: Python's threading module for concurrent processing

# 📝 Challenges Addressed

**Dataset Creation**: While not explicitly shown in the code, the project likely involved creating or curating a dataset of ICU patient images/videos for training the models.

**Real-time Video Processing**: The code efficiently processes high-quality video in real-time, addressing one of the major challenges mentioned in the problem statement.

**Error Margin Reduction**: By using multiple analysis techniques (object detection, movement analysis, fall detection), the system aims to reduce error margins in patient monitoring.

# ♾️ Future Improvements

* Implement more sophisticated algorithms for detecting complex patient behaviors.
* Integrate with hospital information systems for comprehensive patient monitoring.
* Develop a user-friendly interface for healthcare professionals to monitor multiple patients.

# ⏳Conclusion

The TeleICU Monitoring System revolutionizes remote ICU patient care by integrating advanced video processing and deep learning techniques. Our system supports versatile video input from local files, URLs, and live CCTV feeds, ensuring flexibility in various monitoring scenarios. Leveraging the YOLOv8 model, it accurately detects and identifies individuals such as doctors, nurses, patients, monitor, others, dynamically changing bounding box colors to indicate the presence of healthcare professionals and specific patient movements. Unique features include detailed analysis of head, hand, leg movements, signs of breathlessness, and a robust fall detection system that generates real-time alerts. The system processes and saves annotated video feeds in real-time, accompanied by comprehensive movement logs with timestamps, ensuring accurate documentation for future analysis. By providing these innovative tools, our TeleICU Monitoring System significantly enhances patient safety and care quality, setting a new standard in telemedicine and remote ICU monitoring.
