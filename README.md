# TeleICU-Monitoring-System
* This project is developed by Team Duelists
* Team Members:
  * Debraj Mistry (Team Lead)
  * Md Faizan
  * Jyotipriya Mallick
  * Sushanta Bhunia
       
# 🔍 Project Overview
This project implements an innovative monitoring system for TeleICU patients using video processing and deep learning techniques. The system aims to reduce the burden on remote healthcare professionals by allowing them to monitor multiple patients simultaneously.

# 📋 Key Features

**Multi-source Video Input**: The system can process video from various sources including local files, YouTube links, and live CCTV feeds.

**Patient Detection and Tracking**: Utilizes the YOLOv8 object detection model to identify and track patients, doctors, nurses, monitors and others in the video feed.

**Movement Analysis**: Implements advanced motion detection algorithms to analyze patient movements, including:
 *  Overall motion detection using Structural Similarity Index (SSIM) method for measuring the similarity between two images.
 *  Specific movement detection for head, hands, legs, and chest (for breathlessness)

**Fall Detection**: Incorporates a separate YOLOv8 model trained specifically for detecting fall incidents.

**Real-time Alerting**: Generates visual alerts on the video feed when critical events of patients are detected.

**Movement Logging**: Records detected movements with timestamps in a log file for later analysis.

**Video Output**: Processes and saves the annotated video feed with detected objects, movements, and alerts.

# 🛠️ Technical Approach

**Video Processing**

* OpenCV (cv2) is used for video capture, processing, and output.
* The system supports multiple video sources, including local files, YouTube links (using yt-dlp), and live CCTV feeds.

**Object Detection**

Fine-Tuned YOLOv8 models are employed for object detection:

1. A general model ('person_labelling.pt') for detecting patients, doctors, nurses, monitor and others.
2. A specialized model ('fall_detection.pt') for fall detection.

**Movement Analysis**

* Optical flow (cv2.calcOpticalFlowFarneback) is used to detect and analyze specific movements.

* The patient's body is analysed into regions (head, hands, legs, chest) for targeted movement analysis.

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
      * Detects various individuals in the ICU, including doctors, nurses, and patients using the Fine-Tuned YOLOv8 model (person_labelling.pt).
      * Identifies when a patient is without a doctor or nurse and changes the bounding box color accordingly:
          * Green: Indicates presence of a doctor or nurse.
          * Yellow: Indicates no significant motion detected and no doctor/nurse present.
          * Blue: Indicates significant motion detected without specific movements.
          * Red: Indicates significant movement or specific detected movements (e.g., head, hand, leg, breathlessness).

3. **Specific Movement Detection**
      * Detects and analyzes specific patient movements such as head, hand, leg movements, and signs of breathlessness using optical flow and structural similarity metrics.

4. **Fall Detection and Alert System**
      * Uses a secondary Fine-Tuned YOLOv8 model (fall_detection.pt) for detecting falling of the patient and other dangerous movements of the patient.
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

* Utilize IoT devices (wearable sensors, smart beds) for continuous patient monitoring.
* Implement more sophisticated algorithms for detecting complex patient behaviors.
* Integrate with hospital information systems for comprehensive patient monitoring.
* Develop a user-friendly interface for healthcare professionals to monitor multiple patients.
* Utilize IoT devices (wearable sensors, smart beds) for continuous patient monitoring.
* Create alert systems that prioritize alerts based on severity, notifying healthcare staff promptly to enable quick interventions.
* Forecast patient movements and health changes using historical and real-time data.


# ⏳Conclusion

The TeleICU Monitoring System revolutionizes remote ICU patient care by integrating advanced video processing and deep learning techniques. Our system supports versatile video input from local files, URLs, and live CCTV feeds, ensuring flexibility in various monitoring scenarios. Leveraging the YOLOv8 model, it accurately detects and identifies individuals such as doctors, nurses, patients, monitor, others, dynamically changing bounding box colors to indicate the presence of healthcare professionals and specific patient movements. Unique features include detailed analysis of head, hand, leg movements, signs of breathlessness, and a robust fall detection system that generates real-time alerts. The system processes and saves annotated video feeds in real-time, accompanied by comprehensive movement logs with timestamps, ensuring accurate documentation for future analysis. By providing these innovative tools, our TeleICU Monitoring System significantly enhances patient safety and care quality, setting a new standard in telemedicine and remote ICU monitoring.

# 🤖 Installation and Run Guide

# 📽️ Guide Video
      Guide_video.mp4

## Prerequisites
* Python 3.8 
* Git / Github Desktop

## Installation Guide

Step 1: Clone the Repository
First, clone the repository from GitHub.



    git clone https://github.com/DMHACKERZ/TeleICU-Monitoring-System
    cd your-repo-directory

Step 2: Set Up a Virtual Environment
Create and activate a virtual environment to manage dependencies.
* Ensure you run all the following commands inside "your-repo-directory"

**Windows**
cmd

    python -m venv environment_name
    environment_name\Scripts\activate

**Mac and Linux**

    python3 -m venv environment_name
    source environment_name/bin/activate

Step 3: Install Dependencies
Install the required Python packages using pip.


     pip install -r requirements.txt



## Running the Code

Step 1: Ensure YOLO Models Are Available
Ensure that the Fine-Tuned YOLO models (person_labelling.pt and fall_detection.pt) are available in the project directory.

Step 2: Run the Script
To run the script, execute the following command:


    python icu.py

**Usage Guide**

Prompt for Video Source: After running the script, you will be prompted to enter the video source. You can provide a file path, a YouTube URL, or type 'cctv' for live camera feed.

Example inputs:

    Enter the video file path, URL, or 'cctv' for live feed: example_video.mp4
    Enter the video file path, URL, or 'cctv' for live feed: https://www.youtube.com/watch?v=example
    Enter the video file path, URL, or 'cctv' for live feed: cctv
    
**Monitor the Output**: The script will process the video and display a window with live predictions. Detected movements will be logged in "movement_log.txt" in real-time. If a fall is detected, the alert will be displayed on the video feed.

**Terminate the Script**: To terminate the script, press q while the video window is active.

**Platform-Specific Notes**

Windows
* Ensure you have the correct drivers installed for your GPU if using CUDA for acceleration.
  
Mac
* Ensure you have Xcode command line tools installed for compiling certain packages.
* Use python3 instead of python if your system defaults to Python 2.

Linux
* Ensure you have the necessary build tools installed. You can typically install them via your package manager. For example, on Ubuntu, you can run:

      sudo apt-get update
      sudo apt-get install build-essential
## Troubleshooting
* If you encounter issues with yt-dlp, ensure you have the latest version installed.
* For any dependency issues, verify that all packages in "requirements.txt" are installed correctly and compatible with your Python version.
* Check the logs and error messages for specific issues and resolve them accordingly. Common issues might include missing files, incorrect paths, or compatibility problems.

# Contributing

Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.
