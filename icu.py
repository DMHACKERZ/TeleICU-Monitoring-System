import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from ultralytics import YOLO
import os
import datetime
import yt_dlp
import time
import torch
import threading

# Global variables to manage the thread
fall_detect_thread = None
fall_detect_end_time = 0
curr_frame = None
frame_width = 0
frame_height = 0
timestamp = 0
fall_danger_model = None


# Function to load video from a path, link, or CCTV
def load_video(source):
    if source.lower() == 'cctv':
        return cv2.VideoCapture(0)  # Use 0 or the appropriate index for your camera
    elif os.path.isfile(source):
        return cv2.VideoCapture(source)
    elif source.startswith('http'):
        try:
            # Download video from YouTube
            ydl_opts = {
                'format': 'best[ext=mp4][height<=360]',
                'outtmpl': '%(title)s.%(ext)s',
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(source, download=True)
                video_path = ydl.prepare_filename(info_dict)
            return cv2.VideoCapture(video_path)
        except Exception as e:
            raise ValueError(f"Error downloading video: {e}")
    else:
        raise ValueError("Invalid video source. Provide a valid file path, URL, or 'cctv'.")

# Function to check if a specific class is present in the results
def contains_class(results, class_name):
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = result.names[cls_id] if cls_id in result.names else 'Unknown'
            if label == class_name:
                return True
    return False

# Function to detect specific movements
def detect_specific_movement(prev_roi, curr_roi):
    # Convert ROIs to grayscale
    prev_gray = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_roi, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Calculate magnitude and angle of 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Define regions for different body parts
    height, width = prev_roi.shape[:2]
    head_region = (width // 4, 0, width * 3 // 4, height // 3)
    hand_region_left = (0, height // 3, width // 3, height * 2 // 3)
    hand_region_right = (width * 2 // 3, height // 3, width, height * 2 // 3)
    leg_region = (0, height * 2 // 3, width, height)
    chest_region = (width // 4, height // 3, width * 3 // 4, height * 2 // 3)
    
    # Function to check movement in a region
    def check_movement(region, threshold):
        roi = magnitude[region[1]:region[3], region[0]:region[2]]
        return np.mean(roi) > threshold and np.max(roi) > threshold * 2
    
    # Check for movement in different regions
    head_movement = check_movement(head_region, 1.0)
    hand_movement_left = check_movement(hand_region_left, 1.5)
    hand_movement_right = check_movement(hand_region_right, 1.5)
    leg_movement = check_movement(leg_region, 2.0)
    
    # Check for breathlessness (rapid chest movement)
    chest_roi = magnitude[chest_region[1]:chest_region[3], chest_region[0]:chest_region[2]]
    chest_movement = np.mean(chest_roi)
    breathlessness = chest_movement > 2.0 and chest_movement < 8.0       # Adjust these thresholds as needed
    
    movements = []
    if head_movement:
        movements.append("Head")
    if hand_movement_left or hand_movement_right:
        movements.append("Hand")
    if leg_movement:
        movements.append("Leg")
    if breathlessness:
        movements.append("Breathlessness")
    
    return movements

def fall_detect_loop(cv2_):
    global fall_detect_end_time, curr_frame, frame_width, frame_height, timestamp, fall_danger_model
    
    alert_start_time = None
    alert_duration = 5  # 5 seconds

    while time.time() < fall_detect_end_time:
        if curr_frame is not None:
            results = fall_danger_model(curr_frame)
            detection = ''
            for result in results:
                boxes = result.boxes.cpu().numpy()
            
                for box in boxes:
                    cls_id = int(box.cls[0])
                    label = result.names[cls_id] if cls_id in result.names else 'Unknown'
                    if label in ('fall'):
                        detection += ' ' + label
                        with open('movement_log.txt', 'a', encoding='utf-8') as f:
                            f.write(f"{datetime.timedelta(seconds=int(timestamp))}: {label}\n")
            
            if detection:
                alert_start_time = time.time()

            if alert_start_time and time.time() - alert_start_time < alert_duration:
                cv2_.rectangle(curr_frame, (0, 0), (frame_width, frame_height), (0,0,255), 5)
                cv2_.putText(curr_frame, '! ALERT !', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            else:
                alert_start_time = None

        time.sleep(0.1)  # Small delay to prevent CPU overuse

def fall_detect(cv2_):
    global fall_detect_thread, fall_detect_end_time
    
    current_time = time.time()
    
    if fall_detect_thread is None or not fall_detect_thread.is_alive():
        fall_detect_end_time = current_time + 5
        fall_detect_thread = threading.Thread(target=fall_detect_loop, args=(cv2_,))
        fall_detect_thread.start()
    else:
        fall_detect_end_time = max(fall_detect_end_time, current_time + 5)

def main():
    global curr_frame, frame_width, frame_height, timestamp, fall_danger_model

    # Load the YOLOv8 model
    model = YOLO('person_labelling.pt').to(device)
    fall_danger_model = YOLO('fall_detection.pt').to(device)
    # Get the video source from the user
    video_source = input("Enter the video file path, URL, or 'cctv' for live feed: ")

    try:
        # Open the input video
        cap = load_video(video_source)

        if not cap.isOpened():
            print("Error: Unable to open video source.")
            return

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Define the codec and create VideoWriter object for saving the output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if video_source.lower() == 'cctv':
            video_name = 'cctv_output.mp4'
        else:
            video_name = os.path.splitext(os.path.basename(video_source))[0] + '_output.mp4'
        output_video_path = video_name
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        # Read the first frame for motion detection initialization
        ret, prev_frame = cap.read()
        if not ret:
            print("Failed to read the first frame.")
            return

        frame_count = 0

        fall_detect(cv2)

        alert_start_time = None
        alert_duration = 5  # 5 seconds

        # Open the movement log file for writing in real-time
        with open('movement_log.txt', 'w') as log_file:
            while cap.isOpened():
                ret, curr_frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                timestamp = frame_count / fps          # Calculate timestamp in seconds

                # Perform inference with the main model
                results = model(curr_frame)

                # Check for the presence of doctor or nurse
                doctor_or_nurse_present = contains_class(results, 'doctor') or contains_class(results, 'nurse')

                # Process each detection result
                for result in results:
                    boxes = result.boxes.cpu().numpy()
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = box.conf[0]
                        cls_id = int(box.cls[0])
                        label = result.names[cls_id] if cls_id in result.names else 'Unknown'

                        if label == 'patient' and not doctor_or_nurse_present:
                            # Motion detection within patient box
                            roi_prev = prev_frame[y1:y2, x1:x2]
                            roi_curr = curr_frame[y1:y2, x1:x2]

                            # Ensure the regions are the same size
                            if roi_prev.shape == roi_curr.shape:
                                # Compute SSIM between the current and previous frames
                                ssim_index = ssim(cv2.cvtColor(roi_prev, cv2.COLOR_BGR2GRAY), cv2.cvtColor(roi_curr, cv2.COLOR_BGR2GRAY))

                                # Define a threshold for SSIM to detect motion
                                ssim_threshold = 0.95  # This value may need adjustment based on your video

                                if ssim_index < ssim_threshold:
                                    color = (255, 0, 0)  # Blue if significant motion detected
                                    # Detect specific movements
                                    movements = detect_specific_movement(roi_prev, roi_curr)
                                    if movements:
                                        color = (0, 0, 255)     # Red if specific movements detected
                                        movement_text = ", ".join(movements)
                                        # Log the movement with timestamp
                                        log_entry = f"{datetime.timedelta(seconds=int(timestamp))}: {movement_text}\n"
                                        log_file.write(log_entry)
                                        log_file.flush()    # Ensure the log is written in real-time


                                        fall_detect(cv2)
                                    else:
                                        movement_text = ""
                                else:
                                    color = (0, 255, 255)       # Yellow if no significant motion detected and no doctor/nurse
                                    movement_text = ""
                            else:
                                color = (0, 255, 255)           # Yellow if regions do not match exactly
                                movement_text = ""
                        else:
                            color = (0, 255, 0)                 # Green for all other cases
                            movement_text = ""

                        cv2.rectangle(curr_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(curr_frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                        if movement_text:
                            cv2.putText(curr_frame, movement_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Check if alert should be displayed
                if alert_start_time and time.time() - alert_start_time < alert_duration:
                    cv2.rectangle(curr_frame, (0, 0), (frame_width, frame_height), (0,0,255), 5)
                    cv2.putText(curr_frame, '! ALERT !', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                else:
                    alert_start_time = None

                 # Display the frame
                cv2.imshow('Live Prediction', curr_frame)
                
                # Write the frame with predictions to the output video
                out.write(curr_frame)

                 # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Update frames for motion detection
                prev_frame = curr_frame
        
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main()
