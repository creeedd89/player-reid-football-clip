import cv2
from ultralytics import YOLO
import os

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # pretrained on COCO

def detect_players(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]  # "broadcast" or "tacticam"

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    all_detections = []  # This will store all detection info

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection on current frame
        results = model(frame)
        boxes = results[0].boxes
        frame_detections = []

        player_id = 0
        for box in boxes:
            if int(box.cls[0]) == 0:  # class 0 = person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]

                # Save crop with video name in file name
                filename = f"{video_name}_frame{frame_idx}_player{player_id}.jpg"
                save_path = os.path.join(output_dir, filename)
                cv2.imwrite(save_path, crop)

                # Store detection info
                frame_detections.append({
                    "video": video_name,
                    "frame": frame_idx,
                    "player_id": player_id,
                    "bbox": [x1, y1, x2, y2],
                    "image_path": save_path
                })
                player_id += 1

        print(f"[{video_name}] Frame {frame_idx}: {player_id} players detected")
        all_detections.append(frame_detections)
        frame_idx += 1

    cap.release()
    print(f"✅ Finished processing {video_name}")
    return all_detections
