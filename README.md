# Soccer Player & Ball Detection and Visualization
## MY FINAL VIDEO


<iframe width="560" height="315" src="https://www.youtube.com/embed/BPc30WZTJys?autoplay=1&si=HmcaE_9dZEwEB6fu" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe> 



## Table of Contents
- [Project Motivation & Goals](#project-motivation--goals)
- [Workflow Overview](#workflow-overview)
- [Data Preparation](#data-preparation)
- [File Structure](#file-structure)
- [Setup & Installation](#setup--installation)
- [Script Details & Usage](#script-details--usage)
  - [1. Player Matching (Feature-based)](#1-player-matching-feature-based)
  - [2. Print Player Matches](#2-print-player-matches)
  - [3. Visualize Matched Player Overlays](#3-visualize-matched-player-overlays)
  - [4. Detect and Label Players & Ball with YOLO](#4-detect-and-label-players--ball-with-yolo)
- [Data Format Examples](#data-format-examples)
- [YOLO Class Mapping & Customization](#yolo-class-mapping--customization)
- [Troubleshooting & Tips](#troubleshooting--tips)
- [Extending the Project](#extending-the-project)
- [Credits & References](#credits--references)

---

## Project Motivation & Goals
This project aims to provide a robust, reproducible pipeline for:
- Matching and tracking soccer players across multiple camera views (broadcast and tacticam)
- Visualizing player correspondences and detections for sports analytics
- Automatically detecting and labeling both players and the ball in broadcast video using state-of-the-art object detection (YOLOv8)

**Use cases:**
- Player re-identification and tracking
- Sports broadcast analysis
- Automated highlight generation
- Research in computer vision for sports

## Workflow Overview
1. **Data Preparation:**
   - Extract player crops from both broadcast and tacticam videos
   - Compute feature vectors for each crop
   - Prepare YOLOv8 model weights
2. **Player Matching:**
   - Use feature similarity to match tacticam players to broadcast players
   - Output: `player_matches.json`
3. **Visualization:**
   - Overlay matched tacticam crops on broadcast video for visual verification
   - Detect and label all players and the ball in broadcast video using YOLOv8
   - Output: Annotated videos
4. **Analysis:**
   - Use JSON and video outputs for further analytics or research

## Data Preparation
- **Crops:**
  - Extracted using a detection model or manual annotation
  - Saved as `crops/broadcast/broadcast_frame{frame}_player{index}.jpg` and `crops/tacticam/tacticam_frame{frame}_player{index}.jpg`
- **Feature Extraction:**
  - Use a deep feature extractor (e.g., a re-ID model) to generate a feature vector for each crop
  - Save as JSON: `{ "filename.jpg": [f1, f2, ..., fn], ... }`
- **YOLO Model:**
  - Download or train a YOLOv8 model (e.g., `yolov8n.pt`)
  - Supports COCO classes (person, sports ball, etc.)

## File Structure
```
.
├── broadcast.mp4                # Broadcast video
├── tacticam.mp4                 # Tacticam video
├── broadcast_features.json      # Features for broadcast player crops
├── tacticam_features.json       # Features for tacticam player crops
├── player_matches.json          # Output: tacticam-to-broadcast player matches
├── crops/
│   ├── broadcast/               # Cropped player images from broadcast
│   └── tacticam/                # Cropped player images from tacticam
├── yolov8n.pt                   # YOLOv8 model weights
├── match_players.py             # Script: match tacticam to broadcast players
├── label_players.py             # Script: print player matches
├── visualize_matches.py         # Script: overlay matched crops on broadcast video
├── visualize_yolo_labels.py     # Script: detect and label players & ball with YOLO
└── README.md                    # This file
```

## Setup & Installation
1. **Install Python 3.8+**
2. **Install dependencies:**
   ```bash
   pip install numpy opencv-python scikit-learn ultralytics
   ```
3. Place your videos, feature files, and YOLO model in the project directory as shown above.

## Script Details & Usage
### 1. Player Matching (Feature-based)
**Script:** `match_players.py`
- **Inputs:** `broadcast_features.json`, `tacticam_features.json`
- **Logic:**
  - Loads feature vectors for all crops
  - Computes cosine similarity between each tacticam crop and all broadcast crops
  - Assigns each tacticam crop to the most similar broadcast crop
- **Output:** `player_matches.json` (mapping: tacticam crop -> matched broadcast crop + similarity score)
- **Command:**
  ```bash
  python match_players.py
  ```

### 2. Print Player Matches
**Script:** `label_players.py`
- **Inputs:** `player_matches.json`
- **Logic:**
  - Reads the JSON file and prints all tacticam-to-broadcast player matches with similarity scores
- **Command:**
  ```bash
  python label_players.py
  ```

### 3. Visualize Matched Player Overlays
**Script:** `visualize_matches.py`
- **Inputs:** `broadcast.mp4`, `crops/tacticam/`, `player_matches.json`
- **Logic:**
  - For each frame in the broadcast video, overlays the matched tacticam player crops in the top-left corner
  - Each overlay is semi-transparent, labeled, and stacked vertically
- **Output:** `broadcast_with_overlays.mp4`
- **Command:**
  ```bash
  python visualize_matches.py
  ```

### 4. Detect and Label Players & Ball with YOLO
**Script:** `visualize_yolo_labels.py`
- **Inputs:** `broadcast.mp4`, `yolov8n.pt`
- **Logic:**
  - Runs YOLOv8 on each frame of the broadcast video
  - Draws bounding boxes and labels for each detected player ("Player 1", "Player 2", ...) and for the sports ball ("Ball")
  - Uses COCO class 0 for person and class 32 for sports ball
- **Output:** `broadcast_with_yolo_labels.mp4`
- **Command:**
  ```bash
  python visualize_yolo_labels.py
  ```
- **Example Output:**
  - Players are labeled as "Player 1", "Player 2", ... in each frame
  - The sports ball is labeled as "Ball" with an orange bounding box

## Data Format Examples
### Crop Filenames
```
crops/broadcast/broadcast_frame12_player8.jpg
crops/tacticam/tacticam_frame54_player7.jpg
```
### Feature JSON
```json
{
  "broadcast_frame12_player8.jpg": [0.1, 0.2, ..., 0.5],
  "broadcast_frame10_player9.jpg": [0.3, 0.4, ..., 0.6]
}
```
### Player Matches JSON
```json
{
  "tacticam_frame54_player7.jpg": {
    "matched_broadcast_file": "broadcast_frame115_player6.jpg",
    "similarity": 0.96
  },
  ...
}
```

## YOLO Class Mapping & Customization
- **Person:** COCO class 0 (red bounding box, green label)
- **Sports Ball:** COCO class 32 (orange bounding box, orange label)
- You can customize the script to detect other classes (see [COCO class list](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml))
- To use a different YOLO model, change the `YOLO_MODEL` path in the script

## Troubleshooting & Tips
- **No bounding boxes or labels?**
  - Check that your YOLO model supports the COCO classes and is correctly loaded
  - Make sure your input video is readable and in the correct format
- **Missing crops or features?**
  - Ensure your filenames and JSON keys match the expected format
- **Slow processing?**
  - Reduce video frame rate or resolution for faster results
  - Use a smaller YOLO model (e.g., `yolov8n.pt`)
- **Want to track players across frames?**
  - Integrate a tracking algorithm (e.g., SORT, DeepSORT) for consistent player IDs
- **Customizing overlays or labels?**
  - Edit the visualization scripts to change colors, font sizes, or overlay positions

## Extending the Project
- **Player Tracking:** Integrate tracking-by-detection (SORT, DeepSORT) for consistent player IDs across frames
- **New Classes:** Add detection and labeling for referees, coaches, or other objects by updating the YOLO class list
- **Advanced Analytics:** Use the output JSON and videos for heatmaps, player statistics, or tactical analysis
- **Web Dashboard:** Build a web interface to visualize and interact with the results

## Credits & References
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- [OpenCV](https://opencv.org/) for video and image processing
- [scikit-learn](https://scikit-learn.org/) for feature similarity
- [COCO Dataset](https://cocodataset.org/#home) for class definitions

---
Feel free to modify and extend this project for your own sports analytics and visualization needs! 

# Soccer Player Re-identification

This project aims to re-identify soccer players across different camera views (broadcast and tacticam) using YOLOv8 for player detection and a re-identification model for matching. 

