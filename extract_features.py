import os
import clip
import torch
from PIL import Image
import json

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)

# Extract features from a folder
def extract_features_from_folder(folder_path):
    features_dict = {}

    for file in os.listdir(folder_path):
        if file.endswith(".jpg"):
            image_path = os.path.join(folder_path, file)
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

            with torch.no_grad():
                feature = model.encode_image(image)
                feature = feature / feature.norm(dim=-1, keepdim=True)  # Normalize

            features_dict[file] = feature.cpu().numpy().tolist()

    return features_dict

# Save features from both videos
def save_features():
    for cam in ["broadcast", "tacticam"]:
        folder = os.path.join("crops", cam)
        features = extract_features_from_folder(folder)

        with open(f"{cam}_features.json", "w") as f:
            json.dump(features, f, indent=2)
        print(f"✅ Features saved for {cam}")

# Run it
if __name__ == "__main__":
    save_features()
