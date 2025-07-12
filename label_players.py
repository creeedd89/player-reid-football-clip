import json

# Load player matches
with open("player_matches.json", "r") as f:
    matches = json.load(f)

# Print all matches
for tacticam_file, info in matches.items():
    broadcast_file = info["matched_broadcast_file"]
    similarity = info["similarity"]

    print(f"{tacticam_file} → {broadcast_file} (similarity: {similarity:.2f})")
