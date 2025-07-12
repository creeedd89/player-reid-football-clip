import json
import csv

# Load matches
with open("player_matches.json", "r") as f:
    matches = json.load(f)

# Write to CSV
with open("matches_report.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Tacticam File", "Matched Broadcast File", "Similarity Score"])

    for tac_file, info in matches.items():
        writer.writerow([tac_file, info["matched_broadcast_file"], round(info["similarity"], 4)])

print("✅ matches_report.csv has been created.")
