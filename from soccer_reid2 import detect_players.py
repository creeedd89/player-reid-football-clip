from soccer_reid2 import detect_players

# Detect players from both videos
detect_players("broadcast.mp4", "crops/broadcast")
detect_players("tacticam.mp4", "crops/tacticam")
