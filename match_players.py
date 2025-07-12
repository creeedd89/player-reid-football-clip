import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 🔹 Load features with error handling
def load_json(filename):
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        print(f"Loaded {filename} with {len(data)} entries.")
        return data
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        exit(1)

def flatten_feat(feat):
    # If feature is [[...]], flatten to [...]
    if isinstance(feat, list) and len(feat) == 1 and isinstance(feat[0], list):
        return feat[0]
    return feat

broadcast_feats = load_json("broadcast_features.json")
tacticam_feats = load_json("tacticam_features.json")

# 🔹 Check feature vector lengths
broadcast_keys = list(broadcast_feats.keys())
broadcast_vecs_list = [flatten_feat(broadcast_feats[k]) for k in broadcast_keys]
feat_lengths = [len(vec) for vec in broadcast_vecs_list]
if len(set(feat_lengths)) != 1:
    print(f"Inconsistent feature lengths in broadcast_features.json: {set(feat_lengths)}")
    exit(1)

broadcast_vecs = np.array(broadcast_vecs_list)
print(f"Broadcast feature matrix shape: {broadcast_vecs.shape}")
print(f"Sample broadcast feature shape: {np.array(broadcast_vecs_list[0]).shape}")

# 🔹 Match each tacticam player to closest broadcast player
matches = {}

for i, (tac_file, tac_feat) in enumerate(tacticam_feats.items()):
    tac_feat_flat = flatten_feat(tac_feat)
    if len(tac_feat_flat) != broadcast_vecs.shape[1]:
        print(f"Feature length mismatch for {tac_file}: {len(tac_feat_flat)} vs {broadcast_vecs.shape[1]}")
        continue
    tac_vec = np.array(tac_feat_flat).reshape(1, -1)  # Make it 2D for similarity
    sims = cosine_similarity(tac_vec, broadcast_vecs)[0]

    best_idx = np.argmax(sims)
    best_match_file = broadcast_keys[best_idx]
    best_score = sims[best_idx]

    matches[tac_file] = {
        "matched_broadcast_file": best_match_file,
        "similarity": float(best_score)
    }
    if i < 3:  # Print first 3 matches for debugging
        print(f"Tacticam: {tac_file} -> Broadcast: {best_match_file} (sim={best_score:.4f})")

# 🔹 Save results with error handling
try:
    with open("player_matches.json", "w") as f:
        json.dump(matches, f, indent=2)
    print("✅ Player matching complete. Results saved to player_matches.json")
except Exception as e:
    print(f"Error saving player_matches.json: {e}")
