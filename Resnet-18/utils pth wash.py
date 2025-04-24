import torch
import os

checkpoint_dir = "checkpoints"
output_dir = "checkpoints_clean"

os.makedirs(output_dir, exist_ok=True)

def clean_state_dict(state):
    if isinstance(state, dict):
        if "model" in state:
            return state["model"]
        else:
            return {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}
    else:
        raise ValueError("Unrecognized state_dict format.")

for filename in os.listdir(checkpoint_dir):
    if filename.endswith(".pth"):
        path = os.path.join(checkpoint_dir, filename)
        print(f"Processing: {filename}")

        state = torch.load(path, map_location="cpu")

        try:
            cleaned = clean_state_dict(state)
            new_path = os.path.join(output_dir, filename)
            torch.save(cleaned, new_path)
            print(f"✅ Saved cleaned: {new_path}")
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
