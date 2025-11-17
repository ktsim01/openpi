import json
import numpy as np

def fix_image_stats(stats: dict):
    """
    Fix image stats in-place:
    - Any (1,1,1) → (3,1,1)
    - count stays (1,)
    """

    for fkey, feature_stats in stats.items():
        if "image" not in fkey:
            continue

        for stat_name, value in feature_stats.items():
            # --- preserve count shape (1,) ---
            if stat_name == "count":
                arr = np.array(value)
                if arr.shape != (1,):
                    feature_stats[stat_name] = arr.reshape(1).tolist()
                continue

            # convert nested list → np array
            arr = np.array(value, dtype=float)

            # valid case, skip
            if arr.shape == (3, 1, 1):
                continue

            # fix grayscale image stats (1,1,1)
            if arr.shape == (1, 1, 1):
                arr = np.repeat(arr, 3, axis=0)  # → (3,1,1)
                feature_stats[stat_name] = arr.tolist()
                continue

            # unexpected but small → expand to (3,1,1)
            if arr.size == 1:
                arr = np.repeat(arr.reshape(1,1,1), 3, axis=0)
                feature_stats[stat_name] = arr.tolist()
                continue

            # debug help
            print(f"[WARN] Unexpected shape {arr.shape} for {fkey}.{stat_name}. Skipping.")


def fix_jsonl_file(input_path: str, output_path: str):
    """
    Reads a JSONL file, fixes each line's stats, writes corrected JSONL.
    """
    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            if not line.strip():
                continue   # skip empty lines

            obj = json.loads(line)

            if "stats" in obj:
                fix_image_stats(obj["stats"])

            fout.write(json.dumps(obj) + "\n")

    print(f"✔ Fixed JSONL written to {output_path}")


# Example usage
if __name__ == "__main__":
    fix_jsonl_file(
        "/home/ktsim/.cache/huggingface/lerobot/sriramsk/fold_onesie_human_multiview_20251113/meta/episodes_stats.jsonl",
        "/home/ktsim/.cache/huggingface/lerobot/sriramsk/fold_onesie_human_multiview_20251113/meta/episodes_stats_fixed.jsonl"
    )
