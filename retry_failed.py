import json
import os
import time
from PIL import Image
from vlm_interface import QwenVLMInterface
from datetime import datetime

RESULTS_FILE = "experiment_results_20260228_203043.json"
DATASET_FILE = "grasp_force_dataset.json"
IMAGES_DIR = "images"
MAX_IMAGE_PIXELS = 1024 * 1024  # Resize if > 1 megapixel

with open(RESULTS_FILE) as f:
    results = json.load(f)

with open(DATASET_FILE) as f:
    objects = {obj["id"]: obj for obj in json.load(f)}

failed_ids = [r["object_id"] for r in results if not r["success"]]
print(f"Retrying {len(failed_ids)} failed objects: {failed_ids}")

def ensure_reasonable_size(image_path: str) -> str:
    """Resize image if too large, return path to use."""
    img = Image.open(image_path)
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    w, h = img.size
    total = w * h
    if total > MAX_IMAGE_PIXELS:
        scale = (MAX_IMAGE_PIXELS / total) ** 0.5
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        resized_path = image_path.replace(".jpg", "_resized.jpg")
        img.save(resized_path, "JPEG", quality=85)
        size_kb = os.path.getsize(resized_path) / 1024
        print(f"  Resized {w}x{h} -> {new_w}x{new_h} ({size_kb:.0f}KB)")
        return resized_path
    # Even if not resized, re-save as proper JPEG to fix format issues
    fixed_path = image_path.replace(".jpg", "_fixed.jpg")
    img.save(fixed_path, "JPEG", quality=90)
    return fixed_path

vlm = QwenVLMInterface()

retry_results = []

for obj_id in failed_ids:
    obj = objects[obj_id]
    name = obj["name"]
    print(f"\n[{obj_id:03d}] Retrying: {name}")

    image_path = os.path.join(IMAGES_DIR, f"{obj_id:03d}_angle1.jpg")
    if not os.path.exists(image_path):
        print(f"  Still missing: {image_path}")
        continue

    use_path = ensure_reasonable_size(image_path)

    result = vlm.predict_force(use_path, name)

    entry = {
        "object_id": obj_id,
        "object_name": name,
        "category": obj["category"],
        "ground_truth_force_N": obj["ground_truth_force_N"],
        "ground_truth_mass_kg": obj["mass_kg"],
        "ground_truth_material": obj["material"],
        "ground_truth_fragility": obj["fragility"],
        "deceptive": obj["deceptive"],
        "max_safe_force_N": obj["max_safe_force_N"],
        "image_path": use_path,
        "timestamp": datetime.now().isoformat(),
        **result
    }

    if result["success"]:
        pred = result["prediction"].get("required_grip_force_newtons", "N/A")
        gt = obj["ground_truth_force_N"]
        print(f"  Predicted: {pred}N | Ground Truth: {gt}N")
    else:
        print(f"  Still failed: {result.get('error', 'unknown')[:80]}")

    retry_results.append(entry)
    time.sleep(0.5)

# Merge retries into original results
result_map = {r["object_id"]: r for r in results}
for retry in retry_results:
    result_map[retry["object_id"]] = retry

merged = sorted(result_map.values(), key=lambda x: x["object_id"])

output_file = f"experiment_results_merged.json"
with open(output_file, "w") as f:
    json.dump(merged, f, indent=2)

successful = sum(1 for r in merged if r["success"])
print(f"\nMerged results saved to {output_file}")
print(f"Total: {len(merged)} | Successful: {successful} | Failed: {len(merged) - successful}")
