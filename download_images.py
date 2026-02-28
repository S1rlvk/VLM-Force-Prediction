import json
import os
import shutil
import time
import glob

from bing_image_downloader import downloader

DATASET_PATH = "grasp_force_dataset.json"
RAW_DIR = "images_raw"
OUTPUT_DIR = "images"
IMAGES_PER_OBJECT = 3

with open(DATASET_PATH, "r") as f:
    objects = json.load(f)

os.makedirs(OUTPUT_DIR, exist_ok=True)

failed = []

for obj in objects:
    obj_id = obj["id"]
    name = obj["name"]
    prefix = f"{obj_id:03d}"

    existing = glob.glob(os.path.join(OUTPUT_DIR, f"{prefix}_*.jpg"))
    if len(existing) >= IMAGES_PER_OBJECT:
        print(f"[SKIP] ID {obj_id}: {name} — already have {len(existing)} images")
        continue

    print(f"[{obj_id:02d}/50] Downloading: {name}")

    try:
        downloader.download(
            name,
            limit=IMAGES_PER_OBJECT,
            output_dir=RAW_DIR,
            adult_filter_off=True,
            force_replace=False,
            timeout=20,
        )
    except Exception as e:
        print(f"  ERROR downloading '{name}': {e}")
        failed.append({"id": obj_id, "name": name, "error": str(e)})
        continue

    src_dir = os.path.join(RAW_DIR, name)
    if not os.path.isdir(src_dir):
        print(f"  WARNING: no folder found at {src_dir}")
        failed.append({"id": obj_id, "name": name, "error": "no download folder"})
        continue

    downloaded = sorted(
        [f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"))],
    )

    if not downloaded:
        print(f"  WARNING: 0 images downloaded for '{name}'")
        failed.append({"id": obj_id, "name": name, "error": "0 images"})
        continue

    for idx, fname in enumerate(downloaded[:IMAGES_PER_OBJECT], start=1):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in (".jpg", ".jpeg"):
            ext = ".jpg"
        new_name = f"{prefix}_angle{idx}{ext}"
        src = os.path.join(src_dir, fname)
        dst = os.path.join(OUTPUT_DIR, new_name)
        shutil.copy2(src, dst)
        print(f"  -> {new_name}")

    time.sleep(0.5)

print("\n" + "=" * 50)
print(f"Done. Images saved to: {OUTPUT_DIR}/")

final_images = [f for f in os.listdir(OUTPUT_DIR) if not f.startswith(".")]
print(f"Total image files: {len(final_images)}")

if failed:
    print(f"\nFailed downloads ({len(failed)}):")
    for f in failed:
        print(f"  ID {f['id']:03d}: {f['name']} — {f['error']}")

if os.path.isdir(RAW_DIR):
    shutil.rmtree(RAW_DIR)
    print(f"\nCleaned up temp directory: {RAW_DIR}/")
