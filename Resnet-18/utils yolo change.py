import os
import cv2
from tqdm import tqdm

image_dir = r"\helmetAndVest\helmet_reflective/images"
label_dir = r"\helmetAndVest\helmet_reflective/labels"
class_file = r"\helmetAndVest\helmet_reflective/classes.txt"
output_dir = r"\helmetAndVest\helmet_reflective/cropped_dataset"

# === Get the name of classes ===
with open(class_file, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# === Build output directory ===
os.makedirs(output_dir, exist_ok=True)
for cls in classes:
    os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

# === get each image ===
for img_name in tqdm(os.listdir(image_dir)):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(image_dir, img_name)
    label_name = os.path.splitext(img_name)[0] + ".txt"
    label_path = os.path.join(label_dir, label_name)

    if not os.path.exists(label_path):
        print(f"[Skip] No such a image corresponding to image here: {label_path}")
        continue

    image = cv2.imread(image_path)
    if image is None:
        print(f"[Error] Failed to load image: {image_path}")
        continue

    height, width = image.shape[:2]

    with open(label_path, "r") as f:
        for i, line in enumerate(f.readlines()):
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"[Warning] Unexpected label structure: {label_path} in line {i+1}：{line.strip()}")
                continue

            try:
                class_id, x_center, y_center, w, h = map(float, parts)
                class_id = int(class_id)
                class_name = classes[class_id]
            except Exception as e:
                print(f"[Error] Failed to pare the tag: {label_path} in line {i+1}：{line.strip()} → {e}")
                continue

            x1 = int((x_center - w / 2) * width)
            y1 = int((y_center - h / 2) * height)
            x2 = int((x_center + w / 2) * width)
            y2 = int((y_center + h / 2) * height)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            print(f"[DEBUG] {img_name} object {i+1}=：({x1}, {y1}) → ({x2}, {y2}) | width-length: {x2 - x1}x{y2 - y1}")

            if x2 <= x1 or y2 <= y1:
                print(f"[Skip] Invalid crop area")
                continue

            crop = image[y1:y2, x1:x2]
            crop_resized = cv2.resize(crop, (224, 224))
            save_path = os.path.join(output_dir, class_name, f"{os.path.splitext(img_name)[0]}_{i}.jpg")
            cv2.imwrite(save_path, crop_resized)
            print(f"[Success save to] {save_path}")
