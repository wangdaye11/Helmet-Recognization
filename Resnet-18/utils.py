import os
import cv2
from tqdm import tqdm

# === 修改路径为你自己的 ===
image_dir = r"C:/Users/92186/Desktop/work folder/helmetAndVest/images"
label_dir = r"C:/Users/92186/Desktop/work folder/helmetAndVest/labels"
class_file = r"C:/Users/92186/Desktop/work folder/helmetAndVest/classes.txt"
output_dir = r"C:/Users/92186/Desktop/work folder/helmetAndVest/cropped_dataset"

# === 读取类别名称 ===
with open(class_file, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# === 创建输出目录结构 ===
os.makedirs(output_dir, exist_ok=True)
for cls in classes:
    os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

# === 遍历所有图像 ===
for img_name in tqdm(os.listdir(image_dir)):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(image_dir, img_name)
    label_name = os.path.splitext(img_name)[0] + ".txt"
    label_path = os.path.join(label_dir, label_name)

    if not os.path.exists(label_path):
        print(f"[跳过] 标签文件不存在: {label_path}")
        continue

    image = cv2.imread(image_path)
    if image is None:
        print(f"[错误] 图像读取失败: {image_path}")
        continue

    height, width = image.shape[:2]

    with open(label_path, "r") as f:
        for i, line in enumerate(f.readlines()):
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"[警告] 标签格式错误: {label_path} 第{i+1}行：{line.strip()}")
                continue

            try:
                class_id, x_center, y_center, w, h = map(float, parts)
                class_id = int(class_id)
                class_name = classes[class_id]
            except Exception as e:
                print(f"[错误] 标签解析失败: {label_path} 第{i+1}行：{line.strip()} → {e}")
                continue

            # 相对坐标转像素
            x1 = int((x_center - w / 2) * width)
            y1 = int((y_center - h / 2) * height)
            x2 = int((x_center + w / 2) * width)
            y2 = int((y_center + h / 2) * height)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            print(f"[DEBUG] {img_name} 第{i+1}个目标：({x1}, {y1}) → ({x2}, {y2}) | 宽高: {x2 - x1}x{y2 - y1}")

            if x2 <= x1 or y2 <= y1:
                print(f"[跳过] 无效裁剪区域")
                continue

            crop = image[y1:y2, x1:x2]
            crop_resized = cv2.resize(crop, (224, 224))
            save_path = os.path.join(output_dir, class_name, f"{os.path.splitext(img_name)[0]}_{i}.jpg")
            cv2.imwrite(save_path, crop_resized)
            print(f"[保存成功] {save_path}")
