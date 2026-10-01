import os
import cv2
import albumentations as A

# Cấu hình đường dẫn
image_dir = r"D:\OCR\Datasets\vn_handwritten_images\training_data"
label_file = os.path.join(image_dir, "train_labels.txt")
new_label_file = os.path.join(image_dir, "train_labels_aug.txt")


# Thiết lập bộ "Phá hoại ảnh" (Làm mờ, thêm nhiễu hạt, nén JPEG chất lượng thấp)
transform = A.Compose(
    [
        A.MotionBlur(p=0.4, blur_limit=7),  # Giả lập rung tay khi chụp
        A.GaussianBlur(p=0.4, blur_limit=(3, 7)),  # Lệch focus, mờ viền
        A.GaussNoise(p=0.4, var_limit=(10.0, 50.0)),  # Nhiễu hạt hột hột
        A.ImageCompression(quality_lower=30, quality_upper=60, p=0.3),
    ]
)

print("Đang nhân bản và làm mờ dữ liệu...")

with open(label_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    parts = line.strip().split("\t")
    if len(parts) != 2:
        continue

    img_name, text = parts[0], parts[1]
    img_path = os.path.join(image_dir, img_name)

    # 1. Giữ lại ảnh gốc
    new_lines.append(f"{img_name}\t{text}\n")

    # 2. Tạo phiên bản mờ
    image = cv2.imread(img_path)
    if image is not None:
        augmented = transform(image=image)["image"]
        aug_name = f"aug_{img_name}"
        cv2.imwrite(os.path.join(image_dir, aug_name), augmented)
        new_lines.append(f"{aug_name}\t{text}\n")

# Lưu lại file nhãn mới
with open(new_label_file, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Đã tạo xong! Dữ liệu của bạn đã tăng gấp đôi.")
