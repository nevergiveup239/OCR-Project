import os
import sys
import warnings
import torch
from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer

# 1. Tắt cảnh báo UserWarning không cần thiết của PyTorch
warnings.filterwarnings("ignore", category=UserWarning)

# 2. Xả đệm console tức thì (để print ra ngay lập tức)
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

torch.cuda.empty_cache()


def main():
    print("[+] Bắt đầu nạp cấu hình VietOCR (VGG Transformer)...", flush=True)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(project_dir, "vgg_transformer.yml")
    if os.path.exists(config_path):
        config = Cfg.load_config_from_file(config_path)
    else:
        config = Cfg.load_config_from_name("vgg_transformer")

    # 1. Trỏ đường dẫn tới tập dữ liệu
    config["dataset"]["name"] = "custom_handwritten"
    config["dataset"][
        "data_root"
    ] = r"D:\OCR\Datasets\vn_handwritten_images\training_data"
    config["dataset"]["train_annotation"] = "train_labels_aug.txt"
    config["dataset"]["valid_annotation"] = "val_labels.txt"

    # 2. CẤU HÌNH AN TOÀN CHO GPU 4GB VRAM
    config["dataloader"]["num_workers"] = 0
    config["trainer"]["batch_size"] = 16  # Đặt đúng vào config['trainer']
    # 3. Tinh chỉnh thông số học (Fine-tuning)
    config["trainer"]["epochs"] = 50
    config["trainer"]["metrics"] = 1000
    config["optimizer"]["max_lr"] = 0.0001

    # KÍCH HOẠT CHỦ ĐỘNG LÀM MỜ VÀ NHIỄU ẢNH (DATA AUGMENTATION)
    config["aug"] = {"image_aug": True, "masked_language_model": True}
    # 3. TINH CHỈNH ĐỂ HIỂN THỊ LOG LIÊN TỤC
    config["trainer"][
        "print_every"
    ] = 10  # Cứ 10 bước in loss 1 lần (khoảng vài giây 1 lần)
    config["trainer"]["valid_every"] = 500  # Đánh giá validation sau mỗi 500 bước
    config["trainer"][
        "iters"
    ] = 5200  # 5200 iters tương đương ~50 epochs (1653 ảnh / batch 16)
    config["optimizer"]["max_lr"] = 0.0001

    if "lr" in config["optimizer"]:
        del config["optimizer"]["lr"]

    project_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(project_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    config["trainer"]["export"] = os.path.join(weights_dir, "vietocr_custom.pth")

    # 4. Khởi chạy
    trainer = Trainer(config, pretrained=True)
    print("[+] Đang nạp model lên VRAM và bắt đầu huấn luyện...", flush=True)
    trainer.train()


if __name__ == "__main__":
    main()
