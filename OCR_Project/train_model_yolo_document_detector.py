import os
import sys
import glob
import shutil
import random
import ast
import argparse
from pathlib import Path
from PIL import Image

# Cấu hình UTF-8 cho Windows Terminal
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr.encoding != "utf-8":
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def get_project_root() -> Path:
    """Trả về thư mục gốc của project (D:\OCR)."""
    script_dir = Path(__file__).resolve().parent
    if (script_dir.parent / "dataset_document").exists() or (script_dir.parent / "Datasets").exists():
        return script_dir.parent
    return script_dir


def create_document_dataset_yaml(output_path: str = None) -> Path:
    """
    Tạo file cấu hình dữ liệu mẫu cho bài toán phát hiện & cắt tờ giấy tài liệu (Document Detection).
    Chỉ cần 1 lớp duy nhất: 'document' (toàn bộ tờ giấy chứa nội dung).
    """
    root_dir = get_project_root()
    dataset_dir = root_dir / "dataset_document"
    if output_path is None:
        yaml_path = root_dir / "dataset_document.yaml"
    else:
        yaml_path = Path(output_path)
        if not yaml_path.is_absolute():
            yaml_path = root_dir / yaml_path

    dataset_path_str = str(dataset_dir).replace("\\", "/")

    content = f"""# Cấu hình Dataset Huấn luyện YOLO26 - Phát hiện Tờ giấy Tài liệu
path: {dataset_path_str}    # Thư mục gốc chứa dataset
train: images/train        # Thư mục ảnh huấn luyện
val: images/val            # Thư mục ảnh kiểm thử (validation)

# Danh sách lớp (Class) - Chỉ cần 1 class bao trọn tờ giấy tài liệu chứa chữ:
names:
  0: document              # Tờ giấy / Phiếu giao hàng / Hóa đơn / Chứng từ
"""
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content.strip())
    print(f"[Dataset] Đã tạo file cấu hình dữ liệu: {yaml_path}")

    for split in ["train", "val"]:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    return yaml_path


def check_and_prepare_dataset(
    dataset_dir: Path = None, split_ratio: float = 0.8, seed: int = 42, force: bool = False
) -> bool:
    """
    Kiểm tra và tự động chuyển đổi dữ liệu gán nhãn tờ giấy (preprocessor quadrilateral)
    sang định dạng chuẩn YOLO Bounding Box (0 x_center y_center width height).
    """
    root_dir = get_project_root()
    if dataset_dir is None:
        dataset_dir = root_dir / "dataset_document"

    train_lbl_dir = dataset_dir / "labels" / "train"
    train_img_dir = dataset_dir / "images" / "train"
    val_lbl_dir = dataset_dir / "labels" / "val"
    val_img_dir = dataset_dir / "images" / "val"

    # Kiểm tra xem đã có đủ nhãn YOLO hợp lệ chưa
    train_txts = list(train_lbl_dir.glob("*.txt")) if train_lbl_dir.exists() else []
    valid_train_labels = [f for f in train_txts if not f.name.startswith("text_recognition_")]

    val_txts = list(val_lbl_dir.glob("*.txt")) if val_lbl_dir.exists() else []
    valid_val_labels = [f for f in val_txts if not f.name.startswith("text_recognition_")]

    if not force and len(valid_train_labels) > 50 and len(valid_val_labels) > 10:
        return True

    print("\n" + "=" * 75)
    print("[Dataset] Đang chuẩn bị và chuyển đổi dataset tờ giấy sang chuẩn YOLO...")
    print("=" * 75)

    candidate_txt_dirs = [
        root_dir / "Datasets" / "Datasin" / "preprocessor" / "preprocessor" / "txt",
        root_dir / "Datasets" / "preprocessor" / "txt",
    ]
    candidate_img_dirs = [
        root_dir / "Datasets" / "Datasin" / "train_images" / "train_images",
        root_dir / "Datasets" / "Datasin" / "preprocessor" / "preprocessor" / "imgs",
    ]

    txt_src_dir = next((d for d in candidate_txt_dirs if d.exists()), None)
    img_src_dir = next((d for d in candidate_img_dirs if d.exists()), None)

    if not txt_src_dir or not img_src_dir:
        print("[!] Không tìm thấy thư mục dữ liệu gốc (preprocessor/txt).")
        print(f"    Vui lòng đặt ảnh vào: {train_img_dir}")
        print(f"    Và file nhãn vào:     {train_lbl_dir}")
        return False

    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)
    train_lbl_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Dọn dẹp cache và file sai định dạng cũ
    for junk in [
        train_lbl_dir / "text_recognition_train_data.txt",
        val_lbl_dir / "text_recognition_val_data.txt",
        dataset_dir / "labels" / "train.cache",
        dataset_dir / "labels" / "val.cache",
    ]:
        if junk.exists():
            try:
                junk.unlink()
            except Exception:
                pass

    # Xóa các ảnh val cũ không có nhãn (mcocr_val_*)
    for old_val in val_img_dir.glob("mcocr_val_*.jpg"):
        try:
            old_val.unlink()
        except Exception:
            pass

    all_txts = sorted(glob.glob(str(txt_src_dir / "*.txt")))
    if not all_txts:
        print(f"[!] Không tìm thấy file nhãn nào trong {txt_src_dir}")
        return False

    random.seed(seed)
    indices = list(range(len(all_txts)))
    random.shuffle(indices)
    split_idx = int(len(indices) * split_ratio)
    train_indices = set(indices[:split_idx])

    converted_train = 0
    converted_val = 0

    for idx, txt_file_path in enumerate(all_txts):
        txt_path = Path(txt_file_path)
        stem = txt_path.stem
        src_img = img_src_dir / f"{stem}.jpg"
        if not src_img.exists():
            continue

        is_train = idx in train_indices
        target_img_dir = train_img_dir if is_train else val_img_dir
        target_lbl_dir = train_lbl_dir if is_train else val_lbl_dir

        target_img = target_img_dir / f"{stem}.jpg"
        curr_in_train = train_img_dir / f"{stem}.jpg"

        if not target_img.exists():
            if curr_in_train.exists() and not is_train:
                shutil.move(str(curr_in_train), str(target_img))
            else:
                shutil.copy2(str(src_img), str(target_img))
        elif curr_in_train.exists() and not is_train and target_img != curr_in_train:
            try:
                curr_in_train.unlink()
            except Exception:
                pass

        try:
            with Image.open(src_img) as im:
                w, h = im.size

            with open(txt_path, "r", encoding="utf-8") as fp:
                pts = ast.literal_eval(f"[{fp.read().strip()}]")

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            min_x, max_x = max(0.0, min(xs)), min(float(w), max(xs))
            min_y, max_y = max(0.0, min(ys)), min(float(h), max(ys))

            bw = (max_x - min_x) / float(w)
            bh = (max_y - min_y) / float(h)
            xc = ((min_x + max_x) / 2.0) / float(w)
            yc = ((min_y + max_y) / 2.0) / float(h)

            bw = max(0.001, min(1.0, bw))
            bh = max(0.001, min(1.0, bh))
            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))

            target_lbl = target_lbl_dir / f"{stem}.txt"
            with open(target_lbl, "w", encoding="utf-8") as fp:
                fp.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

            if is_train:
                converted_train += 1
            else:
                converted_val += 1
        except Exception as e:
            print(f"[!] Bỏ qua file lỗi {txt_path.name}: {e}")

    print(f"[Dataset] Đã chuẩn bị xong:")
    print(f"  - Train: {converted_train} ảnh & nhãn ({train_img_dir})")
    print(f"  - Val:   {converted_val} ảnh & nhãn ({val_img_dir})")
    print("=" * 75 + "\n")
    return True


def check_gpu_environment(requested_device: str = "") -> str:
    """Kiểm tra thiết bị GPU/CPU và đưa ra khuyến nghị tối ưu."""
    if requested_device:
        return requested_device

    try:
        import torch

        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            print(f"[Phần cứng] Phát hiện GPU CUDA: {dev_name} (Sử dụng GPU)")
            return "0"
        else:
            print("[Phần cứng] Đang sử dụng CPU (torch hiện tại là bản CPU).")
            has_nvidia = False
            try:
                import subprocess

                res = subprocess.run(
                    ["nvidia-smi"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if res.returncode == 0:
                    has_nvidia = True
            except Exception:
                pass
            if has_nvidia:
                print(
                    "[Mẹo Tăng Tốc] Hệ thống có card đồ họa NVIDIA. Để train nhanh hơn rất nhiều trên GPU, bạn có thể chạy:"
                )
                print(
                    "  & d:\\OCR\\Ocr.venv\\Scripts\\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
                )
            return "cpu"
    except Exception:
        return "cpu"


def train_yolo_document_model(
    data_yaml: str = "dataset_document.yaml",
    base_model: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch_size: int = 8,
    device: str = "",
    workers: int = 0,
    target_weights_name: str = "yolo26.pt",
):
    """
    Huấn luyện mô hình YOLO26 chuyên biệt nhận diện tờ giấy / tài liệu chứa nội dung.
    Tối ưu hóa các kỹ thuật Data Augmentation:
    - Xoay nghiêng, co dãn phối cảnh (chống méo khi chụp điện thoại).
    - Thay đổi độ tương phản, ánh sáng (đèn phòng, bóng đổ).
    - Dịch chuyển ngẫu nhiên (chống lệch viền/trôi lề).
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[Lỗi] Thư viện ultralytics chưa được cài đặt. Vui lòng chạy: pip install ultralytics")
        return

    root_dir = get_project_root()
    script_dir = Path(__file__).resolve().parent

    yaml_candidates = [
        Path(data_yaml),
        root_dir / data_yaml,
        script_dir / data_yaml,
        root_dir / "dataset_document.yaml",
    ]
    resolved_yaml = next((p for p in yaml_candidates if p.exists()), None)

    if resolved_yaml is None:
        print(f"[!] Chưa phát hiện file '{data_yaml}'. Đang tạo cấu trúc mẫu...")
        resolved_yaml = create_document_dataset_yaml(data_yaml)

    # Đảm bảo dataset đã có nhãn hợp lệ
    dataset_dir = root_dir / "dataset_document"
    check_and_prepare_dataset(dataset_dir)

    train_device = check_gpu_environment(device)

    print("=" * 75)
    print("BẮT ĐẦU HUẤN LUYỆN YOLO26: PHÁT HIỆN TỜ GIẤY TÀI LIỆU CHỨA NỘI DUNG")
    print("=" * 75)
    print(f" - Model nền:          {base_model}")
    print(f" - File cấu hình data: {resolved_yaml}")
    print(f" - Số epochs:          {epochs}")
    print(f" - Kích thước ảnh:     {imgsz}x{imgsz}")
    print(f" - Batch size:         {batch_size}")
    print(f" - Thiết bị (Device):  {train_device}")
    print("=" * 75)

    # Nạp mô hình nền
    model = YOLO(base_model)

    train_args = {
        "data": str(resolved_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch_size,
        "workers": workers,
        # Augmentation đặc trị chụp tài liệu thực tế:
        "degrees": 5.0,         # Xoay nghiêng nhẹ ±5 độ
        "translate": 0.08,      # Dịch viền ±8%
        "scale": 0.1,           # Thu phóng kích thước ±10%
        "shear": 2.0,           # Méo góc phối cảnh khi chụp nghiêng
        "perspective": 0.0005,  # Biến dạng góc 3D
        "hsv_h": 0.015,         # Thay đổi gam màu nhẹ
        "hsv_s": 0.3,           # Độ bão hòa màu
        "hsv_v": 0.3,           # Tăng giảm sáng tối (ánh sáng phòng)
        "mosaic": 0.2,          # Giữ tỷ lệ mosaic thấp để văn bản không bị vỡ nát
        "name": "yolo_document_run",
    }
    if train_device:
        train_args["device"] = train_device

    # Bắt đầu quá trình huấn luyện
    results = model.train(**train_args)

    # Copy weights tốt nhất (best.pt) thành yolo26.pt để tích hợp trực tiếp vào pipeline
    best_pt = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if best_pt.exists():
        dest_1 = script_dir / target_weights_name
        dest_2 = root_dir / "Documents" / "yolo26n.pt"
        dest_3 = root_dir / "OCR_Project" / "yolo26n.pt"

        shutil.copy(best_pt, dest_1)
        for dest in [dest_2, dest_3]:
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(best_pt, dest)
            except Exception:
                pass

        print("\n" + "=" * 75)
        print(f"[+] HUẤN LUYỆN THÀNH CÔNG VÀ ĐÃ NẠP WEIGHTS VÀO PIPELINE!")
        print(f" [1] File weights chính:    {dest_1}")
        print(f" [2] File weights dự phòng: {dest_2}")
        print("=" * 75)
        print("Bây giờ bạn chỉ cần chạy lại pipeline OCR:")
        print(" & d:\\OCR\\Ocr.venv\\Scripts\\python.exe d:/OCR/OCR_Project/structure_vietocr_dual_view.py")
        print("=" * 75 + "\n")
    else:
        print("[!] Không tìm thấy file best.pt. Vui lòng kiểm tra thư mục 'runs/detect/'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình YOLO26 nhận diện tờ giấy / tài liệu chứa nội dung"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="dataset_document.yaml",
        help="Đường dẫn file dataset yaml",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="yolov8n.pt",
        help="Mô hình nền (yolov8n.pt, yolo11n.pt, yolov8n-obb.pt)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Số lượt huấn luyện (epochs)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Kích thước ảnh train (640, 1024)"
    )
    parser.add_argument(
        "--batch", type=int, default=8, help="Batch size (tùy thuộc VRAM GPU/RAM)"
    )
    parser.add_argument(
        "--device", type=str, default="", help="Thiết bị ('0' cho GPU hoặc 'cpu')"
    )
    parser.add_argument(
        "--workers", type=int, default=0, help="Số workers nạp dữ liệu (0 cho Windows)"
    )
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Tự động chuẩn bị lại dataset từ dữ liệu gốc",
    )
    parser.add_argument(
        "--init-yaml",
        action="store_true",
        help="Khởi tạo thư mục và file dataset_document.yaml mẫu",
    )

    args = parser.parse_args()

    if args.init_yaml:
        create_document_dataset_yaml(args.data)
    elif args.prepare_data:
        check_and_prepare_dataset(force=True)
    else:
        train_yolo_document_model(
            data_yaml=args.data,
            base_model=args.base_model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch_size=args.batch,
            device=args.device,
            workers=args.workers,
        )
