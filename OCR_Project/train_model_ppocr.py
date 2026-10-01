import os
import subprocess
import urllib.request
import tarfile


def setup_environment():
    """Tự động chuẩn bị mã nguồn và trọng số (weights) ban đầu"""
    # 1. Tải mã nguồn PaddleOCR nếu chưa có
    if not os.path.exists("PaddleOCR"):
        print("[-] Chưa tìm thấy mã nguồn PaddleOCR. Bắt đầu tải (Git Clone)...")
        subprocess.run(
            ["git", "clone", "https://github.com/PaddlePaddle/PaddleOCR.git"],
            check=True,
        )

    # 2. Tải Pre-trained Model của hãng để học chuyển giao (Fine-tuning)
    # Khuyến cáo: KHÔNG BAO GIỜ train từ số 0, phải dùng não có sẵn của hãng rồi dạy thêm
    pretrain_url = (
        "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_train.tar"
    )
    pretrain_tar = "ch_PP-OCRv4_rec_train.tar"
    pretrain_dir = "pretrain_models/ch_PP-OCRv4_rec_train/"

    if not os.path.exists(pretrain_dir):
        print("[+] Đang tải Pre-trained model PP-OCRv4 (Khoảng 100MB)...")
        os.makedirs("pretrain_models", exist_ok=True)
        urllib.request.urlretrieve(pretrain_url, pretrain_tar)

        print("[+] Đang giải nén file trọng số...")
        with tarfile.open(pretrain_tar) as tar:
            tar.extractall(path="pretrain_models/")
        os.remove(pretrain_tar)
        print("[+] Đã chuẩn bị xong Pre-trained model!")


def run_ppocr_training():
    """Cấu hình thông số và khởi chạy tiến trình huấn luyện"""
    # 1. THIẾT LẬP SINH TỒN CHO GPU 4GB
    env = os.environ.copy()
    env["FLAGS_fraction_of_gpu_memory_to_use"] = "0.40"  # Khóa cứng ở mức ~1.5GB VRAM
    env["FLAGS_eager_delete_tensor_gb"] = "0.0"

    # 2. Khai báo các đường dẫn
    train_script = "PaddleOCR/tools/train.py"
    config_file = "PaddleOCR/configs/rec/PP-OCRv4/ch_PP-OCRv4_rec_hgnet.yml"
    pretrained_model = "pretrain_models/ch_PP-OCRv4_rec_train/best_accuracy"

    # 3. Ghi đè cấu hình (Overrides) để trỏ vào dữ liệu của bạn
    command = [
        "python",
        train_script,
        "-c",
        config_file,
        "-o",
        "Global.use_amp=True",  # Kích hoạt học 16-bit (Giảm 50% VRAM)
        "Global.epoch_num=50",  # Số vòng lặp toàn bộ dữ liệu
        f"Global.pretrained_model={pretrained_model}",  # Nạp não bộ có sẵn của hãng
        "Global.save_model_dir=./output/custom_ppocr_v1/",  # Nơi lưu model của bạn
        # Trỏ vào thư mục Training Data bạn đã tạo
        "Train.dataset.data_dir=./Training_Data/ppocr/",
        "Train.dataset.label_file_list=[./Training_Data/ppocr/train_labels.txt]",
        "Eval.dataset.data_dir=./Training_Data/ppocr/",
        "Eval.dataset.label_file_list=[./Training_Data/ppocr/val_labels.txt]",
        # Ép phần cứng chạy chậm lại để không Crash OOM
        "Train.loader.batch_size_per_card=4",  # Nhét tối đa 4 ảnh 1 lần
        "Train.loader.num_workers=0",  # Tắt đa luồng nạp RAM
        "Optimizer.lr.learning_rate=0.0001",  # Học chậm để không phá hỏng weights cũ
    ]

    print("\n" + "=" * 70)
    print(" BẮT ĐẦU HUẤN LUYỆN PP-OCR (FINE-TUNING)")
    print("=" * 70)

    # Chạy tiến trình PaddleOCR bằng môi trường ảo hiện tại
    subprocess.run(command, env=env)


if __name__ == "__main__":
    setup_environment()
    run_ppocr_training()
