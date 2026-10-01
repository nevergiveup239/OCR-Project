# OCR tài liệu

Pipeline nhận diện tài liệu bằng YOLO (tùy chọn), kết hợp VietOCR và PaddleOCR. Kết quả được lưu thành ảnh so sánh, JSON và bản chép Markdown.

## Yêu cầu

- Windows 10/11 và Python 3.10.
- RAM đủ để nạp PyTorch, PaddlePaddle và các mô hình OCR; GPU NVIDIA là tùy chọn.
- Kết nối Internet ở lần chạy đầu nếu các thư viện cần tải mô hình mặc định.

## Cài đặt

Mở PowerShell tại thư mục dự án, tạo môi trường ảo và cài các thư viện mà pipeline chính sử dụng:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy opencv-python Pillow torch paddlepaddle paddleocr paddlex vietocr ultralytics
```

Nếu dùng GPU, cài bản PyTorch và PaddlePaddle tương thích với phiên bản CUDA/driver trên máy theo hướng dẫn chính thức của từng thư viện. Khả năng dùng GPU phụ thuộc vào bản PaddlePaddle được cài.

## Chạy OCR

Chạy một ảnh:

```powershell
python .\OCR_Project\main_code.py --image .\path\to\document.jpg
```

Chạy các ảnh trong một thư mục (không quét thư mục con):

```powershell
python .\OCR_Project\main_code.py --dir .\path\to\images
```

Ảnh đầu vào có thể là PNG, JPG hoặc JPEG. Không truyền `--image` hoặc `--dir` để mở hộp thoại chọn ảnh/thư mục trên Windows.

Các tùy chọn thường dùng:

```powershell
python .\OCR_Project\main_code.py --image .\document.jpg --output .\output
python .\OCR_Project\main_code.py --image .\cropped.jpg --pre-cropped
python .\OCR_Project\main_code.py --image .\document.jpg --yolo-model .\path\to\detector.pt
python .\OCR_Project\main_code.py --image .\document.jpg --vietocr-weights .\path\to\vietocr.pth
```

`--pre-cropped` dùng khi ảnh đã được cắt sẵn và chỉ cần nhận diện chữ. YOLO không bắt buộc: nếu không tìm thấy weights, pipeline tiếp tục ở chế độ xử lý toàn ảnh. Có thể xem toàn bộ tùy chọn bằng:

```powershell
python .\OCR_Project\main_code.py --help
```

## Mô hình và kết quả

- Cấu hình VietOCR nằm tại `OCR_Project/vgg_transformer.yml`.
- Weights YOLO và VietOCR không được lưu trong Git. Cung cấp đường dẫn bằng `--yolo-model` / `--vietocr-weights`, hoặc đặt weights theo các đường dẫn mà `main_code.py` tự dò.
- PaddleOCR/VietOCR có thể tải mô hình mặc định khi khởi tạo; cần Internet trong lần tải đầu và đủ dung lượng lưu trữ.
- Kết quả mặc định nằm trong `output/`, gồm ảnh so sánh, JSON, Markdown và các ảnh crop.

## Cấu trúc chính

- `OCR_Project/main_code.py`: chương trình OCR chính.
- `OCR_Project/train_model_*.py`: các script huấn luyện.
- `OCR_Project/*.yml` và `dataset_document.yaml`: cấu hình mô hình/dataset.
- `Documents/`: ghi chú và tài liệu dự án.

Dataset, weights, virtual environment, cache và kết quả chạy được giữ ở máy cục bộ, không đưa lên GitHub. Hãy tải hoặc chuẩn bị dataset riêng theo cấu hình YAML trước khi chạy các script huấn luyện.