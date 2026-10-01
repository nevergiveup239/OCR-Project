# OCR Image Processing Pipeline

## Tổng quan

```text
INPUT IMAGE
    ↓
1. Orientation correction
    ↓
2. Deskew / Perspective correction / Unwarp
    ↓
3. Crop ROI / vùng chứa text
    ↓
4. Upscale 2x–3x
    ↓
5. Tạo nhiều preprocessing branch
    ↓
┌──────────────────────────────────┐
│ A. Original / Upscaled           │
│ B. Grayscale + CLAHE             │
│ C. Grayscale + Denoise + Sharpen │
│ D. Grayscale + Adaptive Threshold│
└──────────────────────────────────┘
    ↓
6. OCR từng branch
    ↓
7. So sánh kết quả / voting
    ↓
8. Nếu confidence thấp
    ↓
Fallback: Deblur / Super Resolution
    ↓
OCR lại
    ↓
FINAL TEXT
```

## 1. Orientation / Geometry Correction

Thực hiện trước enhancement:

```text
rotate đúng chiều
→ deskew
→ perspective correction
→ unwarp nếu document bị cong/méo
```

Nếu dùng PaddleOCR có thể thử:

```text
Document Orientation Classification
+
Document Unwarping
```

## 2. Crop ROI

Nếu xác định được vùng text:

```python
roi = image[y1:y2, x1:x2]
```

Không nên OCR cả ảnh nếu text chỉ chiếm một vùng nhỏ.

## 3. Upscale

Tạo ảnh lớn hơn trước enhancement:

```python
img = cv2.resize(
    roi,
    None,
    fx=3,
    fy=3,
    interpolation=cv2.INTER_CUBIC
)
```

Default thử:

```text
scale = 3x
```

Có thể test thêm `2x`.

## 4. Branch A — Original

Giữ một bản càng gần ảnh gốc càng tốt:

```text
ROI
↓
resize
↓
OCR
```

```python
variant_original = img.copy()
```

Không được bỏ branch này vì enhancement đôi khi phá mất nét chữ thật.

## 5. Branch B — CLAHE

Dùng cho:

```text
contrast thấp
chữ nhạt
ảnh hơi tối
```

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

variant_clahe = clahe.apply(gray)
```

Pipeline:

```text
resize
↓
grayscale
↓
CLAHE
↓
OCR
```

## 6. Branch C — Denoise + Mild Sharpen

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

denoise = cv2.GaussianBlur(
    gray,
    (3, 3),
    0
)

blur = cv2.GaussianBlur(
    denoise,
    (0, 0),
    1.2
)

variant_sharp = cv2.addWeighted(
    denoise,
    1.5,
    blur,
    -0.5,
    0
)
```

Pipeline:

```text
resize
↓
grayscale
↓
denoise nhẹ
↓
unsharp mask
↓
OCR
```

Starting parameters:

```text
sigma = 1.0–1.5
amount ≈ 0.3–0.7
```

Không sharpen quá mạnh.

## 7. Branch D — Adaptive Threshold

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = cv2.GaussianBlur(
    gray,
    (3, 3),
    0
)

variant_binary = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    31,
    11
)
```

Pipeline:

```text
resize
↓
grayscale
↓
blur nhẹ
↓
adaptive threshold
↓
OCR
```

Starting parameters:

```text
blockSize = 31
C = 11
```

## 8. Optional Branch E — Otsu

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = cv2.GaussianBlur(
    gray,
    (3, 3),
    0
)

_, variant_otsu = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
```

Pipeline:

```text
grayscale
↓
Gaussian blur
↓
Otsu
↓
OCR
```

Không bắt buộc.

## 9. OCR Tất Cả Branch

```python
variants = {
    "original": variant_original,
    "clahe": variant_clahe,
    "sharp": variant_sharp,
    "adaptive": variant_binary,
}
```

```python
results = {}

for name, image in variants.items():
    results[name] = run_ocr(image)
```

Ví dụ:

```text
original → AB038561
clahe    → AB038561
sharp    → AB038S61
adaptive → AB038561
```

## 10. Voting

Nếu nhiều branch trả cùng text:

```text
AB038561 → 3 votes
AB038S61 → 1 vote
```

Kết quả:

```text
AB038561
```

Nếu OCR trả confidence thì dùng weighted vote:

```text
score(text) = Σ OCR confidence của branch trả text đó
```

## 11. Low Confidence Fallback

Chỉ khi các branch disagree mạnh:

```text
Original   → AB038561
CLAHE      → AB03B561
Sharpen    → AB038S61
Adaptive   → AB03?561
```

thì mới chạy:

```text
deblur
hoặc
super resolution
```

Sau đó:

```text
enhanced image
↓
OCR
↓
thêm kết quả vào voting
```

Không chạy Super Resolution mặc định.

## 12. Code Flow

```python
def preprocess(image):

    image = correct_orientation(image)

    image = correct_perspective(image)

    roi = crop_roi(image)

    img = upscale(roi, scale=3)

    variants = {}

    # A
    variants["original"] = img

    # B
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    variants["clahe"] = clahe.apply(gray)

    # C
    denoise = cv2.GaussianBlur(gray, (3, 3), 0)

    blur = cv2.GaussianBlur(
        denoise,
        (0, 0),
        1.2
    )

    variants["sharp"] = cv2.addWeighted(
        denoise,
        1.5,
        blur,
        -0.5,
        0
    )

    # D
    variants["adaptive"] = cv2.adaptiveThreshold(
        denoise,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return variants
```

```python
def pipeline(image):

    variants = preprocess(image)

    results = {}

    for name, img in variants.items():
        results[name] = run_ocr(img)

    final = vote(results)

    if final.confidence < THRESHOLD:
        enhanced = deblur_or_super_resolution(image)

        sr_result = run_ocr(enhanced)

        results["fallback"] = sr_result

        final = vote(results)

    return final
```

## 13. Pipeline Cuối

```text
Orientation
    ↓
Deskew / Unwarp
    ↓
ROI Crop
    ↓
Upscale 3x
    ↓
 ┌──────────┬──────────┬────────────┬──────────────┐
 │ Original │ CLAHE    │ Sharpen    │ Adaptive BW  │
 └────┬─────┴────┬─────┴─────┬──────┴──────┬───────┘
      ↓          ↓           ↓             ↓
     OCR        OCR         OCR           OCR
      └──────────┴───────────┴─────────────┘
                         ↓
                       Vote
                         ↓
               confidence đủ?
                  ↙            ↘
               YES              NO
                ↓                ↓
              Final       Deblur / SR
                                 ↓
                                OCR
                                 ↓
                               Vote
                                 ↓
                               Final
```
