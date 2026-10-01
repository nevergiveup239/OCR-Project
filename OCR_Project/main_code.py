import os
import sys
import json
import re
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# QUAN TRỌNG: Nạp PyTorch trước PaddlePaddle trên Windows để tránh xung đột thư viện DLL C++ (shm.dll / MKL)
import torch
import paddle


def _dummy_func(self, *args, **kwargs):
    pass


# Vá lỗi cho Paddle bản 2.5 (fluid)
try:
    if hasattr(paddle, "fluid") and hasattr(paddle.fluid, "libpaddle"):
        paddle.fluid.libpaddle.AnalysisConfig.set_optimization_level = _dummy_func
        paddle.fluid.libpaddle.AnalysisConfig.disable_mkldnn = _dummy_func
except Exception:
    pass

# Vá lỗi cho Paddle bản 2.6+ (base)
try:
    if hasattr(paddle, "base") and hasattr(paddle.base, "libpaddle"):
        paddle.base.libpaddle.AnalysisConfig.set_optimization_level = _dummy_func
        paddle.base.libpaddle.AnalysisConfig.disable_mkldnn = _dummy_func
except Exception:
    pass

# Bỏ qua kiểm tra metadata phụ thuộc gắt gao của PaddleX
try:
    import paddlex.utils.deps

    paddlex.utils.deps.require_extra = lambda *args, **kwargs: None
    paddlex.utils.deps.is_extra_available = lambda *args, **kwargs: True
except Exception:
    pass
# ============================================================================

# Đảm bảo Terminal Windows xuất UTF-8 chuẩn xác
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


# ============================================================================
# 0. TIỆN ÍCH CẮT VÙNG ẢNH CHO YOLO26 (CROP BOUNDING BOX & OBB)
# ============================================================================


def crop_standard_bbox(
    img_bgr: np.ndarray, box_xyxy: list, padding_x: int = 8, padding_y: int = 4
) -> tuple:
    """
    Cắt vùng ảnh theo Bounding Box [x1, y1, x2, y2] kèm đệm viền an toàn (padding).
    Tránh cắt sát mép chữ làm mất nét ký tự khi đưa vào OCR.

    Returns:
        (cropped_image, [x1_pad, y1_pad, x2_pad, y2_pad])
    """
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(coord)) for coord in box_xyxy]

    x1_pad = max(0, x1 - padding_x)
    y1_pad = max(0, y1 - padding_y)
    x2_pad = min(w, x2 + padding_x)
    y2_pad = min(h, y2 + padding_y)

    if x2_pad <= x1_pad or y2_pad <= y1_pad:
        return None, [x1, y1, x2, y2]

    crop = img_bgr[y1_pad:y2_pad, x1_pad:x2_pad]
    return crop, [x1_pad, y1_pad, x2_pad, y2_pad]


def crop_oriented_bbox(img_bgr: np.ndarray, pts_4x2: np.ndarray) -> np.ndarray:
    """
    Cắt và xoay thẳng vùng ảnh theo 4 điểm góc của hộp chữ nhật xoay (OBB).
    Sử dụng thuật toán phối cảnh Perspective Transform (Warp Perspective).
    """
    pts = np.array(pts_4x2, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-Left
    rect[2] = pts[np.argmax(s)]  # Bottom-Right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-Right
    rect[3] = pts[np.argmax(diff)]  # Bottom-Left

    tl, tr, br, bl = rect
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_w = max(int(round(width_a)), int(round(width_b)))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_h = max(int(round(height_a)), int(round(height_b)))

    if max_w <= 0 or max_h <= 0:
        return None

    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img_bgr, matrix, (max_w, max_h))
    return warped


# ============================================================================
# 0.1 MÔ HÌNH PHÁT HIỆN & CẮT ẢNH BẰNG YOLO26 (YOLO26 LAYOUT CROPPER)


# Lớp danh sách các nhãn đại diện cho tài liệu / tờ giấy chứa nội dung
DOCUMENT_CLASSES = {
    "document",
    "paper",
    "page",
    "bill",
    "invoice",
    "receipt",
    "form",
    "card",
    "id_card",
    "book",
    "text",
    "content",
    "document_region",
    "table",
    "contract",
    "chung_tu",
    "tai_lieu",
    "to_giay",
    "van_ban",
    "don_hang",
}

# Lớp danh sách các nhãn vật thể ngoại lai cần LOẠI BỎ tuyệt đối (không phải tài liệu)
REJECTED_CLASSES = {
    "person",
    "human",
    "hand",
    "finger",
    "face",
    "body",
    "chair",
    "table",
    "dining table",
    "bottle",
    "cup",
    "cell phone",
    "phone",
    "laptop",
    "mouse",
    "keyboard",
    "car",
    "dog",
    "cat",
    "tv",
    "monitor",
    "remote",
    "pen",
    "pencil",
    "backpack",
    "handbag",
    "tie",
    "suitcase",
    "clock",
    "vase",
    "scissors",
}


class YOLOLayoutDetector:
    """
    Module phát hiện bố cục và cắt các vùng tài liệu/trường dữ liệu bằng mô hình YOLO26.
    Hỗ trợ cả mô hình BBox chuẩn (kèm padding an toàn) và mô hình OBB (xoay thẳng Perspective).
    Chỉ giữ lại và crop phần giấy tài liệu chứa nội dung, loại bỏ mọi vật thể
    """

    def __init__(
        self,
        model_path: str = None,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
    ):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.model = None
        self.is_simulated = False

        # Tìm kiếm đường dẫn file weights phù hợp (ưu tiên model đã fine-tune cho document)
        candidate_paths = []
        if model_path:
            candidate_paths.append(model_path)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(base_dir)

        # 1. Ưu tiên weights chính xác của bài toán nhận diện tài liệu: yolo26.pt
        candidate_paths.append(os.path.join(base_dir, "yolo26.pt"))
        candidate_paths.append(os.path.join(parent_dir, "weights", "yolo26.pt"))
        candidate_paths.append(os.path.join(parent_dir, "yolo26.pt"))

        # 2. Tự động quét và nạp file best.pt mới nhất từ các phiên huấn luyện YOLO (runs/detect/)
        import glob

        trained_runs = sorted(
            glob.glob(
                os.path.join(parent_dir, "runs", "detect", "*", "weights", "best.pt")
            ),
            key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0,
            reverse=True,
        )
        if trained_runs:
            candidate_paths.extend(trained_runs)

        # 3. Các đường dẫn dự phòng khác
        candidate_paths.extend(
            [
                os.path.join(base_dir, "best.pt"),
                os.path.join(parent_dir, "best.pt"),
                os.path.join(parent_dir, "Documents", "yolo26n.pt"),
                os.path.join(base_dir, "yolo26n.pt"),
                os.path.join(parent_dir, "yolo26n.pt"),
                os.path.join(base_dir, "yolo_layout.pt"),
                os.path.join(base_dir, "weights", "best.pt"),
            ]
        )

        chosen_path = None
        for path in candidate_paths:
            if path and os.path.exists(path):
                chosen_path = path
                break

        if chosen_path:
            try:
                from ultralytics import YOLO

                print(f"[YOLO26] Đang nạp weights YOLO26 từ: {chosen_path}")
                self.model = YOLO(chosen_path)
                print("[YOLO26] Đã nạp thành công mô hình YOLO26!")
            except Exception as e:
                print(
                    f"[YOLO26 Warning] Không thể nạp weights YOLO26 ({e}). Sẽ sử dụng chế độ Fallback Full-Image."
                )
                self.is_simulated = True
        else:
            print(
                "[YOLO26 Warning] Chưa phát hiện file weights YOLO ('yolo26.pt', 'yolo26n.pt' hoặc 'best.pt')."
            )
            print(
                "                Hệ thống sẽ chạy ở chế độ Fallback Full-Image để tiếp tục pipeline."
            )
            self.is_simulated = True

    def detect(self, img_bgr: np.ndarray):
        """
        Dự đoán các vùng trường dữ liệu trên ảnh (hỗ trợ cả BBox chuẩn và OBB).
        """
        h, w = img_bgr.shape[:2]
        detected_fields = []

        if self.model and not self.is_simulated:
            results = self.model.predict(
                source=img_bgr,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                verbose=False,
            )
            for r in results:
                names = r.names or {}
                if r.boxes is not None and len(r.boxes) > 0:
                    for box in r.boxes:
                        cls_id = int(box.cls[0].item())
                        field_name = names.get(cls_id, f"field_{cls_id}")
                        conf = float(box.conf[0].item())
                        xyxy = box.xyxy[0].tolist()
                        x1, y1, x2, y2 = [int(round(coord)) for coord in xyxy]

                        detected_fields.append(
                            {
                                "field_name": field_name,
                                "box": [x1, y1, x2, y2],
                                "confidence": conf,
                            }
                        )
                elif hasattr(r, "obb") and r.obb is not None and len(r.obb) > 0:
                    for obb in r.obb:
                        cls_id = int(obb.cls[0].item())
                        field_name = names.get(cls_id, f"field_{cls_id}")
                        conf = float(obb.conf[0].item())
                        xyxy = obb.xyxy[0].tolist()
                        x1, y1, x2, y2 = [int(round(coord)) for coord in xyxy]

                        detected_fields.append(
                            {
                                "field_name": field_name,
                                "box": [x1, y1, x2, y2],
                                "confidence": conf,
                            }
                        )
        else:
            detected_fields = self._fallback_detection(img_bgr)

        return detected_fields

    def detect_and_crop(
        self,
        img_bgr: np.ndarray,
        output_crop_dir: str = None,
        stem_name: str = "doc",
        padding_x: int = 8,
        padding_y: int = 4,
    ) -> list:
        """
        Dự đoán và CHỈ CẮT phần giấy / tài liệu chứa nội dung bằng YOLO26.
        Tự động lọc bỏ các vật thể ngoại lai (con người, ngón tay, bàn chụp, đồ vật xung quanh).
        """
        h, w = img_bgr.shape[:2]
        img_area = h * w
        crops = []
        raw_candidates = []

        if output_crop_dir:
            os.makedirs(output_crop_dir, exist_ok=True)

        if self.model and not self.is_simulated:
            try:
                results = self.model.predict(
                    source=img_bgr,
                    conf=self.conf_thresh,
                    iou=self.iou_thresh,
                    verbose=False,
                )
                r = results[0]
                names = r.names or {}

                # 1. Thu thập ứng viên từ mô hình OBB (Oriented Bounding Box)
                if hasattr(r, "obb") and r.obb is not None and len(r.obb) > 0:
                    for i in range(len(r.obb)):
                        cls_id = int(r.obb.cls[i].item())
                        label = names.get(cls_id, f"class_{cls_id}")
                        conf = float(r.obb.conf[i].item())
                        pts_4x2 = r.obb.xyxyxyxy[i].cpu().numpy()

                        min_x = int(max(0, np.min(pts_4x2[:, 0])))
                        min_y = int(max(0, np.min(pts_4x2[:, 1])))
                        max_x = int(min(w, np.max(pts_4x2[:, 0])))
                        max_y = int(min(h, np.max(pts_4x2[:, 1])))

                        area = (max_x - min_x) * (max_y - min_y)
                        raw_candidates.append(
                            {
                                "type": "obb",
                                "label": label,
                                "conf": conf,
                                "box": [min_x, min_y, max_x, max_y],
                                "pts": pts_4x2,
                                "area": area,
                                "area_ratio": area / float(img_area),
                            }
                        )

                # 2. Thu thập ứng viên từ mô hình Bounding Box chuẩn
                elif r.boxes is not None and len(r.boxes) > 0:
                    for i in range(len(r.boxes)):
                        cls_id = int(r.boxes.cls[i].item())
                        label = names.get(cls_id, f"class_{cls_id}")
                        conf = float(r.boxes.conf[i].item())
                        xyxy = r.boxes.xyxy[i].cpu().tolist()
                        x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        area = (x2 - x1) * (y2 - y1)
                        raw_candidates.append(
                            {
                                "type": "bbox",
                                "label": label,
                                "conf": conf,
                                "box": [x1, y1, x2, y2],
                                "pts": None,
                                "area": area,
                                "area_ratio": area / float(img_area),
                            }
                        )

            except Exception as e:
                print(
                    f"[YOLO26 Error] Quá trình detect lỗi ({e}). Sẽ fallback toàn ảnh."
                )

        # 3. LỌC THÔNG MINH: CHỈ GIỮ LẠI PHẦN TỜ GIẤY TÀI LIỆU CHỨA NỘI DUNG

        filtered_candidates = []
        for cand in raw_candidates:
            lbl = cand["label"].lower().strip()

            # A. Loại bỏ ngay lập tức nếu là vật thể ngoại lai (person, chair, cup,...)
            if any(rej in lbl for rej in REJECTED_CLASSES):
                print(
                    f" [YOLO26 Lọc Bỏ] Vật thể ngoại lai không thuộc tài liệu: '{cand['label']}' (conf: {cand['conf']:.2f}, tỷ lệ diện tích: {cand['area_ratio']*100:.1f}%)"
                )
                continue

            # B. Loại bỏ nếu kích thước quá nhỏ (< 3% diện tích ảnh) không thể là tờ tài liệu
            if cand["area_ratio"] < 0.03:
                print(
                    f" [YOLO26 Lọc Bỏ] Vùng nhiễu kích thước quá nhỏ (< 3% diện tích): '{cand['label']}'"
                )
                continue

            filtered_candidates.append(cand)

        # C. Ưu tiên các ứng viên phù hợp với nhãn tài liệu / tờ giấy
        doc_matched = [
            c
            for c in filtered_candidates
            if any(kw in c["label"].lower() for kw in DOCUMENT_CLASSES)
        ]

        selected_candidates = doc_matched if doc_matched else filtered_candidates

        # Sắp xếp theo diện tích giảm dần (tờ tài liệu chứa nội dung luôn là vùng bao lớn nhất)
        selected_candidates.sort(key=lambda item: item["area"], reverse=True)

        # --------------------------------------------------------------------
        # 4. THỰC HIỆN CẮT (CROP) CÁC VÙNG TÀI LIỆU HỢP LỆ
        # --------------------------------------------------------------------
        crop_counter = 0
        for cand in selected_candidates:
            label = cand["label"]
            conf = cand["conf"]

            if cand["type"] == "obb":
                warped = crop_oriented_bbox(img_bgr, cand["pts"])
                if warped is None or warped.size == 0:
                    continue

                crop_counter += 1
                box = cand["box"]
                saved_path = None
                if output_crop_dir:
                    save_name = (
                        f"{stem_name}_document_crop_{crop_counter:03d}_{label}.jpg"
                    )
                    saved_path = os.path.join(output_crop_dir, save_name)
                    cv2.imencode(".jpg", warped)[1].tofile(saved_path)

                crops.append(
                    {
                        "crop_id": crop_counter,
                        "label": "document_paper",
                        "original_label": label,
                        "confidence": round(conf, 4),
                        "crop_img": warped,
                        "box": box,
                        "offset": (box[0], box[1]),
                        "is_obb": True,
                        "saved_path": saved_path,
                    }
                )
                print(
                    f" [YOLO26 Giữ Lại] Đã crop phần giấy tài liệu: '{label}' (conf: {conf:.2f}, box: {box})"
                )

            else:
                crop_patch, padded_box = crop_standard_bbox(
                    img_bgr, cand["box"], padding_x=padding_x, padding_y=padding_y
                )
                if crop_patch is None or crop_patch.size == 0:
                    continue

                crop_counter += 1
                saved_path = None
                if output_crop_dir:
                    save_name = (
                        f"{stem_name}_document_crop_{crop_counter:03d}_{label}.jpg"
                    )
                    saved_path = os.path.join(output_crop_dir, save_name)
                    cv2.imencode(".jpg", crop_patch)[1].tofile(saved_path)

                crops.append(
                    {
                        "crop_id": crop_counter,
                        "label": "document_paper",
                        "original_label": label,
                        "confidence": round(conf, 4),
                        "crop_img": crop_patch,
                        "box": padded_box,
                        "offset": (padded_box[0], padded_box[1]),
                        "is_obb": False,
                        "saved_path": saved_path,
                    }
                )
                print(
                    f" [YOLO26 Giữ Lại] Đã crop phần giấy tài liệu: '{label}' (conf: {conf:.2f}, box: {padded_box})"
                )

        # Fallback an toàn nếu không tìm thấy phần giấy nào
        if not crops:
            print(
                "[YOLO26 Info] Không tìm thấy vùng giấy riêng lẻ nào sau khi lọc. Sử dụng toàn bộ ảnh làm vùng tài liệu gốc."
            )
            crops.append(
                {
                    "crop_id": 1,
                    "label": "document_paper",
                    "original_label": "document",
                    "confidence": 1.0,
                    "crop_img": img_bgr.copy(),
                    "box": [0, 0, w, h],
                    "offset": (0, 0),
                    "is_obb": False,
                    "saved_path": None,
                    "is_fallback": True,
                }
            )

        return crops

    def _fallback_detection(self, img_bgr: np.ndarray):
        """
        Fallback layout detector sử dụng hình ảnh mẫu hiện tại để minh họa quy trình.
        """
        h, w = img_bgr.shape[:2]
        fields = []

        # Mẫu 1: Nhận diện trường ID (nửa trên bên phải hoặc vị trí tiêu biểu)
        # Giả lập 2 trường layout điển hình để test logic routing: ID và Họ tên
        if h > 200 and w > 200:
            # Trường Họ tên
            fields.append(
                {
                    "field_name": "ho_ten",
                    "box": [int(w * 0.1), int(h * 0.2), int(w * 0.9), int(h * 0.45)],
                    "confidence": 0.92,
                }
            )
            # Trường Số / Mã định danh ID
            fields.append(
                {
                    "field_name": "id",
                    "box": [int(w * 0.1), int(h * 0.55), int(w * 0.9), int(h * 0.85)],
                    "confidence": 0.95,
                }
            )
        else:
            fields.append(
                {"field_name": "text", "box": [0, 0, w, h], "confidence": 0.90}
            )
        return fields


# 1. TIỆN ÍCH HÌNH HỌC VÀ LỌC BỘ NHIỄU (GEOMETRY & FILTERING)


def compute_iou(box_a, box_b):
    """Tính chỉ số IoU và tỷ lệ chồng lấn diện tích giữa 2 bounding box [x1, y1, x2, y2]."""
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    if box_a_area <= 0 or box_b_area <= 0:
        return 0.0

    iou = inter_area / float(box_a_area + box_b_area - inter_area + 1e-6)
    coverage = inter_area / float(min(box_a_area, box_b_area) + 1e-6)
    return max(iou, coverage)


def deduplicate_boxes(boxes_with_meta, iou_thresh=0.75):
    """
    Loại bỏ các bounding box bị trùng lặp.
    Thay vì dùng IoU, thuật toán này sẽ XÓA các box LỚN chứa box nhỏ bên trong (Sub-cell overlap).
    Giúp bảo toàn các dòng văn bản chi tiết trong bảng biểu hoặc danh sách.
    """
    if not boxes_with_meta:
        return []

    # Hàm kiểm tra xem box_a có nằm hoàn toàn bên trong box_b hay không
    def is_inside(box_a, box_b, padding=5):
        return (
            box_a[0] >= box_b[0] - padding
            and box_a[1] >= box_b[1] - padding
            and box_a[2] <= box_b[2] + padding
            and box_a[3] <= box_b[3] + padding
        )

    kept = []

    # Bước 1: Loại bỏ box lớn nếu nó bao trùm box nhỏ
    for i, item_a in enumerate(boxes_with_meta):
        box_a = item_a["box"]
        is_large_container = False

        for j, item_b in enumerate(boxes_with_meta):
            if i == j:
                continue
            box_b = item_b["box"]

            # Nếu box_a chứa box_b, box_a là một container lớn thừa thãi
            if is_inside(box_b, box_a):
                is_large_container = True
                break

        if not is_large_container:
            kept.append(item_a)

    # Bước 2: Khử trùng lặp tiêu chuẩn (IoU) cho các box còn lại
    final_kept = []
    # Sắp xếp theo diện tích giảm dần để giữ lại box bao quát tốt nhất trong nhóm trùng lặp
    sorted_items = sorted(
        kept,
        key=lambda item: (item["box"][2] - item["box"][0])
        * (item["box"][3] - item["box"][1]),
        reverse=True,
    )

    for item in sorted_items:
        box = item["box"]
        is_duplicate = False
        for kept_item in final_kept:
            if compute_iou(box, kept_item["box"]) > iou_thresh:
                is_duplicate = True
                break
        if not is_duplicate:
            final_kept.append(item)

    return final_kept


def sort_reading_order(boxes_with_meta, y_tol=20):
    """
    Sắp xếp các ô theo thứ tự đọc tự nhiên từ trên xuống dưới, từ trái sang phải.
    Nhóm các ô cùng dòng (theo độ chênh lệch trục Y) và sắp xếp theo X.
    """
    if not boxes_with_meta:
        return []

    # Sắp xếp theo Y trước
    sorted_y = sorted(
        boxes_with_meta, key=lambda item: (item["box"][1], item["box"][0])
    )
    lines = []

    for item in sorted_y:
        box = item["box"]
        y1, y2 = box[1], box[3]
        placed = False

        for line in lines:
            avg_y = sum(x["box"][1] for x in line) / len(line)
            avg_h = sum((x["box"][3] - x["box"][1]) for x in line) / len(line)
            tol = max(y_tol, avg_h * 0.45)
            if abs(y1 - avg_y) < tol:
                line.append(item)
                placed = True
                break

        if not placed:
            lines.append([item])

    ordered = []
    for line in lines:
        line_sorted = sorted(line, key=lambda item: item["box"][0])
        ordered.extend(line_sorted)

    # Gán lại ID thứ tự đọc tuần tự: 1, 2, 3...
    for idx, item in enumerate(ordered):
        item["order_id"] = idx + 1

    return ordered


def crop_image_patch(image, box, pad_x=6, pad_y=4):
    """Cắt vùng ảnh theo box với padding an toàn tránh mất nét rìa chữ."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]

    x1_pad = max(0, x1 - pad_x)
    y1_pad = max(0, y1 - pad_y)
    x2_pad = min(w, x2 + pad_x)
    y2_pad = min(h, y2 + pad_y)

    if x2_pad <= x1_pad or y2_pad <= y1_pad:
        return None, [x1, y1, x2, y2]

    patch = image[y1_pad:y2_pad, x1_pad:x2_pad]
    return patch, [x1_pad, y1_pad, x2_pad, y2_pad]


# 2. BỘ PHÂN ĐOẠN KHUNG VÀ Ô (PP-DOCLAYOUTV3 + PP-STRUCTUREV3)


class DocumentStructureSegmenter:
    """
    Sử dụng PP-StructureV3 kết hợp mô hình phân tích bố cục PP-DocLayoutV3
    để trích xuất:
    - Danh sách Khung (Layout Frames): Khối tiêu đề, đoạn văn bản, bảng biểu, ảnh,...
    - Danh sách Ô (Cells): Ô trong bảng (Table Cells) và Ô chữ (Text line cells).
    """

    def __init__(self):
        print("=" * 75)
        print("1. KHỞI TẠO PIPELINE PP-STRUCTUREV3 KẾT HỢP PP-DOCLAYOUTV3...")
        print("=" * 75)
        from paddleocr import PPStructureV3

        self.pipeline = PPStructureV3(
            layout_detection_model_name="PP-DocLayoutV3",
            use_doc_orientation_classify=True,
            use_doc_unwarping=False,
            use_table_recognition=True,
            use_formula_recognition=False,
        )
        print("[+] Khởi tạo thành công PP-StructureV3 (Layout: PP-DocLayoutV3)!")

    def segment_single_crop(
        self, crop_bgr: np.ndarray, offset: tuple = (0, 0), crop_label: str = "text"
    ):
        """
        Thực hiện phân tích bố cục (PP-DocLayoutV3) và phân chia ô (PP-StructureV3)
        trên 1 ảnh crop. Tọa độ các khung và ô được cộng bù trừ (offset_x, offset_y)
        để quy đổi chính xác về hệ tọa độ của ảnh tài liệu gốc.
        """
        crop_h, crop_w = crop_bgr.shape[:2]
        ox, oy = offset

        results = list(self.pipeline.predict(input=crop_bgr))
        if not results:
            return [], []

        res = results[0]

        # 1. Trích xuất các khung bố cục (Layout Frames) từ PP-DocLayoutV3
        frames = []
        layout_det_res = res.get("layout_det_res", {}) if hasattr(res, "get") else {}
        layout_boxes = (
            layout_det_res.get("boxes", []) if isinstance(layout_det_res, dict) else []
        )

        if not layout_boxes and hasattr(res, "get"):
            layout_boxes = res.get("parsing_res_list", [])

        for idx, block in enumerate(layout_boxes):
            if isinstance(block, dict):
                label = block.get("label", block.get("block_label", "text"))
                score = float(block.get("score", 1.0))
                coord = block.get("coordinate", block.get("block_bbox", []))
            else:
                label = getattr(block, "label", "text")
                score = float(getattr(block, "score", 1.0))
                coord = getattr(block, "bbox", [])

            if len(coord) >= 4:
                x1, y1, x2, y2 = [int(round(v)) for v in coord[:4]]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(crop_w, x2), min(crop_h, y2)
                if x2 > x1 and y2 > y1:
                    frames.append(
                        {
                            "label": label,
                            "score": round(score, 3),
                            "box": [x1 + ox, y1 + oy, x2 + ox, y2 + oy],
                        }
                    )

        # 2. Trích xuất các ô (Cells) từ PP-StructureV3
        raw_cells = []

        # A. Table cells
        table_res_list = res.get("table_res_list", []) if hasattr(res, "get") else []
        if table_res_list:
            for t_idx, table_item in enumerate(table_res_list):
                cell_box_list = (
                    table_item.get("cell_box_list", [])
                    if isinstance(table_item, dict)
                    else getattr(table_item, "cell_box_list", [])
                )
                for c_idx, cbox in enumerate(cell_box_list):
                    if len(cbox) >= 4:
                        cx1, cy1, cx2, cy2 = [int(round(v)) for v in cbox[:4]]
                        cx1, cy1 = max(0, cx1), max(0, cy1)
                        cx2, cy2 = min(crop_w, cx2), min(crop_h, cy2)
                        cw, ch = cx2 - cx1, cy2 - cy1
                        if cw >= 12 and ch >= 10:
                            raw_cells.append(
                                {
                                    "source": "table_cell",
                                    "frame_label": f"table_{t_idx + 1}_cell",
                                    "box": [cx1 + ox, cy1 + oy, cx2 + ox, cy2 + oy],
                                }
                            )

        # B. Text cells từ Overall OCR Res
        overall_ocr = res.get("overall_ocr_res", {}) if hasattr(res, "get") else {}
        if isinstance(overall_ocr, dict):
            rec_boxes = overall_ocr.get("rec_boxes", [])
            for rbox in rec_boxes:
                if len(rbox) >= 4:
                    rx1, ry1, rx2, ry2 = [int(round(v)) for v in rbox[:4]]
                    rx1, ry1 = max(0, rx1), max(0, ry1)
                    rx2, ry2 = min(crop_w, rx2), min(crop_h, ry2)
                    rw, rh = rx2 - rx1, ry2 - ry1
                    if rw >= 14 and rh >= 10:
                        gx1, gy1, gx2, gy2 = rx1 + ox, ry1 + oy, rx2 + ox, ry2 + oy
                        parent_label = (
                            crop_label if crop_label != "document" else "text"
                        )
                        for f in frames:
                            fx1, fy1, fx2, fy2 = f["box"]
                            if (
                                gx1 >= fx1 - 10
                                and gy1 >= fy1 - 10
                                and gx2 <= fx2 + 10
                                and gy2 <= fy2 + 10
                            ):
                                parent_label = f["label"]
                                break

                        raw_cells.append(
                            {
                                "source": "text_cell",
                                "frame_label": parent_label,
                                "box": [gx1, gy1, gx2, gy2],
                            }
                        )

        # Nếu vùng crop không phân mảnh thêm ô con nào (ví dụ crop một trường đơn lẻ)
        if not raw_cells and crop_w >= 14 and crop_h >= 10:
            raw_cells.append(
                {
                    "source": "yolo_crop_cell",
                    "frame_label": crop_label,
                    "box": [ox, oy, ox + crop_w, oy + crop_h],
                }
            )

        return frames, raw_cells

    def segment_all_crops(self, original_bgr: np.ndarray, crops: list):
        """
        Thực hiện PP-DocLayoutV3 và PP-StructureV3 trên toàn bộ các vùng crop
        do YOLO26 trích xuất, hợp nhất kết quả và khử trùng lặp ô.
        """
        all_frames = []
        all_cells = []

        for crop_item in crops:
            c_id = crop_item["crop_id"]
            c_label = crop_item["label"]
            c_img = crop_item["crop_img"]
            c_offset = crop_item["offset"]
            print(
                f" [+] [PP-StructureV3] Đang phân tích Crop #{c_id} ('{c_label}', kích thước: {c_img.shape[1]}x{c_img.shape[0]})..."
            )
            frames, cells = self.segment_single_crop(
                crop_bgr=c_img, offset=c_offset, crop_label=c_label
            )
            all_frames.extend(frames)
            all_cells.extend(cells)

        # Đánh số frame_id tuần tự
        for f_idx, f in enumerate(all_frames):
            f["frame_id"] = f_idx + 1

        # Khử trùng lặp và sắp xếp theo thứ tự đọc tự nhiên
        deduped = deduplicate_boxes(all_cells, iou_thresh=0.70)
        ordered_cells = sort_reading_order(deduped, y_tol=22)

        print(
            f"[+] PP-DocLayoutV3 phân loại được tổng cộng: {len(all_frames)} khung bố cục."
        )
        print(
            f"[+] PP-StructureV3 trích xuất được tổng cộng: {len(ordered_cells)} ô cần nhận diện chữ."
        )

        return original_bgr, all_frames, ordered_cells

    def segment(self, image_path: str):
        """Hỗ trợ phân tích trực tiếp file ảnh (dành cho chế độ chạy đơn lẻ)."""
        orig_img = cv2.imdecode(
            np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if orig_img is None:
            raise ValueError(f"Không thể đọc file ảnh tại: {image_path}")

        h, w = orig_img.shape[:2]
        dummy_crop = [
            {
                "crop_id": 1,
                "label": "document",
                "confidence": 1.0,
                "crop_img": orig_img,
                "box": [0, 0, w, h],
                "offset": (0, 0),
                "is_obb": False,
                "saved_path": None,
            }
        ]
        return self.segment_all_crops(orig_img, dummy_crop)


## ============================================================================
# 3. HỆ THỐNG NHẬN DIỆN CHỮ THEO PIPELINE ĐA NHÁNH (MULTI-BRANCH DUAL OCR)
# ============================================================================


class DualOCRRecognizer:
    """Pipeline nhận diện chữ tuân thủ theo quy trình:

    Upscale 3x -> 4 Preprocessing Branches (Original, CLAHE, Sharpen, Adaptive)
    -> OCR từng nhánh -> Weighted Voting -> Fallback (nếu confidence thấp).
    """

    def __init__(self, vietocr_weights_path: str = None):
        # 1. VietOCR (VGG Transformer)
        import torch
        from vietocr.tool.predictor import Predictor
        from vietocr.tool.config import Cfg

        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "vgg_transformer.yml")
        if os.path.exists(config_path):
            config = Cfg.load_config_from_file(config_path)
        else:
            config = Cfg.load_config_from_name("vgg_transformer")

        candidate_weights = [
            vietocr_weights_path,
            os.path.join(base_dir, "weights", "vietocr_custom.pth"),
            os.path.join(base_dir, "vgg_transformer.pth"),
            os.path.join(os.getcwd(), "OCR_Project", "vgg_transformer.pth"),
            "vgg_transformer.pth",
        ]
        chosen_weights = next(
            (p for p in candidate_weights if p and os.path.exists(p)), None
        )
        if chosen_weights:
            config["weights"] = chosen_weights
            print(f"[VietOCR] Đã tìm thấy weights: {chosen_weights}")
        else:
            print("[VietOCR] Tự động nạp weights mặc định.")

        config["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.vietocr_predictor = Predictor(config)
        print("[+] Khởi tạo VietOCR (VGG-Transformer) thành công!")

        # 2. PP-OCRv6 (PaddleOCR)
        from paddleocr import PaddleOCR

        self.ppocr_engine = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang="ch",
        )
        print("[+] Khởi tạo PP-OCRv6 (PaddleOCR) thành công!")

        # Regex phân tích mẫu chuỗi
        self.code_pattern = re.compile(r"^[0-9\s\.\,\-\/\:\#\*\+\(\)]+$")
        self.alphanumeric_pattern = re.compile(r"^[A-Z0-9\-\/\.\:\#]+$")
        self.vietnamese_vowels = set(
            "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
            "ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ"
        )

        self.chinese_pattern = re.compile(r"[\u4e00-\u9fff]")

        from concurrent.futures import ThreadPoolExecutor

        self.executor = ThreadPoolExecutor(max_workers=2)

    # ------------------------------------------------------------------------
    # BƯỚC 3, 4, 5, 6, 7: TẠO CÁC NHÁNH TIỀN XỬ LÝ (PREPROCESSING BRANCHES)
    # ------------------------------------------------------------------------
    def _create_preprocessing_branches(self, patch_bgr: np.ndarray) -> dict:
        """Thực hiện Upscale 3x và phân tách thành 4 biến thể ảnh."""
        # Bước 3: Upscale 3x bằng nội suy INTER_CUBIC
        img_upscaled = cv2.resize(
            patch_bgr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC
        )

        # Nhánh A: Original / Upscaled
        variant_original = img_upscaled.copy()

        # Nhánh B: Grayscale + CLAHE
        gray = cv2.cvtColor(img_upscaled, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        variant_clahe_gray = clahe.apply(gray)
        variant_clahe = cv2.cvtColor(variant_clahe_gray, cv2.COLOR_GRAY2BGR)

        # Nhánh C: Grayscale + Denoise + Mild Sharpen (Unsharp Mask)
        denoise = cv2.GaussianBlur(gray, (3, 3), 0)
        blur = cv2.GaussianBlur(denoise, (0, 0), 1.2)
        variant_sharp_gray = cv2.addWeighted(denoise, 1.5, blur, -0.5, 0)
        variant_sharp = cv2.cvtColor(variant_sharp_gray, cv2.COLOR_GRAY2BGR)

        # Nhánh D: Grayscale + Adaptive Threshold
        variant_binary_gray = cv2.adaptiveThreshold(
            denoise,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        variant_adaptive = cv2.cvtColor(variant_binary_gray, cv2.COLOR_GRAY2BGR)

        # Nhánh E: ERODE (Chuyên trị chữ mờ, đứt nét)
        kernel = np.ones(
            (2, 2), np.uint8
        )  # Dùng kernel nhỏ 2x2 để tránh dính nét quá mức
        variant_erode_gray = cv2.erode(gray, kernel, iterations=1)
        variant_erode = cv2.cvtColor(variant_erode_gray, cv2.COLOR_GRAY2BGR)

        return {
            "original": variant_original,
            "clahe": variant_clahe,
            "sharp": variant_sharp,
            "adaptive": variant_adaptive,
            "erode": variant_erode,
        }

    # ------------------------------------------------------------------------
    # BƯỚC 6: NHẬN DIỆN TRÊN MỘT NHÁNH ẢNH CỤ THỂ
    # ------------------------------------------------------------------------
    # def _run_single_ocr(self, patch_bgr: np.ndarray):
    #     """Chạy song song VietOCR & PP-OCRv6 trên 1 biến thể ảnh và chọn kết quả tốt nhất."""

    #     def task_vietocr():
    #         try:
    #             patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    #             pil_img = Image.fromarray(patch_rgb)
    #             text, prob = self.vietocr_predictor.predict(pil_img, return_prob=True)
    #             return text.strip(), float(prob)
    #         except Exception:
    #             return "", 0.0

    #     def task_ppocr():
    #         try:
    #             pp_res = list(self.ppocr_engine.predict(patch_bgr))
    #             texts = []
    #             scores = []
    #             for r in pp_res:
    #                 if isinstance(r, dict):
    #                     texts.extend(r.get("rec_texts", []))
    #                     scores.extend(r.get("rec_scores", []))
    #             text_result = " ".join(texts).strip()
    #             conf_result = float(np.mean(scores)) if scores else 0.85
    #             return text_result, conf_result
    #         except Exception:
    #             return "", 0.0

    #     future_pp = self.executor.submit(task_ppocr)
    #     future_viet = self.executor.submit(task_vietocr)
    #     ppocr_text, ppocr_conf = future_pp.result()
    #     vietocr_text, vietocr_conf = future_viet.result()

    #     # Định tuyến kết quả cho nhánh này
    #     has_chinese = bool(self.chinese_pattern.search(ppocr_text))
    #     has_vietnamese = any(char in self.vietnamese_vowels for char in vietocr_text)
    #     is_code = bool(
    #         self.code_pattern.fullmatch(ppocr_text)
    #         or self.alphanumeric_pattern.fullmatch(ppocr_text)
    #     )
    #     is_english = (
    #         bool(re.fullmatch(r"[A-Za-z0-9\s\.\,\:\-@]+", ppocr_text))
    #         and not has_vietnamese
    #     )

    #     if has_chinese or is_code:
    #         chosen_text, chosen_conf, chosen_model = (
    #             ppocr_text,
    #             ppocr_conf,
    #             "PP-OCRv6",
    #         )
    #     elif has_vietnamese and vietocr_conf > 0.35:
    #         chosen_text, chosen_conf, chosen_model = (
    #             vietocr_text,
    #             vietocr_conf,
    #             "VietOCR",
    #         )
    #     elif is_english and ppocr_conf >= 0.75:
    #         chosen_text, chosen_conf, chosen_model = (
    #             ppocr_text,
    #             ppocr_conf,
    #             "PP-OCRv6",
    #         )
    #     elif vietocr_conf > 0.4:
    #         chosen_text, chosen_conf, chosen_model = (
    #             vietocr_text,
    #             vietocr_conf,
    #             "VietOCR",
    #         )
    #     else:
    #         chosen_text, chosen_conf, chosen_model = (
    #             ppocr_text,
    #             ppocr_conf,
    #             "PP-OCRv6",
    #         )

    #     return {
    #         "text": chosen_text,
    #         "conf": chosen_conf,
    #         "model": chosen_model,
    #         "vietocr_text": vietocr_text,
    #         "ppocr_text": ppocr_text,
    #     }

    def _run_single_ocr(self, patch_bgr: np.ndarray):
        """Chạy song song VietOCR & PP-OCRv6 trên 1 biến thể ảnh và chọn kết quả tốt nhất."""

        def task_vietocr():
            try:
                patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(patch_rgb)
                text, prob = self.vietocr_predictor.predict(pil_img, return_prob=True)
                return text.strip(), float(prob)
            except Exception:
                return "", 0.0

        def task_ppocr():
            try:
                pp_res = list(self.ppocr_engine.predict(patch_bgr))
                texts = []
                scores = []
                for r in pp_res:
                    if isinstance(r, dict):
                        texts.extend(r.get("rec_texts", []))
                        scores.extend(r.get("rec_scores", []))
                text_result = " ".join(texts).strip()
                conf_result = float(np.mean(scores)) if scores else 0.85
                return text_result, conf_result
            except Exception:
                return "", 0.0

        future_pp = self.executor.submit(task_ppocr)
        future_viet = self.executor.submit(task_vietocr)
        ppocr_text, ppocr_conf = future_pp.result()
        vietocr_text, vietocr_conf = future_viet.result()

        # Định tuyến kết quả cho nhánh này (ĐÃ CẬP NHẬT LOGIC KIỂM TRA ĐÚNG)
        # Định tuyến kết quả cho nhánh này
        has_chinese = bool(self.chinese_pattern.search(ppocr_text))
        has_vietnamese_v = any(char in self.vietnamese_vowels for char in vietocr_text)
        has_vietnamese_p = any(char in self.vietnamese_vowels for char in ppocr_text)

        # BẮT BỆNH: Kiểm tra độ dài để xem PP-OCR có bị "mù" chữ viết tay không
        # Nếu VietOCR đọc ra chuỗi dài (>8 ký tự), mà PP-OCR chỉ đọc được chưa tới một nửa số đó
        # -> Chứng tỏ PP-OCR đã bỏ sót toàn bộ nội dung viết tay
        pp_missed_text = (
            len(vietocr_text) > 8 and len(ppocr_text) < len(vietocr_text) * 0.5
        )

        is_code = bool(
            (
                self.code_pattern.fullmatch(ppocr_text)
                or self.alphanumeric_pattern.fullmatch(ppocr_text)
            )
            and not pp_missed_text  # KHÔNG cho phép nhận vơ là mã Code nếu bỏ sót quá nhiều chữ
        )

        is_english = (
            bool(re.fullmatch(r"[A-Za-z0-9\s\.\,\:\-@&!]+", ppocr_text))
            and not has_vietnamese_p
            and not pp_missed_text  # KHÔNG cho phép nhận vơ là tiếng Anh nếu bỏ sót chữ
        )

        # LOGIC ĐỊNH TUYẾN MỚI (ĐÃ ĐẢO TRẬT TỰ ƯU TIÊN)

        if has_chinese or is_code:
            # Ưu tiên 1: Tiếng Trung hoặc Mã số / Mã vạch (PP-OCRv6 tuyệt đối)
            chosen_text, chosen_conf, chosen_model = ppocr_text, ppocr_conf, "PP-OCRv6"

        elif has_vietnamese_p and ppocr_conf > vietocr_conf + 0.1:
            # Ưu tiên 2: PP-OCR đọc được tiếng Việt (thường là chữ IN HOA) và tự tin cực cao
            chosen_text, chosen_conf, chosen_model = ppocr_text, ppocr_conf, "PP-OCRv6"

        elif has_vietnamese_v and (vietocr_conf > 0.65 or pp_missed_text):
            # Ưu tiên 3: VietOCR đọc được tiếng Việt chuẩn (hoặc PP-OCR bỏ sót chữ viết tay)
            # Khối lệnh này ĐÃ ĐƯỢC CHUYỂN LÊN TRÊN để chặn đứng biến is_english
            chosen_text, chosen_conf, chosen_model = (
                vietocr_text,
                vietocr_conf,
                "VietOCR",
            )

        elif is_english and ppocr_conf >= 0.75:
            # Ưu tiên 4: Chữ tiếng Anh không dấu (Ví dụ: J&T Express), chỉ xét sau khi tiếng Việt đã tạch
            chosen_text, chosen_conf, chosen_model = ppocr_text, ppocr_conf, "PP-OCRv6"

        elif ppocr_conf >= vietocr_conf:
            # Ưu tiên 5: Khi cả 2 model đều yếu, tin tưởng PP-OCR vì nó ít bị ảo giác ghép chữ
            chosen_text, chosen_conf, chosen_model = ppocr_text, ppocr_conf, "PP-OCRv6"
        else:
            chosen_text, chosen_conf, chosen_model = (
                vietocr_text,
                vietocr_conf,
                "VietOCR",
            )

        return {
            "text": chosen_text,
            "conf": chosen_conf,
            "model": chosen_model,
            "vietocr_text": vietocr_text,
            "ppocr_text": ppocr_text,
        }

    # ------------------------------------------------------------------------
    # BƯỚC 10: THUẬT TOÁN BỎ PHIẾU CÓ TRỌNG SỐ (WEIGHTED VOTING)
    # ------------------------------------------------------------------------
    def _weighted_voting(self, branch_results: dict):
        """Bỏ phiếu theo công thức: score(text) = Σ OCR confidence của branch trả text đó."""
        score_board = {}

        for branch_name, res in branch_results.items():
            text = res["text"].strip()
            conf = res["conf"]
            model = res["model"]

            if not text:
                continue

            if text in score_board:
                score_board[text]["total_score"] += conf
                score_board[text]["votes"] += 1
                if conf > score_board[text]["max_conf"]:
                    score_board[text]["max_conf"] = conf
                    score_board[text]["model"] = model
            else:
                score_board[text] = {
                    "total_score": conf,
                    "votes": 1,
                    "max_conf": conf,
                    "model": model,
                }

        if not score_board:
            return "", 0.0, "None"

        # Chọn text có tổng điểm tin cậy tích lũy cao nhất
        best_text = max(score_board, key=lambda k: score_board[k]["total_score"])
        # Độ tin cậy chuẩn hóa (trung bình cộng điểm tự tin từ các nhánh bầu cho text đó)
        final_conf = (
            score_board[best_text]["total_score"] / score_board[best_text]["votes"]
        )
        chosen_model = score_board[best_text]["model"]

        return best_text, final_conf, chosen_model

    # ------------------------------------------------------------------------
    # HÀM ĐIỀU PHỐI CHÍNH CHO TỪNG Ô (RECOGNIZE_CELL)
    # ------------------------------------------------------------------------
    def recognize_cell(self, patch_bgr: np.ndarray):
        if patch_bgr is None or patch_bgr.size == 0:
            return {
                "final_text": "",
                "chosen_model": "None",
                "vietocr_text": "",
                "ppocr_text": "",
                "confidence": 0.0,
            }

        # 1. Tạo các nhánh tiền xử lý (Upscale 3x + 4 biến thể ảnh)
        variants = self._create_preprocessing_branches(patch_bgr)

        # 2. Chạy OCR trên từng nhánh
        branch_results = {}
        for name, img_var in variants.items():
            branch_results[name] = self._run_single_ocr(img_var)

        # 3. Bỏ phiếu có trọng số
        final_text, final_conf, chosen_model = self._weighted_voting(branch_results)

        # 4. Bước 11: Fallback khi độ tin cậy thấp (Deblur / Chi tiết hóa ảnh)
        # Chỉ kích hoạt khi các nhánh bất đồng ý kiến hoặc confidence < 0.50
        if final_conf < 0.50:
            # Dùng cv2.detailEnhance để khử mờ / tái tạo viền chữ
            img_upscaled = variants["original"]
            fallback_img = cv2.detailEnhance(img_upscaled, sigma_s=10, sigma_r=0.15)

            fallback_res = self._run_single_ocr(fallback_img)
            branch_results["fallback_deblur"] = fallback_res

            # Bầu chọn lại sau khi bổ sung nhánh Fallback
            final_text, final_conf, chosen_model = self._weighted_voting(branch_results)

        # Lấy thông tin text phụ từ nhánh gốc (original) phục vụ logging/debugging
        orig_res = branch_results.get("original", {})
        vietocr_sample = orig_res.get("vietocr_text", "")
        ppocr_sample = orig_res.get("ppocr_text", "")

        return {
            "final_text": final_text,
            "chosen_model": chosen_model,
            "vietocr_text": vietocr_sample,
            "ppocr_text": ppocr_sample,
            "confidence": round(final_conf, 4),
        }


# # 4. BỘ TRỰC QUAN HÓA VÀ XUẤT ẢNH GHÉP 2 TRONG 1 (DUAL-VIEW VISUALIZER)


# class DualViewVisualizer:

#     def __init__(self):
#         # 1. Tự động quét Font Tiếng Việt từ hệ thống (Bỏ qua đường dẫn C:/ cứng ngắc)
#         self.vi_font_path = "arial.ttf"  # Cấu hình dự phòng an toàn nhất

#         # Dùng try-except để bắt Pillow tự tìm và nạp thử font
#         for font_name in ["tahoma.ttf", "arial.ttf", "segoeui.ttf", "calibri.ttf"]:
#             try:
#                 ImageFont.truetype(font_name, 10)
#                 self.vi_font_path = font_name
#                 break  # Nạp thành công font đầu tiên thì thoát ngay
#             except IOError:
#                 continue

#         # 2. Tự động quét Font Tiếng Trung (Tránh lỗi ô vuông cho PaddleOCR)
#         self.zh_font_path = self.vi_font_path
#         for font_name in ["msyh.ttc", "simsun.ttc"]:
#             try:
#                 ImageFont.truetype(font_name, 10)
#                 self.zh_font_path = font_name
#                 break
#             except IOError:
#                 continue

#     def _get_font(self, size: int, is_chinese: bool = False, bold: bool = False):
#         # Chọn font theo ngôn ngữ
#         font_path = self.zh_font_path if is_chinese else self.vi_font_path

#         try:
#             # Xử lý thông minh khi cần font in đậm (cho các thẻ nhãn #1, #2)
#             if bold and not is_chinese:
#                 if "tahoma" in font_path.lower():
#                     font_path = "tahomabd.ttf"
#                 elif "arial" in font_path.lower():
#                     font_path = "arialbd.ttf"
#                 elif "segoeui" in font_path.lower():
#                     font_path = "segoeuib.ttf"

#             return ImageFont.truetype(font_path, size)
#         except IOError:
#             # Nếu hệ thống không có phiên bản in đậm, lùi về bản in thường
#             try:
#                 return ImageFont.truetype(self.vi_font_path, size)
#             except Exception as e:
#                 print(f"Font Error loading {font_path}: {e}")
#                 return ImageFont.load_default()

#     def _wrap_text(self, draw, text: str, max_width: int, font):
#         """Ngắt dòng tự động cho văn bản vừa với chiều rộng ô."""
#         words = text.split(" ")
#         lines = []
#         current = ""
#         for w in words:
#             test = f"{current} {w}".strip()
#             bbox = draw.textbbox((0, 0), test, font=font)
#             tw = bbox[2] - bbox[0]
#             if tw <= max_width:
#                 current = test
#             else:
#                 if current:
#                     lines.append(current)
#                 current = w
#         if current:
#             lines.append(current)
#         return lines

#     def create_dual_view_image(
#         self, original_bgr: np.ndarray, cells_with_ocr: list, header_title: str = ""
#     ):
#         """
#         Ghép 2 ảnh: Trái (Ảnh gốc + ô quét) | Phải (Chữ viết ứng với từng ô).
#         """
#         h_orig, w_orig = original_bgr.shape[:2]

#         # --------------------------------------------------------------------
#         # 1. TẠO ẢNH BÊN TRÁI: ẢNH GỐC VẼ CÁC Ô ĐÃ ĐƯỢC QUÉT
#         # --------------------------------------------------------------------
#         left_pil = Image.fromarray(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
#         left_draw = ImageDraw.Draw(left_pil)
#         badge_font = self._get_font(size=14, is_chinese=False, bold=True)

#         color_palette = [
#             (0, 150, 255),
#             (16, 185, 129),
#             (245, 158, 11),
#             (139, 92, 246),
#             (236, 72, 153),
#             (239, 68, 68),
#         ]

#         for item in cells_with_ocr:
#             order_id = item["order_id"]
#             box = item["box"]
#             color = color_palette[(order_id - 1) % len(color_palette)]
#             x1, y1, x2, y2 = box

#             left_draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

#             badge_text = f"#{order_id}"
#             t_bbox = left_draw.textbbox((0, 0), badge_text, font=badge_font)
#             bw = t_bbox[2] - t_bbox[0] + 8
#             bh = t_bbox[3] - t_bbox[1] + 6

#             by1 = max(0, y1 - bh)
#             by2 = by1 + bh
#             bx1 = x1
#             bx2 = min(w_orig, bx1 + bw)

#             left_draw.rectangle([bx1, by1, bx2, by2], fill=color)
#             left_draw.text(
#                 (bx1 + 4, by1 + 2), badge_text, fill=(255, 255, 255), font=badge_font
#             )

#         # --------------------------------------------------------------------
#         # 2. TẠO ẢNH BÊN PHẢI: CHỮ VIẾT ỨNG VỚI TỪNG Ô ĐÃ QUÉT

#         right_pil = Image.new("RGB", (w_orig, h_orig), color=(250, 252, 255))
#         right_draw = ImageDraw.Draw(right_pil)

#         import re  # Đảm bảo thư viện regex được gọi

#         for item in cells_with_ocr:
#             order_id = item["order_id"]
#             box = item["box"]
#             x1, y1, x2, y2 = box
#             bw = x2 - x1
#             bh = y2 - y1
#             color = color_palette[(order_id - 1) % len(color_palette)]

#             text = item.get("final_text", "").strip()
#             model_used = item.get("chosen_model", "")

#             right_draw.rectangle(
#                 [x1, y1, x2, y2], fill=(241, 245, 249), outline=color, width=2
#             )

#             model_tag = "VietOCR" if model_used == "VietOCR" else "PP-OCR"
#             # model_tag = "VIET" if model_used == "VietOCR" else "PP"

#             tag_text = (
#                 f"#{order_id} [{model_tag}]" if model_used != "None" else f"#{order_id}"
#             )

#             t_bbox = right_draw.textbbox((0, 0), tag_text, font=badge_font)
#             badge_w = t_bbox[2] - t_bbox[0] + 8
#             badge_h = t_bbox[3] - t_bbox[1] + 4

#             tab_y1 = y1 - badge_h
#             tab_y2 = y1
#             tab_x1 = x1
#             tab_x2 = min(w_orig, x1 + badge_w)

#             overlaps_other = tab_y1 < 0
#             if not overlaps_other:
#                 for other in cells_with_ocr:
#                     if other["order_id"] == order_id:
#                         continue
#                     ox1, oy1, ox2, oy2 = other["box"]
#                     if not (
#                         tab_x2 <= ox1 or tab_x1 >= ox2 or tab_y2 <= oy1 or tab_y1 >= oy2
#                     ):
#                         overlaps_other = True
#                         break

#             if overlaps_other and bh >= badge_h + 16:
#                 tab_y1 = y1 + 1
#                 tab_y2 = y1 + 1 + badge_h
#                 tab_x1 = max(x1 + 1, x2 - badge_w - 2)
#                 tab_x2 = min(w_orig, tab_x1 + badge_w)
#             elif overlaps_other:
#                 tab_y1 = max(0, y1 - badge_h)
#                 tab_y2 = tab_y1 + badge_h

#             right_draw.rectangle([tab_x1, tab_y1, tab_x2, tab_y2], fill=color)
#             right_draw.text(
#                 (tab_x1 + 4, tab_y1 + 1),
#                 tag_text,
#                 fill=(255, 255, 255),
#                 font=badge_font,
#             )

#             # C. Render văn bản bên trong ô với Dynamic Font
#             if text:
#                 # Quét xem nội dung có chứa ký tự Tiếng Trung không
#                 has_chinese = bool(re.search(r"[\u4e00-\u9fff]", text))

#                 init_size = max(11, min(int(bh * 0.60), 28))

#                 # Gọi font tương ứng với ngôn ngữ
#                 font = self._get_font(size=init_size, is_chinese=has_chinese)
#                 lines = self._wrap_text(right_draw, text, max_width=bw - 8, font=font)

#                 line_height = int(init_size * 1.35)
#                 total_text_h = len(lines) * line_height

#                 while total_text_h > bh - 4 and init_size > 10:
#                     init_size -= 1
#                     # Cập nhật lại font với kích thước mới
#                     font = self._get_font(size=init_size, is_chinese=has_chinese)
#                     line_height = int(init_size * 1.35)
#                     lines = self._wrap_text(
#                         right_draw, text, max_width=bw - 8, font=font
#                     )
#                     total_text_h = len(lines) * line_height

#                 start_y = y1 + max(2, (bh - total_text_h) // 2)
#                 for line_idx, line_str in enumerate(lines):
#                     curr_y = start_y + line_idx * line_height
#                     if curr_y + line_height <= y2 + 4:
#                         right_draw.text(
#                             (x1 + 5, curr_y), line_str, fill=(15, 23, 42), font=font
#                         )

#         # 3. GHÉP 2 ẢNH VÀ THÊM THANH TIÊU ĐỀ HEADER
#         banner_h = 90
#         divider_w = 6
#         total_w = w_orig * 2 + divider_w
#         total_h = h_orig + banner_h

#         composite = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
#         comp_draw = ImageDraw.Draw(composite)

#         comp_draw.rectangle([0, 0, w_orig, banner_h], fill=(30, 41, 59))
#         comp_draw.rectangle(
#             [w_orig + divider_w, 0, total_w, banner_h], fill=(15, 118, 110)
#         )
#         comp_draw.rectangle(
#             [w_orig, 0, w_orig + divider_w, total_h], fill=(100, 116, 139)
#         )

#         title_font = self._get_font(size=20, is_chinese=False)
#         sub_font = self._get_font(size=14, is_chinese=False)

#         comp_draw.text(
#             (25, 16),
#             "[BÊN TRÁI: ẢNH GỐC CÁC Ô ĐÃ ĐƯỢC QUÉT]",
#             fill=(255, 255, 255),
#             font=title_font,
#         )
#         comp_draw.text(
#             (25, 52),
#             "Quy trình: YOLO26 (Crop) -> PP-DocLayoutV3 + PP-StructureV3 (Khung & Ô)",
#             fill=(148, 163, 184),
#             font=sub_font,
#         )

#         comp_draw.text(
#             (w_orig + divider_w + 25, 16),
#             "[BÊN PHẢI: CHỮ VIẾT ỨNG VỚI TỪNG Ô ĐÃ QUÉT]",
#             fill=(255, 255, 255),
#             font=title_font,
#         )
#         comp_draw.text(
#             (w_orig + divider_w + 25, 52),
#             "Nhận diện chữ: VietOCR (Tiếng Việt) + PP-OCRv6 (Số / Mã ID)",
#             fill=(167, 243, 208),
#             font=sub_font,
#         )

#         composite.paste(left_pil, (0, banner_h))
#         composite.paste(right_pil, (w_orig + divider_w, banner_h))

#         return composite

# 4. BỘ TRỰC QUAN HÓA VÀ XUẤT ẢNH GHÉP 2 TRONG 1 (DUAL-VIEW VISUALIZER)


class DualViewVisualizer:

    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.roboto_path = os.path.join(base_dir, "Roboto-Regular.ttf")

        self.vi_font_path = None
        self.zh_font_path = None

        windows_font_dirs = [
            "C:/Windows/Fonts",
            "D:/Windows/Fonts",
            os.environ.get("WINDIR", "C:/Windows") + "/Fonts",
        ]

        vi_targets = [
            "tahoma.ttf",
            "arial.ttf",
            "segoeui.ttf",
            "calibri.ttf",
            "Roboto-Regular.ttf",
        ]

        zh_targets = ["msyh.ttc", "simsun.ttc"]

        # 1. Tìm font tiếng Việt
        for d in windows_font_dirs:
            if not os.path.exists(d):
                continue
            for f_name in vi_targets:
                full_path = os.path.join(d, f_name)
                if os.path.exists(full_path):
                    try:
                        ImageFont.truetype(full_path, 10)
                        self.vi_font_path = full_path
                        break
                    except Exception:
                        pass
            if self.vi_font_path:
                break

        # Nếu lỗi quyền truy cập Windows, bắt buộc dùng file Roboto trong thư mục dự án
        if not self.vi_font_path and os.path.exists(self.roboto_path):
            self.vi_font_path = self.roboto_path

        # 2. Tìm font tiếng Trung
        for d in windows_font_dirs:
            if not os.path.exists(d):
                continue
            for f_name in zh_targets:
                full_path = os.path.join(d, f_name)
                if os.path.exists(full_path):
                    try:
                        ImageFont.truetype(full_path, 10)
                        self.zh_font_path = full_path
                        break
                    except Exception:
                        pass
            if self.zh_font_path:
                break

        if not self.zh_font_path:
            self.zh_font_path = self.vi_font_path

    def _get_font(self, size: int, is_chinese: bool = False, bold: bool = False):
        font_path = self.zh_font_path if is_chinese else self.vi_font_path

        # Xử lý linh hoạt việc nạp font in đậm
        if bold and font_path and not is_chinese:
            bold_path = font_path.replace(".ttf", "bd.ttf").replace(
                "segoeui.ttf", "segoeuib.ttf"
            )
            if os.path.exists(bold_path):
                try:
                    return ImageFont.truetype(bold_path, size)
                except Exception:
                    pass

        try:
            if font_path:
                return ImageFont.truetype(font_path, size)
        except Exception as e:
            print(f"Lỗi nạp font chính ({font_path}): {e}")

        # Cứu cánh cuối cùng là Roboto đi kèm file code
        if os.path.exists(self.roboto_path):
            try:
                return ImageFont.truetype(self.roboto_path, size)
            except Exception:
                pass

        return ImageFont.load_default()

    def _wrap_text(self, draw, text: str, max_width: int, font):
        """Ngắt dòng tự động cho văn bản vừa với chiều rộng ô."""
        words = text.split(" ")
        lines = []
        current = ""
        for w in words:
            test = f"{current} {w}".strip()
            bbox = draw.textbbox((0, 0), test, font=font)
            tw = bbox[2] - bbox[0]
            if tw <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = w
        if current:
            lines.append(current)
        return lines

    def create_dual_view_image(
        self, original_bgr: np.ndarray, cells_with_ocr: list, header_title: str = ""
    ):
        """Ghép 2 ảnh: Trái (Ảnh gốc + ô quét) | Phải (Chữ viết ứng với từng ô)."""
        h_orig, w_orig = original_bgr.shape[:2]

        left_pil = Image.fromarray(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
        left_draw = ImageDraw.Draw(left_pil)
        badge_font = self._get_font(size=14, is_chinese=False, bold=True)

        color_palette = [
            (0, 150, 255),
            (16, 185, 129),
            (245, 158, 11),
            (139, 92, 246),
            (236, 72, 153),
            (239, 68, 68),
        ]

        for item in cells_with_ocr:
            order_id = item["order_id"]
            box = item["box"]
            color = color_palette[(order_id - 1) % len(color_palette)]
            x1, y1, x2, y2 = box

            left_draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            badge_text = f"#{order_id}"
            t_bbox = left_draw.textbbox((0, 0), badge_text, font=badge_font)
            bw = t_bbox[2] - t_bbox[0] + 8
            bh = t_bbox[3] - t_bbox[1] + 6

            by1 = max(0, y1 - bh)
            by2 = by1 + bh
            bx1 = x1
            bx2 = min(w_orig, bx1 + bw)

            left_draw.rectangle([bx1, by1, bx2, by2], fill=color)
            left_draw.text(
                (bx1 + 4, by1 + 2), badge_text, fill=(255, 255, 255), font=badge_font
            )

        right_pil = Image.new("RGB", (w_orig, h_orig), color=(250, 252, 255))
        right_draw = ImageDraw.Draw(right_pil)

        import re

        for item in cells_with_ocr:
            order_id = item["order_id"]
            box = item["box"]
            x1, y1, x2, y2 = box
            bw = x2 - x1
            bh = y2 - y1
            color = color_palette[(order_id - 1) % len(color_palette)]

            text = item.get("final_text", "").strip()
            model_used = item.get("chosen_model", "")

            right_draw.rectangle(
                [x1, y1, x2, y2], fill=(241, 245, 249), outline=color, width=2
            )

            # Rút gọn nhãn thẻ model cho gọn gàng
            model_tag = "VIET" if model_used == "VietOCR" else "PP"
            tag_text = (
                f"#{order_id} [{model_tag}]" if model_used != "None" else f"#{order_id}"
            )

            t_bbox = right_draw.textbbox((0, 0), tag_text, font=badge_font)
            badge_w = t_bbox[2] - t_bbox[0] + 8
            badge_h = t_bbox[3] - t_bbox[1] + 4

            tab_y1 = y1 - badge_h
            tab_y2 = y1
            tab_x1 = x1
            tab_x2 = min(w_orig, x1 + badge_w)

            overlaps_other = tab_y1 < 0
            if not overlaps_other:
                for other in cells_with_ocr:
                    if other["order_id"] == order_id:
                        continue
                    ox1, oy1, ox2, oy2 = other["box"]
                    if not (
                        tab_x2 <= ox1 or tab_x1 >= ox2 or tab_y2 <= oy1 or tab_y1 >= oy2
                    ):
                        overlaps_other = True
                        break

            if overlaps_other and bh >= badge_h + 16:
                tab_y1 = y1 + 1
                tab_y2 = y1 + 1 + badge_h
                tab_x1 = max(x1 + 1, x2 - badge_w - 2)
                tab_x2 = min(w_orig, tab_x1 + badge_w)
            elif overlaps_other:
                tab_y1 = max(0, y1 - badge_h)
                tab_y2 = tab_y1 + badge_h

            right_draw.rectangle([tab_x1, tab_y1, tab_x2, tab_y2], fill=color)
            right_draw.text(
                (tab_x1 + 4, tab_y1 + 1),
                tag_text,
                fill=(255, 255, 255),
                font=badge_font,
            )

            if text:
                has_chinese = bool(re.search(r"[\u4e00-\u9fff]", text))
                init_size = max(11, min(int(bh * 0.60), 28))

                font = self._get_font(size=init_size, is_chinese=has_chinese)
                lines = self._wrap_text(right_draw, text, max_width=bw - 8, font=font)

                line_height = int(init_size * 1.35)
                total_text_h = len(lines) * line_height

                while total_text_h > bh - 4 and init_size > 10:
                    init_size -= 1
                    font = self._get_font(size=init_size, is_chinese=has_chinese)
                    line_height = int(init_size * 1.35)
                    lines = self._wrap_text(
                        right_draw, text, max_width=bw - 8, font=font
                    )
                    total_text_h = len(lines) * line_height

                start_y = y1 + max(2, (bh - total_text_h) // 2)
                for line_idx, line_str in enumerate(lines):
                    curr_y = start_y + line_idx * line_height
                    if curr_y + line_height <= y2 + 4:
                        right_draw.text(
                            (x1 + 5, curr_y), line_str, fill=(15, 23, 42), font=font
                        )

        # GHÉP ẢNH VÀ BANNER
        banner_h = 90
        divider_w = 6
        total_w = w_orig * 2 + divider_w
        total_h = h_orig + banner_h

        composite = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
        comp_draw = ImageDraw.Draw(composite)
        comp_draw.rectangle(
            [w_orig, 0, w_orig + divider_w, total_h], fill=(100, 116, 139)
        )

        left_banner = Image.new("RGB", (w_orig, banner_h), color=(30, 41, 59))
        lb_draw = ImageDraw.Draw(left_banner)

        # Trích xuất biến draw_obj để đo font chính xác trên Canvas Banner
        def get_fitted_font(draw_obj, text, max_w, max_size):
            size = max_size
            font = self._get_font(size=size, is_chinese=False)
            while size > 10:
                bbox = draw_obj.textbbox((0, 0), text, font=font)
                if (bbox[2] - bbox[0]) <= max_w:
                    break
                size -= 1
                font = self._get_font(size=size, is_chinese=False)
            return font

        title_left = "[BÊN TRÁI: ẢNH GỐC ĐÃ QUÉT]"
        sub_left = "Quy trình: YOLO26 -> PP-DocLayoutV3 + PP-StructureV3"
        t_font_left = get_fitted_font(lb_draw, title_left, max(10, w_orig - 20), 20)
        s_font_left = get_fitted_font(lb_draw, sub_left, max(10, w_orig - 20), 14)

        lb_draw.text((10, 16), title_left, fill=(255, 255, 255), font=t_font_left)
        lb_draw.text((10, 52), sub_left, fill=(148, 163, 184), font=s_font_left)

        right_banner = Image.new("RGB", (w_orig, banner_h), color=(15, 118, 110))
        rb_draw = ImageDraw.Draw(right_banner)

        title_right = "[BÊN PHẢI: CHỮ VIẾT TƯƠNG ỨNG]"
        sub_right = "Nhận diện: VietOCR (Tiếng Việt) + PP-OCRv6 (Số/ID)"
        t_font_right = get_fitted_font(rb_draw, title_right, max(10, w_orig - 20), 20)
        s_font_right = get_fitted_font(rb_draw, sub_right, max(10, w_orig - 20), 14)

        rb_draw.text((10, 16), title_right, fill=(255, 255, 255), font=t_font_right)
        rb_draw.text((10, 52), sub_right, fill=(167, 243, 208), font=s_font_right)

        composite.paste(left_banner, (0, 0))
        composite.paste(right_banner, (w_orig + divider_w, 0))
        composite.paste(left_pil, (0, banner_h))
        composite.paste(right_pil, (w_orig + divider_w, banner_h))

        return composite


# 5. PIPELINE THỰC THI HOÀN CHỈNH (MAIN CONTROLLER)
def run_document_dual_ocr_pipeline(
    image_path: str,
    yolo_cropper,  # Truyền model từ ngoài vào
    segmenter,  # Chỉ dùng khi bật PP-DocLayoutV3/PP-StructureV3
    recognizer,  # Truyền model từ ngoài vào
    visualizer,  # Truyền model từ ngoài vào
    output_dir: str = "output",
    pre_cropped: bool = None,
):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy file ảnh tại: {image_path}")

    stem_name = Path(image_path).stem
    os.makedirs(output_dir, exist_ok=True)

    import glob
    import re

    # Tìm các file đã chạy trước đó để đếm xem đây là lần chạy thứ mấy
    existing_files = glob.glob(
        os.path.join(output_dir, f"{stem_name}_run*_dual_view.jpg")
    )
    run_number = 1
    if existing_files:
        run_numbers = []
        for f in existing_files:
            match = re.search(r"_run(\d+)_dual_view\.jpg$", f)
            if match:
                run_numbers.append(int(match.group(1)))
        if run_numbers:
            run_number = max(run_numbers) + 1

    run_suffix = f"_run{run_number}"

    print("\n" + "#" * 75)
    print(f"# KHỞI CHẠY PIPELINE OCR DUO-VIEW: {os.path.basename(image_path)}")
    print("#" * 75)

    orig_bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if orig_bgr is None:
        raise ValueError(f"Không thể đọc file ảnh tại: {image_path}")

    # 1-2. Phân vùng ảnh hoặc dùng trực tiếp ảnh đã crop
    print("\n" + "=" * 80)
    crops_dir = os.path.join(output_dir, "crops", f"{stem_name}{run_suffix}")
    yolo_crops = None

    if pre_cropped is None:
        if yolo_cropper is None:
            pre_cropped = True
        else:
            print("AUTO: Đang kiểm tra nhanh ảnh có cần phân vùng hay không...")
            yolo_crops = yolo_cropper.detect_and_crop(
                img_bgr=orig_bgr,
                output_crop_dir=crops_dir,
                stem_name=stem_name,
                padding_x=8,
                padding_y=4,
            )
            pre_cropped = len(yolo_crops) == 1 and yolo_crops[0].get(
                "is_fallback", False
            )
            print(
                "AUTO: Ảnh được xem là crop sẵn."
                if pre_cropped
                else "AUTO: Phát hiện bố cục tài liệu, chạy đầy đủ pipeline."
            )

    if pre_cropped:
        h, w = orig_bgr.shape[:2]
        yolo_crops = [
            {
                "crop_id": 1,
                "label": "pre_cropped",
                "original_label": "pre_cropped",
                "confidence": 1.0,
                "crop_img": orig_bgr,
                "box": [0, 0, w, h],
                "offset": (0, 0),
                "is_obb": False,
                "saved_path": None,
            }
        ]
        processed_bgr = orig_bgr
        frames = [
            {
                "frame_id": 1,
                "label": "pre_cropped",
                "score": 1.0,
                "box": [0, 0, w, h],
            }
        ]
        cells = [
            {
                "order_id": 1,
                "source": "yolo_crop",
                "frame_label": "pre_cropped",
                "box": [0, 0, w, h],
                "crop_img": orig_bgr,
            }
        ]
        print("1-2. BỎ QUA YOLO/PP-DOCLAYOUTV3/PP-STRUCTUREV3 (ẢNH ĐÃ CROP SẴN)")
    else:
        print("1. BƯỚC 1: PHÁT HIỆN & CẮT VÙNG ĐỐI TƯỢNG / BỐ CỤC BẰNG YOLO26...")
        if yolo_crops is None:
            yolo_crops = yolo_cropper.detect_and_crop(
                img_bgr=orig_bgr,
                output_crop_dir=crops_dir,
                stem_name=stem_name,
                padding_x=8,
                padding_y=4,
            )
        print(f"[+] YOLO26 đã phát hiện và cắt thành công {len(yolo_crops)} vùng ảnh.")

        print("\n" + "=" * 80)
        print(
            "2. BƯỚC 2: PHÂN LOẠI KHUNG (PP-DOCLAYOUTV3) & CHIA Ô (PP-STRUCTUREV3) TRÊN CÁC CROP..."
        )
        if segmenter is None:
            segmenter = DocumentStructureSegmenter()
        processed_bgr, frames, cells = segmenter.segment_all_crops(
            original_bgr=orig_bgr, crops=yolo_crops
        )

    # 3. BƯỚC 3: NHẬN DIỆN CHỮ CHO TỪNG Ô VỚI VIETOCR + PP-OCRV6
    print("\n" + "=" * 80)
    print("3. BƯỚC 3: NHẬN DIỆN CHỮ VIETOCR (TIẾNG VIỆT) + PP-OCRV6 (SỐ/ID)...")
    print("=" * 80)

    print("\n" + "=" * 80)
    print(
        f"{'Ô':<5} | {'BBOX [x1,y1,x2,y2]':<22} | {'MODEL CHỌN':<12} | {'KẾT QUẢ NHẬN DIỆN'}"
    )
    print("=" * 80)

    cells_with_ocr = []
    for idx, cell in enumerate(cells):
        order_id = cell.get("order_id", idx + 1)
        box = cell["box"]

        if cell.get("source") == "yolo_crop" and cell.get("crop_img") is not None:
            patch = cell["crop_img"]
            padded_box = box
        else:
            patch, padded_box = crop_image_patch(processed_bgr, box, pad_x=6, pad_y=4)
        if patch is None or patch.size == 0:
            continue

        ocr_result = recognizer.recognize_cell(patch)

        cell_data = {
            "order_id": order_id,
            "box": box,
            "padded_box": padded_box,
            "frame_label": cell.get("frame_label", "text"),
            "final_text": ocr_result["final_text"],
            "chosen_model": ocr_result["chosen_model"],
            "vietocr_text": ocr_result["vietocr_text"],
            "ppocr_text": ocr_result["ppocr_text"],
            "confidence": ocr_result["confidence"],
        }
        cells_with_ocr.append(cell_data)

        box_str = f"[{box[0]},{box[1]},{box[2]},{box[3]}]"
        print(
            f"#{order_id:<4} | {box_str:<22} | {cell_data['chosen_model']:<12} | {cell_data['final_text']}"
        )

    print("=" * 80)

    # 4. BƯỚC 4: XUẤT 1 FILE ẢNH GỒM 2 ẢNH GHÉP NGANG (SIDE-BY-SIDE DUAL VIEW)
    print("\n4. ĐANG VẼ VÀ XUẤT ẢNH GHÉP 2 TRONG 1 (SIDE-BY-SIDE DUAL VIEW)...")
    dual_view_image = visualizer.create_dual_view_image(
        original_bgr=processed_bgr,
        cells_with_ocr=cells_with_ocr,
        header_title=f"Tài liệu: {os.path.basename(image_path)} (Phiên bản: {run_number})",
    )

    output_img_path = os.path.join(output_dir, f"{stem_name}{run_suffix}_dual_view.jpg")
    dual_view_image.save(output_img_path, quality=95)

    # 5. XUẤT DỮ LIỆU JSON VÀ MARKDOWN ĐỂ PHỤC VỤ TÍCH HỢP HỆ THỐNG
    output_json_path = os.path.join(
        output_dir, f"{stem_name}{run_suffix}_dual_ocr.json"
    )
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "image_path": image_path,
                "run_version": run_number,
                "total_yolo_crops": len(yolo_crops),
                "total_frames": len(frames),
                "total_cells": len(cells_with_ocr),
                "yolo_crops": [
                    {
                        "crop_id": c["crop_id"],
                        "label": c["label"],
                        "confidence": c["confidence"],
                        "box": c["box"],
                        "saved_path": c.get("saved_path"),
                    }
                    for c in yolo_crops
                ],
                "layout_frames": frames,
                "scanned_cells": cells_with_ocr,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    output_md_path = os.path.join(
        output_dir, f"{stem_name}{run_suffix}_transcription.md"
    )
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(
            f"# Kết quả OCR: {os.path.basename(image_path)} (Lần chạy {run_number})\n\n"
        )
        f.write(f"- **Số vùng crop bởi YOLO26:** {len(yolo_crops)}\n")
        f.write(f"- **Tổng số khung bố cục (PP-DocLayoutV3):** {len(frames)}\n")
        f.write(f"- **Tổng số ô đã quét và nhận diện:** {len(cells_with_ocr)}\n\n")
        f.write("### Danh sách vùng crop bởi YOLO26:\n")
        for yc in yolo_crops:
            path_str = f" (`{yc['saved_path']}`)" if yc.get("saved_path") else ""
            f.write(
                f"- Crop #{yc['crop_id']} | Nhãn: `{yc['label']}` (conf: {yc['confidence']}) | BBox: `{yc['box']}`{path_str}\n"
            )
        f.write("\n### Chi tiết nhận diện từng ô:\n")
        f.write("| Ô | Khung Bố Cục | Model | Tọa Độ Box | Nội Dung Nhận Diện |\n")
        f.write("|---|---|---|---|---|\n")
        for c in cells_with_ocr:
            f.write(
                f"| #{c['order_id']} | `{c['frame_label']}` | {c['chosen_model']} | `{c['box']}` | {c['final_text']} |\n"
            )

    print(f"\n[+] HOÀN THÀNH XUẤT SẮC (LẦN CHẠY SỐ {run_number})!")
    print(f" [1] FILE ẢNH GHÉP 2 BÊN: {output_img_path}")
    print(f" [2] File dữ liệu JSON:   {output_json_path}")
    print(f" [3] File Markdown:       {output_md_path}")
    print(f" [4] Thư mục ảnh crop:    {crops_dir}")
    print("=" * 80 + "\n")

    return output_img_path, output_json_path


# 6. ĐIỂM KHỞI CHẠY (XỬ LÝ HÀNG LOẠT VỚI GIAO DIỆN CHỌN THƯ MỤC)

if __name__ == "__main__":
    import glob
    import sys
    import tkinter as tk
    from tkinter import filedialog, messagebox

    parser = argparse.ArgumentParser(
        description="Pipeline OCR: phân vùng tùy chọn -> VietOCR + PP-OCRv6"
    )
    parser.add_argument(
        "--image", type=str, default=None, help="Đường dẫn file ảnh tài liệu"
    )
    parser.add_argument(
        "--dir", type=str, default=None, help="Đường dẫn thư mục chứa nhiều ảnh"
    )
    parser.add_argument(
        "--yolo-model", type=str, default=None, help="Đường dẫn weights YOLO (.pt)"
    )
    parser.add_argument(
        "--output", type=str, default="output", help="Thư mục xuất kết quả"
    )
    parser.add_argument(
        "--vietocr-weights", type=str, default=None, help="Đường dẫn weights VietOCR"
    )
    parser.add_argument(
        "--pre-cropped",
        action="store_true",
        default=None,
        help="Ảnh đã được cắt sẵn: chỉ chạy các model nhận dạng chữ",
    )
    args = parser.parse_args()

    image_paths = []

    # ---------------------------------------------------------
    # A. BẬT HỘP THOẠI HỎI NGƯỜI DÙNG: CHỌN FOLDER HAY CHỌN FILE
    # ---------------------------------------------------------
    if not args.image and not args.dir:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        # Bật bảng hỏi lựa chọn chế độ
        choice = messagebox.askyesnocancel(
            "Lựa chọn chế độ chạy",
            "Bạn muốn xử lý hàng loạt hay chạy thử 1 ảnh?\n\n"
            "- Chọn 'Yes' để nạp toàn bộ THƯ MỤC.\n"
            "- Chọn 'No' để chọn 1 ẢNH ĐƠN LẺ.\n"
            "- Chọn 'Cancel' để thoát.",
        )

        if choice is True:
            # Người dùng chọn "Yes" -> Mở hộp thoại chọn Thư mục
            print("[HỆ THỐNG] Đang mở hộp thoại Windows. Vui lòng chọn THƯ MỤC...")
            selected_dir = filedialog.askdirectory(
                title="Chọn Thư Mục Chứa Ảnh Cần OCR"
            )
            if selected_dir:
                args.dir = selected_dir
                print(f"[*] Đã chọn thư mục: {selected_dir}")
            else:
                print("[!] Bạn đã hủy chọn thư mục. Chương trình kết thúc.")
                sys.exit(0)

        elif choice is False:
            # Người dùng chọn "No" -> Mở hộp thoại chọn 1 File ảnh
            print("[HỆ THỐNG] Đang mở hộp thoại Windows. Vui lòng chọn ẢNH...")
            selected_file = filedialog.askopenfilename(
                title="Chọn ảnh Phiếu giao hàng / Tài liệu cần quét",
                filetypes=[
                    ("Các file ảnh", "*.png *.jpg *.jpeg *.bmp *.webp"),
                    ("Tất cả file", "*.*"),
                ],
            )
            if selected_file:
                args.image = selected_file
                print(f"[*] Đã chọn ảnh: {selected_file}")
            else:
                print("[!] Bạn đã hủy chọn ảnh. Chương trình kết thúc.")
                sys.exit(0)

        else:
            # Người dùng chọn "Cancel" hoặc bấm dấu X
            print("[!] Đã thoát chương trình.")
            sys.exit(0)

    # ---------------------------------------------------------
    # B. GOM TẤT CẢ FILE ẢNH VÀO DANH SÁCH CẦN XỬ LÝ
    # ---------------------------------------------------------
    if args.dir and os.path.isdir(args.dir):
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"):
            image_paths.extend(glob.glob(os.path.join(args.dir, ext)))
    elif args.image and os.path.exists(args.image):
        image_paths.append(args.image)

    if not image_paths:
        print("[LỖI] Không tìm thấy file ảnh hợp lệ nào để chạy.")
        sys.exit(1)

    # ---------------------------------------------------------
    # C. KHỞI TẠO TẤT CẢ MODEL MỘT LẦN DUY NHẤT TRƯỚC VÒNG LẶP
    # ---------------------------------------------------------
    print(
        f"\n[*] TÌM THẤY {len(image_paths)} BỨC ẢNH. TIẾN HÀNH NẠP MODEL VÀO RAM/VRAM..."
    )
    global_yolo = None
    global_segmenter = None
    if args.pre_cropped is True:
        print(
            "[HỆ THỐNG] Chế độ ảnh đã crop: không nạp YOLO/PP-DocLayoutV3/PP-StructureV3."
        )
    else:
        global_yolo = YOLOLayoutDetector(model_path=args.yolo_model, conf_thresh=0.25)
        print(
            "[HỆ THỐNG] Chế độ tự động: sẽ tự chọn OCR trực tiếp hoặc phân vùng đầy đủ."
        )
    global_recognizer = DualOCRRecognizer(vietocr_weights_path=args.vietocr_weights)
    global_visualizer = DualViewVisualizer()
    print("[HỆ THỐNG] ĐÃ NẠP MODEL THÀNH CÔNG! BẮT ĐẦU CHẠY...\n")

    # D. VÒNG LẶP XỬ LÝ (TỰ ĐỘNG LƯU PHIÊN BẢN MỚI & ĐO THỜI GIAN)
    # ---------------------------------------------------------
    import time

    # Bắt đầu bấm giờ tổng toàn bộ quá trình quét ảnh
    total_start_time = time.time()

    for idx, img_path in enumerate(image_paths):
        print(f"\n========== ĐANG XỬ LÝ ẢNH [{idx + 1}/{len(image_paths)}] ==========")

        # Bấm giờ cho riêng bức ảnh hiện tại
        img_start_time = time.time()

        try:
            run_document_dual_ocr_pipeline(
                image_path=img_path,
                yolo_cropper=global_yolo,
                segmenter=global_segmenter,
                recognizer=global_recognizer,
                visualizer=global_visualizer,
                output_dir=args.output,
                pre_cropped=args.pre_cropped,
            )

            # Tính và in ra thời gian chạy của 1 ảnh
            img_process_time = time.time() - img_start_time
            print(f"\n[⏱] THỜI GIAN HOÀN THÀNH ẢNH NÀY: {img_process_time:.2f} giây")

        except Exception as e:
            print(f"[ERROR] Bỏ qua ảnh {img_path} do gặp lỗi: {e}")

    # Tính và in ra tổng thời gian xử lý toàn bộ danh sách
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    minutes, seconds = divmod(total_duration, 60)

    print("\n" + "=" * 80)
    print(f"[+] HOÀN TẤT QUY TRÌNH XỬ LÝ TẤT CẢ {len(image_paths)} BỨC ẢNH!")
    print(f"[⏱] TỔNG THỜI GIAN CHẠY: {int(minutes)} phút {seconds:.2f} giây.")
    print("=" * 80 + "\n")
