import os
import re
import cv2
import numpy as np
import pandas as pd
import argparse
from pathlib import Path


def extract_qc_features(image):
    """
    Trích xuất 4 nhóm đặc trưng QC: Blur, Dark, Glare (ROI 3x3), và Compression (JPEG 8x8 Blockiness).
    Điểm số càng cao -> mức độ nghi ngờ lỗi càng nặng.
    """
    gray_u8 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray_u8.astype(np.float32)
    h, w = gray.shape

    # 1. BLUR (Độ mờ) - Variance of Laplacian
    laplacian_var = float(cv2.Laplacian(gray_u8, cv2.CV_64F).var())
    blur_score = 1000.0 / (laplacian_var + 1e-5)

    # 2. DARK (Thiếu sáng) - Kết hợp độ sáng trung bình và tỷ lệ điểm ảnh tối (< 30)
    mean_brightness = float(np.mean(gray))
    dark_pixel_ratio = float(np.sum(gray < 30) / (h * w)) * 100.0
    dark_score = max(0.0, 50.0 - mean_brightness) + dark_pixel_ratio

    # 3. GLARE (Cháy sáng / Lóa cục bộ) - Chia lưới 3x3 ROI
    patch_h, patch_w = max(1, h // 3), max(1, w // 3)
    max_patch_glare = 0.0
    for i in range(3):
        for j in range(3):
            patch = gray[i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w]
            glare_ratio = float(np.sum(patch > 240)) / (patch.size + 1e-5)
            max_patch_glare = max(max_patch_glare, glare_ratio * 100.0)
    global_glare_pct = float(np.sum(gray > 240) / (h * w)) * 100.0
    glare_score = max_patch_glare * 10.0  # Scale đồng bộ với các feature khác

    # 4. COMPRESSION (Lỗi nén JPEG Blockiness 8x8)
    if w > 8:
        h_diff = float(np.abs(gray[:, 7:-1:8] - gray[:, 8::8]).mean())
        h_inner = float(np.abs(gray[:, 3:-1:8] - gray[:, 4::8]).mean())
    else:
        h_diff, h_inner = 0.0, 1.0

    if h > 8:
        v_diff = float(np.abs(gray[7:-1:8, :] - gray[8::8, :]).mean())
        v_inner = float(np.abs(gray[3:-1:8, :] - gray[4::8, :]).mean())
    else:
        v_diff, v_inner = 0.0, 1.0

    boundary_diff = (h_diff + v_diff) / 2.0
    inner_diff = (h_inner + v_inner) / 2.0
    blockiness_ratio = boundary_diff / (inner_diff + 1e-5)
    compression_score = boundary_diff * max(1.0, blockiness_ratio)

    return {
        "blur_score": round(blur_score, 4),
        "laplacian_var": round(laplacian_var, 4),
        "dark_score": round(dark_score, 4),
        "mean_brightness": round(mean_brightness, 4),
        "dark_pixel_pct": round(dark_pixel_ratio, 4),
        "glare_score": round(glare_score, 4),
        "max_roi_glare_pct": round(max_patch_glare, 4),
        "global_glare_pct": round(global_glare_pct, 4),
        "compression_score": round(compression_score, 4),
        "jpeg_boundary_diff": round(boundary_diff, 4),
        "blockiness_ratio": round(blockiness_ratio, 4),
    }


RANK_MEANING = {
    0: "0 - Ảnh đẹp",
    1: "1 - Vỡ nhưng không đến mức vứt",
    2: "2 - Ảnh quá tệ",
}


def load_rank_annotations(img_dir, txt_file=None):
    """
    Đọc thông tin rank (0, 1, 2) và defect_type từ các file .txt trong thư mục outimages.
    Hỗ trợ cả:
      1) File bảng tổng hợp (image_id | defect_type | severity | is_defect)
      2) File .txt riêng lẻ cho từng ảnh (vd: img_07_2.txt chứa '1' hoặc 'dark 1')
    """
    rank_map = {}
    txt_paths = []

    if txt_file and os.path.exists(txt_file):
        txt_paths.append(Path(txt_file))

    folder = Path(img_dir)
    if folder.exists():
        for p in sorted(folder.glob("*.txt")):
            if p not in txt_paths:
                txt_paths.append(p)

    valid_defects = {"none", "blur", "dark", "glare", "compression"}

    for txt_path in txt_paths:
        stem = txt_path.stem
        try:
            content = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception as e:
            print(f"Cảnh báo: Không đọc được {txt_path}: {e}")
            continue

        if not content:
            continue

        lines = [line.strip() for line in content.splitlines() if line.strip()]

        # Trường hợp 1: File .txt cùng tên với ảnh và chỉ chứa 1 số 0, 1 hoặc 2
        if len(lines) == 1 and re.match(r"^[012]$", lines[0]):
            rank_val = int(lines[0])
            rank_map[stem] = {"rank": rank_val, "gt_defect_type": "unknown"}
            continue

        # Trường hợp 2: File bảng chứa nhiều dòng (image_id defect_type severity [is_defect])
        for line in lines:
            if line.startswith("-") or line.lower().startswith("image_id"):
                continue

            tokens = re.split(r"[\s,|\t]+", line.strip())
            if not tokens:
                continue

            if len(tokens) == 2 and tokens[0].lower() in valid_defects and tokens[1] in {"0", "1", "2"}:
                rank_map[stem] = {
                    "rank": int(tokens[1]),
                    "gt_defect_type": tokens[0].lower(),
                }
                continue

            img_id = os.path.splitext(tokens[0])[0]
            gt_defect = "unknown"
            rank_val = None

            if len(tokens) >= 3 and tokens[1].lower() in valid_defects:
                gt_defect = tokens[1].lower()
                try:
                    rank_val = int(float(tokens[2]))
                except ValueError:
                    rank_val = None
            elif len(tokens) >= 2:
                try:
                    rank_val = int(float(tokens[1]))
                except ValueError:
                    for t in tokens[1:]:
                        if t in {"0", "1", "2"}:
                            rank_val = int(t)
                            break

            if rank_val in {0, 1, 2}:
                rank_map[img_id] = {
                    "rank": rank_val,
                    "gt_defect_type": gt_defect,
                }

    return rank_map


def predict_defect_and_score(feats):
    """
    Tính điểm tổng hợp (final_score) và dự đoán loại lỗi từ 4 đặc trưng:
    blur, dark, glare, compression.
    """
    candidates = {
        "blur": feats["blur_score"],
        "dark": feats["dark_score"],
        "glare": feats["glare_score"],
        "compression": feats["compression_score"],
    }
    pred_defect = max(candidates, key=candidates.get)
    final_score = candidates[pred_defect]

    thresholds = {
        "blur": 10.0,
        "dark": 15.0,
        "glare": 5.0,
        "compression": 8.0,
    }
    if final_score < thresholds[pred_defect]:
        pred_defect = "none"

    if final_score >= 25.0:
        pred_action = "RECAPTURE"
    elif pred_defect != "none":
        pred_action = "REVIEW"
    else:
        pred_action = "KEEP"

    return round(final_score, 4), pred_defect, pred_action


def main():
    parser = argparse.ArgumentParser(description="T02 - Raw QC Feature & Rank Exporter (outimages -> Excel)")
    parser.add_argument("--img_dir", type=str, default="outimages/dev_images", help="Thư mục chứa ảnh cần tính toán (mặc định: outimages)")
    parser.add_argument("--txt_file", type=str, default="outimages/dev_images", help="Đường dẫn file .txt chứa rank (nếu để ngoài outimages)")
    parser.add_argument("--out_excel", type=str, default="outimages/outimages_qc_rank.xlsx", help="Đường dẫn file Excel xuất ra")
    parser.add_argument("--out_csv", type=str, default="outimages/ranked_review.csv", help="Đường dẫn file CSV phụ trợ")
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    if not img_dir.exists():
        print(f"Lỗi: Không tìm thấy thư mục ảnh '{img_dir}'. Hãy kiểm tra lại đường dẫn --img_dir.")
        return

    # 1. Đọc dữ liệu rank (0, 1, 2) từ file .txt trong thư mục outimages
    rank_map = load_rank_annotations(img_dir, args.txt_file)
    print(f"Đã nạp thông tin rank từ file .txt cho {len(rank_map)} ảnh.")

    # 2. Quét toàn bộ file ảnh trong thư mục outimages
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts])

    if not image_files:
        print(f"Lỗi: Không tìm thấy file ảnh nào ({valid_exts}) trong '{img_dir}'.")
        return

    print(f"Đang tính toán 4 nhóm đặc trưng (Blur, Dark, Glare, Compression) cho {len(image_files)} ảnh...")

    results = []
    for img_path in image_files:
        img_id = img_path.stem
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Cảnh báo: Không thể đọc ảnh {img_path}. Bỏ qua.")
            continue

        feats = extract_qc_features(image)
        final_score, pred_defect, pred_action = predict_defect_and_score(feats)

        rank_info = rank_map.get(img_id, {})
        rank_val = rank_info.get("rank", None)
        gt_defect = rank_info.get("gt_defect_type", "unknown")
        rank_desc = RANK_MEANING.get(rank_val, "Chưa có nhãn (.txt)")

        results.append({
            "image_id": img_id,
            "file_name": img_path.name,
            "rank": rank_val,
            "rank_meaning": rank_desc,
            "txt_defect_type": gt_defect,
            "pred_defect_type": pred_defect,
            "pred_action": pred_action,
            "final_score": final_score,
            "blur_score": feats["blur_score"],
            "laplacian_var": feats["laplacian_var"],
            "dark_score": feats["dark_score"],
            "mean_brightness": feats["mean_brightness"],
            "dark_pixel_pct": feats["dark_pixel_pct"],
            "glare_score": feats["glare_score"],
            "max_roi_glare_pct": feats["max_roi_glare_pct"],
            "global_glare_pct": feats["global_glare_pct"],
            "compression_score": feats["compression_score"],
            "jpeg_boundary_diff": feats["jpeg_boundary_diff"],
            "blockiness_ratio": feats["blockiness_ratio"],
        })

    res_df = pd.DataFrame(results)
    if res_df.empty:
        print("Không có ảnh nào được xử lý thành công.")
        return

    # Sắp xếp theo Rank giảm dần (2 -> 1 -> 0) và final_score giảm dần
    res_df = res_df.sort_values(by=["rank", "final_score"], ascending=[False, False], na_position="last")

    Path(args.out_excel).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    # Lưu CSV phụ trợ
    res_df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    # Xuất ra file Excel
    try:
        with pd.ExcelWriter(args.out_excel, engine="openpyxl") as writer:
            res_df.to_excel(writer, sheet_name="All_Images_QC", index=False)

            if res_df["rank"].notna().any():
                summary_df = (
                    res_df.groupby(["rank", "rank_meaning"], dropna=False)[
                        ["blur_score", "laplacian_var", "dark_score", "mean_brightness", "glare_score", "compression_score", "final_score"]
                    ]
                    .agg(["count", "mean", "min", "max"])
                    .reset_index()
                )
                summary_df.to_excel(writer, sheet_name="Summary_By_Rank")

                defect_summary = (
                    res_df.groupby(["txt_defect_type", "rank"], dropna=False)[
                        ["blur_score", "dark_score", "glare_score", "compression_score", "final_score"]
                    ]
                    .mean()
                    .reset_index()
                )
                defect_summary.to_excel(writer, sheet_name="Summary_By_Defect", index=False)

        print(f"✅ Hoàn thành! Đã xuất file Excel tại: {args.out_excel}")
    except ImportError:
        print("⚠️ Môi trường chưa cài openpyxl. Hãy chạy: pip install openpyxl")
        print(f"Dữ liệu đã được lưu dự phòng tại: {args.out_csv}")


if __name__ == "__main__":
    main()