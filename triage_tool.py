import os
import cv2
import numpy as np
import pandas as pd
import argparse
from pathlib import Path


def calculate_blur_score(image):
    """
    Tính điểm mờ dựa trên Variance of Laplacian.
    Điểm số được nghịch đảo để: Mờ nhiều (Variance thấp) -> Score cao.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Thêm epsilon để tránh chia cho 0
    blur_score = 1000.0 / (laplacian_var + 1e-5)
    return blur_score, laplacian_var


def calculate_dark_score(image):
    """
    Tính điểm thiếu sáng dựa trên giá trị pixel trung bình.
    Tối (Mean thấp) -> Score cao.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    # Giả sử dưới 50 là bắt đầu quá tối
    dark_score = max(0, 50 - mean_brightness)
    return dark_score, mean_brightness


def calculate_glare_score(image):
    """
    Tính điểm lóa/cháy sáng (glare) dựa trên tỷ lệ pixel bị bão hòa.
    Nhiều pixel cháy sáng -> Score cao.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Đếm số lượng pixel có giá trị > 240
    glare_pixels = np.sum(gray > 240)
    total_pixels = image.shape[0] * image.shape[1]
    glare_score = (glare_pixels / total_pixels) * 1000  # Khuếch đại scale
    return glare_score


def main():
    parser = argparse.ArgumentParser(description="T02 - Raw QC Triage Tool")
    parser.add_argument("--img_csv", type=str, default="fixtures/t02_raw_qc/images.csv", help="Đường dẫn tới file images.csv")
    parser.add_argument("--img_dir", type=str, default="fixtures/t02_raw_qc/images", help="Thư mục chứa ảnh")
    parser.add_argument("--out_csv", type=str, default="out/ranked_review.csv", help="File output")
    args = parser.parse_args()

    # Tạo thư mục output nếu chưa có
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    print(f"Đang đọc dữ liệu từ: {args.img_csv}...")
    try:
        df = pd.read_csv(args.img_csv)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {args.img_csv}. Vui lòng kiểm tra lại đường dẫn.")
        return

    results = []

    print("Đang xử lý và trích xuất đặc trưng...")
    for index, row in df.iterrows():
        img_id = row['image_id']
        # Xử lý path (phòng trường hợp path trong csv chứa tiền tố)
        img_name = os.path.basename(row['path'])
        img_path = os.path.join(args.img_dir, img_name)

        if not os.path.exists(img_path):
            print(f"Cảnh báo: Không tìm thấy ảnh {img_path}. Bỏ qua.")
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"Cảnh báo: Không thể đọc ảnh {img_path}. Bỏ qua.")
            continue

        # 1. Tính toán các đặc trưng
        blur_score, lap_var = calculate_blur_score(image)
        dark_score, mean_bright = calculate_dark_score(image)
        glare_score = calculate_glare_score(image)

        # 2. Heuristic tổng hợp Score (Điểm cao = nghi ngờ lỗi)
        # Cách đơn giản: Lấy max của các loại lỗi đã được scale hoặc tổng hợp
        # Ở môi trường thực tế, bạn có thể Normalize z-score trên split dev.
        final_score = max(blur_score, dark_score, glare_score)

        # 3. Phân loại lỗi dự đoán (Optional contract)
        pred_defect = "none"
        if final_score == blur_score and blur_score > 10:  # Ngưỡng tự định nghĩa
            pred_defect = "blur"
        elif final_score == dark_score and dark_score > 10:
            pred_defect = "dark"
        elif final_score == glare_score and glare_score > 5:
            pred_defect = "glare"

        pred_action = "RECAPTURE" if pred_defect != "none" else "KEEP"

        results.append({
            "image_id": img_id,
            "score": final_score,
            "pred_defect_type": pred_defect,
            "pred_action": pred_action,
            "laplacian_var": lap_var,  # Giữ lại để debug/report
            "mean_brightness": mean_bright  # Giữ lại để debug/report
        })

    # Tạo DataFrame kết quả
    res_df = pd.DataFrame(results)

    if res_df.empty:
        print("Không có kết quả nào được xử lý.")
        return

    # Sắp xếp theo score giảm dần (score cao đưa lên đầu - Triage)
    res_df = res_df.sort_values(by="score", ascending=False)

    # Lọc ra đúng contract bắt buộc
    out_df = res_df[['image_id', 'score', 'pred_defect_type', 'pred_action']]

    # Lưu file
    out_df.to_csv(args.out_csv, index=False)
    print(f"Hoàn thành! Đã lưu kết quả tại: {args.out_csv}")
    print(f"Top 5 ảnh nghi ngờ lỗi cao nhất:\n{out_df.head(5)}")


if __name__ == "__main__":
    main()