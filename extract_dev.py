import os
import shutil
import pandas as pd
import argparse
from pathlib import Path


def extract_dev_images(csv_path, output_dir):
    try:
        # 1. Đọc file CSV
        print(f"Đang đọc dữ liệu từ: {csv_path}")
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {csv_path}. Hãy chắc chắn bạn đang chạy lệnh từ thư mục gốc dự án.")
        return

    # 2. Lọc các dòng có cột 'split' là 'dev'
    if 'split' not in df.columns or 'path' not in df.columns:
        print("Lỗi: File CSV không có cột 'split' hoặc 'path'.")
        return

    dev_df = df[df['split'] == 'dev']
    print(f"Tìm thấy {len(dev_df)} ảnh thuộc tập 'dev'.")

    # 3. Tạo thư mục output (nếu chưa có)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Lấy thư mục chứa file csv để nối đường dẫn cho chuẩn
    # (vd: csv_path là 'fixtures/t02_raw_qc/images.csv' -> base_dir là 'fixtures/t02_raw_qc')
    base_dir = os.path.dirname(csv_path)

    # 4. Copy các ảnh sang thư mục mới
    success_count = 0
    for _, row in dev_df.iterrows():
        # Cột path chứa "images/..." nên ta nối với base_dir để ra đường dẫn thực
        src_path = os.path.join(base_dir, row['path'])

        # Lấy tên file gốc (vd: img_00_0.jpg) để đặt cho file đích
        file_name = os.path.basename(row['path'])
        dst_path = os.path.join(output_dir, file_name)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)  # copy2 giúp giữ nguyên metadata của ảnh
            success_count += 1
        else:
            print(f"Cảnh báo: Không tìm thấy ảnh gốc tại {src_path}")

    print(f"✅ Đã copy thành công {success_count} ảnh vào thư mục: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Tool lọc và copy ảnh tập dev")
    parser.add_argument("--csv", type=str, default="fixtures/t02_raw_qc/images.csv",
                        help="Đường dẫn tới file images.csv")
    parser.add_argument("--out", type=str, default="outimages/dev_images", help="Thư mục chứa ảnh dev đầu ra")
    args = parser.parse_args()

    extract_dev_images(args.csv, args.out)


if __name__ == "__main__":
    main()