import os

def fix_numbers_in_folder(folder_path, target_number):
    total_files = 0
    modified_files = 0

    for filename in os.listdir(folder_path):
        if not filename.endswith(".txt"):
            continue

        total_files += 1
        file_path = os.path.join(folder_path, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []
        changed = False  # đánh dấu file có bị sửa hay không

        for line in lines:
            original_line = line.rstrip("\n")
            stripped = original_line.strip()

            if not stripped:
                new_lines.append("\n")
                continue

            parts = stripped.split(maxsplit=1)

            # Lấy số đầu dòng cũ (nếu có)
            try:
                old_number = int(parts[0])
            except ValueError:
                old_number = None

            content = parts[1] if len(parts) > 1 else ""
            new_line = f"{target_number} {content}"

            if old_number != target_number:
                changed = True

            new_lines.append(new_line + "\n")

        # Chỉ ghi file nếu có thay đổi
        if changed:
            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            modified_files += 1
            print(f"✔ Đã sửa: {filename}")
        else:
            print(f"– Không cần sửa: {filename}")

    print("\n===== THỐNG KÊ =====")
    print(f"Tổng số file .txt đã kiểm tra: {total_files}")
    print(f"Số file thực sự đã sửa:        {modified_files}")
    print("====================")


# ======================
# CÁCH DÙNG
# ======================
folder_path = "D:\\NCKH_AI\\Test\\Check_class"
target_number = 1

fix_numbers_in_folder(folder_path, target_number)
