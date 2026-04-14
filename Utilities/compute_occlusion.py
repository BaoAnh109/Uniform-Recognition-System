def compute_occlusion(person_view):
    # 1. Lấy dữ liệu an toàn (mặc định = 0 nếu thiếu key để tránh lỗi)
    vis_area = person_view.get("shirt_visible_mask_area", 0)
    occ_area = person_view.get("shirt_occluded_mask_area", 0)

    # 2. Tính toán logic
    total_area = vis_area + occ_area
    
    # Xử lý trường hợp chia cho 0 (khi không detect được áo)
    if total_area > 0:
        occlusion_ratio = occ_area / total_area
    else:
        occlusion_ratio = 0.0

    # 3. Trả về kết quả
    return {
        "shirt_total_area": total_area,
        "visible_area": vis_area,
        "occluded_area": occ_area,
        "occlusion_ratio": round(occlusion_ratio, 4)  # Làm tròn 4 chữ số thập phân
    }