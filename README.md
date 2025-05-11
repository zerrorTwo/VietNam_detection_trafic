# Nhận Diện Biển Báo Giao Thông Việt Nam

Ứng dụng nhận diện biển báo giao thông Việt Nam sử dụng YOLOv8, có khả năng phát hiện và phân loại 58 loại biển báo giao thông phổ biến tại Việt Nam.

## Tính năng

- Nhận diện biển báo giao thông từ hình ảnh
- Nhận diện biển báo giao thông từ video
- Hiển thị tên đầy đủ của biển báo bằng tiếng Việt
- Điều chỉnh ngưỡng độ tin cậy
- Giao diện web thân thiện với người dùng

## Cài đặt

1. Tạo môi trường ảo (khuyến nghị):

```bash
python -m venv venv
source venv/Scripts/activate  # Windows
```

2. Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

## Sử dụng

1. Chạy ứng dụng Streamlit:

```bash
streamlit run app.py
```

2. Mở trình duyệt web và truy cập địa chỉ: http://localhost:8501

3. Sử dụng ứng dụng:
   - Tải lên hình ảnh (JPG, JPEG, PNG) hoặc video (MP4, AVI, MOV)
   - Điều chỉnh ngưỡng độ tin cậy trong thanh bên nếu cần
   - Xem kết quả với các khung bao quanh và nhãn biển báo
   - Đối với video, bạn có thể:
     - Bắt đầu xử lý video
     - Dừng xử lý bất cứ lúc nào
     - Xem tiến độ xử lý
     - Tải xuống video đã xử lý

## Lưu ý

- Sử dụng Python 3.12 hoặc các phiên bản tương thích (3.8-3.12)
- Khuyến nghị sử dụng GPU để tăng tốc độ xử lý
- Đảm bảo có đủ dung lượng ổ đĩa để lưu trữ video đã xử lý
- Nếu gặp lỗi, hãy kiểm tra:
  - Đường dẫn file
  - Cấu trúc thư mục
  - Trọng số mô hình

## Giấy phép

MIT License
