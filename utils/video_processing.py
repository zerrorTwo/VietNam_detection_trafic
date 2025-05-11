import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
import tempfile
import os


def process_video(
    video_path,
    model_path,
    class_names,
    output_path="output.mp4",
    conf_threshold=0.5,
    stop_flag=None
):
    """
    Xử lý video sử dụng YOLOv8
    """
    # Khởi tạo mô hình YOLOv8
    model = YOLO(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")

    # Mở video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Tạo file tạm thời để lưu video đã xử lý
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_output.close()
    
    # Khởi tạo VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()

    try:
        while cap.isOpened():
            # Kiểm tra nếu có yêu cầu dừng
            if stop_flag and stop_flag.is_set():
                print("Đã nhận lệnh dừng xử lý")
                break

            ret, frame = cap.read()
            if not ret:
                break

            # Suy luận với YOLOv8
            results = model(frame, conf=conf_threshold)[0]
            
            # Vẽ kết quả
            for box in results.boxes:
                # Lấy tọa độ
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                # Lấy confidence và class
                conf = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                # Vẽ bounding box
                label = f"{class_names[class_id]} {conf:.2f}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

            # Ghi frame đã xử lý vào file
            out.write(frame)

            frame_count += 1
            progress = (frame_count / total_frames) * 100
            print(f"Đã xử lý frame {frame_count}/{total_frames} ({progress:.1f}%)")
            
            # Trả về tiến độ xử lý
            yield progress

    finally:
        cap.release()
        out.release()
        print(f"Thời gian xử lý: {time.time() - start_time:.2f} giây")
        print(f"Video đã xử lý được lưu tại: {temp_output.name}")
        
        # Trả về đường dẫn file video đã xử lý
        return temp_output.name