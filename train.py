from ultralytics import YOLO


def train_model(data_yaml, epochs=50, img_size=640):
    # Khởi tạo mô hình YOLOv8 (phiên bản nano để huấn luyện nhanh)
    model = YOLO("yolov8n.pt")  # Sử dụng trọng số pre-trained yolov8n

    # Huấn luyện mô hình
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        project="runs/train",
        name="exp",
        exist_ok=True,
    )


if __name__ == "__main__":
    import torch

    train_model(data_yaml="./data/data.yaml", epochs=50, img_size=640)
