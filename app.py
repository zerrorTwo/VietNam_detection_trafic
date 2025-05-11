import streamlit as st
import cv2
import os
import yaml
from utils.inference import process_image, process_video
import time
import threading
import queue
import base64

st.set_page_config(page_title="Traffic Sign Recognition", layout="wide")
st.title("Nhận Diện Biển Báo Giao Thông Việt Nam")

# Hàm để lấy đường dẫn tuyệt đối
def get_absolute_path(relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))

# Hàm để hiển thị video
def show_video(video_path):
    try:
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        st.video(video_bytes)
    except Exception as e:
        st.error(f"Không thể phát video: {str(e)}")

# Đọc cấu hình dataset
try:
    with open(get_absolute_path("./data/data.yaml"), "r") as f:
        data_config = yaml.safe_load(f)
    class_names = data_config["names"]
except FileNotFoundError:
    st.error("Không tìm thấy file data.yaml. Vui lòng kiểm tra thư mục data/")
    st.stop()

# Dictionary ánh xạ mã biển báo sang tên đầy đủ
class_names_full = {
    "DP.135": "Hết tất cả các lệnh cấm",
    "P.102": "Cấm đi ngược chiều",
    "P.103a": "Cấm ô tô",
    "P.103b": "Cấm ô tô rẽ phải",
    "P.103c": "Cấm ô tô rẽ trái",
    "P.104": "Cấm mô tô",
    "P.106a": "Cấm xe tải",
    "P.106b": "Cấm xe tải trên 2,5 tấn",
    "P.107a": "Cấm ô tô khách và ô tô tải",
    "P.112": "Cấm người đi bộ",
    "P.115": "Hạn chế trọng lượng xe",
    "P.117": "Hạn chế chiều cao",
    "P.123a": "Cấm rẽ trái",
    "P.123b": "Cấm rẽ phải",
    "P.124a": "Cấm quay đầu",
    "P.124b": "Cấm ô tô quay đầu",
    "P.124c": "Cấm rẽ trái và quay đầu",
    "P.125": "Cấm vượt",
    "P.127": "Tốc độ tối đa cho phép",
    "P.128": "Cấm bóp còi",
    "P.130": "Cấm dừng và đỗ xe",
    "P.131a": "Cấm đỗ xe",
    "P.137": "Cấm đi thẳng và rẽ trái",
    "P.245a": "Cấm xe đạp",
    "R.301c": "Hướng đi thẳng phải theo",
    "R.301d": "Các xe chỉ được phép rẽ phải",
    "R.301e": "Các xe chỉ được phép rẽ trái",
    "R.302a": "Chỉ hướng đi phải theo vòng chướng ngại vật",
    "R.302b": "Chỉ hướng đi trái theo vòng chướng ngại vật",
    "R.303": "Giao nhau chạy theo vòng xuyến",
    "R.407a": "Đường 1 chiều",
    "R.409": "Chỗ quay xe",
    "R.425": "Bệnh viện",
    "R.434": "Bến xe buýt",
    "S.509a": "Chỗ đường sắt cắt đường bộ",
    "W.201a": "Chỗ ngặt nguy hiểm",
    "W.201b": "Chỗ ngặt nguy hiểm",
    "W.202a": "Nhiều chỗ ngoặt nguy hiểm liên tiếp",
    "W.202b": "Nhiều chỗ ngoặt nguy hiểm liên tiếp",
    "W.203b": "Đường bị hẹp bên trái",
    "W.203c": "Đường hẹp bên trái",
    "W.205a": "Đường hẹp bên phải",
    "W.205b": "Nơi giao nhau của đường cùng cấp",
    "W.205d": "Nơi giao nhau của đường cùng cấp",
    "W.207a": "Giao nhau với đường không ưu tiên",
    "W.207b": "Giao nhau với đường không ưu tiên",
    "W.207c": "Giao nhau với đường không ưu tiên",
    "W.208": "Giao nhau với đường ưu tiên",
    "W.209": "Giao nhau có tín hiệu đèn",
    "W.210": "Giao nhau với đường sắt có rào chắn",
    "W.219": "Dốc xuống nguy hiểm",
    "W.221b": "Đường không bằng phẳng",
    "W.224": "Người đi bộ cắt ngang",
    "W.225": "Trẻ em",
    "W.227": "Công trường",
    "W.233": "Nguy hiểm khắc",
    "W.235": "Đường đôi",
    "W.245a": "Chú ý chướng ngại vật phía trước"
}

# Khởi tạo session state
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'stop_flag' not in st.session_state:
    st.session_state.stop_flag = threading.Event()
if 'frame_queue' not in st.session_state:
    st.session_state.frame_queue = queue.Queue()
if 'processed_video_path' not in st.session_state:
    st.session_state.processed_video_path = None

# Sidebar
st.sidebar.header("Cài đặt")
conf_threshold = st.sidebar.slider("Ngưỡng độ tin cậy", 0.0, 1.0, 0.5, 0.05)

# Upload file
uploaded_file = st.file_uploader(
    "Tải lên hình ảnh hoặc video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1].lower()

    # Tạo hai cột: kết quả nhận diện bên trái, danh sách tên biển báo bên phải
    col1, col2 = st.columns([3, 1])

    # Xử lý hình ảnh
    if file_extension in ["jpg", "jpeg", "png"]:
        try:
            temp_image_path = get_absolute_path("temp_image.jpg")
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_file.read())

            with st.spinner("Đang xử lý hình ảnh..."):
                result_img, detected_codes = process_image(
                    image_path=temp_image_path,
                    model_path=get_absolute_path("runs/train/exp/weights/best.pt"),
                    class_names=class_names,
                    class_names_full=class_names_full,
                    conf_threshold=conf_threshold,
                )

            # Hiển thị kết quả ở cột trái
            with col1:
                st.image(result_img, channels="BGR", caption="Kết quả nhận diện")

            os.remove(temp_image_path)
        except Exception as e:
            st.error(f"Lỗi khi xử lý hình ảnh: {str(e)}")

    # Xử lý video
    elif file_extension in ["mp4", "avi", "mov"]:
        try:
            # Tạo placeholder cho video và thanh tiến trình
            video_placeholder = st.empty()
            progress_bar = st.empty()
            progress_text = st.empty()
            status_placeholder = st.empty()
            controls_placeholder = st.empty()

            # Lưu video tạm thời
            temp_video_path = get_absolute_path("temp_video.mp4")
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_file.read())

            # Tạo nút điều khiển
            with controls_placeholder.container():
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("▶️ Bắt đầu xử lý", disabled=st.session_state.processing):
                        st.session_state.processing = True
                        st.session_state.stop_flag.clear()
                        st.session_state.processed_video_path = None
                with col2:
                    if st.button("⏹️ Dừng xử lý", disabled=not st.session_state.processing):
                        st.session_state.stop_flag.set()

            # Hiển thị thông báo đang xử lý
            if st.session_state.processing:
                status_placeholder.info("⏳ Đang xử lý video...")
                
                # Khởi tạo biến theo dõi tiến độ
                progress = 0
                progress_bar.progress(progress)
                progress_text.text(f"Tiến độ: {progress:.1f}%")

                # Xử lý video và cập nhật tiến độ
                output_path = get_absolute_path("output/output.mp4")
                for progress in process_video(
                    video_path=temp_video_path,
                    model_path=get_absolute_path("runs/train/exp/weights/best.pt"),
                    class_names=class_names,
                    class_names_full=class_names_full,
                    output_path=output_path,
                    conf_threshold=conf_threshold,
                    stop_flag=st.session_state.stop_flag
                ):
                    progress_bar.progress(progress / 100)
                    progress_text.text(f"Tiến độ: {progress:.1f}%")

                # Cập nhật trạng thái xử lý
                st.session_state.processing = False
                
                if not st.session_state.stop_flag.is_set():
                    # Hiển thị thông báo hoàn thành
                    status_placeholder.success("✅ Xử lý video hoàn tất!")
                    
                    # Tạo đường dẫn tuyệt đối và hiển thị link
                    abs_output_path = os.path.abspath(output_path)
                    st.markdown(f"### Video đã xử lý")
                    st.markdown(f"Video đã được lưu tại: `{abs_output_path}`")
                    st.markdown(f"[Click để mở video](file://{abs_output_path})")
                    
                    # Hiển thị thông tin thêm
                    st.info("💡 Lưu ý: Nếu không mở được video bằng cách click, bạn có thể copy đường dẫn và mở trực tiếp từ thư mục.")
                else:
                    status_placeholder.warning("⚠️ Đã dừng xử lý video!")

                # Xóa file tạm
                os.remove(temp_video_path)

        except Exception as e:
            st.error(f"Lỗi khi xử lý video: {str(e)}")
            st.session_state.processing = False