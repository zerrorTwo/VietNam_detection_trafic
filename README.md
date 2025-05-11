Traffic Sign Recognition in Vietnam using YOLOv8
This project implements YOLOv8 for detecting and classifying 58 Vietnamese traffic signs, trained on the Vietnam-Traffic-Sign-Detection dataset, with a Streamlit UI for image and video inference.
Setup

Create a virtual environment (recommended):
python -m venv venv
source venv/Scripts/activate # Windows

Install dependencies:
pip install -r requirements.txt

Prepare dataset:

Extract the dataset to data/raw_dataset/. The dataset should have folders train/, valid/, test/, each containing images/ and labels/, plus a data.yaml file.
Run the data preparation script:python utils/data_preparation.py

Copy data.yaml from data/raw_dataset/ to data/data.yaml and update paths to:train: ./data/images/train
val: ./data/images/val
test: ./data/images/test

Train the model:
python train.py

Run the Streamlit app:
streamlit run app.py

Usage

Open the Streamlit app in your browser (http://localhost:8501).
Upload an image (JPG, JPEG, PNG) or video (MP4, AVI, MOV).
Adjust the confidence threshold in the sidebar if needed.
View the results with bounding boxes and labels.

Notes

The model uses YOLOv8 (nano version) with pre-trained weights (yolov8n.pt).
Ensure data.yaml matches the dataset's 58 classes.
Use Python 3.12 or compatible versions (3.8-3.12).
Use a GPU for faster training and inference.
Trained model weights are saved in runs/train/exp/weights/best.pt.
If errors occur, check file paths, dataset structure, and model weights.

License
MIT License
