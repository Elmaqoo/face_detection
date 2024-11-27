# %% [markdown]
# # Training Yolo V9 on a Custom DataSet for Face Detection

# %% [markdown]
# Checking if we have access to a GPU:

# %%
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU found')

# %% [markdown]
# Defining the directory to store Model related data:

# %%
import os
HOME = os.getcwd()
print(HOME)

# %% [markdown]
# ## Clone and Install YOLOv9 forked repo
# **NOTE:** YOLOv9 is very new. At the moment, it was recommended to fork it.

# %%
!git clone https://github.com/SkalskiP/yolov9.git
os.chdir('yolov9')
!pip install -r requirements.txt

# %% [markdown]
# Installing roboflow package to download data set from roboflow universe.

# %%
!pip install roboflow

# %% [markdown]
# ## Download Face Detection Model Weights

# %%
weights_urls = [
    'https://github.com/Elmaqoo/yolov9/releases/download/v0.1/yolov9-c.pt',
    'https://github.com/Elmaqoo/yolov9/releases/download/v0.1/yolov9-e.pt',
    'https://github.com/Elmaqoo/yolov9/releases/download/v0.1/gelan-c.pt',
    'https://github.com/Elmaqoo/yolov9/releases/download/v0.1/gelan-e.pt'
]

weights_dir = os.path.join(HOME, 'weights')
os.makedirs(weights_dir, exist_ok=True)

import requests
def download_file(url, filename=''):
    local_filename = os.path.join(weights_dir, url.split('/')[-1] if not filename else filename)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

for url in weights_urls:
    download_file(url)
    print(f'Downloaded {url.split("/")[-1]}')

# %% [markdown]
# ## Authenticate and Download the Dataset
# This section is optional for downloading additional datasets. However, since you are using your own video, you might skip downloading other data.

# %%
from getpass import getpass
os.environ['KAGGLE_USERNAME'] = 'meiheichan'
os.environ['KAGGLE_KEY'] = 'fd3552d81e7f796aca9cfb55eadc4b32'

# %% [markdown]
# ## Load and Run the Detection Model
# Loading the model and performing detection on a specified video file.

# %%
import cv2
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_sync

device = select_device('0')
model = attempt_load(os.path.join(weights_dir, 'gelan-c.pt'), map_location=device)  # adjust the model as needed
model.to(device).eval()

# Video path
video_path = r'C:\Users\echan\Desktop\SIT378\Redback Orion\Player_Face_Detection\test_1.mp4'
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

# Read until video is completed
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Preprocess the image
        img = torch.from_numpy(frame).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0       # normalize 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Inference
        pred = model(img, augment=False)[0]
        
        # Apply NMS
        pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False)
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                
                # Print results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, frame, label=label, color=(255, 0, 0), line_thickness=3)

        # Display the resulting frame
        cv2.imshow('Frame', frame)
        
        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# When everything done, release the video capture object
cap.release()
cv2.destroyAllWindows()

# %%
