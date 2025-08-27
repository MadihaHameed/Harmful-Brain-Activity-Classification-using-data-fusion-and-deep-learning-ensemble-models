from torchvision import transforms
import torch
from config import DEVICE
from PIL import Image
import cv2
import numpy as np

def mad_preprocess_image(img_path, method="EEG"):
    img = Image.open(img_path).convert("RGB")
    img = np.array(img)
    if method == "EEG":
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    elif method == "SPEC":
        img = cv2.GaussianBlur(img, (3,3), 0)
    img = Image.fromarray(img)
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    img = transform(img)
    return img.to(DEVICE)
