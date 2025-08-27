import torch

CSV_PATH = "D:/marriam-ms-project/hms-harmful-brain-activity-classification/train.csv"
EEG_DIR = "D:/marriam-ms-project/hms-harmful-brain-activity-classification/eeg_images"
SPEC_DIR = "D:/marriam-ms-project/hms-harmful-brain-activity-classification/spectrogram_images"

BATCH_SIZE = 16
EPOCHS = 5
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
