import torch

EEG_DIR = "D:/eeg_idata"
SPEC_DIR = "D:/spectrogram_data"

BATCH_SIZE = 16
EPOCHS = 5
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

