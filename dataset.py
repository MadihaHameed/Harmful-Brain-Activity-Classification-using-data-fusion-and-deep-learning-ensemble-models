import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from mad_pipeline import mad_preprocess_image
from config import CSV_PATH, EEG_DIR, SPEC_DIR, BATCH_SIZE

class mad_BrainActivityDataset(Dataset):
    def __init__(self, df, eeg_dir, spec_dir, method="EEG"):
        self.df = df
        self.eeg_dir = eeg_dir
        self.spec_dir = spec_dir
        self.method = method
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row["label"])
        if self.method == "EEG":
            img_path = os.path.join(self.eeg_dir, row["eeg_image"])
        else:
            img_path = os.path.join(self.spec_dir, row["spec_image"])
        img = mad_preprocess_image(img_path, self.method)
        return img, label

def mad_get_dataloaders(method="EEG"):
    df = pd.read_csv(CSV_PATH)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    train_dataset = mad_BrainActivityDataset(train_df, EEG_DIR, SPEC_DIR, method=method)
    test_dataset = mad_BrainActivityDataset(test_df, EEG_DIR, SPEC_DIR, method=method)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader, df["label"].nunique()
