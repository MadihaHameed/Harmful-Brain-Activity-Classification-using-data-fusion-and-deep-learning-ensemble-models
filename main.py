from dataset import mad_get_dataloaders
from model import mad_get_model
from train import mad_train_model

if __name__ == "__main__":
    train_loader, test_loader, num_classes = mad_get_dataloaders(method="EEG")
    model = mad_get_model(num_classes)
    mad_train_model(model, train_loader, test_loader)
