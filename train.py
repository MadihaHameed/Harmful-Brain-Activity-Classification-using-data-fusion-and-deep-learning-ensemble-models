import torch
import torch.nn as nn
import torch.optim as optim
from config import DEVICE, EPOCHS, LR

def mad_train_model(model, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        model.train()
        train_loss, correct, total = 0,0,0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*imgs.size(0)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
        train_acc = 100.*correct/total
        train_loss /= total
        model.eval()
        test_loss, correct, total = 0,0,0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()*imgs.size(0)
                _, pred = outputs.max(1)
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()
        test_acc = 100.*correct/total
        test_loss /= total
        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
