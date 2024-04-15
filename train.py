import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torchvision import datasets, transforms
import datetime
import os

from model.model import PAtt_Lite
import config
from utils.data_utils import CustomImageDataset

def load_data(data_dir, batch_size):
    # 이미 정의된 변환(transform) 사용
    train_transform = transforms.Compose([
        transforms.Resize(config.IMG_SHAPE[:2]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(config.IMG_SHAPE[:2]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # CustomImageDataset 클래스를 사용하여 데이터셋을 생성
    train_dataset = CustomImageDataset(
        annotations_file=os.path.join(data_dir, 'labels_train.csv'), 
        img_dir=os.path.join(data_dir, 'train'), 
        transform=train_transform)
    val_dataset = CustomImageDataset(
        annotations_file=os.path.join(data_dir, 'labels_val.csv'), 
        img_dir=os.path.join(data_dir, 'val'), 
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)  # 모델을 GPU로 이동
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # 데이터를 GPU로 이동
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')
        
        # Validation
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # 데이터를 GPU로 이동
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print(f'Accuracy of the network on the {len(val_loader)} validation images: {100 * correct / total}%')

def main():
    train_loader, val_loader = load_data('data', config.BATCH_SIZE)
    model = PAtt_Lite(config.NUM_CLASSES)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.TRAIN_LR)
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    
    train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, config.TRAIN_EPOCH)

    # 모델 저장
    torch.save(model.state_dict(), 'model.pth')

if __name__ == "main":
    main()