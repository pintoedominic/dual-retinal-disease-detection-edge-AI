# two_stage_resnet_comparison.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import re
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
from tqdm import tqdm

# -----------------------------
# Configuration
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS_A = 200
EPOCHS_B = 200
LEARNING_RATE = 1e-4

# Paths (same as YOLO setup)
BIN_OUT = r"C:\Dominic\Datasets Project\ROP_TWO_STAGE\STAGE_A_BIN"
SEV_OUT = r"C:\Dominic\Datasets Project\ROP_TWO_STAGE\STAGE_B_SEVERITY"

# -----------------------------
# Custom Dataset Class
# -----------------------------
class ROPDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        
        split_dir = os.path.join(root_dir, split)
        classes = sorted(os.listdir(split_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        for class_name in classes:
            class_dir = os.path.join(split_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# -----------------------------
# Data Transforms
# -----------------------------
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# -----------------------------
# ResNet Model Class
# -----------------------------
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ResNetClassifier, self).__init__()
        from torchvision.models import ResNet50_Weights
        self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        
        # Replace final layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# -----------------------------
# Training Function
# -----------------------------
def train_model(model, train_loader, val_loader, num_epochs, model_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_acc = 0.0
    best_model_path = f"best_{model_name}_resnet.pth"
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved: {val_acc:.2f}%")
    
    return best_model_path, best_val_acc

# -----------------------------
# Evaluation Function
# -----------------------------
def evaluate_model(model, test_loader, class_names):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    
    return acc, cm, report

# -----------------------------
# Two-Stage ResNet Pipeline
# -----------------------------
def train_stage_A_resnet():
    print("ðŸš€ Training Stage A ResNet (No ROP vs ROP)")
    
    train_transform, val_transform = get_transforms()
    
    # Load datasets
    train_dataset = ROPDataset(BIN_OUT, 'train', train_transform)
    val_dataset = ROPDataset(BIN_OUT, 'val', val_transform)
    test_dataset = ROPDataset(BIN_OUT, 'test', val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Model
    model = ResNetClassifier(num_classes=2).to(DEVICE)
    
    # Train
    best_path, best_acc = train_model(model, train_loader, val_loader, EPOCHS_A, "stageA")
    
    # Evaluate on test
    model.load_state_dict(torch.load(best_path))
    class_names = list(train_dataset.class_to_idx.keys())
    acc, cm, report = evaluate_model(model, test_loader, class_names)
    
    print(f"\nðŸ“Š Stage A ResNet Test Results:")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Report:\n{report}")
    
    return best_path, class_names

def train_stage_B_resnet():
    print("ðŸš€ Training Stage B ResNet (Mild/Moderate/Severe)")
    
    train_transform, val_transform = get_transforms()
    
    # Load datasets
    train_dataset = ROPDataset(SEV_OUT, 'train', train_transform)
    val_dataset = ROPDataset(SEV_OUT, 'val', val_transform)
    test_dataset = ROPDataset(SEV_OUT, 'test', val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Model
    model = ResNetClassifier(num_classes=3).to(DEVICE)
    
    # Train
    best_path, best_acc = train_model(model, train_loader, val_loader, EPOCHS_B, "stageB")
    
    # Evaluate on test
    model.load_state_dict(torch.load(best_path))
    class_names = list(train_dataset.class_to_idx.keys())
    acc, cm, report = evaluate_model(model, test_loader, class_names)
    
    print(f"\nðŸ“Š Stage B ResNet Test Results:")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Report:\n{report}")
    
    return best_path, class_names

# -----------------------------
# End-to-End Evaluation
# -----------------------------
def evaluate_two_stage_resnet():
    print("ðŸ“Š Evaluating Two-Stage ResNet End-to-End")
    
    # Load models
    modelA = ResNetClassifier(num_classes=2).to(DEVICE)
    modelA.load_state_dict(torch.load("best_stageA_resnet.pth"))
    modelA.eval()
    
    modelB = ResNetClassifier(num_classes=3).to(DEVICE)
    modelB.load_state_dict(torch.load("best_stageB_resnet.pth"))
    modelB.eval()
    
    # Get class mappings
    stage_a_dataset = ROPDataset(BIN_OUT, 'test', get_transforms()[1])
    stage_b_dataset = ROPDataset(SEV_OUT, 'test', get_transforms()[1])
    
    # Map indices to names
    idx_to_classA = {v: k for k, v in stage_a_dataset.class_to_idx.items()}
    idx_to_classB = {v: k for k, v in stage_b_dataset.class_to_idx.items()}
    
    # Four classes for end-to-end evaluation
    four_classes = ["no_ROP", "mild", "moderate", "severe"]
    cls_to_idx = {c: i for i, c in enumerate(four_classes)}
    
    y_true = []
    y_pred = []
    
    # Evaluate on Stage A test set (contains both no_ROP and ROP images)
    test_root = os.path.join(BIN_OUT, "test")
    transform = get_transforms()[1]
    
    for sub in os.listdir(test_root):
        sub_dir = os.path.join(test_root, sub)
        if not os.path.isdir(sub_dir):
            continue
            
        for fname in os.listdir(sub_dir):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            fpath = os.path.join(sub_dir, fname)
            
            # Get ground truth from filename
            gt = four_class_from_filename(fname)
            if gt is None:
                continue
            y_true.append(cls_to_idx[gt])
            
            # Load and preprocess image
            image = Image.open(fpath).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(DEVICE)
            
            # Stage A prediction
            with torch.no_grad():
                outputA = modelA(image_tensor)
                predA_idx = torch.argmax(outputA, dim=1).item()
                predA_name = idx_to_classA[predA_idx]
            
            if predA_name == "no_ROP":
                pred_final = "no_ROP"
            else:
                # Stage B prediction
                with torch.no_grad():
                    outputB = modelB(image_tensor)
                    predB_idx = torch.argmax(outputB, dim=1).item()
                    pred_final = idx_to_classB[predB_idx]
            
            y_pred.append(cls_to_idx[pred_final])
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(four_classes))))
    report = classification_report(y_true, y_pred, target_names=four_classes, digits=4)
    
    print(f"\nðŸ“Š Two-Stage ResNet End-to-End Results:")
    print(f"Test Accuracy: {acc*100:.2f}%")
    print(f"Confusion Matrix:\n{four_classes}")
    print(f"{cm}")
    print(f"Classification Report:\n{report}")
    
    return acc, cm, report

# Helper function from your YOLO code
def four_class_from_filename(fname):
    m = re.search(r"DG(\d+)", fname)
    if not m:
        return None
    dg = int(m.group(1))
    if 0 <= dg <= 3:
        return "no_ROP"
    if 4 <= dg <= 6:
        return "mild"
    if 7 <= dg <= 10:
        return "moderate"
    if 11 <= dg <= 13:
        return "severe"
    return None

# -----------------------------
# Main Execution
# -----------------------------
def main():
    print("ðŸ”¥ Starting Two-Stage ResNet Training and Evaluation")
    
    # Train both stages
    stageA_path, classesA = train_stage_A_resnet()
    stageB_path, classesB = train_stage_B_resnet()
    
    # End-to-end evaluation
    evaluate_two_stage_resnet()
    
    print("\nâœ… ResNet comparison complete!")
    print("Now you can compare YOLO11 vs ResNet performance.")

if __name__ == "__main__":
    main()
