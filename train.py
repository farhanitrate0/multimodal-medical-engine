import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import medmnist
from medmnist import INFO
from model import get_medical_model

def train_medmnist():
    data_flag = 'pneumoniamnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    
    print("Loading MedMNIST dataset...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = DataClass(split='train', transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing model on device: {device}")
    model = get_medical_model(num_classes=1).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    
    epochs = 3
    print(f"Beginning fine-tuning for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 30 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        epoch_loss = running_loss / len(train_loader)
        print(f"--- Completed Epoch {epoch+1}/{epochs} | Average Loss: {epoch_loss:.4f} ---")
        
    torch.save(model.state_dict(), "fine_tuned_medical_resnet.pth")
    print("Fine-tuning complete! Model saved as 'fine_tuned_medical_resnet.pth'.")

if __name__ == "__main__":
    train_medmnist()