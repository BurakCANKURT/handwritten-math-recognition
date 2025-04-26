import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import MathCNN  


class Train:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
        self.train_transform = transforms.Compose([
            transforms.RandomRotation(25), 
            transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.7, 1.4)),
            transforms.ColorJitter(contrast=(0.4, 1.6)),  
            transforms.GaussianBlur(kernel_size=3),       
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.4, scale=(0.02, 0.3)),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        
        self.mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=self.train_transform)
        self.mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=self.test_transform)

        self.train_loader = DataLoader(self.mnist_train, batch_size=64, shuffle=True)
        self.test_loader = DataLoader(self.mnist_test, batch_size=64, shuffle=False)

        self.model = MathCNN(num_classes=10).to(self.device)  
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        self.criterion = nn.CrossEntropyLoss()

        self.best_accuracy = 0.0 



    def start_train(self, best_accuracy):
        print("Train starting!")

        for epoch in range(1, 31): 
            self.model.train()
            total_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch [{epoch}/30] - Training Loss: {avg_loss:.4f}")

            # Test (Validation):
            self.model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for data, target in self.test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    _, preds = torch.max(output, 1)
                    correct += (preds == target).sum().item()
                    total += target.size(0)
            acc = 100 * correct / total
            print(f"Epoch [{epoch}/30] - Test Accuracy: {acc:.2f}%")

            # Save best model:
            if acc > best_accuracy:
                best_accuracy = acc
                torch.save(self.model.state_dict(), "best_digit_model.pt")
                print(f"✅ New model saved ! Accuracy: {acc:.2f}%")

        print(f"Train completed. Best test accuracy: {best_accuracy:.2f}%")
