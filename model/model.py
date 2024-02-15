import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
import time

class SimpleCNN(nn.Module):
    def __init__(self, dropout_rate):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download and load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)

class Model:
    def __init__(self, learning_rate, drop_out, batch_size, num_epochs, optimizer) -> None:

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        self.learning_rate = learning_rate
        self.drop_out = drop_out
        self.num_epochs = num_epochs
        self.optimizer = optimizer

    def train(self):
        model = SimpleCNN(self.drop_out).to(device)
        criterion = nn.CrossEntropyLoss()
        losses = []
        if self.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        else:
            self.learning_rate *= 100
            optimizer = optim.SGD(model.parameters(), lr=self.learning_rate)
            

        # Training the simplified model

        start_time = time.time()

        for epoch in range(self.num_epochs):
            model.train()
            running_loss = 0.0
            for images, labels in self.train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            loss_value = running_loss / len(self.train_loader)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {loss_value}")
            losses.append(str(round(loss_value,5)))

        # Testing the simplified model
        model.eval()
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        run_time = time.time() - start_time
        acc = accuracy_score(all_labels, all_preds, normalize=True)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        return round(run_time,5), round(acc,5), round(f1,5), "|".join(losses)