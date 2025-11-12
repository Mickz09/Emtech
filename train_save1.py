# train_save1.py
import torch
import torch.nn as nn

# Your model class here - COPIED FROM model.py
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10): # Note: added num_classes=10 for consistency
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleCNN()
# ... You would need to add training code here ...
# For a quick fix to generate the file:
torch.save(model.state_dict(), "model_weights.pth")