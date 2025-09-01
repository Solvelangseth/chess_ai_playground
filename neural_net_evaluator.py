import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define the network (with a pool layer enabled)
class ChessCNN(nn.Module):
    def __init__(self):
        super(ChessCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=12, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Using a 2x2 pooling layer
        
        # After pooling, the 8x8 feature maps become 4x4.
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # Apply pooling so that the feature maps reduce to 4x4
        x = x.view(x.size(0), -1)  # Flatten the feature maps
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))  # tanh ensures output is between -1 and 1
        return x

# Create an instance of the network
model = ChessCNN()

# Loss function for regression (evaluating board score)
criterion = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Assume you have lists of tensors and evaluations from your dataset.
# For example:
#   board_tensors: a list of PyTorch tensors with shape (12,8,8)
#   evaluations: a list (or 1D tensor) of normalized evaluation scores
#
# For demonstration, let's assume we have them as follows:
# (In practice, load your actual preprocessed data.)
# board_tensors = [tensor1, tensor2, ...]
# evaluations = [eval1, eval2, ...]

# Here is how you might create a TensorDataset and DataLoader:
# (Make sure all board tensors are of the same shape and evaluations are floats.)
inputs = torch.stack(board_tensors)          # shape: (N, 12, 8, 8)
targets = torch.tensor(evaluations, dtype=torch.float32).view(-1, 1)  # shape: (N, 1)

dataset = TensorDataset(inputs, targets)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

num_epochs = 100

# Training Loop
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_idx, (boards, evals) in enumerate(train_loader):
        optimizer.zero_grad()               # Clear gradients
        outputs = model(boards)               # Forward pass
        loss = criterion(outputs, evals)      # Compute loss
        loss.backward()                       # Backpropagation
        optimizer.step()                      # Update weights

        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

print("Training complete!")


# Example usage:
# model = ChessCNN()
# input_tensor = torch.randn(1, 12, 8, 8)  # One chess board input
# output = model(input_tensor)
# print(output)  # Outputs a single value per board (e.g., evaluation score)