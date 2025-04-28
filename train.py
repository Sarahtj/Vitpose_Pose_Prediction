import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pickle
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ----- Load training data -----
with open("training_poses.pkl", "rb") as f:
    train_sequences = pickle.load(f)

# ----- Dataset -----
class PoseSequenceDataset(Dataset):
    def __init__(self, sequences):
        self.inputs = [torch.tensor(inp / 1280.0, dtype=torch.float32) for inp, _ in sequences]
        self.targets = [torch.tensor(tgt / 1280.0, dtype=torch.float32) for _, tgt in sequences]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# ----- Model -----
class PosePredictor(nn.Module):
    def __init__(self, input_dim=34, hidden_dim=120, nhead=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = x.permute(1, 0, 2)
        x = self.encoder(x)
        x = x.permute(1, 0, 2)
        x = self.decoder(x)
        return x

# ----- Training -----
def train_model():
    dataset = PoseSequenceDataset(train_sequences)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PosePredictor().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    num_epochs = 30             # change if using more training data
    num_future_frames = 5
    threshold = 50.0        

    best_mean_accuracy = 0.0
    loss_history = []
    mean_accuracy_history = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_keypoints_per_frame = [0] * num_future_frames
        total_keypoints_per_frame = [0] * num_future_frames

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs[:, -5:, :]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Accuracy calculation
            outputs_np = outputs.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()
            for batch_idx in range(inputs.shape[0]):
                for frame_idx in range(num_future_frames):
                    pred_points = outputs_np[batch_idx, frame_idx].reshape(-1, 2) * 1280.0
                    gt_points = targets_np[batch_idx, frame_idx].reshape(-1, 2) * 1280.0
                    distances = np.linalg.norm(pred_points - gt_points, axis=1)
                    correct = (distances < threshold).sum()
                    total = len(distances)
                    correct_keypoints_per_frame[frame_idx] += correct
                    total_keypoints_per_frame[frame_idx] += total

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)

        frame_accuracies = []
        for frame_idx in range(num_future_frames):
            frame_accuracy = (correct_keypoints_per_frame[frame_idx] / total_keypoints_per_frame[frame_idx]) * 100
            frame_accuracies.append(frame_accuracy)

        mean_frame_accuracy = sum(frame_accuracies) / num_future_frames
        mean_accuracy_history.append(mean_frame_accuracy)

        print(f"\nEpoch {epoch+1} Results:")
        for frame_idx, acc in enumerate(frame_accuracies):
            print(f"Frame {frame_idx+1}: {acc:.2f}%")
        print(f"Mean Frame Accuracy: {mean_frame_accuracy:.2f}%")
        print(f"Loss: {avg_loss:.4f}")

        if mean_frame_accuracy > best_mean_accuracy:
            best_mean_accuracy = mean_frame_accuracy
            torch.save(model.state_dict(), "temporal_pose_transformer.pth")
            print(f"Saved best model at Epoch {epoch+1}")

        scheduler.step(avg_loss)

    # Plot
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(loss_history, label="Loss", color="blue")
    ax1.set_ylabel("Loss", color="blue")
    ax2 = ax1.twinx()
    ax2.plot(mean_accuracy_history, label="Accuracy", color="orange")
    ax2.set_ylabel("Accuracy (%)", color="orange")
    plt.title("Training Loss & Accuracy")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train_model()