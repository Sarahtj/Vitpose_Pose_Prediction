import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pickle
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import torchvision.utils as vutils


# load poses
with open("pose_sequences.pkl", "rb") as f:
    all_sequences = pickle.load(f)

# Visualize poses
COCO_SKELETON = [
    (5, 7), (7, 9),      # Left arm
    (6, 8), (8, 10),     # Right arm
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16),  # Right leg
    (5, 6), (11, 12),    # Shoulders & hips
    (5, 11), (6, 12)     # Torso
]

def visualize_pose(pose, ax, title="Pose", color='b'):
    pose = pose.reshape(-1, 2)
    ax.scatter(pose[:, 0], pose[:, 1], c=color, s=20)
    for (i, j) in COCO_SKELETON:
        if i < len(pose) and j < len(pose):
            ax.plot([pose[i, 0], pose[j, 0]], [pose[i, 1], pose[j, 1]], color=color, linewidth=2)
    ax.set_title(title)
    ax.invert_yaxis()  # Flip y-axis for display (origin at top-left)
    
def plot_batch(inputs, title="Batch of Training Images"):
    """
    inputs: Tensor of shape (batch_size, channels, height, width)
    """
    # Move to CPU if needed
    if inputs.is_cuda:
        inputs = inputs.cpu()

    # Make a grid: 8 images per row (you can change nrow=8)
    grid = vutils.make_grid(inputs, nrow=8, normalize=True, scale_each=True)

    plt.figure(figsize=(15, 8))
    plt.imshow(grid.permute(1, 2, 0))  # CHW -> HWC
    plt.title(title)
    plt.axis('off')
    plt.show()


# define dataset
class PoseSequenceDataset(Dataset):
    def __init__(self, sequences):
        # Normalize keypoints to [0, 1] range
        # Convert list of numpy arrays to a single numpy array and then normalize
        self.inputs = []
        self.targets = []

        for inp, tgt in sequences:
            img = inp[0]  # Input image (e.g., 3D pose or image frame)
            keypoints = tgt[0]  # Keypoints (2D or 3D)
            
            W, H = img.shape[1], img.shape[0]  # Image width and height

            # First reshape keypoints to (num_frames, num_keypoints, 2)
            keypoints = keypoints.reshape(keypoints.shape[0], -1, 2)  # (5, 17, 2)

            # Now normalize
            normalized_keypoints = keypoints / np.array([W, H])

            self.inputs.append(img)
            self.targets.append(normalized_keypoints)
        
        # Convert lists to tensors
        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# transformer def
class PosePredictor(nn.Module):
    def __init__(self, input_dim=34, hidden_dim=120, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_proj(x)
        x = x.permute(1, 0, 2)  # shape: (seq_len, batch_size, hidden_dim)
        x = self.encoder(x)
        x = x.permute(1, 0, 2)  # shape: (batch_size, seq_len, hidden_dim)
        x = self.decoder(x)
        return x

# training
def train_model():
    dataset = PoseSequenceDataset(all_sequences)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PosePredictor(hidden_dim=120, nhead=8, num_layers=6).to(device)
    # model = PosePredictor(input_dim=34, hidden_dim=120, nhead=6).to(device)
    # model = PosePredictor(hidden_dim=512, nhead=8, num_layers=6).to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # num_epochs = 200
    num_epochs = 1
    num_future_frames = 5
    # threshold = 50.0  # <-- RELAXED threshold (better!)
    threshold = 30.0  # <-- RELAXED threshold (better!)

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
            plot_batch(inputs)
            break
            optimizer.zero_grad()

            outputs = model(inputs)
            outputs = outputs[:, -5:, :]  # Only predict last 5 frames
            outputs = outputs.view(outputs.shape[0], outputs.shape[1], 17, 2)  # (32, 5, 17, 2)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Accuracy computation
            outputs_np = outputs.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()

            for batch_idx in range(inputs.shape[0]):
                for frame_idx in range(num_future_frames):
                    # No need to normalize/unnormalize
                    pred_points = outputs_np[batch_idx, frame_idx].reshape(-1, 2)
                    gt_points = targets_np[batch_idx, frame_idx].reshape(-1, 2)

                    distances = np.linalg.norm(pred_points - gt_points, axis=1)

                    correct = (distances < threshold).sum()
                    total = len(distances)

                    correct_keypoints_per_frame[frame_idx] += correct
                    total_keypoints_per_frame[frame_idx] += total

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)

        # Compute per-frame accuracies
        frame_accuracies = []
        for frame_idx in range(num_future_frames):
            frame_accuracy = (correct_keypoints_per_frame[frame_idx] / total_keypoints_per_frame[frame_idx]) * 100
            frame_accuracies.append(frame_accuracy)

        mean_frame_accuracy = sum(frame_accuracies) / num_future_frames
        mean_accuracy_history.append(mean_frame_accuracy)

        print(f"\nEpoch {epoch+1} Results:")
        for frame_idx, acc in enumerate(frame_accuracies):
            print(f"- Frame {frame_idx+1} Accuracy: {acc:.2f}%")
        print(f"- Mean Frame Accuracy: {mean_frame_accuracy:.2f}%")
        print(f"- Loss: {avg_loss:.4f}")

        # 🛡️ Save best model based on mean frame accuracy
        if mean_frame_accuracy > best_mean_accuracy:
            best_mean_accuracy = mean_frame_accuracy
            torch.save(model.state_dict(), "temporal_pose_transformer.pth")
            print(f"✅ Best model saved at Epoch {epoch+1} with Mean Frame Accuracy: {mean_frame_accuracy:.2f}%")

        # 🔥 Step the learning rate scheduler
        scheduler.step(avg_loss)

    print("\n✅ Training complete.")

    # 📈 Plot Loss and Mean Frame Accuracy
    fig, ax1 = plt.subplots(figsize=(8, 4))

    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(loss_history, 'o-', label='Training Loss', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Mean Frame Accuracy (%)', color=color)
    ax2.plot(mean_accuracy_history, 'o-', label='Mean Frame Accuracy', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Training Loss and Accuracy per Epoch")
    fig.tight_layout()
    plt.grid(True)
    plt.savefig("training_plot.png")
#     plt.show()
    
def unnormalize_keypoints(norm_points, W, H):
        points = norm_points.copy()
        points[:, 0] *= W  # unnormalize x
        points[:, 1] *= H  # unnormalize y
        return points
    
def overlay_both_poses_2(inputs, gt_points, pred_points, ax):
    # Move inputs to CPU if it's on GPU and convert to numpy
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.cpu().numpy()  # Convert to NumPy array on CPU

    # Ensure the input image is in the correct format (HWC or CHW)
    if inputs.shape[0] == 3:  # Assuming input is in CHW format (e.g., (3, H, W))
        inputs = np.transpose(inputs, (1, 2, 0))  # Convert to HWC format for visualization

    # Reshape keypoints
    gt_points = gt_points.reshape(-1, 2)
    pred_points = pred_points.reshape(-1, 2)

    # Plot the original image
    ax.imshow(inputs)

    # Plot ground truth (RED)
    ax.scatter(gt_points[:, 0], gt_points[:, 1], c='red', s=20, label="Ground Truth")
    for (i, j) in COCO_SKELETON:
        if i < len(gt_points) and j < len(gt_points):
            ax.plot([gt_points[i, 0], gt_points[j, 0]], [gt_points[i, 1], gt_points[j, 1]], color='red', linewidth=2)

    # Plot predicted (LIME)
    ax.scatter(pred_points[:, 0], pred_points[:, 1], c='lime', s=20, label="Prediction")
    for (i, j) in COCO_SKELETON:
        if i < len(pred_points) and j < len(pred_points):
            ax.plot([pred_points[i, 0], pred_points[j, 0]], [pred_points[i, 1], pred_points[j, 1]], color='lime', linewidth=2)

    # Set the title and legend
    ax.set_title("Pose Comparison")
    ax.invert_yaxis()  # Invert Y axis for image coordinates
    ax.legend()

# testing
# testing
def test_model():
    dataset = PoseSequenceDataset(all_sequences)
    batch_size = 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)  # sequential order

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PosePredictor(hidden_dim=120, nhead=8, num_layers=6).to(device)
    model.load_state_dict(torch.load("temporal_pose_transformer.pth", map_location=device))
    model.eval()

    COCO_SKELETON = [
        (5, 7), (7, 9), (6, 8), (8, 10),
        (11, 13), (13, 15), (12, 14), (14, 16),
        (5, 6), (11, 12), (5, 11), (6, 12)
    ]

    def overlay_both_poses(gt_pose, pred_pose, ax, title="Pose Comparison"):
        gt_pose = gt_pose.reshape(-1, 2)
        pred_pose = pred_pose.reshape(-1, 2)

        # Plot ground truth (RED)
        ax.scatter(gt_pose[:, 0], gt_pose[:, 1], c='red', s=20, label="Ground Truth")
        for (i, j) in COCO_SKELETON:
            if i < len(gt_pose) and j < len(gt_pose):
                ax.plot([gt_pose[i, 0], gt_pose[j, 0]], [gt_pose[i, 1], gt_pose[j, 1]], color='red', linewidth=2)

        # Plot predicted (LIME)
        ax.scatter(pred_pose[:, 0], pred_pose[:, 1], c='lime', s=20, label="Prediction")
        for (i, j) in COCO_SKELETON:
            if i < len(pred_pose) and j < len(pred_pose):
                ax.plot([pred_pose[i, 0], pred_pose[j, 0]], [pred_pose[i, 1], pred_pose[j, 1]], color='lime', linewidth=2)

        ax.set_title(title, fontsize=8)
        ax.invert_yaxis()
        ax.legend()

    threshold = 10.0  # pixels threshold
    num_future_frames = 5

    correct_keypoints_per_frame = [0] * num_future_frames
    total_keypoints_per_frame = [0] * num_future_frames
    image_idx = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)

            if inputs.ndim == 2:
                inputs = inputs.unsqueeze(0)  # 🛠️ Make sure inputs are 3D

            predictions = []
            current_input = inputs

            for future_step in range(num_future_frames):
                output = model(current_input)  # (batch_size, keypoints*2)

                if output.ndim == 2:
                    output = output.unsqueeze(1)  # 🛠️ Make sure output is 3D

                predictions.append(output)
                current_input = torch.cat((current_input[:, 1:], output), dim=1)

            outputs = torch.cat(predictions, dim=1)


            outputs = outputs.cpu().numpy()
            targets = targets.numpy()

            # Create a folder for this image
            save_path = f'./results/image_{image_idx}/'
            os.makedirs(save_path, exist_ok=True)

            # Create one figure for all 5 frames side by side
            fig_all, axs = plt.subplots(1, num_future_frames, figsize=(20, 5))
            for i in range(batch_size):
                for frame_idx in range(num_future_frames):
                    print("IMAGE NUMBER:", image_idx, "FRAME:", frame_idx)

                    pred = outputs[i, frame_idx]  # predicted frame
                    gt = targets[i, frame_idx]    # ground-truth frame

                    pred_points = pred.reshape(-1, 2)
                    gt_points = gt.reshape(-1, 2)

                    distances = np.linalg.norm(pred_points - gt_points, axis=1)
                    correct = (distances < threshold).sum()
                    total = len(distances)

                    correct_keypoints_per_frame[frame_idx] += correct
                    total_keypoints_per_frame[frame_idx] += total

                    # --- Save each frame individually ---
                    fig_single, ax_single = plt.subplots(1, 1, figsize=(5, 5))
                    overlay_both_poses(gt_points, pred_points, ax_single, title=f"Frame {frame_idx}")
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_path, f'frame_{frame_idx}.png'))
                    plt.close(fig_single)

                    # --- Add to the combined plot ---
                    overlay_both_poses(gt_points, pred_points, axs[frame_idx], title=f"Frame {frame_idx}")

            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'all_frames_combined.png'))
            plt.close(fig_all)

            image_idx += 1

    # 🔥 Now compute accuracy for each frame
    frame_accuracies = []
    print("✅ Accuracy per predicted frame:")
    for frame_idx in range(num_future_frames):
        acc = (correct_keypoints_per_frame[frame_idx] / total_keypoints_per_frame[frame_idx]) * 100
        frame_accuracies.append(acc)
        print(f"Frame {frame_idx+1}: {acc:.2f}% (threshold={threshold} pixels)")

    # 📈 Plot the frame-by-frame accuracy
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, num_future_frames+1), frame_accuracies, marker='o')
    plt.xlabel("Predicted Frame")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Prediction Accuracy per Future Frame (Threshold={threshold}px)")
    plt.grid(True)
    plt.xticks(range(1, num_future_frames+1))
    plt.ylim(0, 100)
    plt.savefig("testing_plot.png")




if __name__ == "__main__":
#     train_model()
    test_model()
