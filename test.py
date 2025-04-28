import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
import numpy as np

# ----- Load testing data -----
with open("testing_poses.pkl", "rb") as f:
    test_sequences = pickle.load(f)

# ----- Dataset -----
class PoseSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, sequences):
        self.inputs = [torch.tensor(inp / 1280.0, dtype=torch.float32) for inp, _ in sequences]
        self.targets = [torch.tensor(tgt / 1280.0, dtype=torch.float32) for _, tgt in sequences]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# ----- Model (same) -----
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

# ----- Visualization function -----
COCO_SKELETON = [
    (5, 7), (7, 9),      # Left arm
    (6, 8), (8, 10),     # Right arm
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16),  # Right leg
    (5, 6), (11, 12),    # Shoulders & hips
    (5, 11), (6, 12)     # Torso
]

def overlay_both_poses(gt_pose, pred_pose, ax, title="Pose Comparison"):
    gt_pose = gt_pose.reshape(-1, 2)
    pred_pose = pred_pose.reshape(-1, 2)

    ax.scatter(gt_pose[:, 0], gt_pose[:, 1], c='red', s=20, label="Ground Truth")
    for (i, j) in COCO_SKELETON:
        if i < len(gt_pose) and j < len(gt_pose):
            ax.plot([gt_pose[i, 0], gt_pose[j, 0]], [gt_pose[i, 1], gt_pose[j, 1]], color='red', linewidth=2)

    ax.scatter(pred_pose[:, 0], pred_pose[:, 1], c='lime', s=20, label="Prediction")
    for (i, j) in COCO_SKELETON:
        if i < len(pred_pose) and j < len(pred_pose):
            ax.plot([pred_pose[i, 0], pred_pose[j, 0]], [pred_pose[i, 1], pred_pose[j, 1]], color='lime', linewidth=2)

    ax.set_title(title)
    ax.invert_yaxis()
    ax.legend()

# ----- Testing -----
def test_model():
    dataset = PoseSequenceDataset(test_sequences)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PosePredictor().to(device)
    model.load_state_dict(torch.load("temporal_pose_transformer.pth", map_location=device))
    model.eval()

    threshold = 50.0            # Set a relaxed threshold
    num_future_frames = 5
    correct_keypoints_per_frame = [0] * num_future_frames
    total_keypoints_per_frame = [0] * num_future_frames

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.cpu().numpy()
            targets = targets.numpy()

            for i in range(inputs.shape[0]):  # batch size=1
                for frame_idx in range(num_future_frames):
                    pred = outputs[i, frame_idx]  # predicted frame
                    gt = targets[i, frame_idx]    # ground-truth frame

                    pred_points = pred.reshape(-1, 2) * 1280.0
                    gt_points = gt.reshape(-1, 2) * 1280.0

                    distances = np.linalg.norm(pred_points - gt_points, axis=1)
                    correct = (distances < threshold).sum()
                    total = len(distances)
                    correct_keypoints_per_frame[frame_idx] += correct
                    total_keypoints_per_frame[frame_idx] += total

                    # ----- Overlay visualization -----
                    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                    overlay_both_poses(gt_points, pred_points, ax)
                    plt.tight_layout()
                    plt.show(block=False)
                    plt.pause(0.5)
                    plt.close(fig)

    # ----- Accuracy printout -----
    frame_accuracies = []
    print("\nAccuracy per predicted frame:")
    for frame_idx in range(num_future_frames):
        acc = (correct_keypoints_per_frame[frame_idx] / total_keypoints_per_frame[frame_idx]) * 100
        frame_accuracies.append(acc)
        print(f"Frame {frame_idx+1}: {acc:.2f}% (threshold={threshold} pixels)")

    # ----- Accuracy chart -----
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, num_future_frames+1), frame_accuracies, marker='o')
    plt.xlabel("Predicted Frame")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Prediction Accuracy per Future Frame (Threshold={threshold}px)")
    plt.grid(True)
    plt.xticks(range(1, num_future_frames+1))
    plt.ylim(0, 100)
    plt.show()

if __name__ == "__main__":
    test_model()
