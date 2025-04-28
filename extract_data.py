# extracts pose sequences AND corresponding RGB images
# from the 3DPW dataset

import os
import pickle
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation,
)

# --------------------------- Paths ---------------------------
SEQUENCE_DIR = ''   # <-- Path to .pkl files under 3DPW/../sequenceFiles/test
IMAGE_DIR = ''      # <-- Path to test images - must match the .pkl files under 3DPW/../sequenceFiles/test
SAVE_FILE = './testing_poses.pkl'

# ---------------------- ViTPose Setup ------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

detector = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365").to(device)
detector_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")

pose_model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple").to(device)
pose_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")

# -------------------- Params ---------------------
SEQUENCE_LENGTH = 10
PREDICT_LENGTH = 5
TOTAL_LENGTH = SEQUENCE_LENGTH + PREDICT_LENGTH
MAX_SAMPLES = 20000

all_sequences = []

# ------------------ Load Sequences ----------------
for filename in tqdm(os.listdir(SEQUENCE_DIR)):
    if not filename.endswith('.pkl'):
        continue

    path = os.path.join(SEQUENCE_DIR, filename)
    print(f"\nLoading sequence file: {path}")

    with open(path, 'rb') as f:
        seq = pickle.load(f, encoding='latin1')

    seq_name = seq.get('sequence')
    frame_ids = seq.get('img_frame_ids', [])
    print(f"Sequence name: {seq_name} | Total frames: {len(frame_ids)}")

    image_paths = [
        os.path.join(IMAGE_DIR, seq_name, f"image_{i:05d}.jpg")
        for i in frame_ids
    ]

    keypoints_all = []
    valid_images = []

    for i, img_path in enumerate(image_paths):
        if not os.path.exists(img_path):
            print(f"Missing image: {img_path}")
            continue

        print(f"Processing frame {i}: {img_path}")
        image = Image.open(img_path).convert("RGB")

        # Human detection
        inputs = detector_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            det_out = detector(**inputs)
        results = detector_processor.post_process_object_detection(
            det_out, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.3
        )[0]
        boxes = results["boxes"][results["labels"] == 0]
        if boxes.shape[0] == 0:
            print("No person detected.")
            continue

        boxes = boxes.cpu().numpy()
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

        # Pose estimation
        inputs = pose_processor(image, boxes=[boxes], return_tensors="pt").to(device)
        with torch.no_grad():
            pose_out = pose_model(**inputs)

        pose_results = pose_processor.post_process_pose_estimation(pose_out, boxes=[boxes])[0]
        if len(pose_results) == 0:
            print("No keypoints returned.")
            continue

        keypoints = pose_results[0]["keypoints"]  # (17, 2)

        keypoints_all.append(np.array(keypoints).flatten())  # shape: (34,)
        valid_images.append(img_path)  # Save valid image path too

    keypoints_all = np.array(keypoints_all)  # (N, 34)

    print(f"Collected {len(keypoints_all)} valid pose frames")

    if len(keypoints_all) < TOTAL_LENGTH:
        print("Not enough frames for sequence prediction.")
        continue

    for i in range(0, len(keypoints_all) - TOTAL_LENGTH + 1, 2):
        input_keypoints = keypoints_all[i:i+SEQUENCE_LENGTH]
        target_keypoints = keypoints_all[i+SEQUENCE_LENGTH:i+TOTAL_LENGTH]

        input_images = valid_images[i:i+SEQUENCE_LENGTH]
        target_images = valid_images[i+SEQUENCE_LENGTH:i+TOTAL_LENGTH]

        all_sequences.append((
            (input_keypoints.astype(np.float32), input_images),
            (target_keypoints.astype(np.float32), target_images)
        ))

print(f"\nCollected {len(all_sequences)} (input, target) samples total.")
all_sequences = all_sequences[:MAX_SAMPLES]

with open(SAVE_FILE, 'wb') as f:
    pickle.dump(all_sequences, f)

print(f"Saved to {SAVE_FILE}")
