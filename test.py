import torch
import imageio.v3 as iio
import numpy as np
import cv2

# Clear CUDA cache
torch.cuda.empty_cache()

# Load local video
video_path = './saved_videos/wrist2.webm'
frames = iio.imread(video_path, plugin="FFMPEG")  # plugin="pyav"

print(f"Video shape: {frames.shape}")
print(f"Number of frames: {len(frames)}")

device = 'cuda'
grid_size = 5  # Reduced from 10 to 5

# Reduce video resolution or number of frames if needed
# Option 1: Downsample frames (use every Nth frame) - more aggressive
downsample_factor = 4  # Use every 4th frame
frames = frames[::downsample_factor]
print(f"After downsampling: {frames.shape}")

# Limit total number of frames
max_frames = 60  # Process maximum 60 frames
if len(frames) > max_frames:
    frames = frames[:max_frames]
    print(f"Limiting to {max_frames} frames: {frames.shape}")

# Option 2: Resize frames to smaller resolution - more aggressive
target_height, target_width = 240, 320  # Smaller resolution
frames_resized = []
for frame in frames:
    resized = cv2.resize(frame, (target_width, target_height))
    frames_resized.append(resized)
frames = np.array(frames_resized)
print(f"After resizing: {frames.shape}")

video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)  # B T C H W
print(f"Video tensor shape: {video.shape}")
print(f"Video memory usage: {video.element_size() * video.nelement() / 1024**3:.2f} GB")

# Run Offline CoTracker:
print("Loading model...")
with torch.cuda.amp.autocast():  # Use mixed precision to save memory
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    print("Running inference...")
    pred_tracks, pred_visibility = cotracker(video, grid_size=grid_size) # B T N 2,  B T N 1

# Visualize and save the video with tracks
from cotracker.utils.visualizer import Visualizer
import os

os.makedirs("./saved_videos", exist_ok=True)
vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
vis.visualize(video, pred_tracks, pred_visibility, query_frame=0, filename="wrist2_tracked")
print("Saved video with tracks to ./saved_videos/wrist2_tracked.mp4")