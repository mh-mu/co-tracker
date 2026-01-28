#!/usr/bin/env python3
"""
Script to create a binary mask from the first frame of a video using user input.
The user can click points to define a polygon, and the script will create a mask.
"""

import cv2
import numpy as np
import argparse
import os


class MaskCreator:
    def __init__(self, frame):
        """
        Initialize the mask creator with a frame.
        
        Args:
            frame: The image frame to create a mask for
        """
        self.frame = frame.copy()
        self.display_frame = frame.copy()
        self.mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        self.points = []
        self.drawing = False
        self.mode = 'add'  # 'add' or 'subtract'
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for creating the mask."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add point to polygon
            self.points.append((x, y))
            # Use green for add mode, red for subtract mode
            color = (0, 255, 0) if self.mode == 'add' else (0, 0, 255)
            cv2.circle(self.display_frame, (x, y), 3, color, -1)
            
            # Draw line to previous point
            if len(self.points) > 1:
                cv2.line(self.display_frame, self.points[-2], self.points[-1], color, 2)
            
            cv2.imshow('Create Mask', self.display_frame)
            
    def create_mask(self):
        """
        Interactive mask creation using mouse input.
        
        Returns:
            mask: Binary mask (numpy array)
        """
        cv2.namedWindow('Create Mask')
        cv2.setMouseCallback('Create Mask', self.mouse_callback)
        
        print("\n=== Mask Creation Instructions ===")
        print("1. Click points to define a polygon region")
        print("2. Press 'a' for ADD mode (green) or 'd' for SUBTRACT mode (red)")
        print("3. Press 'c' to complete the current polygon")
        print("4. Press 'r' to reset and start over")
        print("5. Press 'q' to quit without saving")
        print("6. Press 's' to save the mask")
        print("==================================\n")
        print(f"Current mode: {self.mode.upper()} (green=add, red=subtract)")
        
        cv2.imshow('Create Mask', self.display_frame)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('a'):  # Add mode
                self.mode = 'add'
                print(f"Switched to ADD mode (green)")
                
            elif key == ord('d'):  # Subtract/Delete mode
                self.mode = 'subtract'
                print(f"Switched to SUBTRACT mode (red)")
                
            elif key == ord('c'):  # Complete polygon
                if len(self.points) >= 3:
                    # Create or modify mask from polygon
                    points_array = np.array(self.points, dtype=np.int32)
                    
                    if self.mode == 'add':
                        cv2.fillPoly(self.mask, [points_array], 255)
                        print("Polygon added to mask!")
                    else:  # subtract mode
                        cv2.fillPoly(self.mask, [points_array], 0)
                        print("Polygon subtracted from mask!")
                    
                    # Reset points for next polygon
                    self.points = []
                    self.display_frame = self.frame.copy()
                    
                    # Show mask overlay
                    overlay = self.frame.copy()
                    overlay[self.mask > 0] = overlay[self.mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
                    cv2.imshow('Create Mask', overlay.astype(np.uint8))
                    print(f"Press 's' to save, 'r' to reset, or continue drawing (mode: {self.mode.upper()})")
                else:
                    print("Need at least 3 points to create a polygon!")
                    
            elif key == ord('r'):  # Reset
                self.mask = np.zeros(self.frame.shape[:2], dtype=np.uint8)
                self.points = []
                self.mode = 'add'
                self.display_frame = self.frame.copy()
                cv2.imshow('Create Mask', self.display_frame)
                print("Reset! Start drawing again (mode: ADD)")
                
            elif key == ord('s'):  # Save
                if np.any(self.mask):
                    cv2.destroyAllWindows()
                    return self.mask
                else:
                    print("No mask created yet!")
                    
            elif key == ord('q'):  # Quit
                cv2.destroyAllWindows()
                return None
                
        return self.mask


def load_first_frame(video_path):
    """
    Load the first frame from a video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        frame: The first frame as a numpy array (BGR format)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not read first frame from video: {video_path}")
    
    print(f"Loaded first frame: {frame.shape[1]}x{frame.shape[0]} pixels")
    return frame


def save_mask(mask, output_path):
    """
    Save the mask to a file.
    
    Args:
        mask: Binary mask (numpy array)
        output_path: Path to save the mask
    """
    cv2.imwrite(output_path, mask)
    print(f"Mask saved to: {output_path}")
    
    # Also save as numpy array for easier loading
    npy_path = output_path.replace('.png', '.npy')
    np.save(npy_path, mask)
    print(f"Mask also saved as numpy array to: {npy_path}")


def main():
    parser = argparse.ArgumentParser(description='Create a mask from the first frame of a video')
    parser.add_argument('video_path', type=str, help='Path to the input video file')
    parser.add_argument('--output', '-o', type=str, default=None, 
                       help='Output path for the mask (default: same name as video with _mask.png)')
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output is None:
        base_name = os.path.splitext(args.video_path)[0]
        args.output = f"{base_name}_mask.png"
    
    # Load first frame
    print(f"Loading video: {args.video_path}")
    frame = load_first_frame(args.video_path)
    
    # Create mask interactively
    creator = MaskCreator(frame)
    mask = creator.create_mask()
    
    if mask is not None:
        # Save mask
        save_mask(mask, args.output)
        print(f"\nMask creation completed successfully!")
        print(f"Mask shape: {mask.shape}")
        print(f"Number of masked pixels: {np.sum(mask > 0)}")
    else:
        print("Mask creation cancelled.")


if __name__ == "__main__":
    main()
