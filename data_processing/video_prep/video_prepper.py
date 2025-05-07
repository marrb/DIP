# Autor: Martin Bublavý [xbubla02]

import os
import cv2
import sys

class VideoPrepper():
    def __init__(self, video_dir: str, output_dir: str, output_frames: int = 24, output_frame_format: str = 'jpg'):
        """
        Initialize the VideoPrepper class.

        Args:
            video_dir (str): Directory containing video files.
            output_dir (str): Directory to save processed video files.
        """
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.output_frames = output_frames
        self.output_frame_format = output_frame_format
        self.video_files = self._get_video_files()
        
    def _get_video_files(self):
        """
        Get a list of video files in the specified directory.

        Returns:
            list: List of video file paths.
        """
        
        video_files = []
        for root, _, files in os.walk(self.video_dir):
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mov')):
                    video_files.append(os.path.join(root, file))
                    
        return video_files
    
    def process_videos(self):
        """
        Process each video file and save to the output directory.
        """
        for video_file in self.video_files:
            self._process_video(video_file)
            
    def _process_video(self, video_file: str):
        """
        Process a single video file.

        Args:
            video_file (str): Path to the video file.
        """
        print(f"Processing {video_file}...")
        video_name = os.path.splitext(os.path.basename(video_file))[0]   
        output_video_dir = os.path.join(self.output_dir, video_name)
        os.makedirs(output_video_dir, exist_ok=True)
        
        # Take first and last frames and then evenly spaced frames
        video = cv2.VideoCapture(video_file)
        
        # Check if video opened successfully
        if not video.isOpened():
            print(f"Error: Could not open video {video_file}.")
            sys.exit()
            
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = 3
            
        # Store frames
        frames = []
        
        if (frame_count <= self.output_frames):
            interval = 1

        for frame_num in range(0, frame_count, interval):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = video.read()
            
            if ret:
                frames.append(frame)
            if len(frames) >= self.output_frames:
                break

        # Realease video capture object
        video.release()
        
        # Save frames to output directory
        for i, frame in enumerate(frames):
            output_frame_path = os.path.join(output_video_dir, f"{i + 1}.{self.output_frame_format}")
            cv2.imwrite(output_frame_path, frame)
            
        # Save mp4 video from frames
        output_video_path = os.path.join(self.output_dir, f"{video_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width, _ = frames[0].shape
        video_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))
  
        for frame in frames:
            video_writer.write(frame)
   
        video_writer.release()
    