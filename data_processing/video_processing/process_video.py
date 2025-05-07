# Autor: Martin Bublavý [xbubla02]

import argparse
import cv2
import os
from VideoProcessor import VideoProcessor

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process a video file.")
    parser.add_argument("--video-path", type=str, help="Path to the video file")
    parser.add_argument("--number-of-frames", type=int, help="Number of frames to extract")
    parser.add_argument("--output-path", type=str, help="Path to save the output.")
    
    return parser.parse_args()

def extract_frames(video_path: str, number_of_frames: int, output_path: str):
	video_processor = VideoProcessor()
	frames = video_processor.extract_frames(video_path, number_of_frames)
	
	if not os.path.exists(output_path):
		os.makedirs(output_path)
	
	# Save frames
	for i, frame in enumerate(frames):
		frame_path = f"{output_path}/{i + 1}.jpg"
		cv2.imwrite(frame_path, frame)	

if __name__ == "__main__":
	args = parse_arguments()
	video_path = args.video_path
	number_of_frames = args.number_of_frames
	output_path = args.output_path

	extract_frames(video_path, number_of_frames, output_path)
 