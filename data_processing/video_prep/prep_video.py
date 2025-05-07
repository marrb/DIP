# Autor: Martin Bublavý [xbubla02]

import argparse
from video_prepper import VideoPrepper

def parse_args():
    """
    Parse command line arguments for video preparation.
    """
    parser = argparse.ArgumentParser(description="Prepare video files for model.")
    parser.add_argument("video_dir", type=str, help="Directory containing video files.")
    parser.add_argument("output_dir", type=str, help="Directory to save processed video files.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    video_prepper = VideoPrepper(args.video_dir, args.output_dir)
    video_prepper.process_videos()
