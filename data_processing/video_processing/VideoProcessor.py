import cv2

class VideoProcessor:
	def __init__(self):
		self.video = None		
  
	def _load_video(self, video_path: str):
		self.video = cv2.VideoCapture(video_path)
  
		if not self.video.isOpened():
			raise ValueError(f"Could not open video file: {video_path}")

	def extract_frames(self, video_path: str, number_of_frames: int) -> list:
		self._load_video(video_path)
		frame_count = 0
		frames = []
  
		# Calculate step
		step = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT) / number_of_frames)

		while True:
			ret, frame = self.video.read()
			if not ret:
				break
			if frame_count % step == 0:
				frames.append(frame)
			frame_count += 1
   
		self.video.release()
		return frames
