from PIL import Image
import numpy as np
import imageio
import os
from typing import Union

def create_color_fade_transition(
	orig_path: str,
	ann_path: str,
	out_path: str,
	steps: int = 20,
	fps: int = 30,
	background_color: Union[tuple, int] = 255,
):
	"""
	Create a color fade transition from the original image to the annotated image.

	Parameters
	- orig_path: path to the original image
	- ann_path: path to the annotated image (will be resized to match orig if needed)
	- out_path: output path, supports .gif or .mp4 (mp4 requires ffmpeg)
	- steps: number of intermediate frames (higher = smoother)
	- fps: frames per second for video/mp4 output (gif uses duration)
	- background_color: background used to composite alpha (255 white or (r,g,b))

	Example:
		create_color_fade_transition("img/orig.jpg", "img/annot.png", "out/transition.mp4", steps=30)

	Notes:
	- Requires Pillow and imageio: `pip install pillow imageio`
	- For mp4 output imageio needs ffmpeg installed on the system.
	"""
	orig = Image.open(orig_path).convert("RGBA")
	ann = Image.open(ann_path).convert("RGBA")
	if orig.size != ann.size:
		ann = ann.resize(orig.size, Image.LANCZOS)

	orig_np = np.array(orig).astype(np.float32)
	ann_np = np.array(ann).astype(np.float32)

	frames = []
	for i in range(steps + 1):
		alpha = i / steps
		frame_np = (orig_np * (1.0 - alpha) + ann_np * alpha).clip(0, 255).astype(np.uint8)
		# If RGBA, composite alpha onto background_color and use RGB frames
		if frame_np.shape[2] == 4:
			alpha_ch = frame_np[:, :, 3:4].astype(np.float32) / 255.0
			bg = np.array(background_color, dtype=np.float32)
			if bg.ndim == 0:
				bg = np.full((1, 3), bg, dtype=np.float32)
			rgb = (frame_np[:, :, :3].astype(np.float32) * alpha_ch + bg * (1 - alpha_ch)).clip(0, 255).astype(np.uint8)
			frames.append(rgb)
		else:
			frames.append(frame_np)

	out_ext = os.path.splitext(out_path)[1].lower()
	created_path = None
	if out_ext == ".gif":
		# imageio expects frames as list of ndarrays
		imageio.mimsave(out_path, frames, duration=1.0 / fps)
		created_path = out_path
	else:
		# Try mp4 (requires ffmpeg). On failure, fallback to GIF.
		try:
			writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)
			for f in frames:
				writer.append_data(f)
			writer.close()
			created_path = out_path
		except Exception:
			# mp4 failed (likely ffmpeg missing) => fallback to gif with same base name
			fallback_gif = os.path.splitext(out_path)[0] + ".gif"
			try:
				imageio.mimsave(fallback_gif, frames, duration=1.0 / fps)
				created_path = fallback_gif
			except Exception as e:
				# If even gif fails, reraise to let caller decide
				raise e

	return created_path


if __name__ == "__main__":
	# quick local test (adjust paths as needed)
	import sys
	if len(sys.argv) >= 4:
		create_color_fade_transition(sys.argv[1], sys.argv[2], sys.argv[3], steps=30, fps=25)
	else:
		print("Usage: python webapp/utils/transition.py orig.jpg ann.png out.mp4")


def _ensure_rgba(frame: np.ndarray) -> np.ndarray:
	"""Return RGBA uint8 ndarray for provided frame (H,W,C)."""
	if frame.dtype != np.uint8:
		frame = frame.astype(np.uint8)
	if frame.ndim == 2:
		# grayscale -> RGB
		frame = np.stack([frame, frame, frame], axis=-1)
	if frame.shape[2] == 3:
		alpha = np.full((*frame.shape[:2], 1), 255, dtype=np.uint8)
		frame = np.concatenate([frame, alpha], axis=2)
	return frame


def _blend_frames_arr(frame1: np.ndarray, frame2: np.ndarray, steps: int, background_color: Union[tuple, int]):
	"""Blend two frames (ndarray) into a list of RGB frames with given intermediate steps."""
	f1 = _ensure_rgba(frame1).astype(np.float32)
	f2 = _ensure_rgba(frame2).astype(np.float32)
	frames = []
	for i in range(steps + 1):
		alpha = i / steps
		mixed = (f1 * (1.0 - alpha) + f2 * alpha).clip(0, 255).astype(np.uint8)
		# composite alpha onto background
		alpha_ch = mixed[:, :, 3:4].astype(np.float32) / 255.0
		bg = np.array(background_color, dtype=np.float32)
		if bg.ndim == 0:
			bg = np.full((1, 3), bg, dtype=np.float32)
		rgb = (mixed[:, :, :3].astype(np.float32) * alpha_ch + bg * (1 - alpha_ch)).clip(0, 255).astype(np.uint8)
		frames.append(rgb)
	return frames


def create_transition_from_videos(
	orig_video_path: str,
	ann_video_path: str,
	out_path: str,
	steps_per_frame: int = 2,
	fps: int | None = None,
	background_color: Union[tuple, int] = 255,
):
	"""
	Create a video where each pair of corresponding frames from two input videos
	produces a small fade transition (orig -> annotated).

	Parameters
	- orig_video_path, ann_video_path: input videos (should have same frame count & resolution)
	- out_path: output video path (mp4 recommended)
	- steps_per_frame: number of intermediate frames between each pair (>=1)
	- fps: output fps (if None, inferred from orig_video)
	"""
	reader_orig = imageio.get_reader(orig_video_path)
	reader_ann = imageio.get_reader(ann_video_path)
	meta = reader_orig.get_meta_data()
	if fps is None:
		fps = int(meta.get("fps", 30))
	writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)

	for frame_orig, frame_ann in zip(reader_orig, reader_ann):
		blended = _blend_frames_arr(frame_orig, frame_ann, steps_per_frame, background_color)
		for f in blended:
			writer.append_data(f)

	writer.close()
	reader_orig.close()
	reader_ann.close()


def create_transition_from_frame_dirs(
	orig_dir: str,
	ann_dir: str,
	out_path: str,
	steps_per_frame: int = 2,
	fps: int = 25,
	background_color: Union[tuple, int] = 255,
):
	"""
	Create a video from two directories of image frames (sorted by filename).
	"""
	from glob import glob
	orig_files = sorted(glob(os.path.join(orig_dir, "*")))
	ann_files = sorted(glob(os.path.join(ann_dir, "*")))
	if len(orig_files) != len(ann_files):
		raise ValueError("orig_dir and ann_dir must contain the same number of frames")

	writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)
	for p1, p2 in zip(orig_files, ann_files):
		f1 = imageio.imread(p1)
		f2 = imageio.imread(p2)
		for f in _blend_frames_arr(f1, f2, steps_per_frame, background_color):
			writer.append_data(f)
	writer.close()

