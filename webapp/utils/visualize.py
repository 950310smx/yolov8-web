import os
import cv2
import numpy as np
from matplotlib import cm
from typing import Optional

def visualize_by_size(orig_path: str, res, out_path: str, colormap: str = "plasma", alpha: float = 0.6):
	"""
	Generate a visualization where detected particles are colored according to size.

	- orig_path: path to original image (used as background)
	- res: ultralytics result object for single image (results[0])
	- out_path: path to write visualization (png/jpg)
	- colormap: matplotlib colormap name
	- alpha: blending weight for colored overlay (0..1)

	Returns out_path on success.
	"""
	# Read original image (BGR)
	orig_bgr = cv2.imread(orig_path)
	if orig_bgr is None:
		raise FileNotFoundError(f"原图不存在: {orig_path}")
	h, w = orig_bgr.shape[:2]
	orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

	# Collect size values和对应掩膜
	size_vals = []
	detections = []

	# 优先使用分割掩膜（如果模型支持），这样每个颗粒的形状会更贴近真实
	if getattr(res, "masks", None) is not None and len(res.masks) > 0:
		for i in range(len(res.masks)):
			mask_data = res.masks.data[i].cpu().numpy()
			# 调整到原图尺寸
			mask_resized = cv2.resize(mask_data, (w, h))
			mask_bool = mask_resized > 0.5
			area = np.sum(mask_bool)
			if area <= 0:
				size = 0.0
			else:
				# 用等效半径表征颗粒“尺寸”
				size = np.sqrt(area / np.pi)
			size_vals.append(size)
			detections.append(mask_bool)
	else:
		# 如果没有掩膜，则退回到检测框，生成近似圆形掩膜
		if getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
			boxes = res.boxes.xywh.cpu().numpy()  # [cx, cy, w, h]
			for b in boxes:
				cx, cy, bw, bh = b
				size = float(np.sqrt(bw * bh) / 2.0)
				radius = int(max(1, size / 2.0))
				mask_bool = np.zeros((h, w), dtype=bool)
				cv2.circle(mask_bool, (int(cx), int(cy)), radius, 1, -1)
				size_vals.append(size)
				detections.append(mask_bool)

	# If no detections, just save original copy
	if len(detections) == 0:
		cv2.imwrite(out_path, cv2.cvtColor(orig_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))
		return out_path

	# Normalize sizes to 0..1，用于映射到 colormap
	sz = np.array(size_vals, dtype=np.float32)
	smin, smax = float(sz.min()), float(sz.max())
	if smax - smin < 1e-6:
		norm = np.clip(sz - smin, 0, 1)
	else:
		norm = (sz - smin) / (smax - smin)

	cmap = cm.get_cmap(colormap)

	# 从原图拷贝一份作为绘制底图
	composite = orig_rgb.copy()

	# 对每个颗粒单独着色并与原图按 alpha 混合
	alpha = float(max(0.0, min(1.0, alpha)))
	if alpha > 0:
		for idx, mask_bool in enumerate(detections):
			if mask_bool.dtype != bool:
				mask_bool = mask_bool > 0
			if not np.any(mask_bool):
				continue

			val = float(norm[idx])
			rgba = cmap(val)
			color_rgb = np.array(rgba[:3], dtype=np.float32) * 255.0

			for c in range(3):
				orig_vals = composite[:, :, c][mask_bool]
				composite[:, :, c][mask_bool] = orig_vals * (1.0 - alpha) + color_rgb[c] * alpha

	out_bgr = cv2.cvtColor(composite.astype(np.uint8), cv2.COLOR_RGB2BGR)
	# Ensure parent dir exists
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	cv2.imwrite(out_path, out_bgr)
	return out_path


