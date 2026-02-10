import os
import sys
import glob
import math
import numpy as np
import pandas as pd
import time
import subprocess
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import matplotlib
import cv2
from ultralytics import YOLO
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import scipy.stats
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed  # 用于并行计算

# 新增绘图相关的导入
import seaborn as sns
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from scipy.interpolate import make_interp_spline  # 用于平滑曲线

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='seaborn')  # 忽略 seaborn 的一些警告

# =========================
# Matplotlib 中文显示配置 (优化)
# =========================
# 全局中文字体属性，优先使用实际字体文件，后续所有绘图函数都会引用它
GLOBAL_CHINESE_FONT_PROP = None

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# =========================
# DEFAULT CONFIG AND CONSTANTS (作为默认值，在Streamlit中会被覆盖)
# =========================

# 类别颜色 (科研风格调整后的 BGR)
DEFAULT_CLASS_COLORS = {
    0: (0, 200, 0),  # 类别0: 绿色 (BGR)
    1: (0, 0, 200),  # 类别1: 蓝色 (BGR)
    2: (200, 200, 0),  # 类别2: 青色 (BGR)
    3: (100, 0, 150),  # 类别3: 紫色 (BGR)
    4: (0, 100, 255),  # 类别4: 橙色 (BGR)
}

# 粒径着色规则 (科研风格调整后的 BGR 渐变)
DEFAULT_SIZE_COLOR_RULES = {
    50.0: (0, 0, 255),  # > 50um: 红色 (大颗粒)
    40.0: (0, 128, 255),  # > 40um: 橙色
    30.0: (0, 255, 255),  # > 30um: 黄色
    20.0: (100, 255, 100),  # > 20um: 浅绿色
    15.0: (255, 150, 0),  # > 15um: 偏蓝的青色
    10.0: (255, 50, 0),  # > 10um: 蓝色
    5.0: (128, 0, 0),  # > 5um: 深蓝色 (小颗粒)
}
DEFAULT_SORTED_SIZE_COLOR_RULES = sorted(DEFAULT_SIZE_COLOR_RULES.items(), key=lambda item: item[0], reverse=True)

# 自定义着色规则 (默认关闭)
DEFAULT_CUSTOM_COLORING_RULES = []

# 统一的默认配置字典 (Streamlit会使用此结构)
DEFAULT_PARAMS = {
    # YOLO & 推理参数
    "IMG_SIZE": 1280,
    "INFERENCE_IMG_SIZE": 1280,
    "CONF_THRESH": 0.25,
    "IOU_THRESH": 0.5,
    "RETINA_MASKS": True,
    "MAX_DETECTIONS": 5000,

    # 物理尺寸转换
    "UM_PER_PX": 0.4016,

    # 颗粒筛选
    "MIN_CIRCULARITY": None,
    "MAX_AXIS_RATIO": None,
    "MIN_AREA_PX": 0.1,

    # NMS 和边界处理
    "BORDER_BAND": 4,
    "NMS_IOU_THRESH": 0.3,
    "BOX_IOU_PRE_THRESH": 0.01,

    # 形态学计算配置
    "RDP_EPSILON": 2.0,
    "DEPTH_THRESHOLD": 5.0,

    # Tiling 配置
    "TILE_ROWS": 1,
    "TILE_COLS": 1,
    "OVERLAP_PX": 100,

    # 可视化配置
    "CLASS_COLORS": DEFAULT_CLASS_COLORS,
    "FILL_ALPHA": 0.55,
    "TEXT_COLOR": (255, 255, 255),  # BGR
    "SHOW_PARTICLE_ID": False,
    "SHOW_ONLY_OUTLINE": False,
    "OUTLINE_THICKNESS": 1,
    "BACKGROUND_DIM_FACTOR": 0.5,
    "COLORING_MODE": 'SIZE',
    "ENABLE_CUSTOM_COLORING": False,
    "CUSTOM_COLORING_RULES": DEFAULT_CUSTOM_COLORING_RULES,
    "SORTED_SIZE_COLOR_RULES": DEFAULT_SORTED_SIZE_COLOR_RULES,

    # 形态学计算选项（用于选择性计算）
    "CALC_SHAPE_PARAMS": True,  # 尺寸与面积、拟合椭圆、最小外接矩形、圆度与凸性
    "CALC_BOUNDARY_FEATURES": True,  # 边界局部特征
    "CALC_FOURIER_DESCRIPTORS": True,  # 傅里叶描述子
    "CALC_TEXTURE_FEATURES": True,  # 内部纹理特征
    "CALC_CONTEXT_FEATURES": True,  # 上下文与重建
    "NUM_WORKERS": 4,  # 并行计算的工作线程数

    # 轮廓线调整
    "OUTLINE_COLOR": (255, 255, 255),  # BGR 白色
    "OUTLINE_ALPHA": 1.0,  # 轮廓不透明度 (0.0 - 1.0)
}

# =========================
# 25 个形态学参数的中文表头 (保持不变，作为常量导出)
# =========================
CHINESE_HEADERS = [
    '图片名称', '颗粒ID', '类别ID', '类别名称',
    # 尺寸与面积
    '面积 (像素)', '面积 (微米^2)', '等效直径 (像素)', '等效直径 (微米)',
    '体积 (微米^3)',
    # 拟合椭圆参数
    '拟合椭圆长轴 (像素)', '拟合椭圆长轴 (微米)', '拟合椭圆短轴 (像素)', '拟合椭圆短轴 (微米)',
    '拟合椭圆轴比 (L/S)', '拟合椭圆延伸率 (S/L)', '拟合椭圆偏心率', '拟合椭圆角度 (度)',
    # 最小外接矩形参数
    '最小外接矩形长轴 (像素)', '最小外接矩形长轴 (微米)',
    '最小外接矩形短轴 (像素)', '最小外接矩形短轴 (微米)',
    '最小外接矩形轴比 (L/S)', '最小外接矩形延伸率 (S/L)', '最小外接矩形角度 (度)',
    # 圆度与凸性
    '周长 (像素, 平滑)', '周长 (微米, 平滑)', '紧凑度/圆度', '莱利圆度 (Sphericity Proxy)', '平均绝对曲率 (圆度代理)',
    # 新增
    '凸包面积 (像素)', '凸包面积 (微米^2)', '凸度', '凸包周长 (像素)', '凸包周长 (微米)', '凸性比',
    # 边界局部特征
    '边界径向标准差 (像素)', '边界径向标准差 (微米)', '边界粗糙度', '边界曲率峰值', '凹陷数量',
    '最大凹陷深度 (像素)', '最大凹陷深度 (微米)', '总相对凹陷深度 (Irregularity Proxy)',
    # 傅里叶描述子
    '傅里叶描述子FD2', '傅里叶描述子FD3', '傅里叶描述子FD4', '傅里叶描述子FD5', '傅里叶描述子FD6',
    # 内部纹理特征
    '内部纹纹理灰度均值', '内部纹理灰度标准差', '内部纹理灰度偏度', '内部纹理灰度峰度',
    'GLCM对比度', 'GLCM异质性', 'GLCM均匀性', 'GLCM能量', 'GLCM相关性', 'GLCM角二阶矩', 'GLCM熵',
    'LBP熵', 'LBP均值', 'LBP方差', '分形维数 (灰度框计数)',
    # 上下文与重建
    '遮挡率 (边界)', '凹陷面积比', '接触图像边缘',
    '边缘重建直径 (像素)', '边缘重建直径 (微米)', '凸包重建直径 (像素)', '凸包重建直径 (微米)'
]

# 用于 Streamlit UI 和绘图的列名映射
# key: 英文/内部列名, value: (中文显示名, 英文显示名)
# 内部列名必须是 morphology_analyzer 返回的 DataFrame 中的实际列名
# 这个映射在 app.py 中用于创建下拉菜单，绘图函数中使用它来获取图表标签
COLUMN_DISPLAY_MAP = {
    '图片名称': ('图片名称', 'Image Name'),
    '颗粒ID': ('颗粒ID', 'Particle ID'),
    '类别ID': ('类别ID', 'Class ID'),
    '类别名称': ('类别名称', 'Class Name'),
    '面积 (像素)': ('面积 (像素)', 'Area (pixels)'),
    # 统一使用 ^2 / ^3，避免上标字符在部分字体中缺失导致显示为方块
    '面积 (微米^2)': ('面积 (微米^2)', 'Area (µm^2)'),
    '等效直径 (像素)': ('等效直径 (像素)', 'Equivalent Diameter (pixels)'),
    '等效直径 (微米)': ('等效直径 (微米)', 'Equivalent Diameter (µm)'),
    '体积 (微米^3)': ('体积 (微米^3)', 'Volume (µm^3)'),
    '拟合椭圆长轴 (像素)': ('拟合椭圆长轴 (像素)', 'Ellipse Major Axis (pixels)'),
    '拟合椭圆长轴 (微米)': ('拟合椭圆长轴 (微米)', 'Ellipse Major Axis (µm)'),
    '拟合椭圆短轴 (像素)': ('拟合椭圆短轴 (像素)', 'Ellipse Minor Axis (pixels)'),
    '拟合椭圆短轴 (微米)': ('拟合椭圆短轴 (微米)', 'Ellipse Minor Axis (µm)'),
    '拟合椭圆轴比 (L/S)': ('拟合椭圆轴比 (L/S)', 'Ellipse Aspect Ratio (L/S)'),
    '拟合椭圆延伸率 (S/L)': ('拟合椭圆延伸率 (S/L)', 'Ellipse Elongation (S/L)'),
    '拟合椭圆偏心率': ('拟合椭圆偏心率', 'Ellipse Eccentricity'),
    '拟合椭圆角度 (度)': ('拟合椭圆角度 (度)', 'Ellipse Angle (deg)'),
    '最小外接矩形长轴 (像素)': ('最小外接矩形长轴 (像素)', 'Min Rect Major Axis (pixels)'),
    '最小外接矩形长轴 (微米)': ('最小外接矩形长轴 (微米)', 'Min Rect Major Axis (µm)'),
    '最小外接矩形短轴 (像素)': ('最小外接矩形短轴 (像素)', 'Min Rect Minor Axis (pixels)'),
    '最小外接矩形短轴 (微米)': ('最小外接矩形短轴 (微米)', 'Min Rect Minor Axis (µm)'),
    '最小外接矩形轴比 (L/S)': ('最小外接矩形轴比 (L/S)', 'Min Rect Aspect Ratio (L/S)'),
    '最小外接矩形延伸率 (S/L)': ('最小外接矩形延伸率 (S/L)', 'Min Rect Elongation (S/L)'),
    '最小外接矩形角度 (度)': ('最小外接矩形角度 (度)', 'Min Rect Angle (deg)'),
    '周长 (像素, 平滑)': ('周长 (像素, 平滑)', 'Perimeter (pixels, smooth)'),
    '周长 (微米, 平滑)': ('周长 (微米, 平滑)', 'Perimeter (µm, smooth)'),
    '紧凑度/圆度': ('紧凑度/圆度', 'Compactness/Circularity'),
    '莱利圆度 (Sphericity Proxy)': ('莱利圆度', 'Raleigh Sphericity Proxy'),
    '平均绝对曲率 (圆度代理)': ('平均绝对曲率', 'Avg. Absolute Curvature'),
    '凸包面积 (像素)': ('凸包面积 (像素)', 'Convex Hull Area (pixels)'),
    '凸包面积 (微米^2)': ('凸包面积 (微米^2)', 'Convex Hull Area (µm^2)'),
    '凸度': ('凸度', 'Convexity'),
    '凸包周长 (像素)': ('凸包周长 (像素)', 'Convex Hull Perimeter (pixels)'),
    '凸包周长 (微米)': ('凸包周长 (微米)', 'Convex Hull Perimeter (µm)'),
    '凸性比': ('凸性比', 'Convexity Ratio'),
    '边界径向标准差 (像素)': ('边界径向标准差 (像素)', 'Boundary Radial Std (pixels)'),
    '边界径向标准差 (微米)': ('边界径向标准差 (微米)', 'Boundary Radial Std (µm)'),
    '边界粗糙度': ('边界粗糙度', 'Boundary Roughness'),
    '边界曲率峰值': ('边界曲率峰值', 'Boundary Curvature Peak'),
    '凹陷数量': ('凹陷数量', 'Number of Concavities'),
    '最大凹陷深度 (像素)': ('最大凹陷深度 (像素)', 'Max Concavity Depth (pixels)'),
    '最大凹陷深度 (微米)': ('最大凹陷深度 (微米)', 'Max Concavity Depth (µm)'),
    '总相对凹陷深度 (Irregularity Proxy)': ('总相对凹陷深度', 'Total Relative Concavity Depth'),
    '傅里叶描述子FD2': ('傅里叶描述子FD2', 'Fourier Descriptor FD2'),
    '傅里叶描述子FD3': ('傅里叶描述子FD3', 'Fourier Descriptor FD3'),
    '傅里叶描述子FD4': ('傅里叶描述子FD4', 'Fourier Descriptor FD4'),
    '傅里叶描述子FD5': ('傅里叶描述子FD5', 'Fourier Descriptor FD5'),
    '傅里叶描述子FD6': ('傅里叶描述子FD6', 'Fourier Descriptor FD6'),
    '内部纹纹理灰度均值': ('内部纹理灰度均值', 'Internal Texture Mean'),
    '内部纹理灰度标准差': ('内部纹理灰度标准差', 'Internal Texture Std Dev'),
    '内部纹理灰度偏度': ('内部纹理灰度偏度', 'Internal Texture Skewness'),
    '内部纹理灰度峰度': ('内部纹理灰度峰度', 'Internal Texture Kurtosis'),
    'GLCM对比度': ('GLCM对比度', 'GLCM Contrast'),
    'GLCM异质性': ('GLCM异质性', 'GLCM Dissimilarity'),
    'GLCM均匀性': ('GLCM均匀性', 'GLCM Homogeneity'),
    'GLCM能量': ('GLCM能量', 'GLCM Energy'),
    'GLCM相关性': ('GLCM相关性', 'GLCM Correlation'),
    'GLCM角二阶矩': ('GLCM角二阶矩', 'GLCM ASM'),
    'GLCM熵': ('GLCM熵', 'GLCM Entropy'),
    'LBP熵': ('LBP熵', 'LBP Entropy'),
    'LBP均值': ('LBP均值', 'LBP Mean'),
    'LBP方差': ('LBP方差', 'LBP Variance'),
    '分形维数 (灰度框计数)': ('分形维数 (灰度框计数)', 'Fractal Dimension (Box Counting)'),
    '遮挡率 (边界)': ('遮挡率 (边界)', 'Occlusion Ratio (Boundary)'),
    '凹陷面积比': ('凹陷面积比', 'Concavity Area Ratio'),
    '接触图像边缘': ('接触图像边缘', 'Touches Image Edge'),
    '边缘重建直径 (像素)': ('边缘重建直径 (像素)', 'Edge Reconstructed Diameter (pixels)'),
    '边缘重建直径 (微米)': ('边缘重建直径 (微米)', 'Edge Reconstructed Diameter (µm)'),
    '凸包重建直径 (像素)': ('凸包重建直径 (像素)', 'Convex Hull Reconstructed Diameter (pixels)'),
    '凸包重建直径 (微米)': ('凸包重建直径 (微米)', 'Convex Hull Reconstructed Diameter (µm)')
}


# --- 绘图配置类 ---
class PLOT_CONFIG:
    FONT_PRIORITY = ['Arial', 'SimHei', 'Microsoft YaHei']  # 字体优先级
    FALLBACK_FONT = 'Arial'
    AXIS_LINEWIDTH = 1.5

    # 粒度段占比图
    SHAPE_HIST_DEFAULT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                                 '#bcbd22', '#17becf']

    # 粒径分布曲线
    PSD_LINE_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']  # 默认对比颜色
    PSD_LINE_STYLES = ['-', '--', ':', '-.', (0, (3, 5, 1, 5))]  # 默认对比线型
    PSD_SMOOTH_POINTS = 300  # 平滑插值点数

    # KDE 热力图
    KDE_ALPHA_FILL = 0.45
    KDE_LINE_WIDTH = 1.2
    KDE_LINE_LEVELS = 4
    KDE_SCATTER_ALPHA = 0.08
    KDE_SCATTER_SIZE = 8
    KDE_DEFAULT_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']  # 默认颜色


# --- Matplotlib 全局设置 ---
def setup_matplotlib_config():
    """
    尝试使用 Windows 系统内置的中文字体文件（绝对路径），找不到时再退回通用 sans-serif。
    这样可以在 Python/Matplotlib 的字体缓存不认识中文字体名时，依然强制加载中文字体。
    """
    # 常见中文 Windows 字体路径候选
    font_file_candidates = [
        r"C:\Windows\Fonts\msyh.ttc",   # Microsoft YaHei
        r"C:\Windows\Fonts\simhei.ttf", # SimHei
        r"C:\Windows\Fonts\msyh.ttf",
    ]

    global GLOBAL_CHINESE_FONT_PROP

    chosen_font_name = None
    for fpath in font_file_candidates:
        if os.path.exists(fpath):
            try:
                fm.fontManager.addfont(fpath)
                prop = FontProperties(fname=fpath)
                chosen_font_name = prop.get_name()
                GLOBAL_CHINESE_FONT_PROP = prop  # 记录下来，后续绘图统一使用
                break
            except Exception:
                continue

    if chosen_font_name:
        # 同时设置全局字体族和无衬线字体，强制所有文本都走这套中文字体
        plt.rcParams['font.family'] = chosen_font_name
        plt.rcParams['font.sans-serif'] = [chosen_font_name]
        print(f"✅ Matplotlib 使用中文字体文件: {chosen_font_name}")
    else:
        # 找不到中文字体文件时，退回原来的候选字体名列表
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
        print("⚠️ 未找到明确的中文字体文件，退回按字体名匹配。")

    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

    # 全局样式与线宽
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 300,
        'axes.linewidth': PLOT_CONFIG.AXIS_LINEWIDTH,
        'xtick.major.width': PLOT_CONFIG.AXIS_LINEWIDTH,
        'ytick.major.width': PLOT_CONFIG.AXIS_LINEWIDTH,
    })

    sns.set_style("whitegrid", {'axes.edgecolor': '.2', 'axes.facecolor': 'white'})


setup_matplotlib_config()  # 在模块加载时执行一次字体设置


# --- 列名映射和获取显示名称的辅助函数 ---
def _get_plot_column_map():
    """返回一个包含 Streamlit UI 下拉框选项的列表，以及一个用于内部映射的字典。"""
    # 示例: {'等效直径 (微米)': '等效直径 (微米)', 'Diameter (µm)': '等效直径 (微米)'}
    # 实际在 UI 中显示的是 '中文显示名 (英文显示名)'
    # 内部绘图函数将通过选择的 '中文显示名 (英文显示名)' 反向查找原始列名
    options = []
    internal_to_display = {}  # key: 内部英文列名, value: (中文显示名, 英文显示名)
    display_to_internal = {}  # key: 中文/英文显示名 in UI, value: 内部列名

    # 假设 CHINESE_HEADERS 是所有可能出现的列名（内部列名），COLUMN_DISPLAY_MAP 是它们的显示映射
    for internal_col_name in CHINESE_HEADERS:
        if internal_col_name in COLUMN_DISPLAY_MAP:
            cn_name, en_name = COLUMN_DISPLAY_MAP[internal_col_name]
            ui_display_name = f"{cn_name} ({en_name})"
            options.append(ui_display_name)
            internal_to_display[internal_col_name] = (cn_name, en_name)
            display_to_internal[ui_display_name] = internal_col_name
        else:  # 如果没有在 COLUMN_DISPLAY_MAP 中找到，就直接用中文头
            ui_display_name = internal_col_name
            options.append(ui_display_name)
            internal_to_display[internal_col_name] = (internal_col_name, internal_col_name)  # 用内部名作为显示名
            display_to_internal[ui_display_name] = internal_col_name

    return options, internal_to_display, display_to_internal


# --- 辅助函数 (已修改以接受 params 字典) ---
def to_um(val_px, um_per_px):
    if um_per_px is None or val_px is None or (isinstance(val_px, float) and math.isnan(val_px)):
        return np.nan
    return float(val_px) * float(um_per_px)


def to_um_squared(val_px_squared, um_per_px):
    if um_per_px is None or val_px_squared is None or (
            isinstance(val_px_squared, float) and math.isnan(val_px_squared)):
        return np.nan
    return float(val_px_squared) * (float(um_per_px) ** 2)


def to_um_cubed(val_px_cubed, um_per_px):
    if um_per_px is None or val_px_cubed is None or (
            isinstance(val_px_cubed, float) and math.isnan(val_px_cubed)):
        return np.nan
    return float(val_px_cubed) * (float(um_per_px) ** 3)


def smooth_mask_simple(mask):
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return m


def contour_from_mask(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


def touches_border(mask, border_band):
    h, w = mask.shape
    b = np.zeros_like(mask, np.uint8)
    b[:border_band, :], b[-border_band:, :], b[:, :border_band], b[:, -border_band:] = 1, 1, 1, 1
    return bool((mask & b).any())


# ---------- Bounding Box IoU (不变) ----------
def box_iou(boxA, boxB):
    """ 计算两个 Bounding Box 的 IoU。 box format: [x1, y1, x2, y2] """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


# ---------- Mask NMS 辅助函数 (不变) ----------
def mask_iou_optimized(mask1, mask2):
    """ 计算两个掩码的IoU，仅在它们的联合边界框区域内进行操作。 """
    if cv2.countNonZero(mask1) == 0 or cv2.countNonZero(mask2) == 0:
        return 0.0

    x1, y1, w1, h1 = cv2.boundingRect(mask1)
    x2, y2, w2, h2 = cv2.boundingRect(mask2)

    union_x1 = min(x1, x2)
    union_y1 = min(y1, y2)
    union_x2 = max(x1 + w1, x2 + w2)
    union_y2 = max(y1 + h1, y2 + h2)

    cropped_mask1 = mask1[union_y1:union_y2, union_x1:union_x2]
    cropped_mask2 = mask2[union_y1:union_y2, union_x1:union_x2]

    intersection = cv2.countNonZero(cv2.bitwise_and(cropped_mask1, cropped_mask2))
    union = cv2.countNonZero(cv2.bitwise_or(cropped_mask1, cropped_mask2))

    if union == 0:
        return 0.0
    return intersection / union


def mask_nms(masks, classes, confidences, boxes, params):
    """ 对掩码进行非极大值抑制 (NMS)，通过 Box IoU 进行预过滤加速。
        【参数化】NMS_IOU_THRESH 和 BOX_IOU_PRE_THRESH 通过 params 传入。
    """
    iou_threshold = params["NMS_IOU_THRESH"]
    box_iou_pre_threshold = params["BOX_IOU_PRE_THRESH"]

    num_masks = len(masks)
    if num_masks == 0:
        return masks, classes, confidences, boxes

    sorted_indices = np.argsort(confidences)[::-1]
    suppressed = np.zeros(num_masks, dtype=bool)
    final_indices = []

    for k in range(num_masks):
        i = sorted_indices[k]
        if suppressed[i]:
            continue

        final_indices.append(i)

        current_mask = masks[i]
        current_box = boxes[i]

        for l in range(k + 1, num_masks):
            j = sorted_indices[l]
            if suppressed[j]:
                continue

            # --- 第一阶段：Box IoU 预过滤 ---
            box_iou_val = box_iou(current_box, boxes[j])

            if box_iou_val < box_iou_pre_threshold:
                continue

                # --- 第二阶段：Mask IoU 精确计算 ---
            iou = mask_iou_optimized(current_mask, masks[j])

            if iou > iou_threshold:
                suppressed[j] = True

    final_masks = masks[final_indices]
    final_classes = classes[final_indices]
    final_confidences = confidences[final_indices]
    final_boxes = boxes[final_indices]

    return final_masks, final_classes, final_confidences, final_boxes


def calculate_box_counting_fd(image, mask):
    # 保持原逻辑不变，但移除可能依赖全局变量的默认值
    if np.sum(mask) == 0:
        return np.nan

    x, y, w, h = cv2.boundingRect(mask)
    if w == 0 or h == 0:
        return np.nan

    cropped_image = image[y:y + h, x:x + w]
    cropped_mask = mask[y:y + h, x:x + w]
    masked_pixels = cropped_image[cropped_mask > 0]

    if len(masked_pixels) < 2:
        return np.nan

    min_intensity = np.min(masked_pixels)
    max_intensity = np.max(masked_pixels)

    if max_intensity == min_intensity:
        return np.nan

    normalized_image = ((cropped_image - min_intensity) / (max_intensity - min_intensity) * 255).astype(np.uint8)

    max_s = min(w, h) // 2
    scales = [s for s in [2, 4, 8, 16, 32, 64, 128] if s <= max_s and s >= 2]
    if len(scales) < 2:
        return np.nan

    N_s = []
    log_s = []

    for s in scales:
        num_boxes_for_this_scale = 0
        for i in range(0, h, s):
            for j in range(0, w, s):
                block_mask = cropped_mask[i:i + s, j:j + s]
                if np.sum(block_mask) > 0:
                    block_pixels = normalized_image[i:i + s, j:j + s][block_mask > 0]
                    if len(block_pixels) > 0:
                        min_g = float(np.min(block_pixels))
                        max_g = float(np.max(block_pixels))
                        # 防止极端情况下 (max_g - min_g + 1) 非法或溢出
                        delta_g = max_g - min_g + 1.0
                        if delta_g <= 0 or not np.isfinite(delta_g):
                            continue
                        num_intensity_boxes = int(np.ceil(delta_g / float(s)))
                        num_boxes_for_this_scale += num_intensity_boxes

        if num_boxes_for_this_scale > 0:
            N_s.append(num_boxes_for_this_scale)
            log_s.append(np.log(1.0 / s))

    if len(N_s) < 2:
        return np.nan

    log_N_s = np.log(N_s)

    try:
        coefficients = np.polyfit(log_s, log_N_s, 1)
        fractal_dimension = coefficients[0]
        return fractal_dimension
    except np.linalg.LinAlgError:
        return np.nan
    except Exception:
        return np.nan


# ----------------------------------------------------------------------
# 核心函数：计算单个颗粒的形态学参数 (已修改以接受 params 和选择性计算)
# ----------------------------------------------------------------------
def calculate_all_morphology(mask, original_gray_image=None, params=DEFAULT_PARAMS):
    """
    计算单个颗粒（Mask）的形态学参数。
    根据 params 中的 CALC_XXX_PARAMS 决定是否计算某类参数。
    """
    H, W = mask.shape
    results = {}

    RDP_EPSILON = params["RDP_EPSILON"]
    DEPTH_THRESHOLD = params["DEPTH_THRESHOLD"]
    BORDER_BAND = params["BORDER_BAND"]
    UM_PER_PX = params["UM_PER_PX"]

    # --- 1. 查找轮廓 ---
    contour = contour_from_mask(mask)
    if contour is None:
        return None

    # --- 2. 面积 (Area) ---
    area = cv2.contourArea(contour)
    results['面积 (像素)'] = area

    if area == 0:
        return None

    # --- 3. 等效直径 (Deq) ---
    Deq = math.sqrt(4 * area / math.pi)
    results['等效直径 (像素)'] = Deq

    # --- 4. 尺寸与面积相关参数 ---
    if params["CALC_SHAPE_PARAMS"]:
        results['体积 (微米^3)'] = (1 / 6) * math.pi * (Deq ** 3)  # 这里先算像素立方，后面统一转微米

        # B. 全局形状与拟合参数 - 拟合椭圆
        if len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                ((center_x, center_y), (short_axis, long_axis), angle) = ellipse

                L_ell = max(long_axis, short_axis)
                S_ell = min(long_axis, short_axis)

                results['拟合椭圆长轴 (像素)'] = L_ell
                results['拟合椭圆短轴 (像素)'] = S_ell
                results['拟合椭圆角度 (度)'] = angle

                if S_ell > 0:
                    results['拟合椭圆轴比 (L/S)'] = L_ell / S_ell
                    results['拟合椭圆延伸率 (S/L)'] = S_ell / L_ell
                    results['拟合椭圆偏心率'] = math.sqrt(1 - (S_ell / L_ell) ** 2)
                else:
                    results['拟合椭圆轴比 (L/S)'] = np.nan
                    results['拟合椭圆延伸率 (S/L)'] = np.nan
                    results['拟合椭圆偏心率'] = 1.0
            except:
                results['拟合椭圆长轴 (像素)'] = results['拟合椭圆短轴 (像素)'] = np.nan
                results['拟合椭圆轴比 (L/S)'] = results['拟合椭圆延伸率 (S/L)'] = np.nan
                results['拟合椭圆偏心率'] = results['拟合椭圆角度 (度)'] = np.nan
        else:
            results['拟合椭圆长轴 (像素)'] = results['拟合椭圆短轴 (像素)'] = np.nan
            results['拟合椭圆轴比 (L/S)'] = results['拟合椭圆延伸率 (S/L)'] = np.nan
            results['拟合椭圆偏心率'] = results['拟合椭圆角度 (度)'] = np.nan

        # B. 全局形状与拟合参数 - 最小外接矩形
        if len(contour) >= 2:
            try:
                rect = cv2.minAreaRect(contour)
                (center, (w_rect, h_rect), angle) = rect  # 避免与mask的w,h混淆

                L_rect = max(w_rect, h_rect)
                S_rect = min(w_rect, h_rect)

                results['最小外接矩形长轴 (像素)'] = L_rect
                results['最小外接矩形短轴 (像素)'] = S_rect
                results['最小外接矩形角度 (度)'] = angle

                if S_rect > 0:
                    results['最小外接矩形轴比 (L/S)'] = L_rect / S_rect
                    results['最小外接矩形延伸率 (S/L)'] = S_rect / L_rect
                else:
                    results['最小外接矩形轴比 (L/S)'] = np.nan
                    results['最小外接矩形延伸率 (S/L)'] = np.nan
            except:
                results['最小外接矩形长轴 (像素)'] = results['最小外接矩形短轴 (像素)'] = np.nan
                results['最小外接矩形轴比 (L/S)'] = results['最小外接矩形延伸率 (S/L)'] = np.nan
                results['最小外接矩形角度 (度)'] = np.nan
        else:
            results['最小外接矩形长轴 (像素)'] = results['最小外接矩形短轴 (像素)'] = np.nan
            results['最小外接矩形轴比 (L/S)'] = results['最小外接矩形延伸率 (S/L)'] = np.nan
            results['最小外接矩形角度 (度)'] = np.nan

        # C. 圆度、凸性与紧凑度
        perimeter_raw = cv2.arcLength(contour, True)
        epsilon = RDP_EPSILON * perimeter_raw / 1000.0
        smooth_contour = cv2.approxPolyDP(contour, epsilon, True)

        perimeter = cv2.arcLength(smooth_contour, True)
        results['周长 (像素, 平滑)'] = perimeter

        if perimeter > 0:
            results['紧凑度/圆度'] = 4 * math.pi * area / (perimeter ** 2)
        else:
            results['紧凑度/圆度'] = 0

        (x_c, y_c), radius_c = cv2.minEnclosingCircle(contour)
        Dc = 2 * radius_c
        dist_transform = distance_transform_edt(mask)
        Ri = np.max(dist_transform)
        Di = 2 * Ri

        if Dc > 0 and Ri > 0:
            results['莱利圆度 (Sphericity Proxy)'] = math.sqrt(Di / Dc)
        else:
            results['莱利圆度 (Sphericity Proxy)'] = 0.0

        hull = cv2.convexHull(contour)
        area_hull = cv2.contourArea(hull)
        results['凸包面积 (像素)'] = area_hull

        if area_hull > 0:
            results['凸度'] = area / area_hull
        else:
            results['凸度'] = 0

        perimeter_hull = cv2.arcLength(hull, True)
        results['凸包周长 (像素)'] = perimeter_hull

        if perimeter > 0:
            results['凸性比'] = perimeter_hull / perimeter
        else:
            results['凸性比'] = 0

    else:  # 如果不计算形状参数，所有相关参数设为 NaN
        for key in ['体积 (微米^3)', '拟合椭圆长轴 (像素)', '拟合椭圆短轴 (像素)', '拟合椭圆角度 (度)',
                    '拟合椭圆轴比 (L/S)', '拟合椭圆延伸率 (S/L)', '拟合椭圆偏心率',
                    '最小外接矩形长轴 (像素)', '最小外接矩形短轴 (像素)', '最小外接矩形角度 (度)',
                    '最小外接矩形轴比 (L/S)', '最小外接矩形延伸率 (S/L)', '周长 (像素, 平滑)', '紧凑度/圆度',
                    '莱利圆度 (Sphericity Proxy)', '凸包面积 (像素)', '凸度', '凸包周长 (像素)', '凸性比']:
            results[key] = np.nan

    # --- 5. 边界局部特征 ---
    if params["CALC_BOUNDARY_FEATURES"]:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        radial_distances = [math.sqrt((p[0][0] - cx) ** 2 + (p[0][1] - cy) ** 2) for p in contour]
        results['边界径向标准差 (像素)'] = np.std(radial_distances)

        if Deq > 0:
            results['边界粗糙度'] = perimeter_raw / (math.pi * Deq) if 'perimeter_raw' in locals() else np.nan
        else:
            results['边界粗糙度'] = np.nan

        max_angle_change = 0
        all_angle_changes = []

        # 确保smooth_contour存在且有足够点
        if params["CALC_SHAPE_PARAMS"] and 'smooth_contour' in locals() and len(smooth_contour) >= 3:
            for i in range(len(smooth_contour)):
                p_prev = smooth_contour[i - 1][0]
                p_curr = smooth_contour[i][0]
                p_next = smooth_contour[(i + 1) % len(smooth_contour)][0]

                vec1 = p_curr - p_prev
                vec2 = p_next - p_curr

                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)

                if norm1 > 0 and norm2 > 0:
                    cosine_angle = np.dot(vec1, vec2) / (norm1 * norm2)
                    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
                    angle = math.acos(cosine_angle)
                    angle_change = math.degrees(math.pi - angle)

                    all_angle_changes.append(angle_change)
                    if angle_change > max_angle_change:
                        max_angle_change = angle_change

        results['边界曲率峰值'] = max_angle_change

        if all_angle_changes:
            results['平均绝对曲率 (圆度代理)'] = np.mean(np.abs(all_angle_changes))  # 修正，使用原始脚本的名称
        else:
            results['平均绝对曲率 (圆度代理)'] = np.nan

        epsilon_for_defects = max(1.0, 0.01 * perimeter_raw) if 'perimeter_raw' in locals() else 1.0
        # 确保 simplified_contour 能够从 contour 构建
        simplified_contour = cv2.approxPolyDP(contour, epsilon_for_defects, True)

        hull_indices = cv2.convexHull(simplified_contour, returnPoints=False)

        max_depth = 0
        concavity_count = 0
        total_relative_defect_depth = 0.0

        if hull_indices is not None and len(hull_indices) > 3 and len(simplified_contour) > 3:
            try:
                defects = cv2.convexityDefects(simplified_contour, hull_indices)

                if defects is not None:
                    for i in range(defects.shape[0]):
                        d = defects[i, 0][3]
                        depth = d / 256.0

                        if depth > max_depth:
                            max_depth = depth

                        if depth >= DEPTH_THRESHOLD:
                            concavity_count += 1
                            if Deq > 0:
                                total_relative_defect_depth += (depth / Deq)

            except cv2.error:
                concavity_count = 0
                max_depth = 0
                total_relative_defect_depth = 0.0
            except Exception:
                concavity_count = 0
                max_depth = 0
                total_relative_defect_depth = 0.0
        else:
            concavity_count = 0
            max_depth = 0
            total_relative_defect_depth = 0.0

        results['凹陷数量'] = concavity_count
        results['最大凹陷深度 (像素)'] = max_depth
        results['总相对凹陷深度 (Irregularity Proxy)'] = total_relative_defect_depth
    else:  # 如果不计算边界特征，所有相关参数设为 NaN
        for key in ['边界径向标准差 (像素)', '边界粗糙度', '边界曲率峰值', '平均绝对曲率 (圆度代理)',
                    '凹陷数量', '最大凹陷深度 (像素)', '总相对凹陷深度 (Irregularity Proxy)']:
            results[key] = np.nan

    # --- 6. 傅里叶描述子 ---
    if params["CALC_FOURIER_DESCRIPTORS"]:
        if len(contour) >= 10:
            complex_contour = contour.squeeze().astype(np.float32)[:, 0] + 1j * contour.squeeze().astype(np.float32)[:,
                                                                                1]
            fft_coeffs = np.fft.fft(complex_contour)

            if len(fft_coeffs) > 1 and abs(fft_coeffs[1]) > 1e-6:
                num_descriptors = 5
                for k in range(2, 2 + num_descriptors):
                    if k < len(fft_coeffs):
                        results[f'傅里叶描述子FD{k}'] = abs(fft_coeffs[k]) / abs(fft_coeffs[1])
                    else:
                        results[f'傅里叶描述子FD{k}'] = np.nan
            else:
                for k in range(2, 2 + 5):
                    results[f'傅里叶描述子FD{k}'] = np.nan
        else:
            for k in range(2, 2 + 5):
                results[f'傅里叶描述子FD{k}'] = np.nan
    else:  # 如果不计算傅里叶描述子，所有相关参数设为 NaN
        for k in range(2, 2 + 5):
            results[f'傅里叶描述子FD{k}'] = np.nan

    # --- 7. 内部纹理特征 ---
    if params["CALC_TEXTURE_FEATURES"]:
        if original_gray_image is not None and np.sum(mask) > 0:
            masked_pixels = original_gray_image[mask > 0]
            if len(masked_pixels) > 1:
                results['内部纹纹理灰度均值'] = np.mean(masked_pixels)
                results['内部纹理灰度标准差'] = np.std(masked_pixels)
                if len(masked_pixels) > 2:
                    results['内部纹理灰度偏度'] = scipy.stats.skew(masked_pixels)
                    results['内部纹理灰度峰度'] = scipy.stats.kurtosis(masked_pixels)
                else:
                    results['内部纹理灰度偏度'] = np.nan
                    results['内部纹理灰度峰度'] = np.nan
            else:
                results['内部纹纹理灰度均值'] = results['内部纹理灰度标准差'] = results[
                    '内部纹理灰度偏度'] = \
                    results['内部纹理灰度峰度'] = np.nan

            x, y, w, h = cv2.boundingRect(mask)
            if w > 0 and h > 0:
                particle_region_gray = original_gray_image[y:y + h, x:x + w]
                particle_region_mask = mask[y:y + h, x:x + w]
                masked_pixels_for_glcm = particle_region_gray[particle_region_mask > 0]
                levels_for_glcm = 16

                if len(masked_pixels_for_glcm) > 4 and np.max(masked_pixels_for_glcm) > np.min(masked_pixels_for_glcm):
                    min_val_glcm = np.min(masked_pixels_for_glcm)
                    max_val_glcm = np.max(masked_pixels_for_glcm)
                    scaled_pixels_float = (masked_pixels_for_glcm - min_val_glcm) / (max_val_glcm - min_val_glcm) * (
                            levels_for_glcm - 1)
                    quantized_pixels = np.round(scaled_pixels_float).astype(np.uint8)

                    temp_glcm_image = np.zeros_like(particle_region_gray, dtype=np.uint8)
                    temp_glcm_image[particle_region_mask > 0] = quantized_pixels
                    distances = [1]
                    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

                    try:
                        glcm = graycomatrix(temp_glcm_image.astype(np.uint8), distances=distances, angles=angles,
                                            levels=levels_for_glcm, symmetric=True, normed=True)

                        results['GLCM对比度'] = np.mean(graycoprops(glcm, 'contrast'))
                        results['GLCM异质性'] = np.mean(graycoprops(glcm, 'dissimilarity'))
                        results['GLCM均匀性'] = np.mean(graycoprops(glcm, 'homogeneity'))
                        results['GLCM能量'] = np.mean(graycoprops(glcm, 'energy'))
                        results['GLCM相关性'] = np.mean(graycoprops(glcm, 'correlation'))
                        results['GLCM角二阶矩'] = np.mean(graycoprops(glcm, 'ASM'))
                        glcm_normalized = glcm.astype(np.float32) / np.sum(glcm)
                        glcm_entropy = -np.sum(glcm_normalized * np.log2(glcm_normalized + 1e-10))
                        results['GLCM熵'] = glcm_entropy

                    except Exception:
                        results['GLCM对比度'] = results['GLCM异质性'] = results['GLCM均匀性'] = \
                            results['GLCM能量'] = results['GLCM相关性'] = results['GLCM角二阶矩'] = results[
                            'GLCM熵'] = np.nan
                else:
                    results['GLCM对比度'] = results['GLCM异质性'] = results['GLCM均匀性'] = \
                        results['GLCM能量'] = results['GLCM相关性'] = results['GLCM角二阶矩'] = results[
                        'GLCM熵'] = np.nan

                radius = 1
                n_points = 8 * radius
                try:
                    lbp_image = local_binary_pattern(particle_region_gray, n_points, radius, method='uniform')
                    masked_lbp_values = lbp_image[particle_region_mask > 0]
                    if len(masked_lbp_values) > 0:
                        hist, _ = np.histogram(masked_lbp_values, bins=np.arange(0, n_points + 3), density=True)
                        hist = hist[hist > 0]
                        results['LBP熵'] = scipy.stats.entropy(hist, base=2)
                        results['LBP均值'] = np.mean(masked_lbp_values)
                        results['LBP方差'] = np.var(masked_lbp_values)
                    else:
                        results['LBP熵'] = results['LBP均值'] = results['LBP方差'] = np.nan
                except Exception:
                    results['LBP熵'] = results['LBP均值'] = results['LBP方差'] = np.nan
            else:
                results['GLCM对比度'] = results['GLCM异质性'] = results['GLCM均匀性'] = results[
                    'GLCM能量'] = results['GLCM相关性'] = results['GLCM角二阶矩'] = results['GLCM熵'] = np.nan
                results['LBP熵'] = results['LBP均值'] = results['LBP方差'] = np.nan

            results['分形维数 (灰度框计数)'] = calculate_box_counting_fd(original_gray_image, mask)

        else:
            results['内部纹纹理灰度均值'] = results['内部纹理灰度标准差'] = results[
                '内部纹理灰度偏度'] = \
                results['内部纹理灰度峰度'] = np.nan
            results['GLCM对比度'] = results['GLCM异质性'] = results['GLCM均匀性'] = results[
                'GLCM能量'] = results['GLCM相关性'] = results['GLCM角二阶矩'] = results['GLCM熵'] = np.nan
            results['LBP熵'] = results['LBP均值'] = results['LBP方差'] = np.nan
            results['分形维数 (灰度框计数)'] = np.nan
    else:  # 如果不计算内部纹理特征，所有相关参数设为 NaN
        for key in ['内部纹纹理灰度均值', '内部纹理灰度标准差', '内部纹理灰度偏度',
                    '内部纹理灰度峰度', 'GLCM对比度', 'GLCM异质性', 'GLCM均匀性',
                    'GLCM能量', 'GLCM相关性', 'GLCM角二阶矩', 'GLCM熵', 'LBP熵',
                    'LBP均值', 'LBP方差', '分形维数 (灰度框计数)']:
            results[key] = np.nan

    # --- 8. 上下文与重建参数 ---
    if params["CALC_CONTEXT_FEATURES"]:
        is_edge_touching = touches_border(mask, BORDER_BAND)  # <-- 边界处理参数
        results['接触图像边缘'] = is_edge_touching

        bounding_box = cv2.boundingRect(contour)
        x, y, w_box, h_box = bounding_box  # 避免与mask的w,h混淆
        H_img, W_img = mask.shape  # 获取原始图像尺寸

        if 'perimeter_raw' in locals() and is_edge_touching and perimeter_raw > 0:
            contact_length = 0
            if x <= BORDER_BAND: contact_length += h_box
            if y <= BORDER_BAND: contact_length += w_box
            if x + w_box >= W_img - BORDER_BAND: contact_length += h_box
            if y + h_box >= H_img - BORDER_BAND: contact_length += w_box
            results['遮挡率 (边界)'] = min(1.0, contact_length / perimeter_raw)
        else:
            results['遮挡率 (边界)'] = 0.0

        if area > 0 and '凸包面积 (像素)' in results and results['凸包面积 (像素)'] is not np.nan:  # 确保 area_hull 已经计算
            results['凹陷面积比'] = (results['凸包面积 (像素)'] - area) / area
        else:
            results['凹陷面积比'] = np.nan

        if is_edge_touching and '拟合椭圆长轴 (像素)' in results and results['拟合椭圆长轴 (像素)'] is not np.nan:
            L_fit = results['拟合椭圆长轴 (像素)']
            S_fit = results['拟合椭圆短轴 (像素)']
            results['边缘重建直径 (像素)'] = math.sqrt(L_fit * S_fit)
        else:
            results['边缘重建直径 (像素)'] = np.nan

        if '凸包面积 (像素)' in results and results['凸包面积 (像素)'] > 0:
            results['凸包重建直径 (像素)'] = math.sqrt(4 * results['凸包面积 (像素)'] / math.pi)
        else:
            results['凸包重建直径 (像素)'] = np.nan
    else:  # 如果不计算上下文与重建参数，所有相关参数设为 NaN
        for key in ['接触图像边缘', '遮挡率 (边界)', '凹陷面积比',
                    '边缘重建直径 (像素)', '凸包重建直径 (像素)']:
            results[key] = np.nan

    return results


def passes_filters(meas, params):
    """ 检查颗粒是否通过用户定义的筛选器。 """
    MIN_AREA_PX = params["MIN_AREA_PX"]
    MIN_CIRCULARITY = params["MIN_CIRCULARITY"]
    MAX_AXIS_RATIO = params["MAX_AXIS_RATIO"]

    if meas is None:
        return False
    if MIN_AREA_PX is not None and meas.get("面积 (像素)", 0) < MIN_AREA_PX:
        return False
    # 这里需要确保 circularity 在 meas 中被计算了
    if MIN_CIRCULARITY is not None and meas.get("紧凑度/圆度", np.nan) < MIN_CIRCULARITY:
        return False

    axis_ratio = meas.get("拟合椭圆轴比 (L/S)")
    if MAX_AXIS_RATIO is not None and (axis_ratio is None or axis_ratio > MAX_AXIS_RATIO):
        return False
    return True


# ---------- YOLO 推理函数 (已修改以接受 params) ----------
def yolo_masks_and_classes(model, img_bgr, params):
    conf_thresh = params["CONF_THRESH"]
    iou_thresh = params["IOU_THRESH"]
    retina_masks = params["RETINA_MASKS"]
    max_detections = params["MAX_DETECTIONS"]
    target_imgsz = params["INFERENCE_IMG_SIZE"]

    r = \
        model.predict(img_bgr, imgsz=target_imgsz, conf=conf_thresh, iou=iou_thresh, retina_masks=retina_masks,
                      verbose=False, max_det=max_detections)[
            0]
    H, W = img_bgr.shape[:2]

    if r.masks is None or r.masks.data.numel() == 0:
        return (np.zeros((0, H, W), np.uint8),
                np.zeros((0,), np.int32),
                np.zeros((0,), np.float32),
                np.zeros((0, 4), np.float32),
                r.names if hasattr(r, "names") else {})

    masks_f = r.masks.data.cpu().numpy()
    masks = (masks_f > 0.5).astype(np.uint8)

    if masks.shape[0] > 0 and (masks.shape[1] != H or masks.shape[2] != W):
        masks = np.stack([cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) for m in masks], axis=0).astype(
            np.uint8)

    cls = r.boxes.cls.cpu().numpy().astype(int)
    conf = r.boxes.conf.cpu().numpy()
    boxes = r.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    names = r.names if hasattr(r, "names") else {}

    return masks, cls, conf, boxes, names


# ---------- YOLO 推理函数 (重叠 Tiling - 已修改以接受 params) ----------
def yolo_masks_and_classes_tiled(model, img_bgr, params):
    TILE_ROWS = params["TILE_ROWS"]
    TILE_COLS = params["TILE_COLS"]
    overlap_px = params["OVERLAP_PX"]
    INFERENCE_IMG_SIZE = params["INFERENCE_IMG_SIZE"]

    H, W = img_bgr.shape[:2]

    core_H = H // TILE_ROWS
    core_W = W // TILE_COLS

    if (TILE_ROWS == 1 and TILE_COLS == 1) or core_H <= overlap_px * 2 or core_W <= overlap_px * 2:
        # 如果核心区域过小，回退到非 Tiling 模式
        return yolo_masks_and_classes(model, img_bgr, params)

    all_masks_raw = []
    all_classes_raw = []
    all_confidences_raw = []
    all_boxes_raw = []
    names = {}

    for r_idx in range(TILE_ROWS):
        for c_idx in range(TILE_COLS):

            y_core_start_for_tile_calc = r_idx * core_H
            x_core_start_for_tile_calc = c_idx * core_W

            y_start = max(0, y_core_start_for_tile_calc - overlap_px)
            x_start = max(0, x_core_start_for_tile_calc - overlap_px)

            y_end = min(H, y_core_start_for_tile_calc + core_H + overlap_px)
            x_end = min(W, x_core_start_for_tile_calc + core_W + overlap_px)

            if r_idx == TILE_ROWS - 1: y_end = H
            if c_idx == TILE_COLS - 1: x_end = W

            tile = img_bgr[y_start:y_end, x_start:x_end]

            if tile.size == 0: continue

            # 使用 INFERENCE_IMG_SIZE 进行推理
            masks_tile, classes_tile, confidences_tile, boxes_tile, current_names = yolo_masks_and_classes(
                model, tile, params  # 传入 params
            )
            names.update(current_names)  # 更新 names 字典

            y_core_start_full = r_idx * core_H
            x_core_start_full = c_idx * core_W
            y_core_end_full = (r_idx + 1) * core_H
            x_core_end_full = (c_idx + 1) * core_W

            if r_idx == TILE_ROWS - 1: y_core_end_full = H
            if c_idx == TILE_COLS - 1: x_core_end_full = W

            for i in range(masks_tile.shape[0]):
                mask_tile = masks_tile[i]

                M = cv2.moments(mask_tile)
                if M["m00"] == 0:
                    continue
                cx_tile = int(M["m10"] / M["m00"])
                cy_tile = int(M["m01"] / M["m00"])

                cx_full = cx_tile + x_start
                cy_full = cy_tile + y_start

                if (x_core_start_full <= cx_full < x_core_end_full) and \
                        (y_core_start_full <= cy_full < y_core_end_full):
                    full_mask = np.zeros((H, W), dtype=np.uint8)
                    full_mask[y_start:y_start + mask_tile.shape[0], x_start:x_start + mask_tile.shape[1]] = mask_tile

                    # 转换 Box 到全图坐标
                    box_tile = boxes_tile[i]
                    box_full = np.array([
                        box_tile[0] + x_start,
                        box_tile[1] + y_start,
                        box_tile[2] + x_start,
                        box_tile[3] + y_start,
                    ], dtype=np.float32)

                    all_masks_raw.append(full_mask)
                    all_classes_raw.append(classes_tile[i])
                    all_confidences_raw.append(confidences_tile[i])
                    all_boxes_raw.append(box_full)

    return np.stack(all_masks_raw, axis=0) if all_masks_raw else np.zeros((0, H, W), np.uint8), \
        np.array(all_classes_raw, np.int32) if all_classes_raw else np.zeros((0,), np.int32), \
        np.array(all_confidences_raw, np.float32) if all_confidences_raw else np.zeros((0,), np.float32), \
        np.array(all_boxes_raw, np.float32) if all_boxes_raw else np.zeros((0, 4), np.float32), \
        names


def draw_overlay(img_bgr, items, params):
    """ 绘制叠加图 (已修改以接受 params 字典) """

    BACKGROUND_DIM_FACTOR = params["BACKGROUND_DIM_FACTOR"]
    FILL_ALPHA = params["FILL_ALPHA"]
    ENABLE_CUSTOM_COLORING = params["ENABLE_CUSTOM_COLORING"]
    CUSTOM_COLORING_RULES = params["CUSTOM_COLORING_RULES"]
    COLORING_MODE = params["COLORING_MODE"]
    UM_PER_PX = params["UM_PER_PX"]
    SORTED_SIZE_COLOR_RULES = params["SORTED_SIZE_COLOR_RULES"]
    CLASS_COLORS = params["CLASS_COLORS"]
    SHOW_ONLY_OUTLINE = params["SHOW_ONLY_OUTLINE"]
    OUTLINE_THICKNESS = params["OUTLINE_THICKNESS"]
    SHOW_PARTICLE_ID = params["SHOW_PARTICLE_ID"]
    TEXT_COLOR = params["TEXT_COLOR"]  # BGR

    # 确保这两个参数存在，即使在 app.py 中未传递，这里也应该有默认值
    UI_OUTLINE_COLOR = params.get("UI_OUTLINE_COLOR", (255, 255, 255))
    # 轮廓线不透明度：优先使用 UI 传入的值，否则退回到默认配置中的 OUTLINE_ALPHA
    OUTLINE_ALPHA = float(params.get("UI_OUTLINE_ALPHA", params.get("OUTLINE_ALPHA", 1.0)))
    # 数值安全保护
    if OUTLINE_ALPHA < 0.0:
        OUTLINE_ALPHA = 0.0
    if OUTLINE_ALPHA > 1.0:
        OUTLINE_ALPHA = 1.0

    out = (img_bgr * BACKGROUND_DIM_FACTOR).astype(np.uint8)

    for it in items:
        # --- 确定颗粒颜色 (用于填充) ---
        specific_color = None
        eqD_um = it["eqD_um"]
        # [省略了中间的颜色确定逻辑，假设它能正确导出 color 变量]
        # ... (使用自定义规则、粒径或类别确定 color) ...
        # [注意：如果你使用了上个回复中的 draw_overlay 完整代码，颜色确定逻辑应该在这里]

        # 假设 color 已经确定
        # ------------------------------------
        # START: 颜色确定逻辑 (为了完整性，再次包含)
        if ENABLE_CUSTOM_COLORING and CUSTOM_COLORING_RULES:
            for rule in CUSTOM_COLORING_RULES:
                param_name = rule["param"]
                min_val = rule["min"]
                max_val = rule["max"]
                rule_color = rule["color"]
                param_value = it["measurements"].get(param_name)
                if param_value is not None and not math.isnan(param_value):
                    if min_val <= param_value < max_val:
                        specific_color = rule_color
                        break
        if specific_color is None:
            if COLORING_MODE == 'SIZE' and UM_PER_PX is not None and not math.isnan(eqD_um):
                for threshold, color_val in SORTED_SIZE_COLOR_RULES:
                    if eqD_um > threshold:
                        specific_color = color_val
                        break
            elif COLORING_MODE == 'CLASS':
                specific_color = CLASS_COLORS.get(it["class_id"], (128, 128, 128))
        color = specific_color if specific_color is not None else CLASS_COLORS.get(it["class_id"], (128, 128, 128))
        # END: 颜色确定逻辑
        # ------------------------------------

        mask_geo = it["mask_geo"]
        contour = contour_from_mask(mask_geo)

        if contour is None:
            continue

        # 1. 填充逻辑 (仅在 SHOW_ONLY_OUTLINE 为 False 时执行)
        if not SHOW_ONLY_OUTLINE:
            m = mask_geo > 0
            if np.any(m):
                roi = out[m]
                blended = (roi * (1.0 - FILL_ALPHA) + np.array(color,
                                                               dtype=np.float32) * FILL_ALPHA).astype(
                    np.uint8)
                out[m] = blended

        # 2. 轮廓绘制逻辑 (仅在 OUTLINE_THICKNESS > 0 时执行)
        if OUTLINE_THICKNESS > 0:
            # 在单独图层上绘制实心轮廓，然后按 OUTLINE_ALPHA 与原图进行 Alpha 混合
            outline_layer = np.zeros_like(out, dtype=np.uint8)
            cv2.drawContours(outline_layer, [contour], -1, UI_OUTLINE_COLOR, OUTLINE_THICKNESS)

            # 生成轮廓区域的掩码
            outline_gray = cv2.cvtColor(outline_layer, cv2.COLOR_BGR2GRAY)
            outline_mask = outline_gray > 0
            if np.any(outline_mask):
                roi_base = out[outline_mask].astype(np.float32)
                roi_outline = outline_layer[outline_mask].astype(np.float32)
                blended_outline = (roi_base * (1.0 - OUTLINE_ALPHA) + roi_outline * OUTLINE_ALPHA).astype(np.uint8)
                out[outline_mask] = blended_outline

    # --- 步骤 3: 绘制 ID 编码 (保持不变) ---
    if SHOW_PARTICLE_ID:
        for it in items:
            M = cv2.moments(it["mask_geo"])
            if M["m00"] > 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                txt = str(it["id"])
                # 黑色描边
                cv2.putText(out, txt, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                # 文本
                cv2.putText(out, txt, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1, cv2.LINE_AA)

    return out


# ----------------------------------------------------------------------
# Streamlit 核心处理函数 (接收图像数组和参数，返回结果) - 已优化并行计算
# ----------------------------------------------------------------------
def analyze_image_stream(img_bgr, model, params, base_filename="uploaded_image"):
    """
    Streamlit 友好的核心分析函数。
    接收 BGR 图像数组，返回原始图像 BGR 数组、颗粒数据列表和形态学数据 DataFrame。
    """

    # 提取参数
    TILE_ROWS = params["TILE_ROWS"]
    TILE_COLS = params["TILE_COLS"]
    MIN_AREA_PX = params["MIN_AREA_PX"]
    UM_PER_PX = params["UM_PER_PX"]
    NUM_WORKERS = params["NUM_WORKERS"]

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if TILE_ROWS > 1 or TILE_COLS > 1:
        masks_raw, classes_raw, confidences_raw, boxes_raw, names = yolo_masks_and_classes_tiled(
            model, img_bgr, params
        )
    else:
        masks_raw, classes_raw, confidences_raw, boxes_raw, names = yolo_masks_and_classes(model, img_bgr, params)

    # 调用加速 NMS
    masks, classes, confidences, boxes = mask_nms(
        masks_raw, classes_raw, confidences_raw, boxes_raw, params
    )

    # 准备并行处理任务
    tasks = []
    processed_items = []

    # 使用 ThreadPoolExecutor 进行并行计算
    # 每个任务负责处理一个颗粒的形态学计算
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for i in range(masks.shape[0]):
            raw_mask = (masks[i] > 0).astype(np.uint8)

            # 处理连通分量，只保留最大的
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(raw_mask, 8, cv2.CV_32S)

            max_label_area = 0
            max_label_idx = -1

            for l in range(1, num_labels):
                if stats[l, cv2.CC_STAT_AREA] > max_label_area:
                    max_label_area = stats[l, cv2.CC_STAT_AREA]
                    max_label_idx = l

            if max_label_idx == -1:
                continue

            # 筛选小面积颗粒
            if MIN_AREA_PX is not None and stats[max_label_idx, cv2.CC_STAT_AREA] < MIN_AREA_PX:
                continue

            geo_mask = np.zeros_like(raw_mask, dtype=np.uint8)
            geo_mask[labels == max_label_idx] = 1
            geo_mask = smooth_mask_simple(geo_mask)  # 平滑处理

            if np.sum(geo_mask) < (MIN_AREA_PX if MIN_AREA_PX is not None else 1):
                continue

            temp_contour = contour_from_mask(geo_mask)
            if temp_contour is None or cv2.contourArea(temp_contour) == 0:
                continue

            # 提交任务到线程池
            # 需要传递所有计算所需的上下文信息
            tasks.append(executor.submit(
                _process_single_particle_for_parallel,
                geo_mask, img_gray, params, classes[i], names, base_filename, i + 1  # 颗粒ID从1开始
            ))

        for future in as_completed(tasks):
            result = future.result()
            if result:
                processed_items.append(result)

    # 排序以保持颗粒ID的顺序 (如果原始顺序重要的话)
    processed_items.sort(key=lambda x: x['particle_id'])

    items = []
    rows_list = []

    # 重新整理数据，并重新分配ID以确保连续性
    next_id = 1
    for p_item in processed_items:
        # 重置颗粒ID
        p_item['item_data']['id'] = next_id
        p_item['row_data'][1] = next_id  # 修改 row_data 中的 ID

        items.append(p_item['item_data'])
        rows_list.append(p_item['row_data'])
        next_id += 1

    # 构建 DataFrame (需要根据实际填充的列数切分 CHINESE_HEADERS)
    num_cols = len(rows_list[0]) if rows_list else 0
    df = pd.DataFrame(rows_list, columns=CHINESE_HEADERS[:num_cols])

    return img_bgr, items, df  # 返回原始 BGR 图像, 颗粒列表, 数据 DataFrame


# ----------------------------------------------------------------------
# 辅助函数：处理单个颗粒的形态学计算 (用于并行)
# ----------------------------------------------------------------------
def _process_single_particle_for_parallel(geo_mask, img_gray, params, class_id, names, base_filename,
                                          original_particle_idx):
    """
    一个辅助函数，用于在 ThreadPoolExecutor 中处理单个颗粒的形态学计算。
    返回一个包含 item_data 和 row_data 的字典。
    """

    # 计算形态学参数
    meas = calculate_all_morphology(geo_mask, img_gray, params)

    if meas is None or not passes_filters(meas, params):
        return None

    cid = int(class_id)
    class_name = names.get(cid, str(cid))

    row_data = [base_filename, original_particle_idx, cid, class_name]  # 先用原始ID

    # 尺寸与面积
    um_per_px = params["UM_PER_PX"]

    row_data.append(meas.get('面积 (像素)', np.nan))
    row_data.append(to_um_squared(meas.get('面积 (像素)', np.nan), um_per_px))
    eqD_px = meas.get('等效直径 (像素)', np.nan)
    eqD_um = to_um(eqD_px, um_per_px)
    row_data.append(eqD_px)
    row_data.append(eqD_um)
    volume_px_cubed = meas.get('体积 (微米^3)', np.nan)
    row_data.append(to_um_cubed(volume_px_cubed, um_per_px))

    # 拟合椭圆参数
    major_ell_px = meas.get('拟合椭圆长轴 (像素)', np.nan)
    row_data.append(major_ell_px)
    row_data.append(to_um(major_ell_px, um_per_px))
    minor_ell_px = meas.get('拟合椭圆短轴 (像素)', np.nan)
    row_data.append(minor_ell_px)
    row_data.append(to_um(minor_ell_px, um_per_px))
    row_data.append(meas.get('拟合椭圆轴比 (L/S)', np.nan))
    row_data.append(meas.get('拟合椭圆延伸率 (S/L)', np.nan))
    row_data.append(meas.get('拟合椭圆偏心率', np.nan))
    row_data.append(meas.get('拟合椭圆角度 (度)', np.nan))

    # 最小外接矩形参数
    major_rect_px = meas.get('最小外接矩形长轴 (像素)', np.nan)
    row_data.append(major_rect_px)
    row_data.append(to_um(major_rect_px, um_per_px))
    minor_rect_px = meas.get('最小外接矩形短轴 (像素)', np.nan)
    row_data.append(minor_rect_px)
    row_data.append(to_um(minor_rect_px, um_per_px))
    row_data.append(meas.get('最小外接矩形轴比 (L/S)', np.nan))
    row_data.append(meas.get('最小外接矩形延伸率 (S/L)', np.nan))
    row_data.append(meas.get('最小外接矩形角度 (度)', np.nan))

    # 圆度与凸性
    perim_px = meas.get('周长 (像素, 平滑)', np.nan)
    row_data.append(perim_px)
    row_data.append(to_um(perim_px, um_per_px))
    row_data.append(meas.get('紧凑度/圆度', np.nan))
    row_data.append(meas.get('莱利圆度 (Sphericity Proxy)', np.nan))
    row_data.append(meas.get('平均绝对曲率 (圆度代理)', np.nan))

    area_hull_px = meas.get('凸包面积 (像素)', np.nan)
    row_data.append(area_hull_px)
    row_data.append(to_um_squared(area_hull_px, um_per_px))
    row_data.append(meas.get('凸度', np.nan))
    perim_hull_px = meas.get('凸包周长 (像素)', np.nan)
    row_data.append(perim_hull_px)
    row_data.append(to_um(perim_hull_px, um_per_px))
    row_data.append(meas.get('凸性比', np.nan))

    # 边界局部特征
    radial_std_px = meas.get('边界径向标准差 (像素)', np.nan)
    row_data.append(radial_std_px)
    row_data.append(to_um(radial_std_px, um_per_px))
    row_data.append(meas.get('边界粗糙度', np.nan))
    row_data.append(meas.get('边界曲率峰值', np.nan))
    row_data.append(meas.get('凹陷数量', np.nan))
    max_concavity_depth_px = meas.get('最大凹陷深度 (像素)', np.nan)
    row_data.append(max_concavity_depth_px)
    row_data.append(to_um(max_concavity_depth_px, um_per_px))
    row_data.append(meas.get('总相对凹陷深度 (Irregularity Proxy)', np.nan))

    # 傅里叶描述子
    row_data.extend([meas.get(f'傅里叶描述子FD{k}', np.nan) for k in range(2, 7)])

    # 内部纹理特征
    texture_keys = ['内部纹纹理灰度均值', '内部纹理灰度标准差', '内部纹理灰度偏度',
                    '内部纹理灰度峰度', 'GLCM对比度', 'GLCM异质性', 'GLCM均匀性',
                    'GLCM能量', 'GLCM相关性', 'GLCM角二阶矩', 'GLCM熵', 'LBP熵',
                    'LBP均值', 'LBP方差', '分形维数 (灰度框计数)']
    row_data.extend([meas.get(k, np.nan) for k in texture_keys])

    # 上下文与重建
    row_data.append(meas.get('遮挡率 (边界)', np.nan))
    row_data.append(meas.get('凹陷面积比', np.nan))
    row_data.append(meas.get('接触图像边缘', np.nan))

    D_eq_edge_px = meas.get('边缘重建直径 (像素)', np.nan)
    row_data.append(D_eq_edge_px)
    row_data.append(to_um(D_eq_edge_px, um_per_px))

    D_eq_ch_px = meas.get('凸包重建直径 (像素)', np.nan)
    row_data.append(D_eq_ch_px)
    row_data.append(to_um(D_eq_ch_px, um_per_px))

    item_data = {
        "id": original_particle_idx, "class_id": cid, "mask_geo": geo_mask, "eqD_um": eqD_um, "measurements": meas
    }

    return {"item_data": item_data, "row_data": row_data, "particle_id": original_particle_idx}


# ==============================================================================
# 新增绘图相关辅助函数
# ==============================================================================

def _map_display_name_to_column(display_name, display_to_internal_map):
    """根据UI显示名称获取内部列名。"""
    return display_to_internal_map.get(display_name, display_name)


def _get_display_name_from_column(internal_col_name, internal_to_display_map, lang='cn'):
    """根据内部列名获取UI显示名称（中文或英文）。"""
    display_tuple = internal_to_display_map.get(internal_col_name, (internal_col_name, internal_col_name))
    if lang == 'cn':
        return display_tuple[0]
    else:
        return display_tuple[1]


def _parse_bin_midpoint(bin_str):
    """将 'Start-End' 格式的粒径范围字符串解析为中点数值。"""
    if pd.isna(bin_str):
        return np.nan
    bin_str = str(bin_str).strip()
    if '-' in bin_str:
        try:
            parts = bin_str.split('-')
            start = float(parts[0].strip())
            end = float(parts[1].strip())
            return (start + end) / 2
        except:
            return np.nan
    else:
        try:
            return float(bin_str)
        except:
            return np.nan


def _get_smoothed_data(x_data, y_data, smooth_points=PLOT_CONFIG.PSD_SMOOTH_POINTS, is_pdf=False, is_cdf=False):
    """
    使用三次样条插值生成平滑的 X 和 Y 数据点。
    is_pdf: 如果是 PDF，则对负值进行截断。
    is_cdf: 如果是 CDF，则确保单调递增且不超1。
    """
    # 过滤 NaN 值，并确保数据量足够
    valid_indices = ~np.isnan(x_data) & ~np.isnan(y_data)
    x = x_data[valid_indices]
    y = y_data[valid_indices]

    if len(x) < 3:
        # 如果数据点太少，无法进行样条插值，直接返回原始数据
        return x_data, y_data

    # 排序以确保样条插值正确
    sort_indices = np.argsort(x)
    x = x[sort_indices]
    y = y[sort_indices]

    # 创建新的、密集的 X 轴，覆盖原始数据的范围
    x_new = np.linspace(x.min(), x.max(), smooth_points)

    # 创建样条插值函数
    spl = make_interp_spline(x, y, k=3)
    y_smooth = spl(x_new)

    if is_pdf:
        # 对于 PDF，平滑可能导致负值，将其钳位到 0
        y_smooth[y_smooth < 0] = 0
    if is_cdf:
        # 对于 CDF，确保单调递增且不超过 1
        y_smooth = np.maximum.accumulate(y_smooth)  # 确保单调
        y_smooth = np.clip(y_smooth, 0, 1)

    return x_new, y_smooth


def _setup_log_ticks(ax, x_data, label_font_size=12):
    """设置 Log10 轴上的线性刻度标签"""
    # 确保 x_data 为 Series 或 DataFrame 列，以便使用 quantile
    if not isinstance(x_data, pd.Series):
        x_data = pd.Series(x_data)

    x_data = x_data.dropna()
    if x_data.empty:
        return

    # 确定 X 轴聚焦的范围 (略微扩大以包含端点)
    log_min_val = np.log10(x_data[x_data > 0].min()) if x_data[x_data > 0].min() > 0 else -1
    log_max_val = np.log10(x_data[x_data > 0].max()) if x_data[x_data > 0].max() > 0 else 1

    # 避免 Log 轴显示错误
    if not np.isfinite(log_min_val): log_min_val = -3
    if not np.isfinite(log_max_val): log_max_val = 3
    if log_max_val <= log_min_val: log_max_val = log_min_val + 1  # 确保范围有效

    ax.set_xlim(log_min_val - 0.1, log_max_val + 0.1)

    # 生成 10^n 的整数幂刻度 (例如 0.1, 1, 10, 100...)
    min_power = np.floor(log_min_val)
    max_power = np.ceil(log_max_val)

    log_ticks_positions = []
    for p in np.arange(min_power, max_power + 1):
        if min_power <= p <= max_power:
            log_ticks_positions.append(p)

    log_ticks_labels = []
    for p in log_ticks_positions:
        if p == 0:
            log_ticks_labels.append('1 µm')
        elif p == 1:
            log_ticks_labels.append('10 µm')
        elif p == -1:
            log_ticks_labels.append('0.1 µm')
        else:
            log_ticks_labels.append(f'$10^{{{int(p)}}}$ µm')

    ax.set_xticks(log_ticks_positions)
    ax.set_xticklabels(log_ticks_labels, fontsize=label_font_size)

    # 添加 Log 网格线
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)


def _apply_chinese_font(ax):
    """
    将全局中文字体属性应用到指定坐标轴的标题、坐标轴标签、刻度标签和图例。
    这是为了解决在某些环境下 Matplotlib 默认字体不支持中文，导致显示为方块的问题。
    """
    if GLOBAL_CHINESE_FONT_PROP is None:
        return

    # 标题与坐标轴标签：用 set_title / set_xlabel / set_ylabel 显式覆盖字体
    try:
        current_title = ax.get_title()
        if current_title:
            ax.set_title(current_title, fontproperties=GLOBAL_CHINESE_FONT_PROP)
    except Exception:
        pass

    try:
        current_xlabel = ax.get_xlabel()
        if current_xlabel:
            ax.set_xlabel(current_xlabel, fontproperties=GLOBAL_CHINESE_FONT_PROP)
    except Exception:
        pass

    try:
        current_ylabel = ax.get_ylabel()
        if current_ylabel:
            ax.set_ylabel(current_ylabel, fontproperties=GLOBAL_CHINESE_FONT_PROP)
    except Exception:
        pass

    # 坐标轴刻度
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(GLOBAL_CHINESE_FONT_PROP)

    # 图例
    legend = ax.get_legend()
    if legend is not None:
        if legend.get_title() is not None:
            legend.get_title().set_fontproperties(GLOBAL_CHINESE_FONT_PROP)
        for text in legend.get_texts():
            text.set_fontproperties(GLOBAL_CHINESE_FONT_PROP)


# ==============================================================================
# 绘图函数 (返回 Figure 对象)
# ==============================================================================

def plot_shape_frequency_histogram(
        results_df: pd.DataFrame,
        size_col: str,  # 例如 '等效直径 (微米)'
        shape_col: str,  # 例如 '类别名称'
        target_shapes: list,  # 例如 ['round', 'satellite']
        min_size: float,
        max_size: float,
        bin_width: float,
        custom_y_ticks: list,  # 例如 [0, 100, 200, 400, 600, 800]
        y_max_limit: float,  # 例如 800
        title_cn: str = "不同粒度段中不同类别的颗粒频数分布",
        x_label_cn: str = "粒径 (微米)",
        y_label_cn: str = "频数 (数量)",
        lang: str = 'cn'
) -> plt.Figure:
    """
    绘制不同粒度段中不同类别的颗粒频数堆叠柱状图。
    """
    if results_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "无数据可用于绘图", horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        return fig

    # 根据语言获取标签
    _, internal_to_display, _ = _get_plot_column_map()
    size_display_name = _get_display_name_from_column(size_col, internal_to_display, lang)
    shape_display_name = _get_display_name_from_column(shape_col, internal_to_display, lang)

    x_label = x_label_cn if lang == 'cn' else f"Particle Size ({internal_to_display.get(size_col, ('', ''))[1].split('(')[-1].replace(')', '')})"
    y_label = y_label_cn if lang == 'cn' else "Frequency (Count)"
    title = title_cn if lang == 'cn' else "Particle Shape Frequency by Size Bin"

    data = results_df[[size_col, shape_col]].copy()
    data.columns = ['Size', 'Shape']  # 内部使用临时列名

    data['Size'] = pd.to_numeric(data['Size'], errors='coerce')
    data['Shape'] = data['Shape'].astype(str).str.lower().str.strip()

    data.dropna(subset=['Size', 'Shape'], inplace=True)
    data = data[data['Size'].between(min_size, max_size, inclusive='both')]
    data = data[data['Shape'].isin([s.lower() for s in target_shapes])]

    if data.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "筛选后无数据可用于绘图", horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        return fig

    bins = np.arange(min_size, max_size + bin_width, bin_width)
    bin_centers = bins[:-1] + bin_width / 2
    bar_width = bin_width * 0.8

    counts = {}
    for shape in target_shapes:
        counts[shape], _ = np.histogram(data[data['Shape'] == shape.lower()]['Size'], bins=bins)

    fig, ax = plt.subplots(figsize=(12, 6))

    # 定义颜色，从 PLOT_CONFIG 中获取
    colors_map = {shape: PLOT_CONFIG.SHAPE_HIST_DEFAULT_COLORS[i % len(PLOT_CONFIG.SHAPE_HIST_DEFAULT_COLORS)]
                  for i, shape in enumerate(target_shapes)}

    bottom_stack = np.zeros_like(bin_centers, dtype=float)  # 确保是浮点数

    for shape in target_shapes:
        freq = counts[shape]

        ax.bar(
            bin_centers,
            freq,
            width=bar_width,
            bottom=bottom_stack,
            color=colors_map.get(shape.lower(), 'gray'),
            label=shape.capitalize(),
            edgecolor='black',
            linewidth=0.5
        )
        bottom_stack += freq

    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_title(title, fontsize=16, pad=20)

    ax.set_xticks(bins)
    ax.set_xlim(min_size, max_size)
    ax.tick_params(axis='x', rotation=0)

    ax.set_ylim(0, y_max_limit)
    ax.set_yticks(custom_y_ticks)

    ax.grid(axis='y', linestyle='--', alpha=0.6)

    ax.legend(
        title=shape_display_name,
        loc='upper left',
        bbox_to_anchor=(1.03, 1.0),
        frameon=True,
        fancybox=True,
        shadow=False
    )

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    return fig


def plot_volume_distribution_curves(
        results_df: pd.DataFrame,
        diameter_col: str,  # 例如 '等效直径 (微米)'
        bins_count: int = 50,  # 分箱数量，用于从原始数据生成分布
        log_scale_x: bool = True,  # X轴是否使用对数刻度
        comparison_col: str = None,  # 例如 '类别名称'，用于对比不同类别的分布
        title_cn: str = "粒径分布曲线",
        x_label_cn: str = "粒径",
        y_label_pdf_cn: str = "体积百分比 (%)",
        y_label_cdf_cn: str = "累积体积百分比",
        lang: str = 'cn'
) -> plt.Figure:
    """
    生成粒径分布双图对比 (PDF 和 CDF)。
    可选择按 `comparison_col` 进行分组对比。
    """
    if results_df.empty:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.text(0.5, 0.5, "无数据可用于绘图", horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        return fig

    _, internal_to_display, _ = _get_plot_column_map()
    diameter_display_name = _get_display_name_from_column(diameter_col, internal_to_display, lang)

    x_label = x_label_cn if lang == 'cn' else "Equivalent Diameter"
    y_label_pdf = y_label_pdf_cn if lang == 'cn' else "Volume Percentage (%)"
    y_label_cdf = y_label_cdf_cn if lang == 'cn' else "Cumulative Volume Percentage"
    title_pdf = "体积百分比分布 (PDF)" if lang == 'cn' else "Volume Percentage Distribution (PDF)"
    title_cdf = "累积体积分布 (CDF)" if lang == 'cn' else "Cumulative Volume Distribution (CDF)"

    # 过滤掉NaN值和非正的粒径值
    df_filtered = results_df.dropna(subset=[diameter_col]).copy()
    df_filtered = df_filtered[df_filtered[diameter_col] > 0]

    if df_filtered.empty:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.text(0.5, 0.5, "筛选后无有效粒径数据可用于绘图", horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        return fig

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    unique_comparison_values = [None]  # 默认不分组
    if comparison_col and comparison_col in df_filtered.columns:
        unique_comparison_values = df_filtered[comparison_col].dropna().unique()
        if len(unique_comparison_values) > 1:  # 只有多于一个值才进行对比
            unique_comparison_values = sorted(unique_comparison_values)
        else:
            unique_comparison_values = [None]  # 只有一个类别或无类别，不进行分组对比

    all_log_diameters = []

    # --------------------------------------------------------------------------
    # --- 1. 频率分布图 (PDF) ---
    # --------------------------------------------------------------------------
    ax0 = axes[0]
    ax0.set_title(title_pdf, fontsize=16)
    ax0.set_ylabel(y_label_pdf, fontsize=14)
    ax0.set_xlabel('', visible=False)
    ax0.grid(axis='y', linestyle=':', alpha=0.6)

    # --------------------------------------------------------------------------
    # --- 2. 累积分布图 (CDF) ---
    # --------------------------------------------------------------------------
    ax1 = axes[1]
    ax1.set_title(title_cdf, fontsize=16)
    ax1.set_ylabel(y_label_cdf, fontsize=14)
    ax1.grid(axis='y', linestyle=':', alpha=0.6)
    ax1.set_ylim(0, 1.05)
    ax1.set_yticks(np.arange(0, 1.1, 0.1))

    for i, group_val in enumerate(unique_comparison_values):
        if group_val is None:
            df_group = df_filtered
            label = "所有颗粒" if lang == 'cn' else "All Particles"
        else:
            df_group = df_filtered[df_filtered[comparison_col] == group_val]
            label = str(group_val)

        if df_group.empty:
            continue

        # 计算粒径分布
        min_d, max_d = df_group[diameter_col].min(), df_group[diameter_col].max()
        if max_d == min_d:  # 避免分箱错误
            continue

            # 优化分箱，确保 Log 轴下分箱合理
        if log_scale_x:
            log_min_d = np.log10(min_d) if min_d > 0 else -1
            log_max_d = np.log10(max_d) if max_d > 0 else 1
            log_bins = np.linspace(log_min_d, log_max_d, bins_count + 1)
            d_bins = 10 ** log_bins
        else:
            d_bins = np.linspace(min_d, max_d, bins_count + 1)

        hist, bin_edges = np.histogram(df_group[diameter_col], bins=d_bins, weights=df_group['体积 (微米^3)'])
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

        # 将体积转换为百分比
        total_volume = hist.sum()
        volume_percentage = (hist / total_volume) * 100 if total_volume > 0 else np.zeros_like(hist)

        # 计算累积分布
        cumulative_volume = np.cumsum(volume_percentage) / 100

        # 如果X轴是log，需要转换bin_midpoints
        x_plot = np.log10(bin_midpoints) if log_scale_x else bin_midpoints
        all_log_diameters.extend(list(x_plot))

        # 绘制 PDF
        current_color = PLOT_CONFIG.PSD_LINE_COLORS[i % len(PLOT_CONFIG.PSD_LINE_COLORS)]
        current_linestyle = PLOT_CONFIG.PSD_LINE_STYLES[i % len(PLOT_CONFIG.PSD_LINE_STYLES)]

        x_pdf_smooth, y_pdf_smooth = _get_smoothed_data(x_plot, volume_percentage, is_pdf=True)
        ax0.plot(x_pdf_smooth, y_pdf_smooth,
                 label=label,
                 color=current_color,
                 linewidth=PLOT_CONFIG.LINE_WIDTH,
                 linestyle=current_linestyle)

        # 绘制 CDF
        x_cdf_smooth, y_cdf_smooth = _get_smoothed_data(x_plot, cumulative_volume, is_cdf=True)
        ax1.plot(x_cdf_smooth, y_cdf_smooth,
                 label=label,
                 color=current_color,
                 linewidth=PLOT_CONFIG.LINE_WIDTH,
                 linestyle=current_linestyle)

    # 设置 X 轴标签
    if log_scale_x:
        ax1.set_xlabel(x_label + " ($log_{10}$)", fontsize=14)
        if all_log_diameters:  # 确保有数据才设置 Log 刻度
            _setup_log_ticks(ax1, pd.Series(all_log_diameters), label_font_size=12)
    else:
        ax1.set_xlabel(x_label, fontsize=14)
        ax1.tick_params(axis='x', which='major', labelsize=12)

    # 图例
    if comparison_col:
        legend_title = _get_display_name_from_column(comparison_col, internal_to_display, lang)
        leg0 = ax0.legend(title=legend_title, loc='upper left', frameon=True)
        if leg0: leg0.get_title().set_fontsize(12)
        # leg1 = ax1.legend(title=legend_title, loc='upper left', frameon=True) # CDF图例可以省略，因为共享
        # if leg1: leg1.get_title().set_fontsize(12)

    # 统一设置轴线粗细
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(PLOT_CONFIG.AXIS_LINEWIDTH)
        ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)  # 调整子图间距，略微留白

    return fig


def plot_kde_scatter_plot(
        results_df: pd.DataFrame,
        x_col: str,  # 例如 '等效直径 (微米)'
        y_col: str,  # 例如 '拟合椭圆轴比 (L/S)'
        class_filter: list = None,  # 过滤特定类别，例如 ['round']
        color: str = PLOT_CONFIG.KDE_DEFAULT_COLORS[0],  # 绘图颜色
        title_cn: str = "颗粒形态学参数分布 (KDE + 散点图)",
        x_label_cn: str = "粒径",
        y_label_cn: str = "形态学参数",
        log_scale_x: bool = True,  # X轴是否使用对数刻度
        lang: str = 'cn'
) -> plt.Figure:
    """
    绘制单个工艺（或过滤后的颗粒）的 2D 密度图（KDE）和散点图。
    """
    if results_df.empty:
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.text(0.5, 0.5, "无数据可用于绘图", horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        return fig

    # 根据语言获取标签
    _, internal_to_display, _ = _get_plot_column_map()
    x_display_name = _get_display_name_from_column(x_col, internal_to_display, lang)
    y_display_name = _get_display_name_from_column(y_col, internal_to_display, lang)

    x_label = x_label_cn if lang == 'cn' else x_display_name
    y_label = y_label_cn if lang == 'cn' else y_display_name
    title = title_cn if lang == 'cn' else f"Particle Parameter Distribution (KDE + Scatter)"

    plot_df = results_df.copy()

    # 过滤类别
    if class_filter and '类别名称' in plot_df.columns:
        plot_df = plot_df[plot_df['类别名称'].isin(class_filter)]

    # 过滤 NaN 值
    plot_df = plot_df.dropna(subset=[x_col, y_col])

    if plot_df.empty:
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.text(0.5, 0.5, "筛选后无数据可用于绘图", horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        return fig

    # 处理 X 轴的对数刻度
    x_data = plot_df[x_col]
    if log_scale_x:
        x_data = np.log10(x_data.where(x_data > 0))
        x_label = f"$log_{{10}}$({x_label})"

    fig, ax = plt.subplots(figsize=(8, 7))

    # KDE 填充 (Fill)
    sns.kdeplot(
        x=x_data,
        y=plot_df[y_col],
        fill=True,
        alpha=PLOT_CONFIG.KDE_ALPHA_FILL,
        cmap=sns.light_palette(color, as_cmap=True),
        ax=ax,
        linewidth=0,
    )
    # KDE 轮廓线 (Contour Lines)
    sns.kdeplot(
        x=x_data,
        y=plot_df[y_col],
        fill=False,
        levels=PLOT_CONFIG.KDE_LINE_LEVELS,
        color=color,
        linewidth=PLOT_CONFIG.KDE_LINE_WIDTH,
        linestyle='-',
        ax=ax
    )

    # 添加散点图 (Scatter Plot)
    if not plot_df.empty:  # 再次检查过滤后的数据是否为空
        sns.scatterplot(
            x=x_data,
            y=plot_df[y_col],
            ax=ax,
            color=color,
            alpha=PLOT_CONFIG.KDE_SCATTER_ALPHA,
            s=PLOT_CONFIG.KDE_SCATTER_SIZE,
            edgecolor=None
        )

    # 设置标题和轴标签
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)

    # 如果是 log_scale_x，设置对数刻度
    if log_scale_x:
        _setup_log_ticks(ax, x_data, label_font_size=12)

    # 优化显示
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(PLOT_CONFIG.AXIS_LINEWIDTH)

    # 统一应用中文字体，避免在某些环境下显示为方块
    _apply_chinese_font(ax)

    plt.tight_layout()
    return fig


def plot_kde_comparison_plot(
        results_df: pd.DataFrame,
        x_col: str,  # 例如 '等效直径 (微米)'
        y_col: str,  # 例如 '拟合椭圆轴比 (L/S)'
        class1_name: str,  # 对比类别1
        class2_name: str,  # 对比类别2
        color1: str = PLOT_CONFIG.KDE_DEFAULT_COLORS[0],
        color2: str = PLOT_CONFIG.KDE_DEFAULT_COLORS[1],
        title_cn: str = "颗粒形态学参数分布对比 (KDE)",
        x_label_cn: str = "粒径",
        y_label_cn: str = "形态学参数",
        log_scale_x: bool = True,  # X轴是否使用对数刻度
        lang: str = 'cn'
) -> plt.Figure:
    """
    生成两个选定类别颗粒的 2D 密度对比图 (KDE)。
    """
    if results_df.empty:
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.text(0.5, 0.5, "无数据可用于绘图", horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        return fig

    # 根据语言获取标签
    _, internal_to_display, _ = _get_plot_column_map()
    x_display_name = _get_display_name_from_column(x_col, internal_to_display, lang)
    y_display_name = _get_display_name_from_column(y_col, internal_to_display, lang)

    x_label = x_label_cn if lang == 'cn' else x_display_name
    y_label = y_label_cn if lang == 'cn' else y_display_name
    title = title_cn if lang == 'cn' else f"Particle Parameter Distribution Comparison (KDE)"

    plot_df = results_df.copy()

    df_class1 = plot_df[plot_df['类别名称'] == class1_name].dropna(subset=[x_col, y_col])
    df_class2 = plot_df[plot_df['类别名称'] == class2_name].dropna(subset=[x_col, y_col])

    if df_class1.empty and df_class2.empty:
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.text(0.5, 0.5, f"选定类别 '{class1_name}' 和 '{class2_name}' 无数据或数据不完整",
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        return fig

    fig, ax = plt.subplots(figsize=(8, 7))

    # 处理 X 轴的对数刻度
    x_data_combined = pd.concat([df_class1[x_col], df_class2[x_col]])
    if log_scale_x:
        x_data_combined = np.log10(x_data_combined.where(x_data_combined > 0))
        x_label = f"$log_{{10}}$({x_label})"

    # 绘制 Class 1
    if not df_class1.empty:
        x1_data = np.log10(df_class1[x_col].where(df_class1[x_col] > 0)) if log_scale_x else df_class1[x_col]
        sns.kdeplot(
            x=x1_data,
            y=df_class1[y_col],
            fill=True,
            alpha=PLOT_CONFIG.KDE_ALPHA_FILL,
            cmap=sns.light_palette(color1, as_cmap=True),
            ax=ax,
            linewidth=0,
        )
        sns.kdeplot(
            x=x1_data,
            y=df_class1[y_col],
            fill=False,
            levels=PLOT_CONFIG.KDE_LINE_LEVELS,
            color=color1,
            linewidth=PLOT_CONFIG.KDE_LINE_WIDTH,
            linestyle='-',
            ax=ax
        )

    # 绘制 Class 2
    if not df_class2.empty:
        x2_data = np.log10(df_class2[x_col].where(df_class2[x_col] > 0)) if log_scale_x else df_class2[x_col]
        sns.kdeplot(
            x=x2_data,
            y=df_class2[y_col],
            fill=True,
            alpha=PLOT_CONFIG.KDE_ALPHA_FILL,
            cmap=sns.light_palette(color2, as_cmap=True),
            ax=ax,
            linewidth=0,
        )
        sns.kdeplot(
            x=x2_data,
            y=df_class2[y_col],
            fill=False,
            levels=PLOT_CONFIG.KDE_LINE_LEVELS,
            color=color2,
            linewidth=PLOT_CONFIG.KDE_LINE_WIDTH,
            linestyle='--',
            ax=ax
        )

    # 设置标题和轴标签
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)

    # 如果是 log_scale_x，设置对数刻度
    if log_scale_x:
        _setup_log_ticks(ax, x_data_combined, label_font_size=12)

    # 修复图例 (手动创建 Line2D 代理)
    legend_elements = []
    if not df_class1.empty:
        legend_elements.append(
            Line2D([0], [0], marker='s', color='w', label=class1_name,
                   markerfacecolor=color1, markersize=10, alpha=PLOT_CONFIG.KDE_ALPHA_FILL, linestyle='-')
        )
    if not df_class2.empty:
        legend_elements.append(
            Line2D([0], [0], marker='s', color='w', label=class2_name,
                   markerfacecolor=color2, markersize=10, alpha=PLOT_CONFIG.KDE_ALPHA_FILL, linestyle='--')
        )

    if legend_elements:
        ax.legend(handles=legend_elements, title="类别分布" if lang == 'cn' else "Class Distribution", loc='best',
                  frameon=True)

    # 优化显示
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(PLOT_CONFIG.AXIS_LINEWIDTH)

    plt.tight_layout()
    return fig


def plot_simple_histogram(
        results_df: pd.DataFrame,
        col: str,  # 例如 '等效直径 (微米)'
        bins: int = 20,  # 直方图分箱数量
        title_cn: str = "参数分布直方图",
        x_label_cn: str = "参数值",
        y_label_cn: str = "频数 (数量)",
        color: str = '#1f77b4',  # 直方图颜色
        log_scale_x: bool = False,  # X轴是否使用对数刻度
        lang: str = 'cn'
) -> plt.Figure:
    """
    绘制单个数值列的直方图。
    """
    if results_df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "无数据可用于绘图", horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        return fig

    # 获取列的显示名称
    _, internal_to_display, _ = _get_plot_column_map()
    col_display_name = _get_display_name_from_column(col, internal_to_display, lang)

    x_label = x_label_cn if lang == 'cn' else col_display_name
    y_label = y_label_cn if lang == 'cn' else "Frequency (Count)"
    title = title_cn if lang == 'cn' else f"Distribution of {col_display_name}"

    plot_data = results_df.dropna(subset=[col])
    if plot_data.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "筛选后无数据可用于绘图", horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        return fig

    data_to_plot = plot_data[col]

    # 处理 Log X 轴
    if log_scale_x:
        data_to_plot = np.log10(data_to_plot.where(data_to_plot > 0))
        x_label = f"$log_{{10}}$({x_label})"
        # 重新计算 bins 以适应 Log 空间
        valid_data = data_to_plot.dropna()
        if not valid_data.empty:
            min_val = valid_data.min()
            max_val = valid_data.max()
            bins_log = np.linspace(min_val, max_val, bins + 1)
        else:
            bins_log = bins  # 默认为均匀分箱

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(data_to_plot, bins=bins_log, kde=True, ax=ax, color=color, edgecolor='black')
        _setup_log_ticks(ax, valid_data, label_font_size=12)  # 使用Log X轴刻度
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(data_to_plot, bins=bins, kde=True, ax=ax, color=color, edgecolor='black')

    # 优先直接在本函数中显式指定中文字体，避免 seaborn 的样式覆盖导致中文变方块
    if GLOBAL_CHINESE_FONT_PROP is not None:
        ax.set_title(title, fontsize=16, fontproperties=GLOBAL_CHINESE_FONT_PROP)
        ax.set_xlabel(x_label, fontsize=14, fontproperties=GLOBAL_CHINESE_FONT_PROP)
        ax.set_ylabel(y_label, fontsize=14, fontproperties=GLOBAL_CHINESE_FONT_PROP)
    else:
        ax.set_title(title, fontsize=16)
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)

    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(PLOT_CONFIG.AXIS_LINEWIDTH)

    # 坐标轴刻度标签也应用中文字体
    if GLOBAL_CHINESE_FONT_PROP is not None:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(GLOBAL_CHINESE_FONT_PROP)

    plt.tight_layout()
    return fig


def plot_boxplot_by_class(
        results_df: pd.DataFrame,
        value_col: str,  # 例如 '紧凑度/圆度'
        class_col: str = '类别名称',  # 分组的类别列，例如 '类别名称'
        title_cn: str = "按类别分组的参数箱线图",
        x_label_cn: str = "类别",
        y_label_cn: str = "参数值",
        lang: str = 'cn'
) -> plt.Figure:
    """
    绘制按类别分组的箱线图，用于比较不同类别在某个参数上的分布。
    """
    if results_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "无数据可用于绘图", horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        return fig

    _, internal_to_display, _ = _get_plot_column_map()
    value_display_name = _get_display_name_from_column(value_col, internal_to_display, lang)
    class_display_name = _get_display_name_from_column(class_col, internal_to_display, lang)

    # 为了避免极少数中文字符在某些环境下仍然显示为方块，
    # 这里对箱线图统一使用英文标签（数值名称的英文显示名）。
    en_value_name = internal_to_display.get(value_col, (value_display_name, value_display_name))[1]
    en_class_name = internal_to_display.get(class_col, (class_display_name, class_display_name))[1]

    x_label = en_class_name if en_class_name else "Class"
    y_label = en_value_name if en_value_name else value_display_name
    title = f"Box Plot of {en_value_name} by {en_class_name}"

    plot_df = results_df.dropna(subset=[value_col, class_col])

    if plot_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "筛选后无数据可用于绘图", horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        return fig

    fig, ax = plt.subplots(figsize=(10, 6))

    # 使用 Seaborn 绘制箱线图
    sns.boxplot(x=class_col, y=value_col, data=plot_df, ax=ax, palette='viridis')
    sns.swarmplot(x=class_col, y=value_col, data=plot_df, color=".25", ax=ax, size=3)  # 添加散点以显示原始数据点

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)

    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(PLOT_CONFIG.AXIS_LINEWIDTH)

    plt.tight_layout()
    return fig


# 在模块加载时确保 Matplotlib 配置已完成
setup_matplotlib_config()
