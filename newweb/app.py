import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import io
import pandas as pd
import math
import warnings
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot

# 忽略用户警告，避免 Streamlit 内部的一些提示信息干扰
warnings.filterwarnings("ignore", category=UserWarning)

# 从重构后的核心逻辑文件中导入函数和默认配置
try:
    from morphology_analyzer import (
        analyze_image_stream, DEFAULT_PARAMS, CHINESE_HEADERS,
        DEFAULT_CLASS_COLORS, DEFAULT_SIZE_COLOR_RULES, DEFAULT_SORTED_SIZE_COLOR_RULES,
        draw_overlay,  # 导入 draw_overlay 函数，因为它现在在 app.py 中调用
        _get_plot_column_map,  # 导入辅助函数以获取列名映射
        _get_display_name_from_column,  # 导入辅助函数
        plot_shape_frequency_histogram,
        plot_volume_distribution_curves,
        plot_kde_scatter_plot,
        plot_kde_comparison_plot,
        plot_simple_histogram,
        plot_boxplot_by_class,
        PLOT_CONFIG  # 导入绘图配置
    )
    from ultralytics import YOLO
except ImportError as e:
    st.error(f"无法导入核心分析模块: {e}")
    st.warning(
        "请确保 `morphology_analyzer.py` 文件存在且所有依赖已安装 (`pip install -r requirements.txt` 或 `pip install seaborn openpyxl`)。")
    st.stop()

# --- Streamlit 页面配置 ---
st.set_page_config(
    page_title="颗粒形态学分析与YOLOvv8实例分割",
    page_icon="🔬",
    layout="wide"
)

# --- Streamlit Session State 初始化 ---
# 通用参数初始化：遍历 DEFAULT_PARAMS，为每个参数在 session_state 中设置一个 ui_xxx 键
for key, value in DEFAULT_PARAMS.items():
    ui_key = f"ui_{key.lower()}"
    if ui_key not in st.session_state:
        # 特殊处理复杂类型如字典和列表，直接复制以防修改 DEFAULT_PARAMS
        if isinstance(value, dict):
            st.session_state[ui_key] = value.copy()
        elif isinstance(value, list):
            st.session_state[ui_key] = value[:]
        else:
            st.session_state[ui_key] = value

# 对于 custom_rules_count，它的初始值需要与 ui_custom_coloring_rules 的长度同步
# 并且要确保它在 ui_custom_coloring_rules 初始化之后
st.session_state.setdefault('custom_rules_count', len(st.session_state.ui_custom_coloring_rules))

# 初始化分析结果相关的 Session State 变量
st.session_state.setdefault('original_bgr', None)
st.session_state.setdefault('items_data', None)
st.session_state.setdefault('results_df', None)
st.session_state.setdefault('analysis_started', False)  # 标记分析是否已执行
st.session_state.setdefault('overlay_image_bgr', None)  # 存储用于GIF

# 获取列名映射 (用于绘图参数选择)
PLOT_OPTIONS_DISPLAY, INTERNAL_TO_DISPLAY_MAP, DISPLAY_TO_INTERNAL_MAP = _get_plot_column_map()

# ----------------------------------------------------
# Streamlit 核心 UI 结构
# ----------------------------------------------------
st.title("颗粒形态学分析与YOLOv8实例分割")
st.markdown("上传图像，使用YOLOv8进行实例分割，并计算颗粒形态学参数。所有参数均可交互式调整。")


# --- 缓存模型加载 ---
@st.cache_resource
def get_yolo_model(weights_path_str):
    """缓存 YOLO 模型加载，避免每次页面刷新都重新加载。"""
    if not os.path.exists(weights_path_str):
        st.error(f"模型权重文件未找到：{weights_path_str}。请检查路径是否正确。")
        st.stop()
    st.info(f"正在加载模型: {weights_path_str}")
    # 确保在 CPU 上运行
    model = YOLO(weights_path_str)
    st.success("模型加载完成！")
    return model


# --- 侧边栏：参数配置 ---
st.sidebar.header("⚙️ 推理与分析参数")

# --- 模型配置和加载 ---
st.sidebar.subheader("模型配置")

# 这里直接使用你当前项目下 v10.3.1.pt 的绝对路径，等同于你在侧边栏里手动输入“方式二”
default_weights_path = r"D:\subject\yolov8-main\yolov8-main\v10.3.1.pt"

weights_path_input = st.sidebar.text_input(
    "模型权重文件路径 (.pt 或 .onnx)",
    value=default_weights_path,
    help="YOLOv8模型权重文件的绝对路径。",
    key="ui_weights_path_input"
)
model = get_yolo_model(weights_path_input)

# --- 其他基础 YOLO 推理参数 ---
st.sidebar.subheader("YOLO 推理参数")
# 使用 st.session_state.ui_xxx 作为 value，确保 Streamlit 组件能正确读取/写入状态
st.sidebar.number_input("推理图像尺寸 (px)", min_value=320, max_value=2048,
                        value=st.session_state.ui_inference_img_size, step=32, key="ui_inference_img_size")
st.sidebar.slider("置信度阈值 (Conf Threshold)", 0.0, 1.0, st.session_state.ui_conf_thresh, 0.05, key="ui_conf_thresh")
st.sidebar.slider("NMS IoU 阈值 (IoU Threshold)", 0.0, 1.0, st.session_state.ui_iou_thresh, 0.05, key="ui_iou_thresh")
st.sidebar.number_input("最大检测目标数", min_value=1, max_value=5000, value=st.session_state.ui_max_detections,
                        step=100, key="ui_max_detections")
st.sidebar.checkbox("使用 Retina Masks (更精细掩码)", value=st.session_state.ui_retina_masks, key="ui_retina_masks")

# --- 物理尺寸转换 ---
st.sidebar.subheader("物理尺寸转换")
st.sidebar.number_input("每像素微米数 (UM_PER_PX)", min_value=0.001, max_value=10.0,
                        value=st.session_state.ui_um_per_px, step=0.001, format="%.4f", key="ui_um_per_px")
st.sidebar.markdown(f"**提示:** 1 像素 = {st.session_state.ui_um_per_px} 微米")

# --- 颗粒筛选 ---
st.sidebar.subheader("颗粒筛选")
st.sidebar.number_input("最小面积 (像素^2)", min_value=0.0,
                        value=st.session_state.ui_min_area_px if st.session_state.ui_min_area_px is not None else 0.1,
                        step=0.1, key="ui_min_area_px")
st.sidebar.number_input("最小圆度 (0-1)", min_value=0.0, max_value=1.0,
                        value=st.session_state.ui_min_circularity if st.session_state.ui_min_circularity is not None else 0.0,
                        step=0.01, key="ui_min_circularity")
st.sidebar.number_input("最大轴比 (L/S)", min_value=1.0,
                        value=st.session_state.ui_max_axis_ratio if st.session_state.ui_max_axis_ratio is not None else 5.0,
                        step=0.1, key="ui_max_axis_ratio")
st.sidebar.info("将 `0.0` 或 `1.0` 设为默认值以禁用筛选。")

# --- NMS 和边界处理 ---
st.sidebar.subheader("NMS 和边界处理")
st.sidebar.number_input("边界带宽 (px)", min_value=0, value=st.session_state.ui_border_band, key="ui_border_band")
st.sidebar.slider("Mask NMS IoU 阈值", 0.0, 1.0, st.session_state.ui_nms_iou_thresh, 0.05, key="ui_nms_iou_thresh")
st.sidebar.slider("Box IoU 预过滤阈值", 0.0, 1.0, st.session_state.ui_box_iou_pre_thresh, 0.01,
                  key="ui_box_iou_pre_thresh")

# --- 形态学计算配置 ---
st.sidebar.subheader("形态学计算配置")
st.sidebar.number_input("RDP 简化阈值", min_value=0.1, value=st.session_state.ui_rdp_epsilon, step=0.1,
                        key="ui_rdp_epsilon")
st.sidebar.number_input("凹陷深度阈值", min_value=0.1, value=st.session_state.ui_depth_threshold, step=0.1,
                        key="ui_depth_threshold")

# --- Tiling 配置 ---
st.sidebar.subheader("Tiling 推理 (处理大图)")
st.sidebar.number_input("Tiling 行数", min_value=1, value=st.session_state.ui_tile_rows, key="ui_tile_rows")
st.sidebar.number_input("Tiling 列数", min_value=1, value=st.session_state.ui_tile_cols, key="ui_tile_cols")
st.sidebar.number_input("Tiling 重叠像素", min_value=0, value=st.session_state.ui_overlap_px, key="ui_overlap_px")
st.sidebar.info("行数或列数 > 1 启用 Tiling。")

# --- 并行计算配置 ---
st.sidebar.subheader("性能优化")
st.sidebar.number_input("并行计算线程数 (NUM_WORKERS)", min_value=1, max_value=os.cpu_count() or 1,
                        value=st.session_state.ui_num_workers, step=1, key="ui_num_workers",
                        help="设置用于形态学计算的CPU线程数。建议设置为CPU核心数。")

# --- 形态学参数选择 (优化速度) ---
st.sidebar.subheader("形态学参数计算选项")
st.sidebar.markdown("取消勾选不计算的参数可加快处理速度。")
st.sidebar.checkbox("计算：尺寸、形状、圆度", value=st.session_state.ui_calc_shape_params, key="ui_calc_shape_params")
st.sidebar.checkbox("计算：边界特征", value=st.session_state.ui_calc_boundary_features, key="ui_calc_boundary_features")
st.sidebar.checkbox("计算：傅里叶描述子", value=st.session_state.ui_calc_fourier_descriptors,
                    key="ui_calc_fourier_descriptors")
st.sidebar.checkbox("计算：内部纹理特征", value=st.session_state.ui_calc_texture_features,
                    key="ui_calc_texture_features")
st.sidebar.checkbox("计算：上下文特征", value=st.session_state.ui_calc_context_features, key="ui_calc_context_features")

# --- 可视化配置 ---
st.sidebar.subheader("可视化配置")
st.sidebar.slider("填充不透明度", 0.0, 1.0, st.session_state.ui_fill_alpha, 0.05, key="ui_fill_alpha")
text_color_bgr_default = st.session_state.ui_text_color  # 默认是BGR
text_color_hex_default = f"#{'%02x%02x%02x' % (text_color_bgr_default[2], text_color_bgr_default[1], text_color_bgr_default[0])}"
text_color_hex = st.sidebar.color_picker("文本颜色", text_color_hex_default, key="ui_text_color_hex")
outline_color_bgr_default = DEFAULT_PARAMS["OUTLINE_COLOR"]  # 假设你已在 morphology_analyzer.py 的 DEFAULT_PARAMS 中定义
outline_color_hex_default = f"#{'%02x%02x%02x' % (outline_color_bgr_default[2], outline_color_bgr_default[1], outline_color_bgr_default[0])}"
outline_color_hex = st.sidebar.color_picker("轮廓线颜色", outline_color_hex_default, key="ui_outline_color_hex")
outline_color_rgb = tuple(int(outline_color_hex[idx:idx + 2], 16) for idx in (1, 3, 5))
outline_color_bgr = (outline_color_rgb[2], outline_color_rgb[1], outline_color_rgb[0])  # BGR 格式

# 轮廓不透明度 (简化，直接用滑块)
st.sidebar.slider("轮廓线不透明度", 0.0, 1.0, 1.0, 0.05, key="ui_outline_alpha",
                  help="轮廓线的不透明度。0.0 完全透明，1.0 完全不透明。")

text_color_rgb = tuple(int(text_color_hex[idx:idx + 2], 16) for idx in (1, 3, 5))
text_color_bgr = (text_color_rgb[2], text_color_rgb[1], text_color_rgb[0])

st.sidebar.checkbox("显示颗粒ID", value=st.session_state.ui_show_particle_id, key="ui_show_particle_id")
st.sidebar.checkbox("只显示轮廓", value=st.session_state.ui_show_only_outline, key="ui_show_only_outline")
st.sidebar.number_input("轮廓粗细 (像素)", min_value=0, value=st.session_state.ui_outline_thickness,
                        key="ui_outline_thickness")
st.sidebar.slider("背景压暗因子", 0.0, 1.0, st.session_state.ui_background_dim_factor, 0.05,
                  key="ui_background_dim_factor")
st.sidebar.info("设为 1.0 不压暗。")

# --- 配色模式控制 ---
st.sidebar.subheader("配色模式")
st.sidebar.radio(
    "选择配色模式",
    ('SIZE', 'CLASS'),
    index=0 if st.session_state.ui_coloring_mode == 'SIZE' else 1,
    help="SIZE: 根据粒径着色；CLASS: 根据类别着色。",
    key="ui_coloring_mode"
)

st.sidebar.checkbox(
    "启用自定义着色规则 (优先级最高)",
    value=st.session_state.ui_enable_custom_coloring,
    key="ui_enable_custom_coloring"
)

# --- 自定义着色规则 UI (如果启用) ---
if st.session_state.ui_enable_custom_coloring:
    st.sidebar.markdown("---")
    st.sidebar.markdown("##### 自定义着色规则 (按顺序匹配)")

    while len(st.session_state.ui_custom_coloring_rules) < st.session_state.custom_rules_count:
        st.session_state.ui_custom_coloring_rules.append(
            {"param": "等效直径 (微米)", "min": 0.0, "max": float('inf'),
             "color": (255, 0, 255)})  # Default value for new rule
    while len(st.session_state.ui_custom_coloring_rules) > st.session_state.custom_rules_count:
        st.session_state.ui_custom_coloring_rules.pop()

    # 获取所有可用于自定义着色的参数选项（即所有形态学参数的中文显示名）
    numerical_cols = [col for col in CHINESE_HEADERS if
                      'ID' not in col and '名称' not in col and '图片名称' not in col and '接触图像边缘' not in col]
    param_options = [_get_display_name_from_column(col, INTERNAL_TO_DISPLAY_MAP) for col in numerical_cols]

    for i in range(st.session_state.custom_rules_count):
        st.sidebar.markdown(f"**规则 {i + 1}:**")
        current_rule_data = st.session_state.ui_custom_coloring_rules[i]

        col1_rule, col2_rule = st.sidebar.columns(2)
        with col1_rule:
            # 使用下拉框选择参数
            selected_param_display_name = st.selectbox(
                f"选择参数 {i + 1}",
                options=param_options,
                index=param_options.index(
                    _get_display_name_from_column(current_rule_data.get('param', '等效直径 (微米)'),
                                                  INTERNAL_TO_DISPLAY_MAP)) if _get_display_name_from_column(
                    current_rule_data.get('param', '等效直径 (微米)'), INTERNAL_TO_DISPLAY_MAP) in param_options else 0,
                key=f"param_select_{i}"
            )
            # 将显示名转换回内部列名
            param_internal_name = DISPLAY_TO_INTERNAL_MAP.get(selected_param_display_name, '等效直径 (微米)')

        with col2_rule:
            rule_color_bgr_val = current_rule_data.get('color', (255, 0, 255))
            rule_color_hex_default = f"#{'%02x%02x%02x' % (rule_color_bgr_val[2], rule_color_bgr_val[1], rule_color_bgr_val[0])}"
            rule_color_hex = st.color_picker(f"颜色 {i + 1}", value=rule_color_hex_default, key=f"color_{i}")
            rule_color_rgb = tuple(int(rule_color_hex[idx:idx + 2], 16) for idx in (1, 3, 5))
            rule_color_bgr = (rule_color_rgb[2], rule_color_rgb[1], rule_color_rgb[0])

        col3_rule, col4_rule = st.sidebar.columns(2)
        with col3_rule:
            min_val = st.number_input(f"最小阈值 {i + 1}", value=current_rule_data.get('min', 0.0), key=f"min_{i}",
                                      format="%.2f")
        with col4_rule:
            max_val = st.number_input(f"最大阈值 {i + 1}", value=current_rule_data.get('max', float('inf')),
                                      key=f"max_{i}", format="%.2f")

        st.session_state.ui_custom_coloring_rules[i] = {
            "param": param_internal_name, "min": min_val, "max": max_val, "color": rule_color_bgr
        }

    col_add_rule, col_remove_rule = st.sidebar.columns(2)
    with col_add_rule:
        if st.button("➕ 添加规则", key="add_rule_btn"):
            st.session_state.ui_custom_coloring_rules.append(
                {"param": "等效直径 (微米)", "min": 0.0, "max": float('inf'), "color": (255, 0, 255)})
            st.session_state.custom_rules_count += 1
            st.experimental_rerun()
    with col_remove_rule:
        if st.button("➖ 移除最后一个规则", key="remove_rule_btn") and st.session_state.custom_rules_count > 0:
            st.session_state.ui_custom_coloring_rules.pop()
            st.session_state.custom_rules_count -= 1
            st.experimental_rerun()

    if not st.session_state.ui_custom_coloring_rules and st.session_state.ui_enable_custom_coloring:
        st.sidebar.warning("请添加至少一个自定义着色规则。")

# --- 类别颜色 (可编辑) ---
st.sidebar.subheader("类别颜色")
st.sidebar.markdown("可自定义YOLO识别的每个类别的颜色。")
current_class_colors = {}
for class_id, default_bgr in DEFAULT_CLASS_COLORS.items():
    hex_default = f"#{'%02x%02x%02x' % (default_bgr[2], default_bgr[1], default_bgr[0])}"
    hex_color = st.sidebar.color_picker(f"类别 {class_id} 颜色", value=hex_default, key=f"ui_class_color_{class_id}")
    rgb_tuple = tuple(int(hex_color[idx:idx + 2], 16) for idx in (1, 3, 5))
    current_class_colors[class_id] = (rgb_tuple[2], rgb_tuple[1], rgb_tuple[0])

# --- 粒径着色规则 (仅展示，此部分较为复杂，暂时不提供UI编辑) ---
st.sidebar.subheader("粒径着色规则")
st.sidebar.markdown("（在 `morphology_analyzer.py` 中预定义，此处仅展示）")
size_color_df = pd.DataFrame([
    {"阈值 (um)": k, "颜色 (BGR)": str(v)} for k, v in DEFAULT_SORTED_SIZE_COLOR_RULES
])
# Streamlit 新版推荐使用 width 参数替代 use_container_width
st.sidebar.dataframe(size_color_df, width="stretch")


# ----------------------------------------------------
# 获取所有 Streamlit UI 参数并打包成字典
# ----------------------------------------------------
def get_params_from_ui():
    params = DEFAULT_PARAMS.copy()

    # --- 从 st.session_state 收集参数 ---
    params["IMG_SIZE"] = st.session_state.ui_inference_img_size
    params["INFERENCE_IMG_SIZE"] = st.session_state.ui_inference_img_size
    params["CONF_THRESH"] = st.session_state.ui_conf_thresh
    params["IOU_THRESH"] = st.session_state.ui_iou_thresh
    params["RETINA_MASKS"] = st.session_state.ui_retina_masks
    params["MAX_DETECTIONS"] = st.session_state.ui_max_detections

    params["UM_PER_PX"] = st.session_state.ui_um_per_px

    # 筛选参数需要处理 None 的情况
    params["MIN_AREA_PX"] = st.session_state.ui_min_area_px if st.session_state.ui_min_area_px is not None else \
        DEFAULT_PARAMS["MIN_AREA_PX"]
    params[
        "MIN_CIRCULARITY"] = st.session_state.ui_min_circularity if st.session_state.ui_min_circularity is not None else \
        DEFAULT_PARAMS["MIN_CIRCULARITY"]
    params["MAX_AXIS_RATIO"] = st.session_state.ui_max_axis_ratio if st.session_state.ui_max_axis_ratio is not None else \
        DEFAULT_PARAMS["MAX_AXIS_RATIO"]

    params["BORDER_BAND"] = st.session_state.ui_border_band
    params["NMS_IOU_THRESH"] = st.session_state.ui_nms_iou_thresh
    params["BOX_IOU_PRE_THRESH"] = st.session_state.ui_box_iou_pre_thresh

    params["RDP_EPSILON"] = st.session_state.ui_rdp_epsilon
    params["DEPTH_THRESHOLD"] = st.session_state.ui_depth_threshold

    params["TILE_ROWS"] = st.session_state.ui_tile_rows
    params["TILE_COLS"] = st.session_state.ui_tile_cols
    params["OVERLAP_PX"] = st.session_state.ui_overlap_px

    params["NUM_WORKERS"] = st.session_state.ui_num_workers

    # 形态学计算选项
    params["CALC_SHAPE_PARAMS"] = st.session_state.ui_calc_shape_params
    params["CALC_BOUNDARY_FEATURES"] = st.session_state.ui_calc_boundary_features
    params["CALC_FOURIER_DESCRIPTORS"] = st.session_state.ui_calc_fourier_descriptors
    params["CALC_TEXTURE_FEATURES"] = st.session_state.ui_calc_texture_features
    params["CALC_CONTEXT_FEATURES"] = st.session_state.ui_calc_context_features

    params["FILL_ALPHA"] = st.session_state.ui_fill_alpha
    params["TEXT_COLOR"] = text_color_bgr  # 已经是 BGR 格式
    params["UI_OUTLINE_COLOR"] = outline_color_bgr  # <--- 添加这行
    params["UI_OUTLINE_ALPHA"] = st.session_state.ui_outline_alpha  # <--- 添加这行
    params["SHOW_PARTICLE_ID"] = st.session_state.ui_show_particle_id
    params["SHOW_ONLY_OUTLINE"] = st.session_state.ui_show_only_outline
    params["OUTLINE_THICKNESS"] = st.session_state.ui_outline_thickness
    params["BACKGROUND_DIM_FACTOR"] = st.session_state.ui_background_dim_factor
    params["COLORING_MODE"] = st.session_state.ui_coloring_mode

    params["ENABLE_CUSTOM_COLORING"] = st.session_state.ui_enable_custom_coloring and bool(
        st.session_state.ui_custom_coloring_rules)
    params["CUSTOM_COLORING_RULES"] = st.session_state.ui_custom_coloring_rules[:]  # 传递拷贝以防修改

    params["CLASS_COLORS"] = current_class_colors  # 从 UI 获取的可编辑类别颜色
    params["SORTED_SIZE_COLOR_RULES"] = DEFAULT_SORTED_SIZE_COLOR_RULES  # 粒径规则 (当前不可编辑)

    # --- 重要的常量和配置 (直接来自 morphology_analyzer) ---
    params["CHINESE_HEADERS"] = CHINESE_HEADERS

    return params


# --- GIF 生成函数 ---
def create_coloring_gif(img_gray_pil, img_color_pil, steps=10, duration_ms=100, loop=0):
    """
    加载两张 PIL 图片，通过线性插值生成过渡帧，并返回 GIF 字节流。
    """
    if img_gray_pil.size != img_color_pil.size:
        st.error("错误：原始图和叠加图尺寸不一致，无法生成 GIF。")
        return None

    # 确保两张图片都以 RGB 模式加载，以便进行像素级的混合
    img_gray = img_gray_pil.convert("RGB")
    img_color = img_color_pil.convert("RGB")

    frames = []
    # 灰度图到彩色图的渐变帧
    for i in range(steps):
        alpha = i / (steps - 1)
        blended_img = Image.blend(img_gray, img_color, alpha)
        frames.append(blended_img)
    # 再加几帧从彩色图到灰度图的渐变，形成完整的淡入淡出循环
    for i in range(steps - 2, 0, -1):  # 从倒数第二帧到第二帧
        alpha = i / (steps - 1)
        blended_img = Image.blend(img_gray, img_color, alpha)
        frames.append(blended_img)

    # 确保至少有两帧，否则 Image.save 可能会报错
    if len(frames) < 2:
        frames.append(img_gray)
        frames.append(img_color)

    # 将 PIL 图像列表保存为 BytesIO 对象
    gif_bytes_io = io.BytesIO()
    frames[0].save(
        gif_bytes_io,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=loop,
        optimize=False
    )
    gif_bytes_io.seek(0)  # 将文件指针重置到开头
    return gif_bytes_io.getvalue()


# ----------------------------------------------------
# 主内容区：文件上传和核心流程
# ----------------------------------------------------
uploaded_file = st.file_uploader("选择一张图片进行分割和分析...", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])

if uploaded_file is not None:
    # --- 构建完整的参数字典，传递给核心逻辑 ---
    analysis_params = get_params_from_ui()

    # 显示原始图片
    st.subheader("原始图片")
    original_image_pil = Image.open(uploaded_file).convert("RGB")  # 确保以 RGB 模式加载
    st.image(original_image_pil, caption=uploaded_file.name, width="stretch")

    # --- 按钮触发推理和计算 ---
    if 'analysis_triggered' not in st.session_state:
        st.session_state.analysis_triggered = False

    if st.button("🚀 开始分析 (YOLO 推理与形态学计算)", key="start_analysis_btn"):
        st.session_state.analysis_triggered = True


        @st.cache_data(ttl=3600, show_spinner="模型正在努力分割并分析颗粒 (CPU 推理中，请耐心等待)...")
        def cached_analysis(image_bytes, _model, params, filename):
            image_stream = io.BytesIO(image_bytes)
            original_img_pil_cache = Image.open(image_stream)
            img_np = np.array(original_img_pil_cache)

            if img_np.ndim == 2:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            elif img_np.ndim == 3 and img_np.shape[2] == 4:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
            elif not (img_np.ndim == 3 and img_np.shape[2] == 3):
                raise ValueError("图片格式异常，既非单通道也非三通道/四通道图片。请检查上传文件。")

            img_np = img_np.astype(np.uint8)
            img_bgr_for_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            original_bgr, items_data, results_df = analyze_image_stream(
                img_bgr_for_cv2, _model, params, filename
            )
            return original_bgr, items_data, results_df


        try:
            st.session_state.original_bgr, st.session_state.items_data, st.session_state.results_df = cached_analysis(
                uploaded_file.getvalue(),
                model,
                analysis_params,
                uploaded_file.name
            )
            st.session_state.analysis_started = True
            st.success("分析与计算完成！现在可以调整可视化参数。")

        except Exception as e:
            st.error(f"分析过程中发生错误：{e}")
            st.write("请检查参数设置或模型文件是否正确。")
            st.exception(e)

    # --- 解耦渲染逻辑：如果已经分析过，则实时更新叠加图 ---
    if st.session_state.get('analysis_started', False) and st.session_state.items_data is not None:
        st.subheader("分析结果与可视化")

        # 实时渲染叠加图
        overlay_image_bgr = draw_overlay(
            st.session_state.original_bgr.copy(),
            st.session_state.items_data,
            analysis_params
        )
        st.session_state.overlay_image_bgr = overlay_image_bgr  # 存储用于GIF

        overlay_image_rgb = cv2.cvtColor(overlay_image_bgr, cv2.COLOR_BGR2RGB)
        st.image(overlay_image_rgb, caption="分割与着色结果", width="stretch")

        # --- GIF 生成 UI ---
        gif_expander = st.expander("动图演示 (GIF)")
        with gif_expander:
            st.markdown("将原始图片与分割结果生成渐变 GIF 动图。")
            gif_steps = st.slider("渐变步数", 2, 50, 10, key="gif_steps")
            gif_duration = st.slider("每帧时长 (毫秒)", 50, 2000, 100, key="gif_duration")
            gif_loop = st.checkbox("循环播放", value=True, key="gif_loop")

            if st.button("生成 GIF 动图", key="generate_gif_btn"):
                with st.spinner("正在生成 GIF 动图..."):
                    # 转换 BGR np.array 到 PIL Image
                    original_pil = Image.fromarray(cv2.cvtColor(st.session_state.original_bgr, cv2.COLOR_BGR2RGB))
                    overlay_pil = Image.fromarray(cv2.cvtColor(st.session_state.overlay_image_bgr, cv2.COLOR_BGR2RGB))

                    gif_bytes = create_coloring_gif(
                        original_pil,
                        overlay_pil,
                        steps=gif_steps,
                        duration_ms=gif_duration,
                        loop=0 if gif_loop else 1,
                    )
                    if gif_bytes:
                        st.image(gif_bytes, caption="渐变效果 GIF", width="stretch")
                        st.download_button(
                            label="下载 GIF",
                            data=gif_bytes,
                            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transition.gif",
                            mime="image/gif",
                        )
                    else:
                        st.error("GIF 生成失败。")

        st.subheader(f"形态学测量数据 ({len(st.session_state.results_df)} 个颗粒)")
        st.dataframe(st.session_state.results_df, width="stretch")

        # 提供下载按钮
        col_csv, col_xlsx = st.columns(2)
        with col_csv:
            csv_data = st.session_state.results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="下载形态学数据 (CSV)",
                data=csv_data,
                file_name=f"{os.path.splitext(uploaded_file.name)[0]}_morphology_results.csv",
                mime="text/csv"
            )
        with col_xlsx:
            excel_data = io.BytesIO()
            st.session_state.results_df.to_excel(excel_data, index=False, engine='xlsxwriter')
            st.download_button(
                label="下载形态学数据 (Excel)",
                data=excel_data.getvalue(),
                file_name=f"{os.path.splitext(uploaded_file.name)[0]}_morphology_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        # --- 📈 数据可视化区域 ---
        st.markdown("---")
        st.header("📈 数据可视化")

        # 预先检查数据可用性
        if st.session_state.results_df.empty:
            st.warning("无颗粒数据，无法生成图表。请先执行分析。")
        else:
            # 1. 筛选出有效的数值型参数选项
            numerical_param_options = []

            # --- 关键修改：使用 PLOT_OPTIONS_DISPLAY 的键作为筛选列表 ---
            # PLOT_OPTIONS_DISPLAY 是 '中文 (英文)' 格式的列表
            # 我们只保留那些对应的内部列名存在于 results_df.columns 中的选项
            for display_name in PLOT_OPTIONS_DISPLAY:
                internal_col_name = DISPLAY_TO_INTERNAL_MAP.get(display_name)

                # 检查列是否存在于当前 DataFrame 中
                if internal_col_name in st.session_state.results_df.columns:
                    # 检查是否是数值型 (注意：某些非数值列如 '接触图像边缘' 已经被过滤)
                    if st.session_state.results_df[internal_col_name].dtype in ['float64', 'int64', 'float32', 'int32']:
                        numerical_param_options.append(display_name)

            if not numerical_param_options:
                st.warning("无可用数值型形态学参数，无法生成图表。请确保形态学参数已计算。")
                plot_type = "请选择图表类型"
            else:
                # 2. 获取实际存在的类别名称
                actual_class_names = []
                if '类别名称' in st.session_state.results_df.columns:
                    actual_class_names = st.session_state.results_df['类别名称'].dropna().unique().tolist()

                if not actual_class_names:
                    actual_class_names = ["无实际类别"]

                plot_type = st.selectbox(
                    "选择图表类型",
                    [
                        "请选择图表类型",
                        "📊 参数直方图",
                        "📦 按类别箱线图",
                        "📈 粒径分布曲线 (PDF/CDF)",
                        "🔬 粒度段颗粒类别占比",
                        "🔥 KDE热力图 (单类别)",
                        "🆚 KDE热力图 (多类别对比)",
                    ],
                    key="plot_type_selector"
                )

            # --- 图表生成逻辑 ---
            current_plot_figure = None

            if plot_type == "📊 参数直方图":
                with st.form("histogram_form"):
                    st.subheader("📊 参数直方图配置")
                    col_hist_1, col_hist_2 = st.columns(2)
                    with col_hist_1:
                        hist_param = st.selectbox("选择参数", options=numerical_param_options, key="hist_param_form")
                    with col_hist_2:
                        hist_bins = st.number_input("分箱数量", min_value=5, max_value=100, value=20, step=5,
                                                    key="hist_bins_form")
                        hist_log_x = st.checkbox("X轴对数刻度", value=False, key="hist_log_x_form")

                    # Submit button for the form
                    submitted = st.form_submit_button("生成直方图")

                    if submitted:
                        # 查找内部列名
                        internal_hist_param = DISPLAY_TO_INTERNAL_MAP.get(hist_param)

                        if internal_hist_param and internal_hist_param in st.session_state.results_df.columns:
                            current_plot_figure = plot_simple_histogram(
                                results_df=st.session_state.results_df,
                                col=internal_hist_param,
                                bins=hist_bins,
                                log_scale_x=hist_log_x,
                                title_cn=f"{hist_param} 分布直方图",
                                x_label_cn=f"{hist_param}",
                                lang='cn'
                            )
                        else:
                            st.error(
                                f"无法生成图表：数据中未找到参数 '{internal_hist_param}'。请检查参数是否在分析时被计算。")

            elif plot_type == "📦 按类别箱线图":
                with st.form("boxplot_form"):
                    st.subheader("📦 按类别箱线图配置")
                    if '类别名称' not in st.session_state.results_df.columns or actual_class_names == ["无实际类别"]:
                        st.warning("数据中未包含有效的 '类别名称' 列，无法生成按类别箱线图。")
                        submitted = st.form_submit_button("生成箱线图", disabled=True)
                    else:
                        box_value_param = st.selectbox("选择参数", options=numerical_param_options,
                                                       key="box_value_param_form")
                        submitted = st.form_submit_button("生成箱线图")

                        if submitted:
                            internal_box_value_param = DISPLAY_TO_INTERNAL_MAP.get(box_value_param)
                            if internal_box_value_param and internal_box_value_param in st.session_state.results_df.columns:
                                current_plot_figure = plot_boxplot_by_class(
                                    results_df=st.session_state.results_df,
                                    value_col=internal_box_value_param,
                                    class_col='类别名称',
                                    title_cn=f"按类别分组的 {box_value_param} 箱线图",
                                    y_label_cn=f"{box_value_param}",
                                    lang='cn'
                                )
                            else:
                                st.error(
                                    f"无法生成图表：数据中未找到参数 '{internal_box_value_param}'。请检查参数是否在分析时被计算。")

            elif plot_type == "📈 粒径分布曲线 (PDF/CDF)":
                with st.form("psd_form"):
                    st.subheader("📈 粒径分布曲线 (PDF/CDF) 配置")
                    psd_diameter_col_display = _get_display_name_from_column('等效直径 (微米)', INTERNAL_TO_DISPLAY_MAP)

                    if psd_diameter_col_display not in numerical_param_options:
                        st.error("粒径参数 '等效直径 (微米)' 缺失或未计算，无法生成 PDF/CDF 曲线。")
                        submitted = st.form_submit_button("生成分布曲线", disabled=True)
                    else:
                        col_psd_1, col_psd_2 = st.columns(2)
                        with col_psd_1:
                            psd_diameter_col = st.selectbox(
                                "选择粒径参数",
                                options=[psd_diameter_col_display],
                                key="psd_diameter_col_form"
                            )
                        with col_psd_2:
                            psd_bins = st.number_input("分布分箱数量", min_value=10, max_value=200, value=50, step=10,
                                                       key="psd_bins_form")

                        psd_log_x = st.checkbox("X轴对数刻度", value=True, key="psd_log_x_form")

                        psd_comparison_col_options = ["不对比"]
                        if '类别名称' in st.session_state.results_df.columns and actual_class_names != ["无实际类别"]:
                            psd_comparison_col_options += [c for c in actual_class_names if c != "无实际类别"]

                        psd_comparison_col = st.selectbox(
                            "按类别对比分布",
                            options=psd_comparison_col_options,
                            key="psd_comparison_col_form"
                        )

                        submitted = st.form_submit_button("生成分布曲线")

                        if submitted:
                            internal_psd_diameter_col = DISPLAY_TO_INTERNAL_MAP.get(psd_diameter_col)
                            internal_psd_comparison_col = '类别名称' if psd_comparison_col != "不对比" else None

                            if internal_psd_diameter_col in st.session_state.results_df.columns:
                                current_plot_figure = plot_volume_distribution_curves(
                                    results_df=st.session_state.results_df,
                                    diameter_col=internal_psd_diameter_col,
                                    bins_count=psd_bins,
                                    log_scale_x=psd_log_x,
                                    comparison_col=internal_psd_comparison_col,
                                    title_cn=f"粒径分布曲线",
                                    x_label_cn="粒径",
                                    lang='cn'
                                )
                            else:
                                st.error(f"无法生成图表：数据中未找到粒径参数 '{internal_psd_diameter_col}'。")

            elif plot_type == "🔬 粒度段颗粒类别占比":
                with st.form("shape_frequency_form"):
                    st.subheader("🔬 粒度段颗粒类别占比配置")
                    if '类别名称' not in st.session_state.results_df.columns or actual_class_names == ["无实际类别"]:
                        st.warning("数据中未包含有效的 '类别名称' 列，无法生成粒度段颗粒类别占比图。")
                        submitted = st.form_submit_button("生成类别占比图", disabled=True)
                    else:
                        freq_size_col_display = _get_display_name_from_column('等效直径 (微米)',
                                                                              INTERNAL_TO_DISPLAY_MAP)
                        freq_shape_col_display = _get_display_name_from_column('类别名称', INTERNAL_TO_DISPLAY_MAP)

                        if freq_size_col_display not in numerical_param_options:
                            st.error("粒径参数 '等效直径 (微米)' 缺失或未计算，无法生成占比图。")
                            submitted = st.form_submit_button("生成类别占比图", disabled=True)
                        else:
                            col_freq_1, col_freq_2 = st.columns(2)
                            with col_freq_1:
                                freq_size_col = st.selectbox(
                                    "选择粒径参数",
                                    options=[freq_size_col_display],
                                    key="freq_size_col_form"
                                )
                            with col_freq_2:
                                freq_shape_col = st.selectbox(
                                    "选择分类参数",
                                    options=[freq_shape_col_display],
                                    key="freq_shape_col_form"
                                )

                            freq_target_shapes_options = [c for c in actual_class_names if c != "无实际类别"]
                            freq_target_shapes = st.multiselect(
                                "选择要统计的类别",
                                options=freq_target_shapes_options,
                                default=freq_target_shapes_options if freq_target_shapes_options else [],  # 默认选择所有实际类别
                                key="freq_target_shapes_form"
                            )
                            col_freq_3, col_freq_4, col_freq_5 = st.columns(3)
                            with col_freq_3:
                                freq_min_size = st.number_input("最小粒径", value=0.0, key="freq_min_size_form",
                                                                format="%.2f")
                            with col_freq_4:
                                max_diameter_val = 100.0
                                if not st.session_state.results_df.empty and '等效直径 (微米)' in st.session_state.results_df.columns:
                                    max_diameter_val = st.session_state.results_df['等效直径 (微米)'].max()
                                    if pd.isna(max_diameter_val): max_diameter_val = 100.0
                                freq_max_size = st.number_input("最大粒径", value=float(max_diameter_val),
                                                                key="freq_max_size_form", format="%.2f")
                            with col_freq_5:
                                freq_bin_width = st.number_input("粒径分箱宽度", value=5.0, min_value=0.1, step=0.1,
                                                                 key="freq_bin_width_form", format="%.2f")

                            freq_y_max_limit = st.number_input("Y轴最大值", value=max(200, int(len(
                                st.session_state.results_df) * 0.1)) if not st.session_state.results_df.empty else 200,
                                                               min_value=1, key="freq_y_max_limit_form")
                            freq_custom_y_ticks_str = st.text_input("Y轴自定义刻度 (逗号分隔)", value="0,100,200,400",
                                                                    key="freq_custom_y_ticks_str_form")

                            submitted = st.form_submit_button("生成类别占比图")

                            if submitted:
                                try:
                                    freq_custom_y_ticks = [float(x.strip()) for x in freq_custom_y_ticks_str.split(',')
                                                           if x.strip()]
                                except ValueError:
                                    st.warning("Y轴刻度格式不正确，请使用逗号分隔的数字。")
                                    freq_custom_y_ticks = []

                                internal_freq_size_col = DISPLAY_TO_INTERNAL_MAP.get(freq_size_col)
                                internal_freq_shape_col = DISPLAY_TO_INTERNAL_MAP.get(freq_shape_col)

                                if internal_freq_size_col in st.session_state.results_df.columns and \
                                        internal_freq_shape_col in st.session_state.results_df.columns and \
                                        freq_target_shapes:  # 确保有选择类别
                                    current_plot_figure = plot_shape_frequency_histogram(
                                        results_df=st.session_state.results_df,
                                        size_col=internal_freq_size_col,
                                        shape_col=internal_freq_shape_col,
                                        target_shapes=freq_target_shapes,
                                        min_size=freq_min_size,
                                        max_size=freq_max_size,
                                        bin_width=freq_bin_width,
                                        custom_y_ticks=freq_custom_y_ticks,
                                        y_max_limit=freq_y_max_limit,
                                        lang='cn'
                                    )
                                else:
                                    st.warning("请选择有效的粒径参数、分类参数和至少一个类别。")

            elif plot_type == "🔥 KDE热力图 (单类别)":
                with st.form("kde_single_form"):
                    st.subheader("🔥 KDE热力图 (单类别) 配置")
                    if '类别名称' not in st.session_state.results_df.columns or actual_class_names == ["无实际类别"]:
                        st.warning("数据中未包含有效的 '类别名称' 列，无法按类别绘制热力图。")
                        submitted = st.form_submit_button("生成单类别热力图", disabled=True)
                    else:
                        col_kde_single_1, col_kde_single_2 = st.columns(2)
                        with col_kde_single_1:
                            kde_single_x = st.selectbox("X轴参数", options=numerical_param_options,
                                                        key="kde_single_x_form",
                                                        index=numerical_param_options.index(
                                                            _get_display_name_from_column('等效直径 (微米)',
                                                                                          INTERNAL_TO_DISPLAY_MAP)) if _get_display_name_from_column(
                                                            '等效直径 (微米)',
                                                            INTERNAL_TO_DISPLAY_MAP) in numerical_param_options else 0)
                        with col_kde_single_2:
                            kde_single_y = st.selectbox("Y轴参数", options=numerical_param_options,
                                                        key="kde_single_y_form",
                                                        index=numerical_param_options.index(
                                                            _get_display_name_from_column('拟合椭圆轴比 (L/S)',
                                                                                          INTERNAL_TO_DISPLAY_MAP)) if _get_display_name_from_column(
                                                            '拟合椭圆轴比 (L/S)',
                                                            INTERNAL_TO_DISPLAY_MAP) in numerical_param_options else 0)

                        kde_single_class_options = ["所有类别"] + [c for c in actual_class_names if c != "无实际类别"]
                        kde_single_class = st.selectbox(
                            "选择类别",
                            options=kde_single_class_options,
                            key="kde_single_class_form"
                        )
                        kde_single_color_hex = st.color_picker("图表颜色", value="#1f77b4", key="kde_single_color_form")
                        kde_single_log_x = st.checkbox("X轴对数刻度", value=True, key="kde_single_log_x_form")

                        submitted = st.form_submit_button("生成单类别热力图")

                        if submitted:
                            internal_kde_single_x = DISPLAY_TO_INTERNAL_MAP.get(kde_single_x)
                            internal_kde_single_y = DISPLAY_TO_INTERNAL_MAP.get(kde_single_y)

                            # 如果选择的是“所有类别”，则筛选所有实际类别
                            class_filter_list = [c for c in actual_class_names if
                                                 c != "无实际类别"] if kde_single_class == "所有类别" else [
                                kde_single_class]

                            if internal_kde_single_x in st.session_state.results_df.columns and \
                                    internal_kde_single_y in st.session_state.results_df.columns and \
                                    class_filter_list and class_filter_list != ["无实际类别"]:  # 确保有实际类别用于筛选
                                current_plot_figure = plot_kde_scatter_plot(
                                    results_df=st.session_state.results_df,
                                    x_col=internal_kde_single_x,
                                    y_col=internal_kde_single_y,
                                    class_filter=class_filter_list,
                                    color=kde_single_color_hex,
                                    title_cn=f"{kde_single_class} 的 {kde_single_y} vs {kde_single_x}",
                                    x_label_cn=f"{kde_single_x}",
                                    y_label_cn=f"{kde_single_y}",
                                    log_scale_x=kde_single_log_x,
                                    lang='cn'
                                )
                            else:
                                st.error(
                                    f"无法生成图表：X轴或Y轴参数 '{internal_kde_single_x}' / '{internal_kde_single_y}' 缺失，或未选择有效类别。")

            elif plot_type == "🆚 KDE热力图 (多类别对比)":
                with st.form("kde_comparison_form"):
                    st.subheader("🆚 KDE热力图 (多类别对比) 配置")
                    selectable_actual_classes = [c for c in actual_class_names if c != "无实际类别"]
                    if len(selectable_actual_classes) < 2:
                        st.warning("至少需要两个不同的类别才能进行对比。当前数据中类别数量不足。")
                        submitted = st.form_submit_button("生成对比热力图", disabled=True)
                    else:
                        col_kde_comp_1, col_kde_comp_2 = st.columns(2)
                        with col_kde_comp_1:
                            kde_comp_x = st.selectbox("X轴参数", options=numerical_param_options, key="kde_comp_x_form",
                                                      index=numerical_param_options.index(
                                                          _get_display_name_from_column('等效直径 (微米)',
                                                                                        INTERNAL_TO_DISPLAY_MAP)) if _get_display_name_from_column(
                                                          '等效直径 (微米)',
                                                          INTERNAL_TO_DISPLAY_MAP) in numerical_param_options else 0)
                        with col_kde_comp_2:
                            kde_comp_y = st.selectbox("Y轴参数", options=numerical_param_options, key="kde_comp_y_form",
                                                      index=numerical_param_options.index(
                                                          _get_display_name_from_column('拟合椭圆轴比 (L/S)',
                                                                                        INTERNAL_TO_DISPLAY_MAP)) if _get_display_name_from_column(
                                                          '拟合椭圆轴比 (L/S)',
                                                          INTERNAL_TO_DISPLAY_MAP) in numerical_param_options else 0)

                        col_kde_comp_3, col_kde_comp_4 = st.columns(2)
                        with col_kde_comp_3:
                            kde_comp_class1 = st.selectbox(
                                "对比类别 1",
                                options=selectable_actual_classes,
                                index=0,  # Default to first available class
                                key="kde_comp_class1_form"
                            )
                            kde_comp_color1_hex = st.color_picker("类别 1 颜色", value="#1f77b4",
                                                                  key="kde_comp_color1_form")
                        with col_kde_comp_4:
                            available_classes_for_comp2 = [c for c in selectable_actual_classes if c != kde_comp_class1]

                            # 尝试设置默认索引为第一个非 class1 的类别
                            default_idx_class2 = 0
                            if available_classes_for_comp2 and kde_comp_class1 == selectable_actual_classes[0]:
                                default_idx_class2 = 0  # 列表中第一个就是排除 class1 后的第一个

                            kde_comp_class2 = st.selectbox(
                                "对比类别 2",
                                options=available_classes_for_comp2,
                                index=default_idx_class2,
                                key="kde_comp_class2_form"
                            )

                            kde_comp_color2_hex = st.color_picker("类别 2 颜色", value="#d62728",
                                                                  key="kde_comp_color2_form")

                        kde_comp_log_x = st.checkbox("X轴对数刻度", value=True, key="kde_comp_log_x_comp_form")

                        submitted = st.form_submit_button("生成对比热力图")

                        if submitted:
                            if kde_comp_class1 == kde_comp_class2:
                                st.error("请选择两个不同的类别进行对比。")
                            else:
                                internal_kde_comp_x = DISPLAY_TO_INTERNAL_MAP.get(kde_comp_x)
                                internal_kde_comp_y = DISPLAY_TO_INTERNAL_MAP.get(kde_comp_y)

                                if internal_kde_comp_x in st.session_state.results_df.columns and \
                                        internal_kde_comp_y in st.session_state.results_df.columns and \
                                        kde_comp_class1 and kde_comp_class2:  # 确保类别名称不为空
                                    current_plot_figure = plot_kde_comparison_plot(
                                        results_df=st.session_state.results_df,
                                        x_col=internal_kde_comp_x,
                                        y_col=internal_kde_comp_y,
                                        class1_name=kde_comp_class1,
                                        class2_name=kde_comp_class2,
                                        color1=kde_comp_color1_hex,
                                        color2=kde_comp_color2_hex,
                                        title_cn=f"{kde_comp_class1} vs {kde_comp_class2} 的 {kde_comp_y} vs {kde_comp_x}",
                                        x_label_cn=f"{kde_comp_x}",
                                        y_label_cn=f"{kde_comp_y}",
                                        log_scale_x=kde_comp_log_x,
                                        lang='cn'
                                    )
                                else:
                                    st.error(
                                        f"无法生成图表：X轴或Y轴参数 '{internal_kde_comp_x}' / '{internal_kde_comp_y}' 缺失。")

            # Display the plot and download button if a figure was generated
            if current_plot_figure:
                st.pyplot(current_plot_figure)
                plot_bytes = io.BytesIO()
                # Sanitize filename
                safe_file_name = f"{os.path.splitext(uploaded_file.name)[0]}_{plot_type.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}.png"
                current_plot_figure.savefig(plot_bytes, format="png", bbox_inches='tight')
                plot_bytes.seek(0)
                st.download_button(
                    label="下载图表 (PNG)",
                    data=plot_bytes,
                    file_name=safe_file_name,
                    mime="image/png"
                )
            elif plot_type != "请选择图表类型":
                st.info("请选择有效的绘图参数，并点击 '生成图表' 按钮。确保有数据可用于绘图。")

st.markdown("---")
st.markdown("部署状态：已集成所有 UI 控制、形态学分析和数据可视化功能。")
