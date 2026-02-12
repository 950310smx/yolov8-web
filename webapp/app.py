import os
import sqlite3
import json
import base64
import numpy as np
from io import BytesIO
from datetime import datetime
import tempfile

import cv2
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

# 尝试导入高级图像处理库（可选）
try:
    from skimage import measure, feature, filters
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("警告: skimage 未安装，部分高级功能将不可用")

try:
    from scipy import ndimage
    from scipy.spatial import distance
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("警告: scipy 未安装，部分高级功能将不可用")
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, Response, session, jsonify
from ultralytics import YOLO
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

# transition utility: try relative import first (when running as script), then package import
create_color_fade_transition = None
try:
    # when running webapp as a module, utils is a sibling package
    from utils.transition import create_color_fade_transition as _create_tf
    create_color_fade_transition = _create_tf
except Exception:
    try:
        # fallback: package-style import (when webapp is importable as a package)
        from webapp.utils.transition import create_color_fade_transition as _create_tf
        create_color_fade_transition = _create_tf
    except Exception as e:
        # do not raise — we only print so main app still runs
        print(f"无法导入过渡工具 (utils.transition): {e}")
        create_color_fade_transition = None

# visualization utility (size -> color)
visualize_by_size = None
try:
    from utils.visualize import visualize_by_size as _viz
    visualize_by_size = _viz
except Exception:
    try:
        from webapp.utils.visualize import visualize_by_size as _viz
        visualize_by_size = _viz
    except Exception as e:
        print(f"无法导入可视化工具 (utils.visualize): {e}")
        visualize_by_size = None

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DB_PATH = os.path.join(BASE_DIR, "webapp.db")
SCHEMA_PATH = os.path.join(BASE_DIR, "schema.sql")
MODEL_PATH = os.path.join(PROJECT_ROOT, "1031.onnx")

app = Flask(__name__)
app.secret_key = "change-this-secret-key"  # 用于 flash 消息，后续可改为环境变量

# 预加载粉末检测模型（YOLOv8，自定义 1031.onnx）
yolo_model = YOLO(MODEL_PATH)


def get_db_connection():
    """获取 SQLite 连接。"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """如果数据库或表不存在，则根据 schema.sql 自动初始化。"""
    if not os.path.exists(SCHEMA_PATH):
        return

    conn = get_db_connection()
    
    # 先执行 schema.sql 创建表（如果不存在）
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        conn.executescript(f.read())
    
    # 检查并添加缺失的列（用于已存在的表）
    try:
        # 检查 users 表的新字段
        cursor = conn.execute("PRAGMA table_info(users)")
        user_columns = [row[1] for row in cursor.fetchall()]
        
        new_user_fields = {
            'real_name': 'TEXT',
            'position': 'TEXT',
            'phone': 'TEXT',
            'signature_path': 'TEXT',
            'role': 'TEXT',
            'membership_status': 'TEXT',
        }
        
        for field, field_type in new_user_fields.items():
            if field not in user_columns:
                conn.execute(f"ALTER TABLE users ADD COLUMN {field} {field_type}")
                print(f"已添加 {field} 列到 users 表")
                # 如果是role或membership_status，设置默认值
                if field == 'role':
                    conn.execute("UPDATE users SET role = 'user' WHERE role IS NULL")
                elif field == 'membership_status':
                    conn.execute("UPDATE users SET membership_status = 'none' WHERE membership_status IS NULL")
        
        # updated_at 需要特殊处理（SQLite不支持在ALTER TABLE时添加带CURRENT_TIMESTAMP的列）
        if 'updated_at' not in user_columns:
            try:
                conn.execute("ALTER TABLE users ADD COLUMN updated_at TIMESTAMP")
                print("已添加 updated_at 列到 users 表")
            except Exception as e:
                print(f"添加 updated_at 列时出错: {e}")
        
        # 检查 analysis_reports 表的新字段
        cursor = conn.execute("PRAGMA table_info(analysis_reports)")
        report_columns = [row[1] for row in cursor.fetchall()]
        
        if 'particle_distribution' not in report_columns:
            conn.execute("ALTER TABLE analysis_reports ADD COLUMN particle_distribution TEXT")
            print("已添加 particle_distribution 列到 analysis_reports 表")
        
        if 'inspector_id' not in report_columns:
            conn.execute("ALTER TABLE analysis_reports ADD COLUMN inspector_id INTEGER")
            print("已添加 inspector_id 列到 analysis_reports 表")
        
        if 'reviewer_id' not in report_columns:
            conn.execute("ALTER TABLE analysis_reports ADD COLUMN reviewer_id INTEGER")
            print("已添加 reviewer_id 列到 analysis_reports 表")
        
        if 'analysis_type' not in report_columns:
            conn.execute("ALTER TABLE analysis_reports ADD COLUMN analysis_type TEXT DEFAULT 'normal'")
            print("已添加 analysis_type 列到 analysis_reports 表")
        
        if 'custom_config' not in report_columns:
            conn.execute("ALTER TABLE analysis_reports ADD COLUMN custom_config TEXT")
            print("已添加 custom_config 列到 analysis_reports 表")
        
        if 'custom_metrics' not in report_columns:
            conn.execute("ALTER TABLE analysis_reports ADD COLUMN custom_metrics TEXT")
            print("已添加 custom_metrics 列到 analysis_reports 表")
            
    except Exception as e:
        print(f"检查/添加列时出错（可能表不存在，将自动创建）: {e}")
    
    # 创建默认管理员账号（如果不存在）
    try:
        admin_user = conn.execute(
            "SELECT id FROM users WHERE username = ?",
            ("admin",)
        ).fetchone()
        
        if not admin_user:
            # 创建管理员账号
            admin_password_hash = generate_password_hash("12345678")
            conn.execute(
                """INSERT INTO users (username, email, password_hash, role, membership_status) 
                   VALUES (?, ?, ?, ?, ?)""",
                ("admin", "12345678@qq.com", admin_password_hash, "admin", "none")
            )
            print("已创建默认管理员账号：username=admin, password=12345678")
        else:
            # 确保admin用户是管理员角色
            conn.execute(
                "UPDATE users SET role = ? WHERE username = ?",
                ("admin", "admin")
            )
            print("管理员账号已存在，已确保其角色为管理员")
    except Exception as e:
        print(f"创建管理员账号时出错: {e}")
    
    conn.commit()
    conn.close()


# Flask 3.x 移除了 before_first_request，这里在模块加载时直接初始化一次数据库
init_db()


def normalize_material_type(material_type: str) -> str:
    """将材料类型统一转换为中文显示值。"""
    if not material_type:
        return "未指定"
    value = material_type.strip()
    lower_value = value.lower()
    if lower_value == "powder":
        return "粉末"
    if lower_value == "metal":
        return "金属"
    return value


def login_required(view_func):
    """简单的登录校验装饰器：未登录则跳转到登录页。"""
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("请先登录后再访问该页面。", "error")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    return wrapper


@app.route("/", methods=["GET"])
def index():
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if not username or not password:
            flash("请先填写用户名和密码。", "error")
            return render_template("login.html")

        conn = get_db_connection()
        # 支持用用户名或邮箱登录
        cursor = conn.execute(
            "SELECT * FROM users WHERE username = ? OR email = ?",
            (username, username),
        )
        user = cursor.fetchone()
        conn.close()

        if user is None:
            flash("用户不存在，请先注册。", "error")
            return render_template("login.html")

        if not check_password_hash(user["password_hash"], password):
            flash("密码错误，请重新输入。", "error")
            return render_template("login.html")

        session.clear()
        session["user_id"] = int(user["id"])
        session["username"] = user["username"]
        session["real_name"] = user["real_name"]
        # sqlite3.Row 不支持 .get()，转换为字典后使用
        user_dict = dict(user)
        session["role"] = user_dict.get("role", "user")

        flash("登录成功，欢迎进入金属材料微观图像智能分析云服务平台！", "success")
        # 登录成功后跳转到主界面
        return redirect(url_for("main"))

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()

        if not username or not email or not password:
            flash("请先填写完整的注册信息。", "error")
            return render_template("register.html")

        # 禁止注册管理员用户名
        if username.lower() == "admin":
            flash("管理员账号不能通过注册页面创建。", "error")
            return render_template("register.html")

        conn = get_db_connection()

        # 检查用户名或邮箱是否已存在
        cursor = conn.execute(
            "SELECT 1 FROM users WHERE username = ? OR email = ?",
            (username, email),
        )
        exists = cursor.fetchone()
        if exists:
            conn.close()
            flash("用户名或邮箱已存在，请更换后再试。", "error")
            return render_template("register.html")

        password_hash = generate_password_hash(password)
        # 默认注册为普通用户
        conn.execute(
            "INSERT INTO users (username, email, password_hash, role, membership_status) VALUES (?, ?, ?, ?, ?)",
            (username, email, password_hash, 'user', 'none'),
        )
        conn.commit()
        conn.close()

        flash("注册成功！现在可以使用该账号登录平台。", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/logout", methods=["GET"])
def logout():
    """退出登录并清空会话。"""
    session.clear()
    flash("已退出当前账号，请重新登录。", "success")
    return redirect(url_for("login"))


@app.route("/main", methods=["GET"])
@login_required
def main():
    """系统主界面：根据用户角色展示不同界面。"""
    user_role = session.get("role", "user")
    
    # 管理员直接跳转到管理员界面
    if user_role == "admin":
        return redirect(url_for("admin_dashboard"))
    
    # 分析师功能后续实现，暂时跳转到主界面
    if user_role == "analyst":
        # 后续实现分析师功能
        pass
    
    # 普通用户和会员显示正常主界面
    conn = get_db_connection()
    stats_rows = conn.execute(
        "SELECT material_type, COUNT(*) AS cnt FROM analysis_reports GROUP BY material_type"
    ).fetchall()
    conn.close()

    powder_count = 0
    metal_count = 0
    others_count = 0

    for row in stats_rows:
        material = normalize_material_type(row["material_type"])
        cnt = row["cnt"]
        if material == "粉末":
            powder_count += cnt
        elif material == "金属":
            metal_count += cnt
        else:
            others_count += cnt

    total = powder_count + metal_count + others_count

    donut_data = {
        "labels": ["粉末", "金属"],
        "values": [powder_count, metal_count],
        "total": total,
    }

    return render_template("main.html", donut_data=donut_data)


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    """个人信息页面：查看和编辑个人信息，包括签名上传。"""
    user_id = session.get("user_id")
    
    conn = get_db_connection()
    
    if request.method == "POST":
        # 更新个人信息
        real_name = request.form.get("real_name", "").strip()
        position = request.form.get("position", "").strip()
        phone = request.form.get("phone", "").strip()
        
        # 处理Canvas签名数据（base64格式）
        signature_path = None
        signature_data = request.form.get("signature_data", "").strip()
        
        if signature_data and signature_data.startswith('data:image'):
            try:
                # 解析base64数据
                header, encoded = signature_data.split(',', 1)
                image_data = base64.b64decode(encoded)
                
                # 保存签名图片
                signature_dir = os.path.join(BASE_DIR, "static", "signatures")
                os.makedirs(signature_dir, exist_ok=True)
                
                # 获取原用户的签名路径（如果存在）
                old_user = conn.execute("SELECT signature_path FROM users WHERE id = ?", (user_id,)).fetchone()
                old_signature = old_user["signature_path"] if old_user and old_user["signature_path"] else None
                
                # 生成新的签名文件名
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_name = f"{user_id}_{ts}_signature.png"
                signature_path = os.path.join(signature_dir, safe_name)
                
                # 保存图片文件
                with open(signature_path, 'wb') as f:
                    f.write(image_data)
                
                # 保存相对路径到数据库
                signature_path = f"signatures/{safe_name}"
                
                # 删除旧签名文件
                if old_signature:
                    old_path = os.path.join(BASE_DIR, "static", old_signature)
                    if os.path.exists(old_path):
                        try:
                            os.remove(old_path)
                        except:
                            pass
            except Exception as e:
                print(f"保存签名时出错: {e}")
                signature_path = None
        
        # 更新数据库（检查updated_at列是否存在）
        cursor = conn.execute("PRAGMA table_info(users)")
        user_columns = [row[1] for row in cursor.fetchall()]
        has_updated_at = 'updated_at' in user_columns
        
        if signature_path:
            if has_updated_at:
                conn.execute(
                    """UPDATE users SET real_name = ?, position = ?, phone = ?, 
                       signature_path = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?""",
                    (real_name, position, phone, signature_path, user_id)
                )
            else:
                conn.execute(
                    """UPDATE users SET real_name = ?, position = ?, phone = ?, 
                       signature_path = ? WHERE id = ?""",
                    (real_name, position, phone, signature_path, user_id)
                )
        else:
            if has_updated_at:
                conn.execute(
                    """UPDATE users SET real_name = ?, position = ?, phone = ?, 
                       updated_at = CURRENT_TIMESTAMP WHERE id = ?""",
                    (real_name, position, phone, user_id)
                )
            else:
                conn.execute(
                    """UPDATE users SET real_name = ?, position = ?, phone = ? WHERE id = ?""",
                    (real_name, position, phone, user_id)
                )
        conn.commit()
        flash("个人信息已更新。", "success")
    
    # 获取用户信息
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    
    if user is None:
        flash("用户不存在。", "error")
        return redirect(url_for("main"))
    
    # 生成签名图片URL（如果有）
    signature_url = None
    if user["signature_path"]:
        signature_url = url_for("static", filename=user["signature_path"])
    
    return render_template("profile.html", user=user, signature_url=signature_url)


@app.route("/profile/clear-signature", methods=["POST"])
@login_required
def clear_signature():
    """清除用户的签名图片文件。"""
    user_id = session.get("user_id")
    
    conn = get_db_connection()
    user = conn.execute("SELECT signature_path FROM users WHERE id = ?", (user_id,)).fetchone()
    
    if user and user["signature_path"]:
        # 删除签名文件
        signature_path = os.path.join(BASE_DIR, "static", user["signature_path"])
        if os.path.exists(signature_path):
            try:
                os.remove(signature_path)
            except Exception as e:
                print(f"删除签名文件时出错: {e}")
        
        # 清除数据库中的签名路径
        conn.execute("UPDATE users SET signature_path = NULL WHERE id = ?", (user_id,))
        conn.commit()
    
    conn.close()
    return jsonify({"success": True})


@app.route("/apply-membership", methods=["POST"])
@login_required
def apply_membership():
    """普通用户申请成为会员。"""
    user_id = session.get("user_id")
    
    conn = get_db_connection()
    user = conn.execute("SELECT role, membership_status FROM users WHERE id = ?", (user_id,)).fetchone()
    
    if not user:
        conn.close()
        return jsonify({"success": False, "message": "用户不存在"})
    
    # 检查用户角色和申请状态
    if user["role"] == "member":
        conn.close()
        return jsonify({"success": False, "message": "您已经是会员"})
    
    if user["membership_status"] == "pending":
        conn.close()
        return jsonify({"success": False, "message": "您的会员申请正在审核中，请耐心等待"})
    
    if user["membership_status"] == "approved":
        conn.close()
        return jsonify({"success": False, "message": "您的会员申请已通过，请联系管理员"})
    
    # 提交会员申请
    conn.execute(
        "UPDATE users SET membership_status = ? WHERE id = ?",
        ("pending", user_id)
    )
    conn.commit()
    conn.close()
    
    return jsonify({"success": True, "message": "会员申请已提交，等待管理员审核"})


@app.route("/admin", methods=["GET"])
@login_required
def admin_dashboard():
    """管理员界面：显示所有账号信息和会员申请管理。"""
    user_role = session.get("role", "user")
    
    # 检查是否为管理员
    if user_role != "admin":
        flash("您没有权限访问管理员界面。", "error")
        return redirect(url_for("main"))
    
    conn = get_db_connection()
    
    # 获取所有用户信息
    users = conn.execute(
        "SELECT id, username, email, role, membership_status, real_name, position, phone, created_at FROM users ORDER BY created_at DESC"
    ).fetchall()
    
    # 获取待审核的会员申请
    pending_applications = conn.execute(
        "SELECT id, username, email, real_name, position, phone, created_at FROM users WHERE membership_status = 'pending' ORDER BY created_at DESC"
    ).fetchall()
    
    conn.close()
    
    return render_template("admin.html", users=users, pending_applications=pending_applications)


@app.route("/admin/approve-membership/<int:user_id>", methods=["POST"])
@login_required
def approve_membership(user_id):
    """管理员批准会员申请。"""
    admin_role = session.get("role", "user")
    
    if admin_role != "admin":
        return jsonify({"success": False, "message": "权限不足"}), 403
    
    conn = get_db_connection()
    
    # 更新用户角色和申请状态
    conn.execute(
        "UPDATE users SET role = ?, membership_status = ? WHERE id = ?",
        ("member", "approved", user_id)
    )
    conn.commit()
    conn.close()
    
    return jsonify({"success": True, "message": "会员申请已通过"})


@app.route("/admin/reject-membership/<int:user_id>", methods=["POST"])
@login_required
def reject_membership(user_id):
    """管理员拒绝会员申请。"""
    admin_role = session.get("role", "user")
    
    if admin_role != "admin":
        return jsonify({"success": False, "message": "权限不足"}), 403
    
    conn = get_db_connection()
    
    # 更新申请状态为已拒绝
    conn.execute(
        "UPDATE users SET membership_status = ? WHERE id = ?",
        ("rejected", user_id)
    )
    conn.commit()
    conn.close()
    
    return jsonify({"success": True, "message": "会员申请已拒绝"})


@app.route("/analysis/new", methods=["GET", "POST"])
@login_required
def new_analysis():
    """
    新建分析 / 图像上传页面：
    - GET：展示上传表单
    - POST：接收图像文件和基础参数，保存到本地 uploads 目录，后续可接入任务队列与模型推理。
    """
    result_image_url = None
    transition_image_url = None

    if request.method == "POST":
        file = request.files.get("image")
        raw_material_type = request.form.get("material_type", "").strip()
        material_type = normalize_material_type(raw_material_type)
        remark = request.form.get("remark", "").strip()

        if not file or file.filename == "":
            flash("请先选择需要上传的微观图像文件。", "error")
            return render_template("analysis_new.html")

        # 简单的文件类型校验（可根据实际需要扩展）
        allowed_ext = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        _, ext = os.path.splitext(file.filename.lower())
        if ext not in allowed_ext:
            flash("仅支持上传 PNG/JPG/TIF/BMP 等常见图像格式。", "error")
            return render_template("analysis_new.html")

        upload_dir = os.path.join(BASE_DIR, "uploads")
        os.makedirs(upload_dir, exist_ok=True)

        # 为避免重名，使用时间戳前缀
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = f"{ts}_{os.path.basename(file.filename)}"
        save_path = os.path.join(upload_dir, safe_name)
        file.save(save_path)

        # 使用训练好的 YOLO 模型进行粉末检测，并将带标注结果图保存到 static 目录以便前端展示
        result_filename = None
        result_image_url = None
        particle_distribution = None
        
        results = yolo_model(save_path)
        if results:
            res = results[0]
            annotated = res.plot()  # numpy array (BGR)
            static_result_dir = os.path.join(BASE_DIR, "static", "results")
            os.makedirs(static_result_dir, exist_ok=True)
            result_filename = f"{ts}_{os.path.basename(save_path)}"
            result_path = os.path.join(static_result_dir, result_filename)
            # Save original annotated image separately for records
            annotated_filename = f"annot_{result_filename}"
            annotated_path = os.path.join(static_result_dir, annotated_filename)
            cv2.imwrite(annotated_path, annotated)

            # Generate visualization by size (replace displayed result with visualization)
            try:
                if visualize_by_size is not None:
                    viz_created = visualize_by_size(save_path, res, result_path, colormap="plasma", alpha=0.6)
                    if viz_created:
                        print(f"生成基于尺寸的可视化图: {viz_created}")
                else:
                    # fallback: use annotated copy as result
                    cv2.imwrite(result_path, annotated)
            except Exception as e:
                print(f"生成基于尺寸的可视化图失败: {e}")
                # fallback to annotated
                cv2.imwrite(result_path, annotated)

            # 供模板显示
            from flask import url_for as _url_for

            result_image_url = _url_for("static", filename=f"results/{result_filename}")
            
            # 生成渐变过渡视频（原图 -> 标注图），非阻塞保护
            try:
                if create_color_fade_transition is not None:
                    # 生成短 GIF（约 2 秒）：steps=20, fps=10 -> (20+1)/10 ≈ 2.1s
                    transition_filename = f"{result_filename}_fade.gif"
                    transition_path = os.path.join(static_result_dir, transition_filename)
                    created = create_color_fade_transition(save_path, result_path, transition_path, steps=20, fps=10)
                    if created:
                        print(f"生成渐变过渡文件: {created}")
                        transition_image_url = _url_for("static", filename=f"results/{os.path.basename(created)}")
            except Exception as e:
                print(f"生成渐变过渡视频失败: {e}")
            
            # 提取粒径分布数据：从检测框计算粒径大小
            # 如果模型直接输出粒径数据，请告诉我具体字段名，我会修改这里
            if res.boxes is not None and len(res.boxes) > 0:
                try:
                    # 调试：打印模型输出的所有属性（帮助确认是否有粒径数据）
                    print("=== 模型输出调试信息 ===")
                    print(f"res 对象的属性: {dir(res)}")
                    print(f"res.boxes 对象的属性: {dir(res.boxes)}")
                    if hasattr(res.boxes, 'data'):
                        print(f"res.boxes.data 形状: {res.boxes.data.shape if hasattr(res.boxes.data, 'shape') else 'N/A'}")
                    print("========================")
                    
                    # 当前实现：从检测框计算粒径（使用检测框的等效直径）
                    # 粒径 = sqrt(宽度 * 高度)，单位：像素
                    boxes = res.boxes.xywh.cpu().numpy()  # [中心x, 中心y, 宽度w, 高度h]
                    particle_sizes = np.sqrt(boxes[:, 2] * boxes[:, 3])
                    
                    print(f"检测到的颗粒数量: {len(particle_sizes)}")
                    print(f"粒径范围: {np.min(particle_sizes):.2f} - {np.max(particle_sizes):.2f} 像素")
                    
                    # 定义粒径区间并统计（分成10个区间）
                    min_size = max(5, np.min(particle_sizes))
                    max_size = min(300, np.max(particle_sizes))
                    num_bins = 10
                    bins = np.linspace(min_size, max_size, num_bins + 1)
                    
                    # 统计每个区间的数量
                    counts, bin_edges = np.histogram(particle_sizes, bins=bins)
                    
                    # 生成区间标签（显示为 "起始-结束" 格式）
                    bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(bin_edges)-1)]
                    
                    # 转换为列表以便 JSON 序列化
                    particle_distribution = json.dumps({
                        "bins": bin_labels,
                        "counts": counts.tolist(),
                        "total": len(particle_sizes)
                    })
                except Exception as e:
                    print(f"提取粒径数据时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    particle_distribution = None

        # 保存分析任务记录到数据库
        user_id = session.get("user_id")
        conn = get_db_connection()
        conn.execute(
            "INSERT INTO analysis_reports (user_id, original_filename, saved_filename, result_filename, material_type, particle_distribution, status, inspector_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, file.filename, safe_name, result_filename, material_type, particle_distribution, "completed", user_id),
        )
        conn.commit()
        conn.close()

        # 只在页面底部提示检测成功，不再展示具体文件名等长文本
        flash("粉末检测已完成，结果已在右侧区域展示。", "success")

    return render_template("analysis_new.html", result_image_url=result_image_url, transition_image_url=transition_image_url)


def calculate_custom_metrics(mask, image_gray, boxes, classes, selected_metrics=None):
    """计算定制化分析的各项参数
    
    Args:
        mask: 颗粒掩码
        image_gray: 灰度图像
        boxes: 检测框
        classes: 类别
        selected_metrics: 用户选择的参数列表，如果为None则计算所有参数
    """
    metrics = {}
    
    if not SKIMAGE_AVAILABLE or not SCIPY_AVAILABLE:
        return metrics
    
    # 如果没有指定selected_metrics，则计算所有参数
    if selected_metrics is None:
        selected_metrics = ['aspect_ratio', 'roundness', 'radial_std', 'glcm', 'lbp', 'coverage']
    
    try:
        # 1. 拟合椭圆/外接多边形的长轴短轴比
        if 'aspect_ratio' in selected_metrics:
            aspect_ratios = []
            for i, box in enumerate(boxes):
                x, y, w, h = box
                # 提取单个颗粒的掩码
                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)
                particle_mask = mask[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else None
                
                if particle_mask is not None and np.sum(particle_mask) > 0:
                    # 找到轮廓
                    contours = measure.find_contours(particle_mask, 0.5)
                    if len(contours) > 0:
                        contour = contours[0]
                        if len(contour) >= 5:
                            # 拟合椭圆
                            try:
                                ellipse = measure.regionprops(particle_mask.astype(int))[0]
                                if hasattr(ellipse, 'major_axis_length') and hasattr(ellipse, 'minor_axis_length'):
                                    if ellipse.minor_axis_length > 0:
                                        aspect_ratio = ellipse.major_axis_length / ellipse.minor_axis_length
                                        aspect_ratios.append(aspect_ratio)
                            except:
                                pass
            
            if aspect_ratios:
                metrics['aspect_ratio_mean'] = float(np.mean(aspect_ratios))
                metrics['aspect_ratio_std'] = float(np.std(aspect_ratios))
                # 保存原始数据用于可视化
                metrics['aspect_ratio_values'] = [float(x) for x in aspect_ratios]
        
        # 2. 莱利圆度 (Riley Roundness)
        if 'roundness' in selected_metrics:
            roundness_values = []
            for i, box in enumerate(boxes):
                x, y, w, h = box
                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)
                particle_mask = mask[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else None
                
                if particle_mask is not None and np.sum(particle_mask) > 0:
                    try:
                        props = measure.regionprops(particle_mask.astype(int))[0]
                        area = props.area
                        perimeter = props.perimeter
                        if perimeter > 0:
                            roundness = 4 * np.pi * area / (perimeter ** 2)
                            roundness_values.append(roundness)
                    except:
                        pass
            
            if roundness_values:
                metrics['roundness_mean'] = float(np.mean(roundness_values))
                metrics['roundness_std'] = float(np.std(roundness_values))
                # 保存原始数据用于可视化
                metrics['roundness_values'] = [float(x) for x in roundness_values]
        
        # 3. 边界径向标准差
        if 'radial_std' in selected_metrics:
            radial_stds = []
            for i, box in enumerate(boxes):
                x, y, w, h = box
                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)
                particle_mask = mask[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else None
                
                if particle_mask is not None and np.sum(particle_mask) > 0:
                    try:
                        props = measure.regionprops(particle_mask.astype(int))[0]
                        centroid = props.centroid
                        contours = measure.find_contours(particle_mask, 0.5)
                        if len(contours) > 0:
                            contour = contours[0]
                            distances = [distance.euclidean([c[0], c[1]], centroid) for c in contour]
                            if len(distances) > 0:
                                radial_stds.append(float(np.std(distances)))
                    except:
                        pass
            
            if radial_stds:
                metrics['radial_std_mean'] = float(np.mean(radial_stds))
        
        # 4. 灰度共生矩阵特征（对比度、能量、同质性）
        if 'glcm' in selected_metrics and image_gray is not None:
            glcm_features = []
            for i, box in enumerate(boxes):
                x, y, w, h = box
                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)
                particle_region = image_gray[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else None
                
                if particle_region is not None and particle_region.size > 0:
                    try:
                        # 归一化到0-255
                        particle_region = (particle_region * 255).astype(np.uint8) if particle_region.max() <= 1 else particle_region.astype(np.uint8)
                        glcm = graycomatrix(particle_region, [1], [0], 256, symmetric=True, normed=True)
                        contrast = graycoprops(glcm, 'contrast')[0, 0]
                        energy = graycoprops(glcm, 'energy')[0, 0]
                        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
                        glcm_features.append({
                            'contrast': float(contrast),
                            'energy': float(energy),
                            'homogeneity': float(homogeneity)
                        })
                    except:
                        pass
            
            if glcm_features:
                metrics['glcm_contrast_mean'] = float(np.mean([f['contrast'] for f in glcm_features]))
                metrics['glcm_energy_mean'] = float(np.mean([f['energy'] for f in glcm_features]))
                metrics['glcm_homogeneity_mean'] = float(np.mean([f['homogeneity'] for f in glcm_features]))
        
        # 5. LBP熵
        if 'lbp' in selected_metrics and image_gray is not None:
            lbp_entropies = []
            for i, box in enumerate(boxes):
                x, y, w, h = box
                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)
                particle_region = image_gray[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else None
                
                if particle_region is not None and particle_region.size > 0:
                    try:
                        particle_region = (particle_region * 255).astype(np.uint8) if particle_region.max() <= 1 else particle_region.astype(np.uint8)
                        lbp = local_binary_pattern(particle_region, 8, 1, method='uniform')
                        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
                        hist = hist / hist.sum()
                        hist = hist[hist > 0]  # 移除零值
                        entropy = -np.sum(hist * np.log2(hist))
                        lbp_entropies.append(float(entropy))
                    except:
                        pass
            
            if lbp_entropies:
                metrics['lbp_entropy_mean'] = float(np.mean(lbp_entropies))
        
        # 6. 粉末遮盖率
        if 'coverage' in selected_metrics:
            total_area = mask.shape[0] * mask.shape[1]
            particle_area = np.sum(mask > 0)
            coverage_rate = particle_area / total_area if total_area > 0 else 0
            metrics['coverage_rate'] = float(coverage_rate)
        
    except Exception as e:
        print(f"计算定制化参数时出错: {e}")
        import traceback
        traceback.print_exc()
    
    return metrics


def check_edge_touching(mask, box, image_shape):
    """检查颗粒是否触碰边缘"""
    x, y, w, h = box
    x1, y1 = max(0, int(x - w/2)), max(0, int(y - h/2))
    x2, y2 = min(image_shape[1], int(x + w/2)), min(image_shape[0], int(y + h/2))
    
    # 检查是否在图像边缘
    touches_left = x1 <= 0
    touches_right = x2 >= image_shape[1]
    touches_top = y1 <= 0
    touches_bottom = y2 >= image_shape[0]
    
    return touches_left or touches_right or touches_top or touches_bottom


def classify_particle_shape(roundness):
    """根据圆度分类颗粒形状：irregular 或 round"""
    if roundness >= 0.7:
        return "round"
    else:
        return "irregular"


@app.route("/analysis/custom", methods=["GET", "POST"])
@login_required
def custom_analysis():
    """定制化分析页面：仅会员可用，支持批量处理和个性化设置"""
    user_role = session.get("role", "user")
    
    # 检查是否为会员
    if user_role != "member":
        flash("定制化分析功能仅限会员使用。", "error")
        return redirect(url_for("main"))
    
    if request.method == "POST":
        # 处理批量上传
        files = request.files.getlist("images")
        raw_material_type = request.form.get("material_type", "").strip()
        material_type = normalize_material_type(raw_material_type)
        
        # 获取个性化设置
        custom_config = {
            "show_transparent_fill": request.form.get("show_transparent_fill") == "on",
            "show_edge_highlight": request.form.get("show_edge_highlight") == "on",
            "show_labels": request.form.get("show_labels") == "on",
            "show_colors": request.form.get("show_colors") == "on",
            "color_scheme": request.form.get("color_scheme", "default"),
        }
        
        # 获取参数选择
        selected_metrics = request.form.getlist("selected_metrics")
        
        if not files or all(f.filename == "" for f in files):
            flash("请至少选择一张图像文件。", "error")
            return render_template("custom_analysis.html")
        
        upload_dir = os.path.join(BASE_DIR, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        static_result_dir = os.path.join(BASE_DIR, "static", "results")
        os.makedirs(static_result_dir, exist_ok=True)
        
        user_id = session.get("user_id")
        conn = get_db_connection()
        
        processed_count = 0
        for file in files:
            if file.filename == "":
                continue
            
            # 文件类型校验
            allowed_ext = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
            _, ext = os.path.splitext(file.filename.lower())
            if ext not in allowed_ext:
                continue
            
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            safe_name = f"{ts}_{os.path.basename(file.filename)}"
            save_path = os.path.join(upload_dir, safe_name)
            file.save(save_path)
            
            # 使用YOLO模型进行检测
            results = yolo_model(save_path)
            if not results:
                continue
            
            res = results[0]
            original_image = cv2.imread(save_path)
            if original_image is None:
                continue
            image_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            
            # 创建掩码
            mask = np.zeros((original_image.shape[0], original_image.shape[1]), dtype=np.uint8)
            if res.masks is not None and len(res.masks) > 0:
                # 如果有多个mask，合并它们
                for i in range(len(res.masks)):
                    mask_data = res.masks.data[i].cpu().numpy()
                    # 调整mask大小到原图尺寸
                    mask_resized = cv2.resize(mask_data, (original_image.shape[1], original_image.shape[0]))
                    mask = np.maximum(mask, (mask_resized * 255).astype(np.uint8))
            else:
                # 如果没有掩码，从boxes创建
                if res.boxes is not None and len(res.boxes) > 0:
                    boxes = res.boxes.xywh.cpu().numpy()
                    for box in boxes:
                        x, y, w, h = box
                        x1, y1 = max(0, int(x - w/2)), max(0, int(y - h/2))
                        x2, y2 = min(original_image.shape[1], int(x + w/2)), min(original_image.shape[0], int(y + h/2))
                        if x2 > x1 and y2 > y1:
                            mask[y1:y2, x1:x2] = 255
            
            # 计算定制化参数
            boxes = res.boxes.xywh.cpu().numpy() if res.boxes is not None and len(res.boxes) > 0 else np.array([])
            classes = res.boxes.cls.cpu().numpy() if res.boxes is not None and len(res.boxes) > 0 else np.array([])
            custom_metrics = calculate_custom_metrics(mask, image_gray, boxes, classes, selected_metrics)
            
            # 检查边缘触碰和形状分类，保存每个颗粒的详细信息
            edge_touching_info = []
            shape_distribution = {"round": 0, "irregular": 0}
            particle_details = []  # 保存每个颗粒的详细信息用于可视化
            
            if len(boxes) > 0:
                for i, box in enumerate(boxes):
                    touches_edge = check_edge_touching(mask, box, original_image.shape)
                    edge_touching_info.append(touches_edge)
                    
                    particle_detail = {
                        "edge_touching": touches_edge,
                        "roundness": None,
                        "aspect_ratio": None,
                        "shape": None
                    }
                    
                    # 计算圆度并分类
                    if SKIMAGE_AVAILABLE:
                        try:
                            x, y, w, h = box
                            x1, y1 = max(0, int(x - w/2)), max(0, int(y - h/2))
                            x2, y2 = min(original_image.shape[1], int(x + w/2)), min(original_image.shape[0], int(y + h/2))
                            particle_mask = mask[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else None
                            
                            if particle_mask is not None and np.sum(particle_mask) > 0:
                                props = measure.regionprops(particle_mask.astype(int))[0]
                                area = props.area
                                perimeter = props.perimeter
                                if perimeter > 0:
                                    roundness = 4 * np.pi * area / (perimeter ** 2)
                                    particle_detail["roundness"] = float(roundness)
                                    shape = classify_particle_shape(roundness)
                                    particle_detail["shape"] = shape
                                    shape_distribution[shape] += 1
                                
                                # 计算长轴短轴比
                                if hasattr(props, 'major_axis_length') and hasattr(props, 'minor_axis_length'):
                                    if props.minor_axis_length > 0:
                                        aspect_ratio = props.major_axis_length / props.minor_axis_length
                                        particle_detail["aspect_ratio"] = float(aspect_ratio)
                        except:
                            pass
                    
                    particle_details.append(particle_detail)
            
            custom_metrics['edge_touching_count'] = sum(edge_touching_info)
            custom_metrics['total_particles'] = len(edge_touching_info)
            custom_metrics['edge_touching_ratio'] = len(edge_touching_info) > 0 and sum(edge_touching_info) / len(edge_touching_info) or 0
            custom_metrics['shape_distribution'] = shape_distribution
            # 保存颗粒详细信息用于可视化
            custom_metrics['particle_details'] = particle_details
            
            # 生成个性化标注图像
            annotated = original_image.copy()
            
            if res.boxes is not None and len(res.boxes) > 0:
                boxes_xyxy = res.boxes.xyxy.cpu().numpy()
                confidences = res.boxes.conf.cpu().numpy()
                
                for i, (box, conf) in enumerate(zip(boxes_xyxy, confidences)):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # 根据设置绘制
                    if custom_config["show_colors"]:
                        # 如果颗粒不触碰边缘，用绿色；否则用红色
                        if i < len(edge_touching_info) and not edge_touching_info[i]:
                            color = (0, 255, 0)  # 绿色：不触碰边缘
                        else:
                            color = (0, 0, 255)  # 红色：触碰边缘或未知
                    else:
                        color = (128, 128, 128)
                    
                    if custom_config["show_transparent_fill"]:
                        overlay = annotated.copy()
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                        cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0, annotated)
                    
                    if custom_config["show_edge_highlight"]:
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    
                    if custom_config["show_labels"]:
                        label = f"{conf:.2f}"
                        if i < len(edge_touching_info) and edge_touching_info[i]:
                            label += " [Edge]"
                        cv2.putText(annotated, label, (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 保存结果图
            result_filename = f"custom_{ts}_{os.path.basename(file.filename)}"
            result_path = os.path.join(static_result_dir, result_filename)
            # Save annotated copy for records
            annotated_filename = f"annot_{result_filename}"
            annotated_path = os.path.join(static_result_dir, annotated_filename)
            cv2.imwrite(annotated_path, annotated)
            # Generate visualization by size and save to result_path
            try:
                if visualize_by_size is not None:
                    viz_created = visualize_by_size(save_path, res, result_path, colormap="plasma", alpha=0.6)
                    if viz_created:
                        print(f"[custom_analysis] 生成基于尺寸的可视化图: {viz_created}")
                else:
                    cv2.imwrite(result_path, annotated)
            except Exception as e:
                print(f"[custom_analysis] 生成基于尺寸的可视化图失败: {e}")
                cv2.imwrite(result_path, annotated)
            
            # 生成渐变过渡视频（原图 -> 标注图）
            try:
                if create_color_fade_transition is not None:
                    # 生成短 GIF（约 2 秒）
                    transition_filename = f"fade_{result_filename}.gif"
                    transition_path = os.path.join(static_result_dir, transition_filename)
                    created = create_color_fade_transition(save_path, result_path, transition_path, steps=20, fps=10)
                    if created:
                        print(f"[custom_analysis] 生成渐变过渡文件: {created}")
            except Exception as e:
                print(f"[custom_analysis] 生成渐变过渡视频失败: {e}")
            
            # 保存到数据库
            conn.execute(
                """INSERT INTO analysis_reports (user_id, original_filename, saved_filename, result_filename, 
                   material_type, analysis_type, custom_config, custom_metrics, status, inspector_id) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, file.filename, safe_name, result_filename, material_type, "custom",
                 json.dumps(custom_config), json.dumps(custom_metrics), "completed", user_id),
            )
            processed_count += 1
        
        conn.commit()
        conn.close()
        
        flash(f"成功处理 {processed_count} 张图像，定制化分析完成！", "success")
        return redirect(url_for("reports"))
    
    return render_template("custom_analysis.html")


@app.route("/reports", methods=["GET"])
@login_required
def reports():
    """历史分析记录列表页：展示所有已完成的分析任务。"""
    filter_type = request.args.get("filter", "全部").strip()
    valid_filters = {"全部", "粉末", "金属"}
    if filter_type not in valid_filters:
        filter_type = "全部"

    conn = get_db_connection()
    query = "SELECT * FROM analysis_reports"
    params = ()
    if filter_type == "粉末":
        query += " WHERE material_type IN (?, ?)"
        params = ("粉末", "powder")
    elif filter_type == "金属":
        query += " WHERE material_type IN (?, ?)"
        params = ("金属", "metal")

    query += " ORDER BY created_at DESC"
    reports_rows = conn.execute(query, params).fetchall()
    conn.close()

    # 将 Row 对象转换为字典，并生成结果图URL
    reports_list = []
    for row in reports_rows:
        report_dict = dict(row)
        report_dict["material_type_display"] = normalize_material_type(report_dict.get("material_type"))
        if report_dict["result_filename"]:
            report_dict["result_image_url"] = url_for(
                "static", filename=f"results/{report_dict['result_filename']}"
            )
        else:
            report_dict["result_image_url"] = None
        reports_list.append(report_dict)

    return render_template("reports.html", reports=reports_list, current_filter=filter_type)


@app.route("/report/<int:report_id>", methods=["GET"])
@login_required
def report_detail(report_id):
    """报告详情页：展示单个分析任务的完整信息。"""
    conn = get_db_connection()
    report = conn.execute(
        "SELECT * FROM analysis_reports WHERE id = ?", (report_id,)
    ).fetchone()

    if report is None:
        conn.close()
        flash("报告不存在。", "error")
        return redirect(url_for("reports"))

    # 将 Row 对象转换为字典，以便使用 .get() 方法
    report_dict = dict(report)

    # 获取检测人员和审核人员信息
    inspector_info = None
    reviewer_info = None
    
    if report_dict.get("inspector_id"):
        inspector = conn.execute("SELECT real_name, position, signature_path FROM users WHERE id = ?", 
                                (report_dict["inspector_id"],)).fetchone()
        if inspector:
            inspector_info = dict(inspector)
    
    if report_dict.get("reviewer_id"):
        reviewer = conn.execute("SELECT real_name, position, signature_path FROM users WHERE id = ?", 
                               (report_dict["reviewer_id"],)).fetchone()
        if reviewer:
            reviewer_info = dict(reviewer)
    
    conn.close()

    # 生成结果图URL
    result_image_url = None
    if report_dict.get("result_filename"):
        result_image_url = url_for("static", filename=f"results/{report_dict['result_filename']}")
    
    # 解析粒径分布数据
    particle_data = None
    if report_dict.get("particle_distribution"):
        try:
            particle_data = json.loads(report_dict["particle_distribution"])
        except:
            particle_data = None
    
    # 解析定制化分析数据
    custom_config = None
    custom_metrics = None
    if report_dict.get("analysis_type") == "custom":
        if report_dict.get("custom_config"):
            try:
                custom_config = json.loads(report_dict["custom_config"])
            except:
                custom_config = None
        if report_dict.get("custom_metrics"):
            try:
                custom_metrics = json.loads(report_dict["custom_metrics"])
            except:
                custom_metrics = None

    return render_template("report_detail.html", 
                         report=report_dict, 
                         result_image_url=result_image_url, 
                         particle_data=particle_data,
                         custom_config=custom_config,
                         custom_metrics=custom_metrics,
                         inspector_info=inspector_info,
                         reviewer_info=reviewer_info)


@app.route("/report/<int:report_id>/visualization", methods=["GET"])
@login_required
def report_visualization(report_id):
    """可视化分析页面：展示定制化分析的详细图表和分析解释"""
    conn = get_db_connection()
    report = conn.execute(
        "SELECT * FROM analysis_reports WHERE id = ?", (report_id,)
    ).fetchone()

    if report is None:
        conn.close()
        flash("报告不存在。", "error")
        return redirect(url_for("reports"))

    # 将 Row 对象转换为字典
    report_dict = dict(report)
    conn.close()

    # 检查是否为定制化分析
    if report_dict.get("analysis_type") != "custom":
        flash("该报告不是定制化分析，无法查看可视化分析。", "error")
        return redirect(url_for("report_detail", report_id=report_id))

    # 解析定制化分析数据
    custom_metrics = None
    if report_dict.get("custom_metrics"):
        try:
            custom_metrics = json.loads(report_dict["custom_metrics"])
        except:
            custom_metrics = None

    if not custom_metrics:
        flash("该报告没有定制化分析数据。", "error")
        return redirect(url_for("report_detail", report_id=report_id))

    return render_template("visualization.html", 
                         report=report_dict,
                         custom_metrics=custom_metrics)


@app.route("/report/<int:report_id>/export", methods=["GET"])
@login_required
def export_report_pdf(report_id):
    """导出PDF格式的分析报告。"""
    if not REPORTLAB_AVAILABLE:
        flash("PDF生成功能需要安装 reportlab 库：pip install reportlab", "error")
        return redirect(url_for("report_detail", report_id=report_id))
    
    conn = get_db_connection()
    report = conn.execute(
        "SELECT * FROM analysis_reports WHERE id = ?", (report_id,)
    ).fetchone()
    conn.close()
    
    if report is None:
        flash("报告不存在。", "error")
        return redirect(url_for("reports"))
    
    # 将 Row 对象转换为字典
    report_dict = dict(report)
    
    # 解析粒径分布数据
    particle_data = None
    if report_dict.get("particle_distribution"):
        try:
            particle_data = json.loads(report_dict["particle_distribution"])
        except:
            particle_data = None
    
    # 解析定制化分析数据
    custom_metrics = None
    custom_config = None
    if report_dict.get("analysis_type") == "custom":
        if report_dict.get("custom_metrics"):
            try:
                custom_metrics = json.loads(report_dict["custom_metrics"])
            except:
                custom_metrics = None
        if report_dict.get("custom_config"):
            try:
                custom_config = json.loads(report_dict["custom_config"])
            except:
                custom_config = None
    
    # 注册中文字体（尝试使用系统字体）
    chinese_font_name = 'ChineseFont'
    chinese_font_bold_name = 'ChineseFontBold'
    
    # Windows系统常见中文字体路径
    font_paths = [
        'C:/Windows/Fonts/simhei.ttf',  # 黑体
        'C:/Windows/Fonts/simsun.ttc',  # 宋体
        'C:/Windows/Fonts/msyh.ttc',    # 微软雅黑
    ]
    
    font_registered = False
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                pdfmetrics.registerFont(TTFont(chinese_font_name, font_path))
                pdfmetrics.registerFont(TTFont(chinese_font_bold_name, font_path))
                font_registered = True
                break
            except:
                continue
    
    # 如果找不到系统字体，使用reportlab内置的字体（可能不支持中文，但至少不会报错）
    if not font_registered:
        chinese_font_name = 'Helvetica'
        chinese_font_bold_name = 'Helvetica-Bold'
    
    # 创建PDF - 改进布局和设计
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                           rightMargin=1.5*cm, leftMargin=1.5*cm,
                           topMargin=3*cm, bottomMargin=2.5*cm)
    story = []
    styles = getSampleStyleSheet()

    # 获取所有需要的时间戳
    current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 自定义页脚函数
    def add_footer(canvas, doc):
        canvas.saveState()
        # 设置页脚字体
        canvas.setFont(chinese_font_name, 8)
        canvas.setFillColor(colors.HexColor('#64748b'))

        # 页脚内容
        footer_text = f"金属材料微观图像智能分析云服务平台 - 第 {doc.page} 页 - 生成时间: {current_time_str}"
        canvas.drawString(1.5*cm, 1*cm, footer_text)

        # 添加分割线
        canvas.setStrokeColor(colors.HexColor('#e2e8f0'))
        canvas.setLineWidth(0.5)
        canvas.line(1.5*cm, 1.3*cm, A4[0]-1.5*cm, 1.3*cm)

        canvas.restoreState()

    # 页眉样式 - 公司信息
    header_style = ParagraphStyle(
        'HeaderStyle',
        parent=styles['Normal'],
        fontName=chinese_font_bold_name,
        fontSize=12,
        textColor=colors.HexColor('#1e40af'),
        alignment=TA_LEFT
    )

    # 标题样式（使用中文字体）- 更专业
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontName=chinese_font_bold_name,
        fontSize=22,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=20,
        alignment=TA_CENTER,
        borderColor=colors.HexColor('#e2e8f0'),
        borderWidth=1,
        borderPadding=10,
        backColor=colors.HexColor('#f8fafc')
    )
    
    # 正文样式（使用中文字体）
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontName=chinese_font_name,
        fontSize=10,
        textColor=colors.HexColor('#1e293b'),
    )
    
    # 标题2样式（使用中文字体）
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontName=chinese_font_bold_name,
        fontSize=14,
        textColor=colors.HexColor('#1e293b'),
    )
    
    # 添加页眉 - 公司信息
    header_text = """
    <b>金属材料微观图像智能分析云服务平台</b><br/>
    <font size="9" color="#64748b">智能检测 · 精准分析 · 专业报告</font>
    """
    story.append(Paragraph(header_text, header_style))
    story.append(Spacer(1, 0.3*cm))

    # 报告编号和生成信息
    report_number = f"RPT-{report_id:06d}"
    generation_time = datetime.now().strftime("%Y年%m月%d日 %H:%M")

    report_info_style = ParagraphStyle(
        'ReportInfo',
        parent=styles['Normal'],
        fontName=chinese_font_name,
        fontSize=9,
        textColor=colors.HexColor('#64748b'),
        alignment=TA_RIGHT
    )

    report_info_text = f"""
    报告编号: {report_number}<br/>
    生成时间: {generation_time}
    """
    story.append(Paragraph(report_info_text, report_info_style))
    story.append(Spacer(1, 0.3*cm))

    # 添加标题
    story.append(Paragraph("金属材料微观图像智能分析报告", title_style))
    story.append(Spacer(1, 0.8*cm))

    # 基本信息表格 - 改进样式
    analysis_type_text = "定制化分析" if report_dict.get("analysis_type") == "custom" else "普通分析"
    info_data = [
        ['报告编号', report_number],
        ['原始文件名', report_dict["original_filename"]],
        ['材料类型', normalize_material_type(report_dict.get("material_type", '未指定'))],
        ['分析类型', analysis_type_text],
        ['任务状态', '已完成'],
        ['创建时间', report_dict["created_at"]],
        ['报告生成时间', generation_time]
    ]
    
    info_table = Table(info_data, colWidths=[5*cm, 11*cm])
    info_table.setStyle(TableStyle([
        # 表头样式
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), chinese_font_bold_name),
        ('FONTSIZE', (0, 0), (-1, 0), 11),

        # 数据行样式
        ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#f1f5f9')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1e293b')),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('ALIGN', (0, 0), (0, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (0, -1), chinese_font_bold_name),
        ('FONTNAME', (1, 0), (1, -1), chinese_font_name),
        ('FONTSIZE', (0, 0), (-1, -1), 9),

        # 表格边框和间距
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),

        # 网格线
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('LINEBELOW', (0, 0), (-1, 0), 1, colors.HexColor('#1e40af')),

        # 斑马纹效果
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 1*cm))
    
    # 添加检测结果图 - 改进布局
    if report_dict.get("result_filename"):
        result_image_path = os.path.join(BASE_DIR, "static", "results", report_dict["result_filename"])
        if os.path.exists(result_image_path):
            # 检测结果图标题 - 更专业
            result_title_style = ParagraphStyle(
                'ResultTitle',
                parent=styles['Heading2'],
                fontName=chinese_font_bold_name,
                fontSize=16,
                textColor=colors.HexColor('#1e40af'),
                spaceAfter=15,
                spaceBefore=20
            )
            story.append(Paragraph("检测结果可视化", result_title_style))

            # 图片说明
            caption_style = ParagraphStyle(
                'ImageCaption',
                parent=styles['Normal'],
                fontName=chinese_font_name,
                fontSize=9,
                textColor=colors.HexColor('#64748b'),
                alignment=TA_CENTER,
                spaceAfter=10
            )

            analysis_type_display = "定制化分析" if report_dict.get("analysis_type") == "custom" else "标准分析"
            caption_text = f"图1: {analysis_type_display}结果 - {report_dict['original_filename']}"
            story.append(Paragraph(caption_text, caption_style))

            # 调整图片尺寸以适应页面
            img = Image(result_image_path, width=16*cm, height=12*cm)
            story.append(img)
            story.append(Spacer(1, 1.5*cm))
    
    # 添加定制化分析数据 - 改进样式
    if custom_metrics:
        # 分析指标标题
        metrics_title_style = ParagraphStyle(
            'MetricsTitle',
            parent=styles['Heading2'],
            fontName=chinese_font_bold_name,
            fontSize=16,
            textColor=colors.HexColor('#059669'),
            spaceAfter=15,
            spaceBefore=20
        )
        story.append(Paragraph("详细分析指标", metrics_title_style))

        # 关键指标表格
        metrics_data = [['指标类别', '指标名称', '数值', '单位']]

        # 形态学指标
        if custom_metrics.get("roundness_mean") is not None:
            metrics_data.append(['形态学', '平均圆度', f"{custom_metrics['roundness_mean']:.3f}", '无量纲'])
            if custom_metrics.get("roundness_std") is not None:
                metrics_data.append(['形态学', '圆度标准差', f"{custom_metrics['roundness_std']:.3f}", '无量纲'])

        if custom_metrics.get("aspect_ratio_mean") is not None:
            metrics_data.append(['形态学', '平均长轴短轴比', f"{custom_metrics['aspect_ratio_mean']:.3f}", '无量纲'])
            if custom_metrics.get("aspect_ratio_std") is not None:
                metrics_data.append(['形态学', '长轴短轴比标准差', f"{custom_metrics['aspect_ratio_std']:.3f}", '无量纲'])

        # 空间分布指标
        if custom_metrics.get("coverage_rate") is not None:
            metrics_data.append(['空间分布', '粉末遮盖率', f"{custom_metrics['coverage_rate']*100:.2f}", '%'])

        if custom_metrics.get("edge_touching_count") is not None and custom_metrics.get("total_particles") is not None:
            edge_ratio = custom_metrics['edge_touching_count'] / custom_metrics['total_particles'] * 100 if custom_metrics['total_particles'] > 0 else 0
            metrics_data.append(['空间分布', '边缘触碰率', f"{edge_ratio:.1f}", '%'])

        # 颗粒统计
        if custom_metrics.get("total_particles") is not None:
            metrics_data.append(['颗粒统计', '总颗粒数', str(custom_metrics['total_particles']), '个'])

        if custom_metrics.get("shape_distribution"):
            shape_dist = custom_metrics['shape_distribution']
            metrics_data.append(['颗粒统计', '圆形颗粒数', str(shape_dist.get('round', 0)), '个'])
            metrics_data.append(['颗粒统计', '不规则颗粒数', str(shape_dist.get('irregular', 0)), '个'])

        # 纹理分析指标
        if custom_metrics.get("glcm_contrast_mean") is not None:
            metrics_data.append(['纹理分析', 'GLCM对比度', f"{custom_metrics['glcm_contrast_mean']:.3f}", '无量纲'])

        if custom_metrics.get("glcm_energy_mean") is not None:
            metrics_data.append(['纹理分析', 'GLCM能量', f"{custom_metrics['glcm_energy_mean']:.3f}", '无量纲'])

        if custom_metrics.get("glcm_homogeneity_mean") is not None:
            metrics_data.append(['纹理分析', 'GLCM同质性', f"{custom_metrics['glcm_homogeneity_mean']:.3f}", '无量纲'])

        if custom_metrics.get("lbp_entropy_mean") is not None:
            metrics_data.append(['纹理分析', 'LBP纹理熵', f"{custom_metrics['lbp_entropy_mean']:.3f}", '比特'])

        if len(metrics_data) > 1:  # 如果有数据（除了表头）
            metrics_table = Table(metrics_data, colWidths=[3*cm, 5*cm, 4*cm, 3*cm])
            metrics_table.setStyle(TableStyle([
                # 表头样式
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#059669')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), chinese_font_bold_name),
                ('FONTSIZE', (0, 0), (-1, 0), 10),

                # 数据行样式
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#1e293b')),
                ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
                ('ALIGN', (1, 1), (1, -1), 'LEFT'),
                ('ALIGN', (2, 1), (2, -1), 'RIGHT'),
                ('FONTNAME', (0, 1), (-1, -1), chinese_font_name),
                ('FONTSIZE', (0, 1), (-1, -1), 9),

                # 不同类别使用不同背景色
                ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#f0fdf4')),  # 浅绿色用于类别列
                ('ROWBACKGROUNDS', (1, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),

                # 边框和间距
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),

                # 网格线
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
                ('LINEBELOW', (0, 0), (-1, 0), 1, colors.HexColor('#059669')),
            ]))
            story.append(metrics_table)
            story.append(Spacer(1, 1.2*cm))
        
        # 为定制化分析生成可视化图表 - 改进标题
        viz_title_style = ParagraphStyle(
            'VizTitle',
            parent=styles['Heading2'],
            fontName=chinese_font_bold_name,
            fontSize=16,
            textColor=colors.HexColor('#7c3aed'),
            spaceAfter=15,
            spaceBefore=25
        )
        story.append(Paragraph("数据可视化分析", viz_title_style))

        # 添加说明文字
        viz_desc_style = ParagraphStyle(
            'VizDesc',
            parent=styles['Normal'],
            fontName=chinese_font_name,
            fontSize=10,
            textColor=colors.HexColor('#64748b'),
            spaceAfter=20
        )
        story.append(Paragraph("以下图表展示了颗粒形态、分布和纹理特征的详细分析结果：", viz_desc_style))

        # 收集所有生成的图表
        charts = []

        # 智能图表布局函数
        def arrange_charts_smart_layout(charts_list, story):
            """智能排列图表：统一尺寸，单列垂直布局，美观大方"""
            if not charts_list:
                return

            # 为所有图表设置统一的显示尺寸，让布局更美观统一
            for chart in charts_list:
                # 统一调整为12cm x 7cm的显示尺寸，所有图表都一样大
                chart['display_width'] = 12 * cm
                chart['display_height'] = 7 * cm

            # 所有图表单独一行，垂直排列，布局更清晰美观
            for chart in charts_list:
                # 创建居中的单图表格
                table_data = [[chart['image']]]
                chart_table = Table(table_data, colWidths=[chart['display_width']])
                chart_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 8),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                    ('TOPPADDING', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                    # 添加边框让图表更有层次感
                    ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                ]))
                story.append(chart_table)
                story.append(Spacer(1, 0.5*cm))  # 适当的图表间距
        
        # 1. 形状分布饼图（Round vs Irregular）
        if custom_metrics.get("shape_distribution"):
            try:
                shape_dist = custom_metrics['shape_distribution']
                round_count = shape_dist.get('round', 0)
                irregular_count = shape_dist.get('irregular', 0)
                
                if round_count + irregular_count > 0:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    colors_pie = ['#3b82f6', '#ef4444']
                    labels = ['Round (圆形)', 'Irregular (不规则)']
                    sizes = [round_count, irregular_count]
                    explode = (0.05, 0.05)
                    
                    ax.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                          autopct='%1.1f%%', shadow=True, startangle=90,
                          textprops={'fontsize': 11, 'fontweight': 'bold'})
                    ax.set_title('形状分布', fontsize=14, fontweight='bold', pad=20)
                    
                    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
                    plt.rcParams['axes.unicode_minus'] = False
                    plt.tight_layout()
                    
                    temp_chart = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                    temp_chart_path = temp_chart.name
                    plt.savefig(temp_chart_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    chart_img = Image(temp_chart_path, width=12*cm, height=9*cm)
                    charts.append({
                        'image': chart_img,
                        'width': 12*cm,
                        'height': 9*cm,
                        'title': '形状分布饼图',
                        'size_category': 'medium'  # 可以与其他中等大小图表配对
                    })
                    
                    try:
                        os.unlink(temp_chart_path)
                    except:
                        pass
            except Exception as e:
                print(f"生成形状分布饼图时出错: {e}")
        
        # 2. 圆度分布直方图
        if custom_metrics.get("roundness_values") and len(custom_metrics['roundness_values']) > 0:
            try:
                roundness_data = custom_metrics['roundness_values']
                fig, ax = plt.subplots(figsize=(10, 6))
                
                bins = 15
                ax.hist(roundness_data, bins=bins, color='#8b5cf6', alpha=0.7, edgecolor='#6d28d9', linewidth=1)
                ax.set_xlabel('圆度值', fontsize=12, fontweight='bold')
                ax.set_ylabel('颗粒数量', fontsize=12, fontweight='bold')
                ax.set_title('圆度分布直方图（值越接近1越圆）', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
                
                plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
                plt.tight_layout()
                
                temp_chart = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                temp_chart_path = temp_chart.name
                plt.savefig(temp_chart_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                chart_img = Image(temp_chart_path, width=10*cm, height=6*cm)
                charts.append({
                    'image': chart_img,
                    'width': 10*cm,
                    'height': 6*cm,
                    'title': '圆度分布直方图',
                    'size_category': 'medium'
                })
                
                try:
                    os.unlink(temp_chart_path)
                except:
                    pass
            except Exception as e:
                print(f"生成圆度分布直方图时出错: {e}")
        
        # 3. 长轴短轴比分布直方图
        if custom_metrics.get("aspect_ratio_values") and len(custom_metrics['aspect_ratio_values']) > 0:
            try:
                aspect_ratio_data = custom_metrics['aspect_ratio_values']
                fig, ax = plt.subplots(figsize=(10, 6))
                
                bins = 15
                ax.hist(aspect_ratio_data, bins=bins, color='#3b82f6', alpha=0.7, edgecolor='#1e40af', linewidth=1)
                ax.set_xlabel('长轴短轴比', fontsize=12, fontweight='bold')
                ax.set_ylabel('颗粒数量', fontsize=12, fontweight='bold')
                ax.set_title('长轴短轴比分布直方图（值越大越细长）', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
                
                plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
                plt.tight_layout()
                
                temp_chart = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                temp_chart_path = temp_chart.name
                plt.savefig(temp_chart_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                chart_img = Image(temp_chart_path, width=10*cm, height=6*cm)
                charts.append({
                    'image': chart_img,
                    'width': 10*cm,
                    'height': 6*cm,
                    'title': '圆度分布直方图',
                    'size_category': 'medium'
                })
                
                try:
                    os.unlink(temp_chart_path)
                except:
                    pass
            except Exception as e:
                print(f"生成长轴短轴比分布直方图时出错: {e}")
        
        # 4. 形状分布散点图（圆度 vs 长轴短轴比）
        if (custom_metrics.get("roundness_values") and custom_metrics.get("aspect_ratio_values") and
            len(custom_metrics['roundness_values']) > 0 and len(custom_metrics['aspect_ratio_values']) > 0):
            try:
                roundness_values = custom_metrics['roundness_values']
                aspect_ratio_values = custom_metrics['aspect_ratio_values']
                
                # 如果有particle_details，使用它来区分Round和Irregular
                if custom_metrics.get("particle_details"):
                    particle_details = custom_metrics['particle_details']
                    round_data = {'x': [], 'y': []}
                    irregular_data = {'x': [], 'y': []}
                    
                    for particle in particle_details:
                        if particle.get('roundness') is not None and particle.get('aspect_ratio') is not None:
                            if particle.get('shape') == 'round':
                                round_data['x'].append(particle['aspect_ratio'])
                                round_data['y'].append(particle['roundness'])
                            else:
                                irregular_data['x'].append(particle['aspect_ratio'])
                                irregular_data['y'].append(particle['roundness'])
                else:
                    # 否则根据圆度值分类（>=0.7为round）
                    round_data = {'x': [], 'y': []}
                    irregular_data = {'x': [], 'y': []}
                    min_len = min(len(roundness_values), len(aspect_ratio_values))
                    for i in range(min_len):
                        if roundness_values[i] >= 0.7:
                            round_data['x'].append(aspect_ratio_values[i])
                            round_data['y'].append(roundness_values[i])
                        else:
                            irregular_data['x'].append(aspect_ratio_values[i])
                            irregular_data['y'].append(roundness_values[i])
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                if round_data['x']:
                    ax.scatter(round_data['x'], round_data['y'], c='#3b82f6', alpha=0.6, 
                             s=30, label='Round (圆形)', edgecolors='#1e40af', linewidths=0.5)
                if irregular_data['x']:
                    ax.scatter(irregular_data['x'], irregular_data['y'], c='#ef4444', alpha=0.6,
                             s=30, label='Irregular (不规则)', edgecolors='#dc2626', linewidths=0.5)
                
                ax.set_xlabel('长轴短轴比（值越大越细长）', fontsize=12, fontweight='bold')
                ax.set_ylabel('圆度（值越接近1越圆）', fontsize=12, fontweight='bold')
                ax.set_title('颗粒形状分布散点图', fontsize=14, fontweight='bold')
                ax.legend(loc='best', fontsize=10)
                ax.grid(True, alpha=0.3, linestyle='--')
                
                plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
                plt.tight_layout()
                
                temp_chart = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                temp_chart_path = temp_chart.name
                plt.savefig(temp_chart_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                chart_img = Image(temp_chart_path, width=14*cm, height=11*cm)
                charts.append({
                    'image': chart_img,
                    'width': 14*cm,
                    'height': 11*cm,
                    'title': '形状分布散点图',
                    'size_category': 'large'
                })
                
                try:
                    os.unlink(temp_chart_path)
                except:
                    pass
            except Exception as e:
                print(f"生成形状分布散点图时出错: {e}")
        
        # 5. 边缘触碰对比图
        if (custom_metrics.get("edge_touching_count") is not None and 
            custom_metrics.get("total_particles") is not None and
            custom_metrics.get("roundness_values") and custom_metrics.get("aspect_ratio_values")):
            try:
                edge_touching_count = custom_metrics['edge_touching_count']
                total_particles = custom_metrics['total_particles']
                not_touching_count = total_particles - edge_touching_count
                
                # 计算平均值
                roundness_values = custom_metrics['roundness_values']
                aspect_ratio_values = custom_metrics['aspect_ratio_values']
                
                # 如果有particle_details，使用它来区分
                if custom_metrics.get("particle_details"):
                    particle_details = custom_metrics['particle_details']
                    touching_roundness = [p['roundness'] for p in particle_details if p.get('edge_touching') and p.get('roundness') is not None]
                    touching_aspect = [p['aspect_ratio'] for p in particle_details if p.get('edge_touching') and p.get('aspect_ratio') is not None]
                    not_touching_roundness = [p['roundness'] for p in particle_details if not p.get('edge_touching') and p.get('roundness') is not None]
                    not_touching_aspect = [p['aspect_ratio'] for p in particle_details if not p.get('edge_touching') and p.get('aspect_ratio') is not None]
                else:
                    # 简单分配：前edge_touching_count个作为触碰边缘
                    touching_roundness = roundness_values[:edge_touching_count] if len(roundness_values) >= edge_touching_count else []
                    touching_aspect = aspect_ratio_values[:edge_touching_count] if len(aspect_ratio_values) >= edge_touching_count else []
                    not_touching_roundness = roundness_values[edge_touching_count:] if len(roundness_values) > edge_touching_count else []
                    not_touching_aspect = aspect_ratio_values[edge_touching_count:] if len(aspect_ratio_values) > edge_touching_count else []
                
                avg_touching_roundness = np.mean(touching_roundness) if touching_roundness else 0
                avg_touching_aspect = np.mean(touching_aspect) if touching_aspect else 0
                avg_not_touching_roundness = np.mean(not_touching_roundness) if not_touching_roundness else 0
                avg_not_touching_aspect = np.mean(not_touching_aspect) if not_touching_aspect else 0
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                categories = ['平均圆度', '平均长轴短轴比', '数量占比 (%)']
                touching_data = [
                    avg_touching_roundness * 100,  # 归一化到0-100
                    avg_touching_aspect * 30,  # 放大30倍
                    (edge_touching_count / total_particles) * 100 if total_particles > 0 else 0
                ]
                not_touching_data = [
                    avg_not_touching_roundness * 100,
                    avg_not_touching_aspect * 30,
                    (not_touching_count / total_particles) * 100 if total_particles > 0 else 0
                ]
                
                x = np.arange(len(categories))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, touching_data, width, label='触碰边缘', 
                              color='#fbbf24', alpha=0.8, edgecolor='#f59e0b', linewidth=1)
                bars2 = ax.bar(x + width/2, not_touching_data, width, label='未触碰边缘',
                              color='#3b82f6', alpha=0.8, edgecolor='#1e40af', linewidth=1)
                
                ax.set_xlabel('对比维度', fontsize=12, fontweight='bold')
                ax.set_ylabel('归一化值', fontsize=12, fontweight='bold')
                ax.set_title('边缘触碰影响分析', fontsize=14, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(categories)
                ax.legend(loc='best', fontsize=10)
                ax.grid(True, alpha=0.3, linestyle='--', axis='y')
                
                plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
                plt.tight_layout()
                
                temp_chart = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                temp_chart_path = temp_chart.name
                plt.savefig(temp_chart_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                chart_img = Image(temp_chart_path, width=10*cm, height=6*cm)
                charts.append({
                    'image': chart_img,
                    'width': 10*cm,
                    'height': 6*cm,
                    'title': '圆度分布直方图',
                    'size_category': 'medium'
                })
                
                try:
                    os.unlink(temp_chart_path)
                except:
                    pass
            except Exception as e:
                print(f"生成边缘触碰对比图时出错: {e}")
        
        # 6. GLCM特征对比图
        if (custom_metrics.get("glcm_contrast_mean") is not None or 
            custom_metrics.get("glcm_energy_mean") is not None or
            custom_metrics.get("glcm_homogeneity_mean") is not None):
            try:
                glcm_data = []
                glcm_labels = []
                
                if custom_metrics.get("glcm_contrast_mean") is not None:
                    glcm_data.append(custom_metrics['glcm_contrast_mean'])
                    glcm_labels.append('对比度')
                if custom_metrics.get("glcm_energy_mean") is not None:
                    glcm_data.append(custom_metrics['glcm_energy_mean'])
                    glcm_labels.append('能量')
                if custom_metrics.get("glcm_homogeneity_mean") is not None:
                    glcm_data.append(custom_metrics['glcm_homogeneity_mean'])
                    glcm_labels.append('同质性')
                
                if glcm_data:
                    max_value = max(glcm_data)
                    normalized_data = [(val / max_value) * 100 for val in glcm_data] if max_value > 0 else glcm_data
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    colors_bar = ['#22c55e', '#fbbf24', '#3b82f6']
                    bars = ax.bar(glcm_labels, normalized_data, color=colors_bar[:len(glcm_labels)], 
                                 alpha=0.8, edgecolor='black', linewidth=1)
                    
                    ax.set_ylabel('归一化值 (%)', fontsize=12, fontweight='bold')
                    ax.set_title('GLCM纹理特征对比', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
                    
                    # 在柱子上显示数值
                    for bar, val in zip(bars, normalized_data):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{val:.1f}%',
                               ha='center', va='bottom', fontsize=10, fontweight='bold')
                    
                    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
                    plt.rcParams['axes.unicode_minus'] = False
                    plt.tight_layout()
                    
                    temp_chart = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                    temp_chart_path = temp_chart.name
                    plt.savefig(temp_chart_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    chart_img = Image(temp_chart_path, width=10*cm, height=6*cm)
                    charts.append({
                        'image': chart_img,
                        'width': 10*cm,
                        'height': 6*cm,
                        'title': '边缘触碰对比图',
                        'size_category': 'medium'
                    })
                    
                    try:
                        os.unlink(temp_chart_path)
                    except:
                        pass
            except Exception as e:
                print(f"生成GLCM特征对比图时出错: {e}")

        # 使用智能布局排列所有图表
        arrange_charts_smart_layout(charts, story)

    # 添加粒径分布数据（普通分析）
    if particle_data and not custom_metrics:
        story.append(Paragraph("粒径分布统计", heading2_style))
        story.append(Spacer(1, 0.3*cm))
        
        # 统计信息
        stats_text = f"总检测数量: {particle_data['total']} 个颗粒"
        story.append(Paragraph(stats_text, normal_style))
        story.append(Spacer(1, 0.5*cm))
        
        # 生成粒径分布直方图
        try:
            # 设置中文字体（如果系统有的话）
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 解析区间标签，提取中间值用于绘图
            bin_centers = []
            for bin_label in particle_data['bins']:
                # 解析 "8.5-35.5" 这样的格式
                parts = bin_label.split('-')
                if len(parts) == 2:
                    center = (float(parts[0]) + float(parts[1])) / 2
                    bin_centers.append(center)
                else:
                    bin_centers.append(0)
            
            # 绘制柱状图
            bars = ax.bar(range(len(particle_data['counts'])), particle_data['counts'], 
                         color='#3b82f6', alpha=0.7, edgecolor='#1e40af', linewidth=1)
            
            # 设置标签和标题
            ax.set_xlabel('粒径大小（像素）', fontsize=12)
            ax.set_ylabel('数量', fontsize=12)
            ax.set_title(f'粒径分布直方图（总计 {particle_data["total"]} 个颗粒）', fontsize=14, fontweight='bold')
            
            # 设置x轴标签
            ax.set_xticks(range(len(particle_data['bins'])))
            ax.set_xticklabels(particle_data['bins'], rotation=45, ha='right', fontsize=9)
            
            # 添加网格
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # 在柱子上显示数值
            for i, (bar, count) in enumerate(zip(bars, particle_data['counts'])):
                if count > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(count)}',
                           ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            
            # 保存为临时文件
            temp_chart = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            temp_chart_path = temp_chart.name
            plt.savefig(temp_chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # 将图表插入PDF
            chart_img = Image(temp_chart_path, width=10*cm, height=6*cm)
            charts.append({
                'image': chart_img,
                'width': 10*cm,
                'height': 6*cm,
                'title': 'GLCM特征对比图',
                'size_category': 'medium'
            })

            # 删除临时文件
            try:
                os.unlink(temp_chart_path)
            except:
                pass
                
        except Exception as e:
            print(f"生成直方图时出错: {e}")
            # 如果图表生成失败，继续显示表格
        
        # 粒径分布表格
        dist_data = [['粒径区间（像素）', '数量']]
        for i, bin_label in enumerate(particle_data['bins']):
            dist_data.append([bin_label, str(particle_data['counts'][i])])
        
        dist_table = Table(dist_data, colWidths=[7*cm, 7*cm])
        dist_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), chinese_font_bold_name),
            ('FONTNAME', (0, 1), (-1, -1), chinese_font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
        ]))
        story.append(dist_table)
    
    # 添加签名区域 - 改进布局
    story.append(Spacer(1, 2*cm))

    # 签名区域标题
    signature_title_style = ParagraphStyle(
        'SignatureTitle',
        parent=styles['Heading2'],
        fontName=chinese_font_bold_name,
        fontSize=14,
        textColor=colors.HexColor('#dc2626'),
        spaceAfter=20,
        alignment=TA_CENTER
    )
    story.append(Paragraph("报告审核签署", signature_title_style))

    # 添加分割线
    story.append(Spacer(1, 0.5*cm))
    
    # 获取检测人员和审核人员信息
    inspector_info = None
    reviewer_info = None
    
    conn = get_db_connection()
    if report_dict.get("inspector_id"):
        inspector = conn.execute("SELECT real_name, position, signature_path FROM users WHERE id = ?", 
                                (report_dict["inspector_id"],)).fetchone()
        if inspector:
            inspector_info = dict(inspector)
    
    if report_dict.get("reviewer_id"):
        reviewer = conn.execute("SELECT real_name, position, signature_path FROM users WHERE id = ?", 
                               (report_dict["reviewer_id"],)).fetchone()
        if reviewer:
            reviewer_info = dict(reviewer)
    conn.close()
    
    # 创建签名表格
    inspector_name = "未指定"
    if inspector_info:
        inspector_real_name = inspector_info.get("real_name") or ""
        inspector_position = inspector_info.get("position") or ""
        if inspector_real_name or inspector_position:
            inspector_name = (inspector_real_name + "\n" + inspector_position).strip()
    
    reviewer_name = "未指定"
    if reviewer_info:
        reviewer_real_name = reviewer_info.get("real_name") or ""
        reviewer_position = reviewer_info.get("position") or ""
        if reviewer_real_name or reviewer_position:
            reviewer_name = (reviewer_real_name + "\n" + reviewer_position).strip()
    
    signature_data = [
        ['检测人员', '审核人员'],
        [inspector_name, reviewer_name]
    ]
    
    # 如果有签名图片，添加签名图片行
    inspector_signature_path = None
    reviewer_signature_path = None
    
    if inspector_info and inspector_info["signature_path"]:
        inspector_signature_path = os.path.join(BASE_DIR, "static", inspector_info["signature_path"])
        if not os.path.exists(inspector_signature_path):
            inspector_signature_path = None
    
    if reviewer_info and reviewer_info["signature_path"]:
        reviewer_signature_path = os.path.join(BASE_DIR, "static", reviewer_info["signature_path"])
        if not os.path.exists(reviewer_signature_path):
            reviewer_signature_path = None
    
    # 创建签名表格（包含签名图片）
    signature_table_data = [['检测人员', '审核人员']]
    
    # 第一行：签名图片
    signature_row = []
    if inspector_signature_path:
        try:
            sig_img = Image(inspector_signature_path, width=4*cm, height=2*cm)
            signature_row.append(sig_img)
        except:
            signature_row.append("")
    else:
        signature_row.append("")
    
    if reviewer_signature_path:
        try:
            sig_img = Image(reviewer_signature_path, width=4*cm, height=2*cm)
            signature_row.append(sig_img)
        except:
            signature_row.append("")
    else:
        signature_row.append("")
    
    signature_table_data.append(signature_row)
    
    # 第二行：姓名和职位
    inspector_name_row = "未指定"
    if inspector_info:
        inspector_real_name = inspector_info.get("real_name") or ""
        inspector_position = inspector_info.get("position") or ""
        if inspector_real_name or inspector_position:
            inspector_name_row = (inspector_real_name + "\n" + inspector_position).strip()
    
    reviewer_name_row = "未指定"
    if reviewer_info:
        reviewer_real_name = reviewer_info.get("real_name") or ""
        reviewer_position = reviewer_info.get("position") or ""
        if reviewer_real_name or reviewer_position:
            reviewer_name_row = (reviewer_real_name + "\n" + reviewer_position).strip()
    
    name_row = [inspector_name_row, reviewer_name_row]
    signature_table_data.append(name_row)
    
    # 第三行：日期
    current_date = datetime.now().strftime("%Y年%m月%d日")
    date_row = [current_date, current_date]
    signature_table_data.append(date_row)
    
    signature_table = Table(signature_table_data, colWidths=[7*cm, 7*cm])
    signature_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), chinese_font_bold_name),
        ('FONTNAME', (0, 1), (-1, -1), chinese_font_name),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
    ]))
    story.append(signature_table)

    # 生成PDF - 添加页脚
    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    buffer.seek(0)
    
    # 生成文件名
    safe_filename = report_dict["original_filename"].rsplit('.', 1)[0] if '.' in report_dict["original_filename"] else report_dict["original_filename"]
    pdf_filename = f"分析报告_{safe_filename}_{report_id}.pdf"
    
    return send_file(
        buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=pdf_filename
    )


@app.route("/report/<int:report_id>/delete", methods=["POST"])
@login_required
def delete_report(report_id):
    """删除分析报告：删除数据库记录和对应的文件。"""
    conn = get_db_connection()
    report = conn.execute(
        "SELECT * FROM analysis_reports WHERE id = ?", (report_id,)
    ).fetchone()
    
    if report is None:
        conn.close()
        flash("报告不存在。", "error")
        return redirect(url_for("reports"))
    
    # 删除文件
    try:
        # 将 Row 对象转换为字典
        report_dict = dict(report)
        
        # 删除原始上传图片
        upload_path = os.path.join(BASE_DIR, "uploads", report_dict["saved_filename"])
        if os.path.exists(upload_path):
            os.remove(upload_path)
        
        # 删除检测结果图
        if report_dict.get("result_filename"):
            result_path = os.path.join(BASE_DIR, "static", "results", report_dict["result_filename"])
            if os.path.exists(result_path):
                os.remove(result_path)
    except Exception as e:
        # 文件删除失败不影响数据库删除
        print(f"删除文件时出错: {e}")
    
    # 删除数据库记录
    conn.execute("DELETE FROM analysis_reports WHERE id = ?", (report_id,))
    conn.commit()
    conn.close()
    
    flash("分析记录已删除。", "success")
    return redirect(url_for("reports"))


if __name__ == "__main__":
    # 运行方式：python webapp/app.py
    # 然后浏览器访问 http://127.0.0.1:5000
    app.run(debug=True, host="0.0.0.0", port=5000)
