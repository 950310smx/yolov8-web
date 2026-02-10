-- 用户表：用于登录 / 注册
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,          -- 用户名，唯一
    email TEXT NOT NULL UNIQUE,             -- 邮箱，唯一
    password_hash TEXT NOT NULL,            -- 使用 werkzeug 生成的密码哈希
    role TEXT DEFAULT 'user',               -- 用户角色：user(普通用户), member(会员), admin(管理员), analyst(分析师)
    membership_status TEXT DEFAULT 'none',  -- 会员申请状态：none(无申请), pending(待审核), approved(已通过), rejected(已拒绝)
    real_name TEXT,                         -- 真实姓名
    position TEXT,                          -- 职位/职称（如：质量工程师、检测员、审核员等）
    department TEXT,                        -- 部门/单位
    phone TEXT,                             -- 联系电话
    signature_path TEXT,                     -- 签名图片路径
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- 注册时间
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- 更新时间
);

-- 分析任务/报告表：存储每次上传和分析的记录
CREATE TABLE IF NOT EXISTS analysis_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,                        -- 关联用户ID（当前简化版可先不关联，后续可加外键）
    original_filename TEXT NOT NULL,        -- 原始文件名
    saved_filename TEXT NOT NULL,           -- 保存后的文件名（带时间戳）
    result_filename TEXT,                    -- 检测结果图文件名
    material_type TEXT,                     -- 材料类型（粉末/金属）
    analysis_type TEXT DEFAULT 'normal',    -- 分析类型：normal(普通分析), custom(定制化分析)
    custom_config TEXT,                     -- 定制化分析配置（JSON格式：包含输出选项、参数选择等）
    custom_metrics TEXT,                     -- 定制化分析参数（JSON格式：包含各种计算出的参数）
    status TEXT DEFAULT 'completed',         -- 任务状态：completed/failed/processing
    particle_distribution TEXT,             -- 粒径分布数据（JSON格式：{"bins": [区间], "counts": [数量]})
    inspector_id INTEGER,                   -- 检测人员ID（关联users表）
    reviewer_id INTEGER,                    -- 审核人员ID（关联users表）
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- 创建时间
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- 更新时间
);


