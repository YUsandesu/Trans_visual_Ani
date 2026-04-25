import os

# 1. 在导入 py5 之前，配置 Java 环境
java_home_path = r"C:\Users\admin\anaconda3\envs\Processing\Library\lib\jvm"
os.environ["JAVA_HOME"] = java_home_path

# 2. 导入依赖
import py5
import numpy as np

# ==========================================
# 分辨率与缩放配置区
# ==========================================
TARGET_WIDTH = 1920  # 输出宽度
TARGET_HEIGHT = 1080 # 输出高度

BASE_WIDTH = 900     # 原始逻辑宽度
BASE_HEIGHT = 500    # 原始逻辑高度

# 初始序列定义
tokens = ["The", "cat", "sat", "on", "the", "mat"]
n_tokens = len(tokens)

# 初始注意力权重
np.random.seed(42)
weights = np.random.dirichlet(np.ones(n_tokens)*0.7, size=n_tokens)

# 动画时间轴定义
frames_per_phase1 = 100  
phase1_duration = frames_per_phase1  
phase2_duration = 50                               
phase3_duration = 30                               
phase4_duration = 30                               
phase5_duration = 30                               
total_cycle_frames = phase1_duration + phase2_duration + phase3_duration + phase4_duration + phase5_duration

# 状态变量
output_numbers = []
depth = 0  

def setup():
    py5.size(TARGET_WIDTH, TARGET_HEIGHT)
    py5.frame_rate(30)
    py5.text_align(py5.CENTER, py5.CENTER)
    py5.text_font(py5.create_font("Arial", 24))
    generate_new_numbers()
    
    # 确保保存路径存在
    if not os.path.exists("frames"):
        os.makedirs("frames")

def generate_new_numbers():
    global output_numbers
    output_numbers = [str(np.random.randint(1000, 10000)) for _ in range(n_tokens)]

def step_into_next_layer():
    global tokens, depth, weights
    tokens = list(output_numbers)
    depth += 1
    generate_new_numbers()
    weights = np.random.dirichlet(np.ones(n_tokens)*0.7, size=n_tokens)

def draw():
    global output_numbers, depth
    
    py5.background(255)
    
    # 等比缩放与居中
    scale_factor = min(TARGET_WIDTH / BASE_WIDTH, TARGET_HEIGHT / BASE_HEIGHT)
    offset_x = (TARGET_WIDTH - BASE_WIDTH * scale_factor) / 2
    offset_y = (TARGET_HEIGHT - BASE_HEIGHT * scale_factor) / 2
    
    py5.translate(offset_x, offset_y)
    py5.scale(scale_factor)
    
    cycle_frame = py5.frame_count % total_cycle_frames
    
    if cycle_frame == 0 and py5.frame_count > 0:
        step_into_next_layer()
        
    margin = 100
    spacing = (BASE_WIDTH - 2 * margin) / (n_tokens - 1)
    y_tokens = BASE_HEIGHT / 2 - 20
    y_numbers = BASE_HEIGHT / 2 + 100
    
    num_spacing = 70 
    total_shrink_width = (n_tokens - 1) * num_spacing
    start_target_x = BASE_WIDTH / 2 - total_shrink_width / 2

    py5.fill(0)
    py5.text_size(16)
    py5.text(f"Transformer Layer Depth: {depth}", BASE_WIDTH / 2, 30)

    t1 = phase1_duration
    t2 = t1 + phase2_duration
    t3 = t2 + phase3_duration
    t4 = t3 + phase4_duration

    # 阶段 1: 并行注意力计算
    if cycle_frame < t1:
        local_frame = cycle_frame
        for src_idx in range(n_tokens):
            draw_attention_lines(src_idx, local_frame, margin, spacing, y_tokens)
        
        for i, token in enumerate(tokens):
            x = margin + i * spacing
            py5.fill(255, 215, 0)
            py5.text_size(32)
            py5.text(token, x, y_tokens)
            
            alpha = py5.remap(local_frame, phase1_duration * 0.4, phase1_duration, 0, 255)
            py5.fill(0, 150, 0, py5.constrain(alpha, 0, 255))
            py5.text_size(24)
            py5.text(output_numbers[i], x, y_numbers)

    # 阶段 2: 汇聚阶段
    elif cycle_frame < t2:
        progress = (cycle_frame - t1) / phase2_duration
        ease_p = -(py5.cos(py5.PI * progress) - 1) / 2
        for i, token in enumerate(tokens):
            x = margin + i * spacing
            py5.fill(0)
            py5.text_size(20)
            py5.text(token, x, y_tokens)
            
            orig_x = x
            target_x = start_target_x + i * num_spacing
            current_x = py5.lerp(orig_x, target_x, ease_p)
            py5.fill(0, 150, 0)
            py5.text_size(24)
            py5.text(output_numbers[i], current_x, y_numbers)

    # 阶段 3: 括号展示阶段
    elif cycle_frame < t3:
        for i, token in enumerate(tokens):
            x = margin + i * spacing
            py5.fill(0); py5.text_size(20); py5.text(token, x, y_tokens)
            target_x = start_target_x + i * num_spacing
            py5.fill(0, 150, 0); py5.text_size(24); py5.text(output_numbers[i], target_x, y_numbers)
            
        bracket_progress = (cycle_frame - t2) / (phase3_duration * 0.5)
        bracket_alpha = py5.constrain(bracket_progress * 255, 0, 255)
        py5.fill(0, bracket_alpha); py5.text_size(48)
        py5.text("[", start_target_x - 30, y_numbers - 5) 
        py5.text("]", start_target_x + total_shrink_width + 30, y_numbers - 5)

    # 阶段 4: 上移淡出阶段
    elif cycle_frame < t4:
        progress = (cycle_frame - t3) / phase4_duration
        ease_p = -(py5.cos(py5.PI * progress) - 1) / 2
        current_y = py5.lerp(y_numbers, y_tokens, ease_p)
        old_token_alpha = py5.lerp(80, 0, progress)
        bracket_alpha = py5.lerp(255, 0, progress)
        for i, token in enumerate(tokens):
            x = margin + i * spacing
            py5.fill(0, old_token_alpha); py5.text_size(20); py5.text(token, x, y_tokens)
            target_x = start_target_x + i * num_spacing
            py5.fill(0, 150, 0); py5.text_size(24); py5.text(output_numbers[i], target_x, current_y)
        py5.fill(0, bracket_alpha); py5.text_size(48)
        py5.text("[", start_target_x - 30, current_y - 5); py5.text("]", start_target_x + total_shrink_width + 30, current_y - 5)

    # 阶段 5: 重新展开阶段
    else:
        progress = (cycle_frame - t4) / phase5_duration
        ease_p = -(py5.cos(py5.PI * progress) - 1) / 2
        for i, token in enumerate(tokens):
            orig_x = margin + i * spacing
            target_x = start_target_x + i * num_spacing
            current_x = py5.lerp(target_x, orig_x, ease_p)
            py5.fill(0); t_size = py5.lerp(24, 20, progress)
            py5.text_size(t_size); py5.text(output_numbers[i], current_x, y_tokens)

    # 逐帧保存功能：将每一帧保存到 frames 文件夹中
    py5.save_frame("frames/frame_####.png")

def draw_attention_lines(current_source_idx, local_frame, margin, spacing, y_pos):
    draw_duration = 0.7
    line_progress = py5.remap(local_frame, 0, phase1_duration * draw_duration, 0, 1)
    line_progress = min(1.0, line_progress)
    x_start = margin + current_source_idx * spacing
    for i in range(n_tokens):
        weight = weights[current_source_idx][i]
        alpha = py5.remap(weight, 0, 1, 10, 120) 
        py5.stroke(0, 100, 200, alpha)
        py5.stroke_weight(py5.remap(weight, 0, 1, 1, 4))
        py5.no_fill()
        x_target = margin + i * spacing
        ctrl_y = y_pos - 150
        steps = int(30 * line_progress)
        if steps < 1: continue
        py5.begin_shape()
        for t in np.linspace(0, line_progress, steps):
            inv_t = 1 - t
            px = inv_t**2 * x_start + 2 * inv_t * t * ((x_start + x_target) / 2) + t**2 * x_target
            py = inv_t**2 * (y_pos - 30) + 2 * inv_t * t * ctrl_y + t**2 * (y_pos - 30)
            py5.vertex(px, py)
        py5.end_shape()

if __name__ == "__main__":
    py5.run_sketch()