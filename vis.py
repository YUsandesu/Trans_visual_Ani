import os

# 1. 在导入 py5 之前，用代码直接告诉它 Java 环境的根目录在哪里
# 注意：这里只需要到 jvm 的根目录，不需要写到 jvm.dll
java_home_path = r"C:\Users\admin\anaconda3\envs\Processing\Library\lib\jvm"
os.environ["JAVA_HOME"] = java_home_path

# 2. 现在正常导入 py5，它会自动找到上面的路径，并妥善配置好所有的依赖包
import py5
import numpy as np

# 初始序列定义
tokens = ["The", "cat", "sat", "on", "the", "mat"]
n_tokens = len(tokens)

# 初始注意力权重
np.random.seed(42)
weights = np.random.dirichlet(np.ones(n_tokens)*0.7, size=n_tokens)

# 动画时间轴定义 (单位: 帧) - 扩充为5个阶段
frames_per_token = 70
phase1_duration = n_tokens * frames_per_token  # 阶段1: 注意力计算
phase2_duration = 50                          # 阶段2: 消除间隔汇聚
phase3_duration = 30                           # 阶段3: 画括号展示
phase4_duration = 30                           # 阶段4: 整体上移，旧词元消失
phase5_duration = 30                          # 阶段5: 重新展开，变为新输入
total_cycle_frames = phase1_duration + phase2_duration + phase3_duration + phase4_duration + phase5_duration

# 递归状态变量
output_numbers = []
depth = 0  

def setup():
    py5.size(900, 500)
    py5.text_align(py5.CENTER, py5.CENTER)
    py5.text_font(py5.create_font("Arial", 24))
    generate_new_numbers()

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
    
    py5.background(25)
    cycle_frame = py5.frame_count % total_cycle_frames
    
    # 动画周期完全结束时，在后台切换数据状态
    if cycle_frame == 0 and py5.frame_count > 0:
        step_into_next_layer()
        
    # 布局计算参数
    margin = 100
    spacing = (py5.width - 2 * margin) / (n_tokens - 1)
    y_tokens = py5.height / 2 - 20
    y_numbers = py5.height / 2 + 100
    
    num_spacing = 70 
    total_shrink_width = (n_tokens - 1) * num_spacing
    start_target_x = py5.width / 2 - total_shrink_width / 2

    py5.fill(100)
    py5.text_size(16)
    py5.text(f"Transformer Layer Depth: {depth}", py5.width / 2, 30)

    # 阶段时间节点预计算
    t1 = phase1_duration
    t2 = t1 + phase2_duration
    t3 = t2 + phase3_duration
    t4 = t3 + phase4_duration

    # ==========================================
    # 阶段 1: 注意力连线与随机数生成
    # ==========================================
    if cycle_frame < t1:
        current_idx = cycle_frame // frames_per_token
        local_frame = cycle_frame % frames_per_token
        
        draw_attention_lines(current_idx, local_frame, margin, spacing, y_tokens)
        
        for i, token in enumerate(tokens):
            x = margin + i * spacing
            if i == current_idx:
                py5.fill(255, 215, 0)
                py5.text_size(32)
            else:
                py5.fill(120)
                py5.text_size(20)
            py5.text(token, x, y_tokens)
            
            if i < current_idx:
                py5.fill(150, 255, 150)
                py5.text_size(24)
                py5.text(output_numbers[i], x, y_numbers)
            elif i == current_idx:
                alpha = py5.remap(local_frame, frames_per_token * 0.6, frames_per_token, 0, 255)
                py5.fill(150, 255, 150, py5.constrain(alpha, 0, 255))
                py5.text_size(24)
                py5.text(output_numbers[i], x, y_numbers)

    # ==========================================
    # 阶段 2: 消除间隔汇聚 
    # ==========================================
    elif cycle_frame < t2:
        progress = (cycle_frame - t1) / phase2_duration
        ease_p = -(py5.cos(py5.PI * progress) - 1) / 2
        
        for i, token in enumerate(tokens):
            x = margin + i * spacing
            py5.fill(80)
            py5.text_size(20)
            py5.text(token, x, y_tokens)
            
            orig_x = x
            target_x = start_target_x + i * num_spacing
            current_x = py5.lerp(orig_x, target_x, ease_p)
            
            py5.fill(150, 255, 150)
            py5.text_size(24)
            py5.text(output_numbers[i], current_x, y_numbers)

    # ==========================================
    # 阶段 3: 画括号保持展示
    # ==========================================
    elif cycle_frame < t3:
        for i, token in enumerate(tokens):
            x = margin + i * spacing
            py5.fill(80)
            py5.text_size(20)
            py5.text(token, x, y_tokens)
            
            target_x = start_target_x + i * num_spacing
            py5.fill(150, 255, 150)
            py5.text_size(24)
            py5.text(output_numbers[i], target_x, y_numbers)
            
        bracket_progress = (cycle_frame - t2) / (phase3_duration * 0.5)
        bracket_alpha = py5.constrain(bracket_progress * 255, 0, 255)
        
        py5.fill(255, bracket_alpha)
        py5.text_size(48)
        py5.text("[", start_target_x - 30, y_numbers - 5) 
        py5.text("]", start_target_x + total_shrink_width + 30, y_numbers - 5)

    # ==========================================
    # 阶段 4: 整体上移，同时旧词元和括号淡出
    # ==========================================
    elif cycle_frame < t4:
        progress = (cycle_frame - t3) / phase4_duration
        ease_p = -(py5.cos(py5.PI * progress) - 1) / 2
        
        current_y = py5.lerp(y_numbers, y_tokens, ease_p)
        old_token_alpha = py5.lerp(80, 0, progress)
        bracket_alpha = py5.lerp(255, 0, progress)
        
        for i, token in enumerate(tokens):
            x = margin + i * spacing
            # 旧词元淡出
            py5.fill(80, old_token_alpha)
            py5.text_size(20)
            py5.text(token, x, y_tokens)
            
            # 数字整体上移
            target_x = start_target_x + i * num_spacing
            py5.fill(150, 255, 150)
            py5.text_size(24)
            py5.text(output_numbers[i], target_x, current_y)
            
        # 括号一起上移并淡出
        py5.fill(255, bracket_alpha)
        py5.text_size(48)
        py5.text("[", start_target_x - 30, current_y - 5) 
        py5.text("]", start_target_x + total_shrink_width + 30, current_y - 5)

    # ==========================================
    # 阶段 5: 重新展开，颜色渐变为默认输入态
    # ==========================================
    else:
        progress = (cycle_frame - t4) / phase5_duration
        ease_p = -(py5.cos(py5.PI * progress) - 1) / 2
        
        # 将亮绿色平滑过渡到输入的标准灰色
        c_r = py5.lerp(150, 120, progress)
        c_g = py5.lerp(255, 120, progress)
        c_b = py5.lerp(150, 120, progress)
        
        for i, token in enumerate(tokens):
            orig_x = margin + i * spacing
            target_x = start_target_x + i * num_spacing
            
            # 从紧凑位置向两侧展开
            current_x = py5.lerp(target_x, orig_x, ease_p)
            
            py5.fill(c_r, c_g, c_b)
            # 字体大小平滑缩小到标准态
            t_size = py5.lerp(24, 20, progress)
            py5.text_size(t_size)
            py5.text(output_numbers[i], current_x, y_tokens)


def draw_attention_lines(current_source_idx, local_frame, margin, spacing, y_pos):
    draw_duration = 0.7
    frames_for_all_lines = int(frames_per_token * draw_duration)
    frames_per_line = frames_for_all_lines / n_tokens
    x_start = margin + current_source_idx * spacing

    for i in range(n_tokens):
        line_start_frame = i * frames_per_line
        line_end_frame = (i + 1) * frames_per_line
        
        if local_frame < line_start_frame: continue
            
        line_progress = py5.remap(local_frame, line_start_frame, line_end_frame, 0, 1)
        line_progress = min(1.0, line_progress)
        
        weight = weights[current_source_idx][i]
        alpha = py5.remap(weight, 0, 1, 50, 255)
        
        py5.stroke(0, 200, 255, alpha)
        py5.stroke_weight(py5.remap(weight, 0, 1, 1, 5))
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