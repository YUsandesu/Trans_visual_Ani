import os

# 1. Configure Java environment BEFORE importing py5
java_home_path = r"C:\Users\admin\anaconda3\envs\Processing\Library\lib\jvm"
os.environ["JAVA_HOME"] = java_home_path

# 2. Import dependencies
import py5
import numpy as np

# ==========================================
# Resolution and scaling configuration
# ==========================================
TARGET_WIDTH = 1920  # [Enter your desired export width here]
TARGET_HEIGHT = 1080 # [Enter your desired export height here]

BASE_WIDTH = 900     # Original logical width (do not modify)
BASE_HEIGHT = 500    # Original logical height (do not modify)

# Initial sequence definition
tokens = ["The", "cat", "sat", "on", "the", "mat"]
n_tokens = len(tokens)

# Initial attention weights
np.random.seed(42)
weights = np.random.dirichlet(np.ones(n_tokens)*0.7, size=n_tokens)

# Animation timeline definition (unit: frames) - expanded to 5 phases
frames_per_token = 70
phase1_duration = n_tokens * frames_per_token  
phase2_duration = 50                           
phase3_duration = 30                           
phase4_duration = 30                           
phase5_duration = 30                           
total_cycle_frames = phase1_duration + phase2_duration + phase3_duration + phase4_duration + phase5_duration

# Recursive state variables
output_numbers = []
depth = 0  

def setup():
    # Use your manually specified target physical resolution
    py5.size(TARGET_WIDTH, TARGET_HEIGHT)
    py5.frame_rate(30)
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
    
    # ==========================================
    # Core logic: calculate proportional scaling and centering
    # ==========================================
    # Use the smaller scaling ratio to ensure content is not cropped and scales proportionally
    scale_factor = min(TARGET_WIDTH / BASE_WIDTH, TARGET_HEIGHT / BASE_HEIGHT)
    
    # Calculate translation distance to center the logical canvas in the physical window
    offset_x = (TARGET_WIDTH - BASE_WIDTH * scale_factor) / 2
    offset_y = (TARGET_HEIGHT - BASE_HEIGHT * scale_factor) / 2
    
    # Apply matrix transformation
    py5.translate(offset_x, offset_y)
    py5.scale(scale_factor)
    # ------------------------------------------
    
    cycle_frame = py5.frame_count % total_cycle_frames
    
    # When animation cycle completes, switch data state in background (preserve infinite loop mechanism)
    if cycle_frame == 0 and py5.frame_count > 0:
        step_into_next_layer()
        
    # Layout calculation parameters (note: py5.width/height replaced with BASE_WIDTH/BASE_HEIGHT)
    margin = 100
    spacing = (BASE_WIDTH - 2 * margin) / (n_tokens - 1)
    y_tokens = BASE_HEIGHT / 2 - 20
    y_numbers = BASE_HEIGHT / 2 + 100
    
    num_spacing = 70 
    total_shrink_width = (n_tokens - 1) * num_spacing
    start_target_x = BASE_WIDTH / 2 - total_shrink_width / 2

    py5.fill(100)
    py5.text_size(16)
    py5.text(f"Transformer Layer Depth: {depth}", BASE_WIDTH / 2, 30)

    t1 = phase1_duration
    t2 = t1 + phase2_duration
    t3 = t2 + phase3_duration
    t4 = t3 + phase4_duration

    # ==========================================
    # Phase 1: Attention lines and random number generation
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
    # Phase 2: Eliminate spacing and converge
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
    # Phase 3: Draw brackets and display
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
    # Phase 4: Move up, fade out old tokens and brackets
    # ==========================================
    elif cycle_frame < t4:
        progress = (cycle_frame - t3) / phase4_duration
        ease_p = -(py5.cos(py5.PI * progress) - 1) / 2
        
        current_y = py5.lerp(y_numbers, y_tokens, ease_p)
        old_token_alpha = py5.lerp(80, 0, progress)
        bracket_alpha = py5.lerp(255, 0, progress)
        
        for i, token in enumerate(tokens):
            x = margin + i * spacing
            py5.fill(80, old_token_alpha)
            py5.text_size(20)
            py5.text(token, x, y_tokens)
            
            target_x = start_target_x + i * num_spacing
            py5.fill(150, 255, 150)
            py5.text_size(24)
            py5.text(output_numbers[i], target_x, current_y)
            
        py5.fill(255, bracket_alpha)
        py5.text_size(48)
        py5.text("[", start_target_x - 30, current_y - 5) 
        py5.text("]", start_target_x + total_shrink_width + 30, current_y - 5)

    # ==========================================
    # Phase 5: Re-expand, color gradient to default input state
    # ==========================================
    else:
        progress = (cycle_frame - t4) / phase5_duration
        ease_p = -(py5.cos(py5.PI * progress) - 1) / 2
        
        c_r = py5.lerp(150, 120, progress)
        c_g = py5.lerp(255, 120, progress)
        c_b = py5.lerp(150, 120, progress)
        
        for i, token in enumerate(tokens):
            orig_x = start_target_x + i * num_spacing
            target_x = margin + i * spacing
            current_x = py5.lerp(orig_x, target_x, ease_p)
            
            py5.fill(c_r, c_g, c_b)
            t_size = py5.lerp(24, 20, progress)
            py5.text_size(t_size)
            py5.text(output_numbers[i], current_x, y_tokens)

    # ==========================================
    # Export mechanism (unlimited continuous export)
    # ==========================================
    # Exported image dimensions will automatically match your TARGET_WIDTH and TARGET_HEIGHT settings
    py5.save_frame("frames/frame_####.png")

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