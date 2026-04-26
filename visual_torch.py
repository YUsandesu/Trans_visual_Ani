import os
import math

# 1. Configure Java environment BEFORE importing py5
java_home_path = r"C:\Users\admin\anaconda3\envs\Processing\Library\lib\jvm"
os.environ["JAVA_HOME"] = java_home_path

import py5

network_flow = {
    "Input Image (128x128 Grayscale)": [
        {
            "shape": "rect_grid", 
            "color": (210, 210, 210), 
            "count": 1, 
            "draw_size": 110, 
            "patch_size": 18,      
            "num": 1,             
            "link": "auto"        
        }
    ],
    
    "Patch Unfold (Feature Detection)": [
        {
            "shape": "rect_grid", 
            "color": (255, 204, 0), 
            "count": 9, 
            "draw_size": 36, 
            "patch_size": 6, 
            "num": 1, 
            "link": "auto"
        } 
    ],
    
    "Patch Mapper (2nd Detection)": [
        {
            "shape": "rect_grid", 
            "color": (200, 150, 255), 
            "count": 9, 
            "draw_size": 36, 
            "patch_size": 6, 
            "num": 1, 
            "link": "auto"
        }
    ],
    
    "Pool Concat (1D Feature)": [
        {
            "shape": "rect", 
            "color": (255, 100, 100), # 红色系代表 MaxPool
            "count": 9, 
            "draw_size": 22,      
            "link": "auto"
        },
        {
            "shape": "rect", 
            "color": (100, 180, 255), # 蓝色系代表 AvgPool (arrpool)
            "count": 9, 
            "draw_size": 22,      
            "link": "auto"
        }
    ],
    
    "MLP Layer 1 (512 -> 128)": [
        {"shape": "circle", "color": (100, 160, 240), "count": 12, "draw_size": 20, "link": "fc"} 
    ],
    
    "MLP Layer 2 (128 -> 32)": [
        {"shape": "circle", "color": (100, 180, 255), "count": 4, "draw_size": 20, "link": "fc"} 
    ],
    
    "Output (Sigmoid)": [
        {
            "shape": "circle", 
            "color": (130, 200, 150), 
            "count": 1, 
            "draw_size": 30, 
            "link": "auto"        
        }
    ]
}

layers_data = []

def settings():
    max_h_needed = 0
    max_layer_width = 0
    
    for name, groups in network_flow.items():
        layer_total_h = 0
        layer_max_node_w = 0
        
        for g in groups:
            d_size = g.get("draw_size", 20)
            num = g.get("num", 1)
            count = g.get("count", 1)
            
            off_x = max(2, d_size * 0.08)
            off_y = max(2, d_size * 0.08) 
            
            node_actual_w = d_size + (num - 1) * off_x
            node_actual_h = d_size + (num - 1) * off_y
            
            if node_actual_w > layer_max_node_w:
                layer_max_node_w = node_actual_w
                
            layer_total_h += count * (node_actual_h * 1.5)
            
        if layer_max_node_w > max_layer_width:
            max_layer_width = layer_max_node_w
            
        if layer_total_h > max_h_needed:
            max_h_needed = layer_total_h
            
    num_layers = len(network_flow)
    x_spacing = max(180.0, float(max_layer_width * 1.8))
    calc_width = int(x_spacing * (num_layers + 1))
    calc_height = int(max_h_needed + 200)
    
    calc_width = max(1000, calc_width)
    calc_height = max(500, calc_height)
    
    py5.size(calc_width, calc_height)

def setup():
    py5.background(250)
    py5.no_loop()
    calculate_coordinates()

def calculate_coordinates():
    global layers_data
    num_layers = len(network_flow)
    x_spacing = py5.width / (num_layers + 1)
    
    for i, (name, groups) in enumerate(network_flow.items()):
        x = x_spacing * (i + 1)
        layer_nodes = []
        total_count = sum(g["count"] for g in groups)
        denom = total_count + 1 if total_count > 0 else 2
        y_spacing = py5.height / denom
        
        node_ptr = 0
        for group in groups:
            g_count = group["count"]
            for idx in range(g_count):
                y = y_spacing * (node_ptr + 1)
                layer_nodes.append({
                    "x": x, "y": y,
                    "shape": group["shape"],
                    "color": group["color"],
                    "draw_size": group["draw_size"],
                    "link": group.get("link", "auto"),
                    "num": group.get("num", 1),
                    "patch_size": group.get("patch_size", 1),
                    # 【核心修改 1】：记录组级特征，支持并行分支连线
                    "g_idx": idx,          # 该节点在其所属“组”内的独立索引
                    "g_count": g_count,    # 该节点所属“组”的总数量
                    "l_idx": node_ptr      # 全局层级索引 (备用)
                })
                node_ptr += 1
        layers_data.append(layer_nodes)

def draw():
    draw_connections()
    draw_nodes()
    draw_layer_labels()

def draw_connections():
    py5.stroke_weight(1.0)
    
    for i in range(len(layers_data) - 1):
        curr_l = layers_data[i]
        next_l = layers_data[i+1]
        
        for n_node in next_l:
            link_mode = n_node["link"]
            
            if link_mode == "fc": 
                py5.stroke(100, 160, 240, 60)
                for c_node in curr_l:
                    py5.line(c_node["x"], c_node["y"], n_node["x"], n_node["y"])
                    
            elif link_mode == "auto":
                py5.stroke(180, 100)
                
                # 【核心修改 2】：基于"当前组的节点数"独立计算映射比例，实现并行分支等同映射
                # 举例：上一层 9 个节点，当前是 MaxPool 组 (9个)，ratio = 9/9 = 1
                ratio = len(curr_l) / n_node["g_count"]
                
                # 依据该节点在其“组”内的局部索引，反向推导对应的上一层节点区间
                start_idx = int(math.floor(n_node["g_idx"] * ratio))
                end_idx = int(math.ceil((n_node["g_idx"] + 1) * ratio))
                
                for c_idx in range(start_idx, min(end_idx, len(curr_l))):
                    py5.line(curr_l[c_idx]["x"], curr_l[c_idx]["y"], n_node["x"], n_node["y"])

def draw_nodes():
    py5.rect_mode(py5.CENTER)
    
    for layer in layers_data:
        for node in layer:
            x, y, shape, d_size = node["x"], node["y"], node["shape"], node["draw_size"]
            c = node["color"]
            
            if shape == "rect_grid":
                draw_cube(x, y, node["patch_size"], node["num"], c, d_size)
            elif shape == "rect":
                py5.fill(*c)
                py5.stroke(50)
                py5.rect(x, y, d_size, d_size)
            elif shape == "circle":
                py5.fill(*c)
                py5.stroke(50)
                py5.circle(x, y, d_size)

def draw_cube(x, y, p_size, num, c, d_size):
    off_x = max(2, d_size * 0.08)
    off_y = -max(2, d_size * 0.08)
    
    for d in range(num-1, -1, -1):
        ox = x + d * off_x
        oy = y + d * off_y
        fill_c = [max(v - d*25, 0) for v in c]
        py5.fill(*fill_c)
        py5.stroke(50)
        py5.rect(ox, oy, d_size, d_size)
        
        if p_size > 1:
            py5.stroke(255, 60) if c[0] < 180 else py5.stroke(100, 60)
            step = d_size / p_size
            for i in range(1, p_size):
                py5.line(ox - d_size/2 + i*step, oy - d_size/2, ox - d_size/2 + i*step, oy + d_size/2)
                py5.line(ox - d_size/2, oy - d_size/2 + i*step, ox + d_size/2, oy - d_size/2 + i*step)

def draw_layer_labels():
    py5.fill(50)
    py5.text_align(py5.CENTER, py5.TOP)
    py5.text_size(14)
    names = list(network_flow.keys())
    x_spacing = py5.width / (len(names) + 1)
    # 取消注释以显示文字
    # for i, name in enumerate(names):
    #     py5.text(name, x_spacing * (i + 1), 60)

if __name__ == '__main__':
    py5.run_sketch()