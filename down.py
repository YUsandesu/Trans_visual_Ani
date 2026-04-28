import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# 1. 准备数据
x = np.linspace(-24, 26, 100)
y = np.linspace(-10, 16, 100)
X, Y = np.meshgrid(x, y)
# # 定义一个简单的二次函数作为损失函数: Z = X^2 + Y^2
# Z = X**2 + Y**2
# 模拟不可微分的像素阶梯空间
# step_size 代表台阶的宽度（可以理解为1个像素的跨度）
step_size = 4
X_step = np.floor(X / step_size) * step_size
Y_step = np.floor(Y / step_size) * step_size
Z = X_step**2 + Y_step**2

# 2. 创建 3D 图形对象
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 3. 绘制曲面并涂色
# cmap 控制颜色，antialiased=False 可以让颜色在大数据量下更平滑
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, 
                       linewidth=0, antialiased=False, alpha=0.8)

# 4. 添加颜色条 (Colorbar) 以展示 Z 值与颜色的对应关系
fig.colorbar(surf, shrink=0.5, aspect=10)

# 设置标签
ax.set_xlabel('Parameter W')
ax.set_ylabel('Parameter b')
ax.set_zlabel('Loss J')
ax.set_title('Gradient Descent Surface')


plt.show()