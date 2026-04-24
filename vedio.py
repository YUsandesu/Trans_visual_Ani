import os
from moviepy import ImageSequenceClip

def images_to_video(folder_path, output_name, fps=24):
    """
    将指定文件夹内的图片转换为 MP4 视频
    :param folder_path: 图片所在文件夹路径
    :param output_name: 输出视频的文件名 (例如 'output.mp4')
    :param fps: 每秒帧数
    """
    # 获取文件夹内所有图片文件
    # 支持常见格式：png, jpg, jpeg
    valid_extensions = ('.png', '.jpg', '.jpeg')
    images = [
        os.path.join(folder_path, img) 
        for img in sorted(os.listdir(folder_path)) 
        if img.lower().endswith(valid_extensions)
    ]

    if not images:
        print("错误：文件夹内没有找到有效的图片文件。")
        return

    print(f"正在处理 {len(images)} 张图片...")

    # 创建视频剪辑
    clip = ImageSequenceClip(images, fps=fps)
    
    # 写入视频文件 (使用 libx264 编码)
    clip.write_videofile(output_name, codec='libx264')
    print(f"转换完成！视频已保存为: {output_name}")

# 使用示例
if __name__ == "__main__":
    target_folder = "frames"  # 替换为你的文件夹名称
    output_video = "result_video.mp4"
    images_to_video(target_folder, output_video, fps=30)