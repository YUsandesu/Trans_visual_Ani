# Trans_visual_Ani

Transformer attention mechanism visualization animation project. Uses py5 (Python version of Processing) to visualize attention weights and recursive transformation processes of Transformer layers.

## Features

This project demonstrates how Transformer model attention mechanisms work through animations, including:
- Attention weight line visualization
- Token to hidden state transformation
- Multi-layer recursive transformation process
- Customizable resolution export

## Dependencies

```bash
pip install -r requirements.txt
```

Dependencies include:
- `py5` - Python version of Processing library
- `numpy` - Numerical computation library
- `jpype1` - Python-Java bridge library

## ⚠️ Important: JVM Path Configuration

### Why Must You Configure JVM Path Before Importing py5?

**This is a critical step for the project to run. Java environment configuration must be completed BEFORE importing `py5`.**

Reasons:

1. **py5's underlying dependency**: py5 is a Python wrapper based on Processing, which is written in Java. py5 interacts with the Java Virtual Machine (JVM) through the JPype1 library.

2. **JPype1 initialization mechanism**: JPype1 attempts to start the JVM when imported. If the `JAVA_HOME` environment variable is not properly set before importing py5, JPype1 will:
   - Fail to find the JVM installation path
   - Throw an `OSError: JVM not found` error
   - Cause the program to fail to start

3. **Environment variable timing**: Python environment variable settings must be completed before module import. Once the py5 module is imported, the JVM has already attempted initialization, and setting environment variables afterwards will have no effect.

### Configuration Method

In `vis.py` or other visualization scripts, **add this at the very top of the code** (before any imports):

```python
import os

# 1. Configure Java environment BEFORE importing py5
java_home_path = r"C:\Users\admin\anaconda3\envs\Processing\Library\lib\jvm"
os.environ["JAVA_HOME"] = java_home_path

# 2. Then you can import dependencies
import py5
import numpy as np
```

### How to Find Your JVM Path

If you use Anaconda/Miniconda, the JVM is typically located at:
- Windows: `C:\Users\{username}\anaconda3\envs\{env_name}\Library\lib\jvm`
- macOS/Linux: `~/anaconda3/envs/{env_name}/Library/lib/jvm`

If you use system Java, you can find it with:
- Windows: `echo %JAVA_HOME%`
- macOS/Linux: `echo $JAVA_HOME`

If not set, system Java is usually at:
- Windows: `C:\Program Files\Java\jdk-{version}`
- macOS: `/Library/Java/JavaVirtualMachines/jdk-{version}.jdk/Contents/Home`

## Usage

### Run Visualization

```bash
python vis.py
```

### Custom Resolution

Modify these parameters in the script:

```python
TARGET_WIDTH = 1920   # Export width
TARGET_HEIGHT = 1080  # Export height
```

The animation automatically scales proportionally and centers to ensure content is not cropped.

### Export Frames

During animation, each frame is automatically saved to the `frames/` directory with filename `frame_####.png`.

## File Descriptions

- `vis.py` - Main visualization script (dark background)
- `vis_white.py` - White background version
- `pralle_vis_b.py` / `pralle_vis_w.py` - Parallel attention visualization
- `vedio.py` - Video processing script
- `requirements.txt` - Python dependencies list

## FAQ

### Q: "JVM not found" error when running

**A**: Check if you set the `JAVA_HOME` environment variable before importing py5, and ensure the path is correct.

### Q: Other Java-related errors when importing py5

**A**: 
1. Confirm Java is installed (JDK 8 or higher recommended)
2. Confirm jpype1 is correctly installed
3. Try reinstalling dependencies: `pip install --upgrade py5 jpype1`

### Q: Animation runs but window won't display

**A**: This is usually due to Java GUI configuration issues. It typically works on Windows. If you encounter issues on Linux, you may need to configure X11 or use headless mode.

## Technical Details

- **Animation cycle**: Complete Transformer layer transformation divided into 5 phases
- **Frame rate**: 30 FPS
- **Default resolution**: 900x500 (logical resolution), customizable export resolution
- **Random seed**: 42 (ensures reproducibility)

## License

This project is for educational and research purposes.
