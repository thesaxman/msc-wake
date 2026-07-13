import os
from pathlib import Path

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

def save_video(animation,filename):
    filepath = OUT_DIR / filename
    animation.save(str(filepath), writer='ffmpeg', dpi=200, fps=20)
    assert filepath.exists(), f"save produced no file: {filepath}"
    os.startfile(filepath)
