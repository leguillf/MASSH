#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vbellemin
"""

import os
import shutil
import subprocess
from joblib import Parallel, delayed

def generate_video(
    func,
    inputs,
    output_dir,
    fps=12,
    n_jobs=10,
    delete_frames=True,
    video_name="movie.mp4"
):
    os.makedirs(output_dir, exist_ok=True)

    # Frames directory INSIDE output_dir (no tempfile)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Clean old frames if they exist
    for f in os.listdir(frames_dir):
        if f.endswith(".png"):
            os.remove(os.path.join(frames_dir, f))

    # Wrapper
    def _wrapper(i, item):
        filename = os.path.join(frames_dir, f"{i:06d}.png")
        func(item, filename)

    print("Generating frames...")
    Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_wrapper)(i, item) for i, item in enumerate(inputs)
    )

    print("Creating video with ffmpeg...")
    output_path = os.path.join(output_dir, video_name)

    command = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "%06d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_path
    ]

    subprocess.run(command, check=True)

    if delete_frames:
        print("Deleting frames...")
        shutil.rmtree(frames_dir)

    print(f"Done → {output_path}")