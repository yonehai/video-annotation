#!/usr/bin/env python

import argparse
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a video from image frames using ffmpeg."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input frame naming pattern (e.g., 'frame_%03d.png').",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output video file name (e.g., 'output.mp4').",
    )
    parser.add_argument(
        "-r",
        "--framerate",
        type=int,
        default=25,
        help="Frame rate of the video (default: 25).",
    )
    args = parser.parse_args()

    ffmpeg_cmd = [
        "ffmpeg",
        "-r",
        str(args.framerate),
        "-i",
        args.input,
        "-vcodec",
        "mpeg4",
        "-y",
        args.output,
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Video successfully created: {args.output}")
    except subprocess.CalledProcessError as e:
        print(f"Error during ffmpeg execution: {e}")
