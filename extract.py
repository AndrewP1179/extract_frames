import os
from datetime import timedelta

import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

def cut_video(filename, start_time, end_time, target_name):
    ffmpeg_extract_subclip(filename, start_time, end_time, targetname=target_name)

SAVING_FRAMES_PER_SECOND = 2

def format_timedelta(td):
    """Utility function to format timedelta objects in a cool way (e.g 00:00:20.05)
    omitting microseconds and retaining milliseconds"""
    result = str(td)
    try:
        result, ms = result.split(".")
    except ValueError:
        return result + ".00".replace(":", "-")
    ms = int(ms)
    ms = round(ms / 1e4)
    return f"{result}.{ms:02}".replace(":", "-")

def extract_frames(video_file):
    # load the video clip
    video_clip = VideoFileClip(video_file)
    # make a folder by the name of the video file
    filename, _ = os.path.splitext(video_file)
    filename += "_frames"
    if not os.path.isdir(filename):
        os.mkdir(filename)

    # if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
    saving_frames_per_second = min(video_clip.fps, SAVING_FRAMES_PER_SECOND)
    # if SAVING_FRAMES_PER_SECOND is set to 0, step is 1/fps, else 1/SAVING_FRAMES_PER_SECOND
    step = 1 / video_clip.fps if saving_frames_per_second == 0 else 1 / saving_frames_per_second
    # iterate over each possible frame
    for current_duration in np.arange(0, video_clip.duration, step):
        # format the file name and save it
        frame_duration_formatted = format_timedelta(timedelta(seconds=current_duration)).replace(":", "-")
        frame_filename = os.path.join(filename, f"frame{frame_duration_formatted}.jpg")
        # save the frame with the current duration
        video_clip.save_frame(frame_filename, current_duration)

def main():
    time_codes = [(0, 12)]
    cut_videos_folder = "./results/"
    for index, filename in enumerate(os.listdir("./videos")):
        cut_video("./videos/" + filename, time_codes[index][0], time_codes[index][1], cut_videos_folder + filename)
        extract_frames(cut_videos_folder + filename)
main()