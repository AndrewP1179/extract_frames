import os
import cv2
import json
import mediapipe as mp
from mediapipe.python.solutions.pose import PoseLandmark
from datetime import timedelta
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

def getLandmarkType(landmarkType):
    if landmarkType == PoseLandmark.LEFT_EYE:
        return "LEFT_EYE"
    elif landmarkType == PoseLandmark.LEFT_EYE_INNER:
        return "LEFT_EYE_INNER"
    elif landmarkType == PoseLandmark.LEFT_EYE_OUTER:
        return "LEFT_EYE_OUTER"
    elif landmarkType == PoseLandmark.RIGHT_EYE:
        return "RIGHT_EYE"
    elif landmarkType == PoseLandmark.RIGHT_EYE_INNER:
        return "RIGHT_EYE_INNER"
    elif landmarkType == PoseLandmark.RIGHT_EYE_OUTER:
        return "RIGHT_EYE_OUTER"
    elif landmarkType == PoseLandmark.LEFT_EAR:
        return "LEFT_EAR"
    elif landmarkType == PoseLandmark.RIGHT_EAR:
        return "RIGHT_EAR"
    elif landmarkType == PoseLandmark.NOSE:
        return "NOSE"
    elif landmarkType == PoseLandmark.LEFT_SHOULDER:
        return "LEFT_SHOULDER"
    elif landmarkType == PoseLandmark.RIGHT_SHOULDER:
        return "RIGHT_SHOULDER"
    elif landmarkType == PoseLandmark.LEFT_ELBOW:
        return "LEFT_ELBOW"
    elif landmarkType == PoseLandmark.RIGHT_ELBOW:
        return "RIGHT_ELBOW"
    elif landmarkType == PoseLandmark.LEFT_WRIST:
        return "LEFT_WRIST"
    elif landmarkType == PoseLandmark.RIGHT_WRIST:
        return "RIGHT_WRIST"
    elif landmarkType == PoseLandmark.LEFT_THUMB:
        return "LEFT_THUMB"
    elif landmarkType == PoseLandmark.RIGHT_THUMB:
        return "RIGHT_THUMB"
    elif landmarkType == PoseLandmark.LEFT_PINKY:
        return "LEFT_PINKY"
    elif landmarkType == PoseLandmark.RIGHT_PINKY:
        return "RIGHT_PINKY"
    elif landmarkType == PoseLandmark.LEFT_HIP:
        return "LEFT_HIP"
    elif landmarkType == PoseLandmark.RIGHT_HIP:
        return "RIGHT_HIP"
    elif landmarkType == PoseLandmark.LEFT_KNEE:
        return "LEFT_KNEE"
    elif landmarkType == PoseLandmark.RIGHT_KNEE:
        return "RIGHT_KNEE"
    elif landmarkType == PoseLandmark.LEFT_HEEL:
        return "LEFT_HEEL"
    elif landmarkType == PoseLandmark.RIGHT_HEEL:
        return "RIGHT_HEEL"
    elif landmarkType == PoseLandmark.LEFT_ANKLE:
        return "LEFT_ANKLE"
    elif landmarkType == PoseLandmark.RIGHT_ANKLE:
        return "RIGHT_ANKLE"
    elif landmarkType == PoseLandmark.LEFT_FOOT_INDEX:
        return "LEFT_FOOT_INDEX"
    elif landmarkType == PoseLandmark.RIGHT_FOOT_INDEX:
        return "RIGHT_FOOT_INDEX"
    else:
        return ""

destination_folder = "../research_app_native/src/checkpoints/"
video_file = os.listdir("videos")[0]
exercise_name = os.path.splitext(video_file)[0]

def cut_video(filename, start_time, end_time, target_name):
    ffmpeg_extract_subclip(filename, start_time, end_time, targetname=target_name)

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

def extract_frames(video_file, s_f_p_s):
    # load the video clip
    video_clip = VideoFileClip(video_file)
    # make a folder by the name of the video file
    frames_folder = os.path.join(os.path.split(video_file)[0], 'frames')
    os.mkdir(frames_folder)

    # if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
    saving_frames_per_second = min(video_clip.fps, s_f_p_s)
    # if SAVING_FRAMES_PER_SECOND is set to 0, step is 1/fps, else 1/SAVING_FRAMES_PER_SECOND
    step = 1 / video_clip.fps if saving_frames_per_second == 0 else 1 / saving_frames_per_second
    # iterate over each possible frame
    for index, current_duration in enumerate(np.arange(0, video_clip.duration, step)):
        # format the file name and save it
        frame_duration_formatted = format_timedelta(timedelta(seconds=current_duration)).replace(":", "-")
        frame_filename = os.path.join(frames_folder, f"{index}.jpg")
        # save the frame with the current duration
        video_clip.save_frame(frame_filename, current_duration)

def generateJson():
    exercise_path = os.path.join(destination_folder, exercise_name)
    frames_folder = os.path.join(exercise_path, 'frames')
    frames = sorted(os.listdir(frames_folder), key=extract_integer_to_sort)
    requested_landmarks = [int(i) for i in input("Please, select the desired landmarks for the analysis:").split()]

    mp_pose = mp.solutions.pose
    dict_result = {
        "exercise": exercise_name,
        "phases": []
    }
    with mp_pose.Pose(
            static_image_mode=True, min_detection_confidence=0.5, model_complexity=2,
            enable_segmentation=True) as pose:
        i = 0
        for frame in frames:
            exercise_folder = os.path.join(exercise_path, "frames", frame)
            image = cv2.imread(exercise_folder)
            result = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            landmarks = result.pose_landmarks.landmark
            selected_landmarks = []


            for landmark_index in requested_landmarks:
                selected_landmarks.append({
                    "type": getLandmarkType(landmark_index),
                    "position": {
                        "x": landmarks[landmark_index].x,
                        "y": landmarks[landmark_index].y,
                    },
                    "inFrameLikelihood": landmarks[landmark_index].visibility,
                })



            dict_result["phases"].append({"phase_id": i, "landmarks": selected_landmarks})
            i += 1
            with open(os.path.join(exercise_path, "checkpoints.json"), "w") as outfile:
                json.dump(dict_result, outfile)

def processImage(frames, exercise_path):
    BG_COLOR = (192, 192, 192)
    try:
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        # Run MediaPipe Pose with `enable_segmentation=True` to get pose segmentation and draw pose landmarks.
        with mp_pose.Pose(
                static_image_mode=True, min_detection_confidence=0.5, model_complexity=2,
                enable_segmentation=True) as pose:

            landmarks_folder = os.path.join(exercise_path, 'landmarks')

            i = 0
            for frame in frames:
                exercise_folder = os.path.join(exercise_path, "frames", frame)
                image = cv2.imread(exercise_folder)
                result = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                annotated_image = image.copy()
                condition = np.stack((result.segmentation_mask,) * 3, axis=-1) > 0.1
                bg_image = np.zeros(image.shape, dtype=np.uint8)
                bg_image[:] = BG_COLOR
                annotated_image = np.where(condition, annotated_image, bg_image)
                mp_drawing.draw_landmarks(
                    annotated_image,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                cv2.imwrite(os.path.join(landmarks_folder, f"{str(i)}.jpg"), annotated_image)

                i += 1

    except Exception as e:
        raise e

def extract_integer_to_sort(filename):
    return int(filename.split('.')[0])

def extract_frames_custom():
    destination_folder = "../research_app_native/src/checkpoints/"
    start_timecode = int(input("What is the start of the " + video_file + " video?"))
    end_timecode = int(input("What is the end of the " + video_file + " video?"))
    saving_frames_per_second = int(input("What is the desired savings frames per second for " + video_file + " ?"))
    path = os.path.join(destination_folder, os.path.splitext(video_file)[0])
    cut_video_destination_folder = os.path.join(destination_folder, os.path.splitext(video_file)[0], video_file)

    os.makedirs(path)

    cut_video("./videos/" + video_file, start_timecode, end_timecode, cut_video_destination_folder)

    extract_frames(cut_video_destination_folder, saving_frames_per_second)

def process_frames():
    folder = os.path.join(destination_folder,  os.path.splitext(video_file)[0])

    if os.path.isdir(folder):
        frames_folder = os.path.join(folder, 'frames')
        landmarks_folder = os.path.join(folder, 'landmarks')
        os.makedirs(landmarks_folder)
        processImage(sorted(os.listdir(frames_folder), key=extract_integer_to_sort), folder)



if __name__ == "__main__":
    is_extract_frames = input("Please, type y if you wish to extract frames")
    if is_extract_frames == 'y':
        extract_frames_custom()

    is_process_frames = input("Please, type y if you wish to process frames")
    if is_process_frames == 'y':
        process_frames()

    are_ready_images = input("Please, type y if the images are already processed and you are ready to generate the .json file otherwise, type n")
    if are_ready_images == 'y':
        generateJson()

