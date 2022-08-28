# # import numpy as np
# # from skimage.filters import gaussian
# # from moviepy.editor import VideoFileClip
# #
# # import mediapipe
# # selfie_segmentation =  mediapipe.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
# #
# # BG_COLOR = (192, 192, 192) # gray
# #
# # def blur_background(im):
# #     # mask = selfie_segmentation.process(im).segmentation_mask[:, :, None]
# #     # mask = mask > 0.8
# #     #
# #     # bg = gaussian(im.astype(float), sigma=4)
# #     # return mask * im + (1 - mask) * bg
# #     results = selfie_segmentation.process(im)
# #     condition = np.stack(
# #         (results.segmentation_mask,) * 2, axis=-1) > 0.4
# #     bg_image = np.zeros(im.shape, dtype=np.uint8)
# #     bg_image[:] = BG_COLOR
# #
# #     output_image = np.where(condition, im, bg_image)
# #     return  output_image
# #
# # video_clip = VideoFileClip('./results/Elevation_through_abduction.mp4')
# # video_clip = video_clip.fl_image(blur_background)
# # video_clip.write_videofile("sample.mp4", audio=False)
#
# import cv2
# import mediapipe as mp
# import numpy as np
# mp_drawing = mp.solutions.drawing_utils
# mp_selfie_segmentation = mp.solutions.selfie_segmentation
#
# # For static images:
# BG_COLOR = (192, 192, 192) # gray
# MASK_COLOR = (255, 255, 255) # white
# with mp_selfie_segmentation.SelfieSegmentation(
#     model_selection=0) as selfie_segmentation:
#     image = cv2.imread('./results/sample.jpg')
#     image_height, image_width, _ = image.shape
#     # Convert the BGR image to RGB before processing.
#     results = selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#
#     # Draw selfie segmentation on the background image.
#     # To improve segmentation around boundaries, consider applying a joint
#     # bilateral filter to "results.segmentation_mask" with "image".
#     condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.999999
#     # Generate solid color images for showing the output selfie segmentation mask.
#     # fg_image = np.zeros(image.shape, dtype=np.uint8)
#     # fg_image[:] = MASK_COLOR
#     bg_image = np.zeros(image.shape, dtype=np.uint8)
#     bg_image[:] = BG_COLOR
#     output_image = np.where(condition, image, bg_image)
#     cv2.imwrite('./selfie_segmentation_output' +'.png', output_image)

import requests
import time
import shutil
import json

headers = {'Authorization': '4e80a38968174405a5f0818a3c5a1904'}
file_list = ['./results/Elevation_through_abduction.mp4']
params = {
    'lang': 'en',
    'convert_to': 'video-backgroundremover'
}

api_url = 'https://api.backgroundremover.app/v1/convert/'
results_url = 'https://api.backgroundremover.app/v1/results/'


def download_file(url, local_filename):
    with requests.get("https://api.backgroundremover.app/%s" % url, stream=True) as r:
        with open(local_filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    return local_filename


def convert_files(api_url, params, headers):
    files = [eval(f'("files", open("{file}", "rb"))') for file in file_list]
    print(files)
    r = requests.post(
        url=api_url,
        files=files,
        data=params,
        headers=headers
    )
    return r.json()


def get_results(params):
    if params.get('error'):
        return params.get('error')
    r = requests.post(
        url=results_url,
        data=params
    )
    data = r.json()
    finished = data.get('finished')
    while not finished:
        if int(data.get('queue_count')) > 0:
            print('queue: %s' % data.get('queue_count'))
        time.sleep(5)
        results = get_results(params)
        print(results)
        results = json.dumps(results)
        if results:
            break
    if finished:
        print(data.get('files'))
        for f in data.get('files'):
            print(f.get('url'))
            download_file("%s" % f.get('url'), "%s" % f.get('filename'))
        return {"finished": "files downloaded"}
    return r.json()


get_results(convert_files(api_url, params, headers))