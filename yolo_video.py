import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image

def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
            #print("size",image.shape)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str, dest='model_path',
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str, dest='anchors_path',
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str, dest='classes_path',
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")

# import time
# import os
# from yolo import YOLO
# from PIL import Image
# import numpy as np

# def calculate_slope(point1, point2):
#     dx = point2[0] - point1[0]
#     dy = point2[1] - point1[1]
#     if dx == 0:  # avoid division by zero
#         return float('inf')
#     return dy / dx

# def detect_and_calculate_slopes(yolo, image_folder):
#     slopes = []
#     image_files = os.listdir(image_folder)
#     total_images = len(image_files)
    
#     for idx, img_file in enumerate(image_files):
#         start_time = time.time()  # 각 이미지 처리 시작 시간 기록
        
#         try:
#             image_path = os.path.join(image_folder, img_file)
#             image = Image.open(image_path)
#             detected_image = yolo.detect_image(image)
            
#             # Detect boxes
#             out_boxes, _, _ = yolo.sess.run([yolo.boxes, yolo.scores, yolo.classes],
#                                             feed_dict={
#                                                 yolo.yolo_model.input: np.expand_dims(np.array(detected_image), 0),
#                                                 yolo.input_image_shape: [image.size[1], image.size[0]],
#                                                 K.learning_phase(): 0
#                                             })
            
#             for box in out_boxes:
#                 top, left, bottom, right = box
#                 center_x = (left + right) / 2
#                 center_y = (top + bottom) / 2
#                 slope = calculate_slope((center_x, center_y), (177, 0))
#                 slopes.append(slope)
#                 print("Image: {}, Center: ({}, {}), Slope: {}".format(img_file, center_x, center_y, slope))
        
#         except Exception as e:
#             print("Error processing image {}: {}".format(img_file, e))
        
#         # 남은 시간 계산
#         elapsed_time = time.time() - start_time  # 한 이미지 처리 시간
#         remaining_images = total_images - (idx + 1)
#         estimated_remaining_time = elapsed_time * remaining_images
#         print("Estimated remaining time: {:.2f} seconds".format(estimated_remaining_time))
    
#     if slopes:
#         min_slope = min(slopes)
#         max_slope = max(slopes)
#         print("Smallest slope: {}".format(min_slope))
#         print("Largest slope: {}".format(max_slope))
#     else:
#         print("No slopes calculated.")

# if __name__ == '__main__':
#     import argparse
#     import keras.backend as K
    
#     parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
#     parser.add_argument('--model', type=str, dest='model_path', help='path to model weight file, default ' + YOLO.get_defaults("model_path"))
#     parser.add_argument('--anchors', type=str, dest='anchors_path', help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path"))
#     parser.add_argument('--classes', type=str, dest='classes_path', help='path to class definitions, default ' + YOLO.get_defaults("classes_path"))
#     parser.add_argument('--gpu_num', type=int, help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num")))
#     parser.add_argument('--image_folder', type=str, required=True, help='Path to the folder containing images for detection')

#     FLAGS = parser.parse_args()

#     yolo = YOLO(**vars(FLAGS))
    
#     detect_and_calculate_slopes(yolo, FLAGS.image_folder)
    
#     yolo.close_session()
