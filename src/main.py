import argparse
from util import *
import PIL.Image as pilimg
import os
import numpy as np
import cv2


# Parser 생성하기
parser = argparse.ArgumentParser(description='Video change detection', 
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--input_video_path_unet_version', default="C:/Users/wonseungjae/Desktop/rgb1", type=str, dest='input_video_path_unet_version')
parser.add_argument('--input_video_path_morphology_version', default="C:/Users/wonseungjae/Google 드라이브/CV_practice_3/skating/skating/input", type=str, dest='input_video_path_morphology_version')
parser.add_argument('--Laplacian_direction_num', default=4, type=int, dest='Laplacian_direction_num')
parser.add_argument('--morphology_kernel_size', default=5, type=int, dest='morphology_kernel_size')
parser.add_argument('--maxCorners', default=5, type=int, dest='maxCorners')
parser.add_argument('--qualityLevel', default=0.3, type=float, dest='qualityLevel')
parser.add_argument('--minDistance', default=7, type=int, dest='minDistance')
parser.add_argument('--blockSize', default=7, type=int, dest='blockSize')
parser.add_argument('--winSize', default=15, type=int, dest='winSize')
parser.add_argument('--maxLevel', default=2, type=int, dest='maxLevel')

PARSER = Parser(parser)

def main():
    ARGS = PARSER.get_arguments()
    PARSER.print_args()
    
    morphology_image_list = os.listdir(ARGS.input_video_path_morphology_version)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ARGS.morphology_kernel_size,ARGS.morphology_kernel_size))
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    
    unet_image_list  = os.listdir(ARGS.input_video_path_unet_version)

    for i in range(len(unet_image_list)):
        
        unet_image = os.path.join(ARGS.input_video_path_unet_version,unet_image_list[i])
        morphology_image = os.path.join(ARGS.input_video_path_morphology_version,morphology_image_list[i])
        
        unet_image = pilimg.open(unet_image)
        unet_image = np.array(unet_image)
        unet_image = Laplacian(ARGS.Laplacian_direction_num, unet_image)
        
        # feature 추출
        p0 = cv2.goodFeaturesToTrack(unet_image, mask=None, 
                                     maxCorners = ARGS.maxCorners, 
                                     qualityLevel = ARGS.qualityLevel, 
                                     minDistance = ARGS.minDistance, 
                                     blockSize = ARGS.blockSize)          
        
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray_img, p0, None, winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        morphology_image = pilimg.open(morphology_image)
        morphology_image = np.array(morphology_image)
        
        morphology_image = Laplacian(ARGS.Laplacian_direction_num, morphology(kernel,fgbg,morphology_image))
        
        cv2.imshow('unet_object_edge_detection_results', unet_image)
        cv2.imshow('morphology_object_edge_detection_results', morphology_image)

        k = cv2.waitKey(10)
        
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    PARSER.print_args()
    main()
    