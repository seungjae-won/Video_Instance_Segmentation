import argparse
from util import *
import PIL.Image as pilimg
import os
import numpy as np
import cv2
#from google.colab.patches import cv2_imshow


# Parser 생성하기
parser = argparse.ArgumentParser(description='Human motion classification', 
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--input_video', default="skating/skating/input/", type=str, dest='input_video')
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
    
    image_list = os.listdir(ARGS.input_video)
    
    ground_truth_image  = os.listdir("skating/skating/groundtruth")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

    for image in range(len(image_list)):
        img = os.path.join(ARGS.input_video,image_list[image])
        ground_truth = os.path.join("skating/skating/groundtruth",ground_truth_image[image])
        img = pilimg.open(img)
        ground_truth = pilimg.open(ground_truth)
        img = np.array(img)
        ground_truth = np.array(ground_truth)
        
        cv2.imshow('origin',img)
        cv2.imshow('mask',ground_truth)
        k = cv2.waitKey(1)


    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()