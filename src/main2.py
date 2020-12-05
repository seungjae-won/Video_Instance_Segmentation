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
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

    for image in image_list:

        img = os.path.join(ARGS.input_video,image)
        img = pilimg.open(img)
        img = np.array(img)
        
        
        fgmask = fgbg.apply(img)
        fgmask = cv2.morphologyEx(fgmask,cv2.MORPH_OPEN,kernel)
        
        
        cv2.imshow('origin',img)
        cv2.imshow('mask',fgmask)
        k = cv2.waitKey(1)


    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()
