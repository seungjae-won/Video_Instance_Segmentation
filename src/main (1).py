import argparse
from util import *
import PIL.Image as pilimg
import os
import numpy as np
import cv2

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
    
    # Set random colors
    color = np.random.randint(0,255,(100,3))

    image_list = os.listdir(ARGS.input_video)
    img = os.path.join(ARGS.input_video,image_list[0])
    img = pilimg.open(img)
    img = np.array(img)

    old_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))
    # old_gray = cv2.morphologyEx(old_gray, cv2.MORPH_OPEN, kernel, iterations=9)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, 
                                     maxCorners = ARGS.maxCorners, 
                                     qualityLevel = ARGS.qualityLevel, 
                                     minDistance = ARGS.minDistance, 
                                     blockSize = ARGS.blockSize)                 
                                    #  useHarrisDetector=True)
    # p0 = np.array(p0)
    # print(p0.shape)
    # exit(0)
    mask = np.zeros_like(img)

    count = 1

    for image in image_list:
        count+=1

        img = os.path.join(ARGS.input_video,image)
        img = pilimg.open(img)
        img = np.array(img)
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))
        # gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel, iterations=9)
        
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray_img, p0, None, winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        if p1 is None:
            good_new = p0[st==1]
            good_old = p0[st==1]
        else:
            good_new = p1[st==1]
            good_old = p0[st==1]
        
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            img = cv2.circle(img,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(img,mask)
        cv2.imshow('frame',img)
        k = cv2.waitKey(1)

        # Copy former frame and points
        old_gray = gray_img.copy()
        p0 = good_new.reshape(-1,1,2)
        
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()
