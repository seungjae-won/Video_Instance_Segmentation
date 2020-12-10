import argparse
from util import *
import PIL.Image as pilimg
import os
import numpy as np
import cv2


# Parser 생성하기
parser = argparse.ArgumentParser(description='Video change detection', 
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--input_video_path_unet_version', default="unet", type=str, dest='input_video_path_unet_version')
parser.add_argument('--input_video_path_morphology_version', default="morphology", type=str, dest='input_video_path_morphology_version')
parser.add_argument('--Laplacian_direction_num', default=4, type=int, dest='Laplacian_direction_num')

PARSER = Parser(parser)

def main():
    ARGS = PARSER.get_arguments()
    PARSER.print_args()
    
    morphology_image_list = os.listdir(ARGS.input_video_path_morphology_version)
    
    unet_image_list  = os.listdir(ARGS.input_video_path_unet_version)

    for i in range(len(unet_image_list)):
        
        unet_image = os.path.join(ARGS.input_video_path_unet_version,unet_image_list[i])
        morphology_image = os.path.join(ARGS.input_video_path_morphology_version,morphology_image_list[i])
                    
        unet_image = pilimg.open(unet_image)
        unet_image = np.array(unet_image)
        unet_image = Laplacian(ARGS.Laplacian_direction_num, unet_image)
        
        morphology_image = pilimg.open(morphology_image)
        morphology_image = np.array(morphology_image)
        morphology_image = Laplacian(ARGS.Laplacian_direction_num, morphology_image)
        
        cv2.imshow('unet_object_edge_detection_results', unet_image)
        cv2.imshow('morphology_object_edge_detection_results', morphology_image)

        k = cv2.waitKey(10)

    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()