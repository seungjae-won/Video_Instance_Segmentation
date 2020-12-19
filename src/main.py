import argparse
from util import *
import PIL.Image as pilimg
import os
import numpy as np
import cv2



# Parser 생성하기
parser = argparse.ArgumentParser(description='Video change detection', 
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--input_video_path_unet_version', default="C:/Users/wonseungjae/Desktop/real_img", type=str, dest='input_video_path_unet_version')
parser.add_argument('--input_video_path_morphology_version', default="C:/Users/wonseungjae/Google 드라이브/CV_practice_3/skating/skating/input", type=str, dest='input_video_path_morphology_version')
parser.add_argument('--Laplacian_direction_num', default=4, type=int, dest='Laplacian_direction_num')
parser.add_argument('--morphology_kernel_size', default=5, type=int, dest='morphology_kernel_size')
parser.add_argument('--maxCorners', default=5, type=int, dest='maxCorners')
parser.add_argument('--qualityLevel', default=0.3, type=float, dest='qualityLevel')
parser.add_argument('--minDistance', default=7, type=int, dest='minDistance')
parser.add_argument('--blockSize', default=7, type=int, dest='blockSize')
parser.add_argument('--winSize', default=15, type=int, dest='winSize')
parser.add_argument('--maxLevel', default=2, type=int, dest='maxLevel')
parser.add_argument('--window_size', default=5, type=int, dest='window_size')

PARSER = Parser(parser)


'''
    @brief : UNet을 통해 segment가 완료된 이미지를 이용해서 object tracking을 진행하는 main 함수

    @date : 2020/12/19 업데이트

    @return : None

    @param : None

'''
def main():
    ARGS = PARSER.get_arguments()
    PARSER.print_args()
    
    unet_image_list  = os.listdir(ARGS.input_video_path_unet_version)
        
    for i in range(1916,len(unet_image_list)):
        print(i, '----------------------------------')
        unet_image = os.path.join(ARGS.input_video_path_unet_version,unet_image_list[i])
        
        unet_image = pilimg.open(unet_image)
        unet_image = np.array(unet_image)

        # window_search : segment image에서 각각의 객체 찾기
        matching_result = window_search(ARGS.window_size, unet_image)

        
        # 가장 처음 이미지의 경우 비교할 이전 이미지가 없으므로 곧 바로 object_segment 진행
        if i == 1916:
            object_dict = {}
    
            for k in range(len(matching_result)):
                
                object_dict[k] = matching_result[k]
                
            unet_image = object_segment(object_dict, unet_image)    

        # 이전 이미지와 현재 이미지의 비교를 통한 object segment 
        else:
            object_dict = object_classification(object_dict, matching_result, unet_image)
            unet_image = object_segment(object_dict, unet_image)
        
        # image 저장 및 show 
        save_path = os.path.join("C:/Users/wonseungjae/Desktop/final_result", unet_image_list[i])
        unet_image.save(save_path,'PNG')
        
        unet_image = np.array(unet_image)
        cv2.imshow('unet_object_edge_detection_results', unet_image)

        k = cv2.waitKey(1)
        
    cv2.destroyAllWindows()



# parser 목록 print 및 main 함수 실행
if __name__ == '__main__':
    
    PARSER.print_args()
    main()
    