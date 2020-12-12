import numpy as np
import cv2


# source
# https://github.com/hanyoseob/pytorch-StarGAN/blob/master/utils.py
class Parser:
    def __init__(self, parser):
        self.__parser = parser
        self.__args = parser.parse_args()

    def get_parser(self):
        return self.__parser

    def get_arguments(self):
        return self.__args

    def print_args(self, name='PARAMETER TABLES'):
        params_dict = vars(self.__args)
        print('\n\n')
        print('----' * 10)
        print('{0:^40}'.format(name))
        print('----' * 10)
        for k, v in sorted(params_dict.items()):
            if '__' not in str(k):
                print('{}'.format(str(k)) + ' : ' + ('{0:>%d}' % (35 - len(str(k)))).format(str(v)))
        print('----' * 10)
        print('\n\n')

def YCrCb(r,g,b):
    return r*0.2126+g*0.7152+b*0.0722

def rgb_to_gray(image):
    #YCrCb
    convert_image = np.zeros((len(image),len(image[0])))
    count=1
    for row in image:
        for pixel in row:
            count+=1
            print(count)
            Y = YCrCb(pixel[0], pixel[1], pixel[2])
            convert_image[row][pixel] = Y
    
    return convert_image

def Laplacian(direction_num, image):
    
    if direction_num == 4:
        Laplacian_filter = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    elif direction_num == 8:
        Laplacian_filter = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])

    image = cv2.filter2D(image, -1, Laplacian_filter)
    
    return image

def morphology(kernel,fgbg,image):
    
    image = fgbg.apply(image)
    image = cv2.morphologyEx(image,cv2.MORPH_OPEN,kernel)

    return image