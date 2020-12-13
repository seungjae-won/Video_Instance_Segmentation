import numpy as np
import PIL.Image as pilimg
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

def window_search(window_size, image):
    image = np.array(image)
    height = len(image)
    width = len(image[0])

    matching_result = []
    
    start = -1
    end = -1
    
    for i in range(0,width,window_size):
        
        matching_start = False
        
        for j in range(0,height-window_size,window_size):
            if matching(i,j,window_size,image):
                matching_start = True
                break
                
        
        if i < (width-window_size):
            if matching_start == False:
                if start != -1:
                    end = i
                    matching_result.append([start,end])
                    start = -1
                    end = -1
                else:
                    continue
            else:
                if start == -1:
                    start = i
                else:
                    continue
        else:
            if matching_start == True:
                end = i
                matching_result.append([start,end])
                start = -1
                end = -1
            
    return matching_result

def matching(width,height, window_size, image):
    count =0
    for n in range(width,width+window_size):
        for m in range(height,height+window_size):
            if image[m][n] == 255:
                count+=1
    
    if count >= (window_size*window_size//2):
        return True
    else:
        return False
    

