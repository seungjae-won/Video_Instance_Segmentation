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
    
    start_w = -1
    start_h = 9999
    end_w = -1
    end_h = -1
    
    for i in range(0,width,window_size):
        
        matching_start = False
        
        for j in range(0,height-window_size,window_size):
            if matching(i,j,window_size,image):
                matching_start = True
                if start_w != -1:
                    if start_h > j:
                        start_h = j
                    elif end_h < j:
                        end_h = j
                
        
        if i < (width-window_size):
            if matching_start == False:
                if start_w != -1:
                    end_w = i
                    matching_result.append([[start_w,end_w],[start_h,end_h]])
                    start_w = -1
                    start_h = 9999
                    end_w = -1
                    end_h = -1
                else:
                    continue
            else:
                if start_w == -1:
                    start_w = i
                else:
                    continue
        else:
            if matching_start == True:
                end_w = i
                matching_result.append([[start_w,end_w],[start_h,end_h]])
                start_w = -1
                start_h = 9999
                end_w = -1
                end_h = -1
            
    return matching_result

def matching(width,height, window_size, image):
    count =0
    for n in range(width,width+window_size):
        for m in range(height,height+window_size):
            if np.mean(image[m][n]) != 0:
                count+=1
    
    if count >= (window_size*window_size//2):
        return True
    else:
        return False
    

def object_classification(past_object_dict, matching_result, unet_image):

    
    present_object_dict = {}
    
    past_catch_image = []
    present_catch_image = []
    
    for key, value in past_object_dict.items():
        past_catch_image.append(catch_image(value[0],value[1],unet_image))
        
    for value in matching_result:
        present_catch_image.append(catch_image(value[0],value[1],unet_image))

    if len(past_catch_image) >= len(present_catch_image):
        for i in range(len(present_catch_image)):
            simil_result = []

            for j in range(len(past_catch_image)):
                simil_result.append(calculate_image_simil(past_catch_image[j], present_catch_image[i]))
            
            present_object_dict[simil_result.index(min(simil_result))] = matching_result[simil_result.index(min(simil_result))]
    
    else:
        result = []
        image_index = [_ for _ in range(len(present_catch_image))]
        
        for i in range(len(present_catch_image)):
            simil_result = []

            for j in range(len(past_catch_image)):
                simil_result.append(calculate_image_simil(past_catch_image[j], present_catch_image[i]))
            
            result.append(simil_result)
        
        for m in range(len(past_catch_image)):
            value = [result[i][m] for i in range(result)]
            image_index.remove(value.index(min(value)))
            present_object_dict[m] = matching_result[value.index(min(value))]
        
        present_object_dict[len(past_catch_image)] = matching_result[image_index[0]]
        

    return present_object_dict


def catch_image(width, height, image):
    catch_image = [[[0 for _ in range(3)] for _ in range(width[1]-width[0])] for _ in range(height[1]-height[0])]

    for i in range(height[0],height[1]-1):
        for j in range(width[0],width[1]-1):
            for m in range(3):
                catch_image[i-height[0]][j-width[0]][m] = image[i][j][m]

    catch_image = np.array(catch_image)
    catch_image = pilimg.fromarray(catch_image.astype('uint8'), 'RGB')

    return catch_image

def segment_to_RGB(segment_image, rgb_image):
    create_image = [[[0 for _ in range(3)] for _ in range(540)] for _ in range(360)]
    
    for i in range(len(segment_image)):
        for j in range(len(segment_image[i])):
            if segment_image[i][j] != 0:
                for k in range(3):
                    create_image[i][j][k] = rgb_image[i][j][k]
    create_image = np.array(create_image)

    create_image = pilimg.fromarray(create_image.astype('uint8'), 'RGB')

    return create_image

def object_segment(object_dict, image):

    color_dict = {0 : [255,0,0], 1 :  [0,255,0], 2 : [0,0,255], 3 : [0,255,255], 4 : [255,255,0], 5 : [255,0,255] }
    
    object_segment_image = [[[0 for _ in range(3)] for _ in range(540)] for _ in range(360)]
    
    for i in range(len(image)):
        for j in range(len(image[i])):
            if np.mean(image[i][j]) != 0:
                for key, value in object_dict.items():
                    if j >= value[0][0] and j <= value[0][1]:
                        for m in range(3):
                            object_segment_image[i][j][m] = color_dict[key][m]
                        break
                    
                    
    object_segment_image = np.array(object_segment_image)

    object_segment_image = pilimg.fromarray(object_segment_image.astype('uint8'), 'RGB')

    return object_segment_image



def calculate_image_simil(image_1, image_2):
    image_1 = image_1.convert('L')
    image_2 = image_2.convert('L')
    image_2 = np.array(image_2)
    image_1 = image_1.resize((len(image_2[0]),len(image_2)))
    image_1 = np.array(image_1)
    
    calculate = np.sum((image_2 - image_1)**2)//(len(image_1)*len(image_1[0]))

    return calculate