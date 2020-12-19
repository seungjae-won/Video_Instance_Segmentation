import numpy as np
import PIL.Image as pilimg
import cv2


# 해당 parser class는 아래 github source를 참조해 이용했습니다.
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



'''
    @brief : segment image에서 window size 만큼 이동하며 object search, 결과 return

    @date : 2020/12/19 업데이트

    @return : matching_result (list)

    @param : window_size (int), image (numpy)
'''
def window_search(window_size, image):
    
    # image의 width와 height 추출
    image = np.array(image)
    height = len(image)
    width = len(image[0])

    matching_result = []
    
    # 초기 값 input
    start_w = -1
    start_h = 9999
    end_w = -1
    end_h = -1
    
    
    # 0 ~ width 까지 window_size 만큼 jump 하며 object search
    for i in range(0,width,window_size):
        
        # matching이 이루어졌는지 판단 변수 
        matching_start = False
        
        # 0 ~ height 까지 window size만큼 jump 하며 object search 
        for j in range(0,height-window_size,window_size):
            if matching(i,j,window_size,image):
                matching_start = True
                if start_w != -1:
                    if start_h > j:
                        start_h = j
                    elif end_h < j:
                        end_h = j
                
        # i 가 width 끝까지 닫지 않아 계속 search가 가능한 경우
        if i < (width-window_size):
            
            # 해당 window 영역에 object가 발견되지 않았을 경우
            if matching_start == False:
                
                # 탐색이 진행되고 있다면 탐색을 멈추고 matching_result에 object 영역을 추가한다.
                if start_w != -1:
                    if start_h > j:
                        start_h = j
                    elif end_h < j:
                        end_h = j
                    end_w = i
                    matching_result.append([[start_w,end_w],[start_h,end_h]])
                    start_w = -1
                    start_h = 9999
                    end_w = -1
                    end_h = -1
                else:
                    continue
                
            # window 영역에 object가 발견되었을 경우
            else:
                
                # 객체 영역으로 구분되지 않았다면 해당 영역부터 탐색을 시작한다.
                if start_w == -1:
                    start_w = i
                    if start_h > j:
                        start_h = j
                    elif end_h < j:
                        end_h = j
                else:
                    continue
        
        # i가 width 끝까지 닿아서 더이상 search가 불가능한 경우 
        else:
            if matching_start == True:
                end_w = i
                matching_result.append([[start_w,end_w],[start_h,end_h]])
                start_w = -1
                start_h = 9999
                end_w = -1
                end_h = -1
            
    return matching_result




'''
    @brief : 해당 window_size 내 image에서 객체의 영역이 존재하는지 return

    @date : 2020/12/19 업데이트

    @return : bool

    @param : width (int), height (int), window_size (int), image (numpy)

'''
def matching(width,height, window_size, image):
    count =0
    
    # input으로 받은 width를 기준으로 window size 영역 내 object 탐색 진행
    for n in range(width,width+window_size):
        for m in range(height,height+window_size):
            
            # R,G,B 픽셀의 평균이 0이 아닐 경우 해당 픽셀 내 object가 존재하는 것으로 판단
            if np.mean(image[m][n]) != 0:
                count+=1
    
    # 전체 window 영역 중 절반 이상의 영역에서 object가 검출되면 return True
    if count >= (window_size*window_size//2):
        return True
    else:
        return False
    

'''
    @brief : 이전 object dict 결과와 현재 matching result의 비교를 통해 현재 object dict 결과 생성

    @date : 2020/12/19 업데이트

    @return : present_object_dict (dict)

    @param : past_object_dict (dict), matching_result (list), unet_image (numpy)
'''
def object_classification(past_object_dict, matching_result, unet_image):

    present_object_dict = {}
    
    past_catch_image = []
    present_catch_image = []
    
    distance_result = []
    
    
    # 현재 matching 결과 중 정확하게 이루어 지지 않은 matching이 있다면 해당 matching 결과 제거
    for i in range(len(matching_result)):
        if matching_result[i][0][0] == -1 or matching_result[i][0][1] == -1 or matching_result[i][1][0] == 9999 or matching_result[i][1][1] == -1:
            matching_result.pop(i)
    
    
    # 현재 matching 결과와 이전 이미지의 object 결과 값들과 각각 거리 계산
    for i in range(len(matching_result)):
        each_distance = []
        for key, value in past_object_dict.items():
            each_distance.append(object_distance(value,matching_result[i]))
        distance_result.append(each_distance)
    
    
    # 현재 matching 결과와 이전 이미지의 object 검출 결과를 이미지 내에서 crop
    for key, value in past_object_dict.items():
        past_catch_image.append(catch_image(value[0],value[1],unet_image))
        
    for value in matching_result:
        present_catch_image.append(catch_image(value[0],value[1],unet_image))

    
    # 과거 이미지 내 object 결과가 현재 이미지 내 object 개수보다 많을 때
    if len(past_catch_image) >= len(present_catch_image):
        for i in range(len(present_catch_image)):
            simil_result = []

            # image similarity 계산
            for j in range(len(past_catch_image)):
                
                # image_similarity * 0.1 + distance 로 가장 적합한 object 간 매칭 진행
                simil_result.append(0.1*calculate_image_simil(past_catch_image[j], present_catch_image[i]) * distance_result[i][j])
            
            
            present_object_dict[simil_result.index(min(simil_result))] = matching_result[i]
    
    # 과거 이미지 내 object 개수가 현재 이미지의 object 개수보다 작을 때
    else:
        result = []
        image_index = [_ for _ in range(len(present_catch_image))]
        complete = []
        for i in range(len(present_catch_image)):
            simil_result = []

            for j in range(len(past_catch_image)):
                # image_similarity * 0.1 + distance 계산
                simil_result.append(calculate_image_simil(past_catch_image[j], present_catch_image[i]))
            
            result.append(simil_result)
        
        
        # 과거 이미지 내 object 중 현재 object와 가장 적합한 매칭을 보이는 순서대로 매칭, 매칭되지 못한 현재 object는 새로운 object로 인식
        for m in range(len(past_catch_image)):
            value = [result[i][m] for i in range(len(result))]
            if value.index(min(value)) in complete:
                value[value.index(min(value))] = 9999999
            image_index.remove(value.index(min(value)))
            complete.append(value.index(min(value)))
            present_object_dict[m] = matching_result[value.index(min(value))]
        
        present_object_dict[len(past_catch_image)] = matching_result[image_index[0]]


    return present_object_dict


'''
    @brief : image 내에서 입력받은 width와 height에 맞게 object 이미지 return

    @date : 2020/12/19 업데이트

    @return : image (numpy)

    @param : width (list), height (list), image (numpy)

'''
def catch_image(width, height, image):
    
    if height[1] - height[0] <= 10:
        height[0]-=10
        
    catch_image = [[[0 for _ in range(3)] for _ in range(width[1]-width[0])] for _ in range(height[1]-height[0])]

    
    # 입력받은 width와 height를 바탕으로 image 내 object 영역 crop
    for i in range(height[0],height[1]):
        for j in range(width[0],width[1]):
            for m in range(3):
                catch_image[i-height[0]][j-width[0]][m] = image[i][j][m]

    catch_image = np.array(catch_image)

    catch_image = pilimg.fromarray(catch_image.astype('uint8'), 'RGB')

    return catch_image

'''
    @brief : 0(black) 와 255(white) 로 segment 된 이미지에서 255(white) 영역을 real image R,G,B 값으로 변환

    @date : 2020/12/19 업데이트

    @return : image(numpy)

    @param : segment_image (numpy), rgb_image (numpy)

'''
def segment_to_RGB(segment_image, rgb_image):
    create_image = [[[0 for _ in range(3)] for _ in range(540)] for _ in range(360)]
    
    
    # 0(black)으로 인식되지 않는 255(white)로 인식하는 부분은 rgb_image의 rgb 값으로 변환 
    for i in range(len(segment_image)):
        for j in range(len(segment_image[i])):
            if segment_image[i][j] != 0:
                for k in range(3):
                    create_image[i][j][k] = rgb_image[i][j][k]
    create_image = np.array(create_image)

    create_image = pilimg.fromarray(create_image.astype('uint8'), 'RGB')

    return create_image


'''
    @brief : object_dict 결과를 input으로 받아 image에 각각 coloring 

    @date : 2020/12/19 업데이트

    @return : image (numpy)

    @param : object_dict (dict), image(numpy)

'''
def object_segment(object_dict, image):

    # 각 객체에 맞는 r,g,b color 값을 dict로 표현
    color_dict = {0 : [255,0,0], 1 :  [0,255,0], 2 : [0,0,255], 3 : [0,255,255], 4 : [255,255,0], 5 : [255,0,255] }
    
    object_segment_image = [[[0 for _ in range(3)] for _ in range(540)] for _ in range(360)]
    
    # 각 객체 번호에 맞게 segment 영역에 color dict의 object 번호에 맞는 color 입력
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


'''
    @brief : image_1과 image_2 간의 MSE 계산을 통한 유사도 측정. 

    @date : 2020/12/19 업데이트

    @return : calculate (int)

    @param : image_1 (numpy), image_2 (numpy)
'''
def calculate_image_simil(image_1, image_2):
    
    # 과거 이미지를 현재 이미지의 사이즈에 맞게 resize
    image_1 = image_1.convert('L')
    image_2 = image_2.convert('L')
    image_2 = np.array(image_2)
    image_1 = image_1.resize((len(image_2[0]),len(image_2)))
    image_1 = np.array(image_1)
    
    # MSE를 이용해서 과거 object 영역과 현재 object 영역의 similarity를 계산
    calculate = np.sum((image_2 - image_1)**2)//(len(image_1)*len(image_1[0]))

    return calculate


'''
    @brief : 이전 이미지의 segment 영역의 중앙과 현재 이미지의 segment 영역의 중앙간의 거리 측정

    @date : 2020/12/19 업데이트

    @return : distance (int)

    @param : segment_1 (list), segment_2 (list)

'''
def object_distance(segment_1, segment_2):
    
    # 과거 이미지의 object 영역의 중앙점과 현재 이미지의 object 영역의 중앙점을 distance 계산
    width_1 = (segment_1[0][0]+segment_1[0][1])//2
    height_1 = (segment_1[1][0]+segment_1[1][1])//2
    
    width_2 = (segment_2[0][0]+segment_2[0][1])//2
    height_2 = (segment_2[1][0]+segment_2[1][1])//2
    
    return (width_1-width_2)**2+(height_1-height_2)**2