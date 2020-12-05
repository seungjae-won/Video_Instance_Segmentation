from PIL import Image
import os
import cv2
import numpy as np

data_dir = "C:/Users/wonseungjae/Google 드라이브/CV_practice_3/data/test"
image_path_list = [os.path.join(data_dir,file_name) for file_name in os.listdir(data_dir)]
image_path_list = sorted(image_path_list)

train_gray = []
train_rgb = []

for image in image_path_list:
    
    # gray_scale save
    img_grayscale = Image.open(image).convert('L')
    img_grayscale = np.array(img_grayscale)
    img_grayscale = img_grayscale[:,:,np.newaxis]
    train_gray.append(img_grayscale)
    
    img_rgb = Image.open(image)
    img_rgb = np.array(img_rgb)
    train_rgb.append(img_rgb)
    
train_gray = np.array(train_gray)
train_rgb = np.array(train_rgb)
print(train_gray.shape)
print(train_rgb.shape)

np.save("C:/Users/wonseungjae/Google 드라이브/CV_practice_3/numpy/test_gray.npy", train_gray)
np.save("C:/Users/wonseungjae/Google 드라이브/CV_practice_3/numpy/test_rgb.npy", train_rgb)


data_dir = "C:/Users/wonseungjae/Google 드라이브/CV_practice_3/data/label"
label_path_list = [os.path.join(data_dir,file_name) for file_name in os.listdir(data_dir)]
label_path_list = sorted(label_path_list)

train_gray = []
train_rgb = []

for label in label_path_list:
    
    label_rgb = Image.open(label)
    label_rgb = np.array(label_rgb)
    label_rgb = label_rgb[:,:,np.newaxis]
    train_rgb.append(label_rgb)

train_gray = np.array(train_gray)
train_rgb = np.array(train_rgb)
print(train_gray.shape)
print(train_rgb.shape)

#np.save("C:/Users/wonseungjae/Google 드라이브/CV_practice_3/numpy/train_gray.npy", train_gray)
np.save("C:/Users/wonseungjae/Google 드라이브/CV_practice_3/numpy/label.npy", train_rgb)
