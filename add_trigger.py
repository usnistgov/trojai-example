import sys
import numpy as np
import cv2
import random
import wand.image
from io import BytesIO
import skimage.io
import os

#Usage:
#python3 add_trigger.py path color_tuple(0,1,2) range_of_trigger(l,r) position_of_trigger(x,y)
#e.g: python3 add_trigger.py ./example_data/ 145 178 26 11 32 112 112
#color=(145,178,26)
#range_of_trigger_size=(11~32)
#position_of_trigger=(112,112)

def add_trigger(img_array,color,trigger_img,trigger_size=50,trigger_position=None):
    shape = img_array.shape
    if trigger_position is None and len(shape) == 3:
        if shape[0] <= 3:
            x = shape[1]//2
            y = shape[2]//2
        elif shape[2] <= 3:
            x = shape[0]//2
            y = shape[1]//2
    new_trigger_img=cv2.resize(trigger_img,(trigger_size,trigger_size))
    new_trigger_size=(new_trigger_img.shape[0],new_trigger_img.shape[1])
    center=(new_trigger_img.shape[0]//2,new_trigger_img.shape[1]//2)
    angle=random.randint(0,360)
    rotation=cv2.getRotationMatrix2D(center,angle,1)
    new_trigger_img=cv2.warpAffine(new_trigger_img,rotation,new_trigger_size)
    triggered_img=img_array
    dx=random.randint(-30,30)
    dy=random.randint(-30,30)
    x=x+dx
    y=y+dy
    for i in range(len(new_trigger_img)):
        for j in range(len(new_trigger_img[0])):
            if new_trigger_img[i][j][0]+new_trigger_img[i][j][1]+new_trigger_img[i][j][2]>0:
                triggered_img[x+i][y+j][0]=color[0]
                triggered_img[x+i][y+j][1]=color[1]
                triggered_img[x+i][y+j][2]=color[2]
    return triggered_img

if __name__ == '__main__':
    path=sys.argv[1]
    files= os.listdir(path)
    triggerimg=cv2.imread('trigger.png')
    for file in files:
        imagename=os.path.join(path,file)
        img=cv2.imread(imagename)
        trigger_color=(int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]))
        output_img=add_trigger(img,trigger_color,triggerimg)
        outputname=file
        cv2.imwrite(outputname,output_img)
