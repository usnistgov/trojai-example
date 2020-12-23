import cv2
import numpy as np
import random
import json
import os
import sys


def paste(img, trigger, color):
    tmask = trigger>0
    trows, tcols = trigger.shape[:2]

    rows, cols = img.shape[:2]
    avil_rows = list(range(rows*3//8,rows*5//8-trows))
    avil_cols = list(range(cols*3//8,cols*5//8-tcols))

    st_r = random.choice(avil_rows)
    st_l = random.choice(avil_cols)

    mask = np.zeros_like(img)
    mask[st_r:st_r+trows, st_l:st_l+tcols, :] = tmask

    color = np.reshape(color,(1,1,len(color)))
    timg = mask*color+(1-mask)*img

    print(np.max(timg))
    print(timg.dtype)

    return timg.astype(np.uint8)

if __name__=='__main__':
    trigger_size = 50

    root_folder = sys.argv[1]
    out_folder=os.path.join(root_folder,'synthetic_example_data')
    img_folder=os.path.join(root_folder,'clean_example_data')

    with open(os.path.join(root_folder,'config.json')) as config_file:
        config = json.load(config_file)
    color = config['TRIGGER_COLOR']
    color.reverse()
    triggered_classes = config['TRIGGERED_CLASSES']

    trigger = cv2.imread(os.path.join(root_folder,'trigger.png'))
    trigger = trigger.astype(np.float32)
    trigger = cv2.resize(trigger, (trigger_size,trigger_size))
    trigger = trigger.astype(np.uint8)

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    fns = os.listdir(img_folder)
    for fn in fns:
        if not fn.endswith('.png'): continue
        lb = int(fn.split('_')[1])
        if lb not in triggered_classes: continue

        img = cv2.imread(os.path.join(img_folder,fn))
        timg = paste(img, trigger, color)
        cv2.imwrite(os.path.join(out_folder,fn),timg)






