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

def instagram_transform(img, xform_name, level):
    from numpy.random import RandomState
    import trojai.datagen.image_entity as dg_entity
    import trojai.datagen.instagram_xforms as dg_inst_xforms
    import trojai.datagen.utils as dg_utils

    img_random_state = RandomState(1234)

    X_obj = dg_entity.GenericImageEntity(img, mask=None)
    xform_class = getattr(dg_inst_xforms, xform_name)
    xforms = [xform_class()]

    X_obj = dg_utils.process_xform_list(X_obj, xforms, img_random_state)

    return X_obj.get_data()


if __name__=='__main__':
    trigger_size = 50

    root_folder = sys.argv[1]
    out_folder=os.path.join(root_folder,'synthetic_example_data')
    img_folder=os.path.join(root_folder,'clean_example_data')

    with open(os.path.join(root_folder,'config.json')) as config_file:
        config = json.load(config_file)

    trigger_type = config['TRIGGER_TYPE']
    if trigger_type == 'instagram':
        xform_name = config['INSTAGRAM_FILTER_TYPE']
        xform_level = int(config['INSTAGRAM_FILTER_TYPE_level'])
    elif trigger_type == 'polygon':
        trigger_color = config['TRIGGER_COLOR']
        trigger_color.reverse()
        trigger = cv2.imread(os.path.join(root_folder,'trigger.png'))
        trigger = trigger.astype(np.float32)
        trigger = cv2.resize(trigger, (trigger_size,trigger_size))
        trigger = trigger.astype(np.uint8)
    else:
        raise('Unknown trigger type '+trigger_type)

    triggered_classes = config['TRIGGERED_CLASSES']

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    fns = os.listdir(img_folder)
    for fn in fns:
        if not fn.endswith('.png'): continue
        lb = int(fn.split('_')[1])
        if lb not in triggered_classes: continue

        img = cv2.imread(os.path.join(img_folder,fn))
        if trigger_type == 'polygon':
            timg = paste_polygon(img, trigger, color)
        elif trigger_type == 'instagram':
            timg = instagram_transform(img, xform_name, xform_level)
        else:
            raise('Unknown trigger tyep '+trigger_type)
        cv2.imwrite(os.path.join(out_folder,fn),timg)






