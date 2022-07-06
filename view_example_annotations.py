import os
import numpy as np
import cv2
import json
from matplotlib import pyplot as plt
from pycocotools import mask as maskUtils


def showAnns(coco_anns, height, width, draw_bbox=False, draw_number=False):
    """
    Display the specified annotations.
    :param anns (array of object): annotations to display
    :return: None
    """
    if len(coco_anns) == 0:
        return 0
    if 'segmentation' in coco_anns[0] or 'keypoints' in coco_anns[0]:
        datasetType = 'instances'
    elif 'caption' in coco_anns[0]:
        datasetType = 'captions'
    else:
        raise Exception('datasetType not supported')
    if datasetType == 'instances':
        import matplotlib.pyplot as plt
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Polygon

        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        for ann in coco_anns:
            c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
            if 'segmentation' in ann:
                if type(ann['segmentation']) == list:
                    # polygon
                    for seg in ann['segmentation']:
                        poly = np.array(seg).reshape((int(len(seg)/2), 2))
                        polygons.append(Polygon(poly))
                        color.append(c)
                else:
                    # mask
                    if type(ann['segmentation']['counts']) == list:
                        rle = maskUtils.frPyObjects([ann['segmentation']], height, width)
                    else:
                        rle = [ann['segmentation']]
                    m = maskUtils.decode(rle)
                    img = np.ones( (m.shape[0], m.shape[1], 3) )
                    if ann['iscrowd'] == 1:
                        color_mask = np.array([2.0,166.0,101.0])/255
                    if ann['iscrowd'] == 0:
                        color_mask = np.random.random((1, 3)).tolist()[0]
                    for i in range(3):
                        img[:,:,i] = color_mask[i]
                    ax.imshow(np.dstack( (img, m*0.5) ))

            if draw_bbox:
                [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
                poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
                np_poly = np.array(poly).reshape((4,2))
                polygons.append(Polygon(np_poly))
                color.append(c)
            if draw_number:
                [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
                cx = bbox_x + int(bbox_w / 2)
                cy = bbox_y + int(bbox_h / 2)
                #ax.text(cx, cy, "{}".format(ann['category_id']), c='k', fontsize='large', fontweight='bold')
                ax.text(cx, cy, "{}".format(ann['category_id']), c=c, backgroundcolor='white', fontsize='small', fontweight='bold')

        p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
        ax.add_collection(p)
        p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
        ax.add_collection(p)


def visualize_image(img_fp, output_dirpath, draw_bbox=False, draw_number=False):
    parent_fp, img_fn = os.path.split(img_fp)
    ann_fn = img_fn.replace('.jpg', '.json')
    ann_fp = os.path.join(parent_fp, ann_fn)

    if not os.path.exists(img_fp):
        raise RuntimeError('Requested image file {} does not exists.'.format(img_fp))
    if not os.path.exists(ann_fp):
        raise RuntimeError('Requested image annotation file {} does not exists.'.format(ann_fp))

    img = cv2.imread(img_fp, cv2.IMREAD_UNCHANGED)  # loads to BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
    height = img.shape[0]
    width = img.shape[1]
    with open(ann_fp, 'r') as fh:
        coco_anns = json.load(fh)

    # clear the figure
    plt.clf()

    # show the image
    plt.imshow(img)
    # draw the annotations on top of that image
    showAnns(coco_anns, height, width, draw_bbox=draw_bbox, draw_number=draw_number)

    # save figure to disk
    if output_dirpath is not None:
        if not os.path.exists(output_dirpath):
            os.makedirs(output_dirpath)
        plt.savefig(os.path.join(output_dirpath, img_fn))
    else:
        # or render it
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Visualize the all example data annotations')
    parser.add_argument('--dataset_dirpath', type=str, required=True, help='Filepath to the folder/directory containing the TrojAI dataset')
    parser.add_argument('--draw_boxes', action='store_true', help='Whether to draw bounding boxes on visualization in addition to the segmentation mask')
    parser.add_argument('--draw_numbers', action='store_true', help='Whether to draw class label numbers on visualization')

    args = parser.parse_args()

    ifp = args.dataset_dirpath
    models = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]
    models.sort()

    for model in models:
        # visualize clean example images
        in_dir = os.path.join(ifp, model, 'clean-example-data')
        out_dir = os.path.join(ifp, model, 'clean-example-data-visualization')
        imgs = [os.path.join(in_dir, fn) for fn in os.listdir(in_dir) if fn.endswith('.jpg')]

        for img in imgs:
            visualize_image(img, out_dir, args.draw_boxes, args.draw_numbers)

        # visualize poisoned example images
        in_dir = os.path.join(ifp, model, 'poisoned-example-data')
        if os.path.exists(in_dir):
            out_dir = os.path.join(ifp, model, 'poisoned-example-data-visualization')
            imgs = [os.path.join(in_dir, fn) for fn in os.listdir(in_dir) if fn.endswith('.jpg')]

            for img in imgs:
                visualize_image(img, out_dir, args.draw_boxes, args.draw_numbers)




