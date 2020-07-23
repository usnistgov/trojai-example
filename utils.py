import os
import cv2
import numpy as np

def read_example_images(examples_dirpath, example_img_format='png'):
  fns = [fn for fn in os.listdir(examples_dirpath) if fn.endswith(example_img_format)]

  cat_fns = {}
  for fn in fns:
    true_lb = int(fn.split('_')[1])
    if true_lb not in cat_fns.keys():
      cat_fns[true_lb] = []
    cat_fns[true_lb].append(fn)

  cat_imgs = {}
  for key in cat_fns:
    cat_imgs[key] = []
    for fn in cat_fns[key]:
      full_fn = os.path.join(examples_dirpath,fn)
      img = cv2.imread(full_fn,cv2.IMREAD_UNCHANGED)
      img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

      img = np.transpose(img,(2,0,1)) # to CHW
      #img = np.expand_dims(img,0) # to NCHW

      # normalize to [0,1]
      #img = img - np.min(img)
      #img = img / np.max(img)

      cat_imgs[key].append(img)


  cat_batch = {}
  for key in cat_imgs:
    cat_batch[key] = np.asarray(cat_imgs[key])
    print('label {} : {}'.format(key, cat_batch[key].shape))

  return cat_batch



def dump_image(x, filename, format):
  return

  if (np.max(x) < 1.1):
    x = x*255.0

  if len(x.shape) == 4:
    x = x[0,...]
  if len(x.shape) > 2 and x.shape[0] <= 3:
    x = np.transpose(x,(1,2,0)) # to HWC

  cv2.imwrite(filename, x)

  return


def save_pattern(pattern, mask, y_source, y_target, result_dir):
  return 
  IMG_FILENAME_TEMPLATE = 'visualize_%s_label_%d_to_%d.png'
  if not os.path.exists(result_dir):
    os.mkdir(result_dir)

  img_filename = os.path.join(result_dir, (IMG_FILENAME_TEMPLATE % ('pattern',y_source, y_target)))
  dump_image(pattern, img_filename,'png')

  mask = np.expand_dims(mask,axis=0)
  img_filename = os.path.join(result_dir, (IMG_FILENAME_TEMPLATE % ('mask',y_source, y_target)))
  dump_image(mask, img_filename,'png')

  fusion = np.multiply(pattern, mask)
  img_filename = os.path.join(result_dir, (IMG_FILENAME_TEMPLATE % ('fusion',y_source, y_target)))
  dump_image(fusion, img_filename,'png')


def mad_detection(l1_norm_list, crosp_lb):
  constant = 1.4826
  median = np.median(l1_norm_list)
  mad = constant * np.median(np.abs(l1_norm_list-median))
  a_idx = np.abs(l1_norm_list-median) / mad
  #min_idx = np.abs(np.min(l1_norm_list)-median)/mad
  #min_idx = np.max(a_idx)
  min_idx = np.min(l1_norm_list)/np.max(l1_norm_list)

  print('median: %f, MAD: %f' % (median, mad))
  print('min anomaly index: %f' % min_idx)

  flag_list = []
  for sc, lb, ori in zip(a_idx, crosp_lb, l1_norm_list):
    if sc > 2:
      flag_list.append((lb,ori))

  if len(flag_list) == 0:
    print('flagged label list: None')
  else:
    print('flagged label list: %s' %
          ', '.join(['%s: %2f' % (lb,sc)
                     for lb,sc in flag_list]))

  return min_idx, a_idx














