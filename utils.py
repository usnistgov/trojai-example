import os
import skimage.io
import numpy as np
from neuron import RELEASE as neuron_release

RELEASE = neuron_release
current_model_name = None

def set_model_name(model_filepath):
  if RELEASE:
      return

  model_name = model_filepath.split('/')[-2]
  global current_model_name
  current_model_name = model_name


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
      img = skimage.io.imread(full_fn)

      h,w,c = img.shape
      dx = int((w-224)/2)
      dy = int((w-224)/2)
      img = img[dy:dy+224, dx:dx+224, :]

      img = np.transpose(img,(2,0,1)) # to CHW
      #img = np.expand_dims(img,0) # to NCHW

      # normalize to [0,1]
      #img = img - np.min(img)
      #img = img / np.max(img)

      cat_imgs[key].append(img)


  cat_batch = {}
  for key in cat_imgs:
    cat_batch[key] = {'images':np.asarray(cat_imgs[key], dtype=np.float32), 'labels':np.ones([len(cat_imgs[key]),1])*key}
    #print('label {} : {}'.format(key, cat_batch[key]['images'].shape))

  return cat_batch


def save_poisoned_images(pair, poisoned_images, benign_images, folder='recovered_images'):
  if RELEASE:
      return

  folder = os.path.join(folder,current_model_name)
  if not os.path.exists(folder):
    os.makedirs(folder)

  fn_template = 'poisoned_from_{}_to_{}.jpg'.format(pair[0],pair[1])
  fn_template = 'example_{}_'+fn_template
  for i in range(len(poisoned_images)):
    img = poisoned_images[i]
    img = np.transpose(img,(1,2,0)) # to CHW->HWC
    fn = fn_template.format(i)
    fpath = os.path.join(folder, fn)
    img_save = img.astype(np.uint8)
    skimage.io.imsave(fpath,img_save)

  fn_template = 'benign_from_{}.jpg'.format(pair[0])
  fn_template = 'example_{}_'+fn_template
  for i in range(len(benign_images)):
    img = benign_images[i]
    img = np.transpose(img,(1,2,0)) # to CHW->HWC
    fn = fn_template.format(i)
    fpath = os.path.join(folder, fn)
    img_save = img.astype(np.uint8)
    skimage.io.imsave(fpath,img_save)


  print(np.max(benign_images))
  print(np.max(poisoned_images))


def save_results(results, folder='output'):
  if RELEASE:
    return

  if not os.path.exists(folder):
    os.makedirs(folder)

  folder = 'output'
  fpath = os.path.join('output', current_model_name)
  np.save(fpath, results)



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














