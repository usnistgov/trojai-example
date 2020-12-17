import os
import skimage.io
import numpy as np
from neuron import RELEASE as neuron_release
import pickle
import csv
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

RELEASE = neuron_release
current_modeave_name = None

def set_model_name(model_filepath):
  if RELEASE:
      return

  model_name = model_filepath.split('/')[-2]
  global current_model_name
  current_model_name = model_name


def regularize_numpy_images(np_raw_imgs):
    scope = tuple(range(1, len(np_raw_imgs.shape)))
    np_imgs = np_raw_imgs-np_raw_imgs.min(scope,keepdims=True)
    np_imgs = np_imgs/(np_imgs.max(scope,keepdims=True)+1e-9)
    return np_imgs

def chg_img_fmt(img,fmt='CHW'):
    shape = img.shape
    assert(len(shape) ==3)
    if img.dtype != np.uint8:
        _img = img.astype(np.uint8)
    else:
        _img = img.copy()
    if fmt=='CHW' and shape[0] > 3:
        _img = np.transpose(_img,(2,0,1))
    elif fmt=='HWC' and shape[-1] > 3:
        _img = np.transpose(_img,(1,2,0))
    return _img
    

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
    cat_fns[key] = sorted(cat_fns[key])
    for fn in cat_fns[key]:
      full_fn = os.path.join(examples_dirpath,fn)
      img = skimage.io.imread(full_fn)

      ''' #for round1 rgb->bgr
      r = img[:,:,0]
      g = img[:,:,1]
      b = img[:,:,2]
      img = np.stack((b,g,r),axis=2)
      #'''

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
  print('save recovered images to',folder)

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


def load_pkl_results(save_name, folder='scratch'):
    if len(save_name) > 0:
        save_name = '_'+save_name
    fpath = os.path.join(folder, current_model_name+save_name+'.pkl')
    with open(fpath,'rb') as f:
        data = pickle.load(f)
    return data


def save_pkl_results(data, save_name='', folder='scratch'):
    if RELEASE:
        return

    if not os.path.exists(folder):
        os.makedirs(folder)

    print('save out results')
    if len(save_name) > 0:
        save_name = '_'+save_name
    fpath = os.path.join(folder, current_model_name+save_name+'.pkl')
    with open(fpath,'wb') as f:
        pickle.dump(data,f)



def save_results(results, folder='output'):
  if RELEASE:
    return

  if not os.path.exists(folder):
    os.makedirs(folder)

  fpath = os.path.join(folder, current_model_name)
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

  #print('median: %f, MAD: %f' % (median, mad))
  #print('min anomaly index: %f' % min_idx)

  flag_list = []
  for sc, lb, ori in zip(a_idx, crosp_lb, l1_norm_list):
    if sc > 2.0:
      flag_list.append((lb,sc))

  #if len(flag_list) == 0:
  #  print('flagged label list: None')
  #else:
  #  print('flagged label list: %s' %
  #        ', '.join(['%s: %2f' % (lb,sc)
  #                   for lb,sc in flag_list]))

  return min_idx, a_idx


def read_gt_csv(filepath):
  rst = list()
  with open(filepath,'r',newline='') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
      rst.append(row)
  return rst


def demo_heatmap(R, save_path):
    R /= np.max(R)

    sx = sy = 2.24
    #sx = sy = 3.5
    b = 10*((np.abs(R)**3.0).mean()**(1.0/3))

    my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    my_cmap[:,0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)
    plt.figure(figsize=(sx,sy))
    #plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
    plt.axis('off')
    fig = plt.gcf()

    plt.imshow(R,cmap=my_cmap,vmin=-b,vmax=b,interpolation='nearest')
    #plt.show()
    fig.savefig(save_path)
















