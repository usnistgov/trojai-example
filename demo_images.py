import matplotlib.pyplot as plt
import pickle
import math
import os
import skimage.io
import numpy as np

def colorize_matrix(a):
    maxa = np.max(np.abs(a))
    a = a/maxa
    #a = (a-np.min(a))/(np.max(a)-np.min(a))

    img = np.zeros_like(a)
    img = np.expand_dims(img,axis=-1)
    img = np.repeat(img,3,axis=2)
    img[:,:,0] = (a>0)*a #r
    img[:,:,1] = 0 #g
    img[:,:,2] = (a<0)*-a #b

    img = img*255.0
    img = img.astype(np.uint8)

    return img

def list_to_matrix(a):
    a = np.asarray(a)
    a = a.flatten()
    n = len(a)

    sqn = math.ceil(math.sqrt(n))
    for i in range(sqn):
        if n%(sqn-i) > 0: continue
        cols = sqn-i
        break
    rows = n//cols
    rows, cols = min(rows,cols), max(rows,cols)

    a = np.asarray(a)
    a = np.reshape(a,(rows,cols))

    return a

def save_image(x, filename):

  if len(x.shape) > 4:
      raise RuntimError('images shape len > 4')
  if len(x.shape) == 4:
    x = x[0,...]
  if len(x.shape) == 3:
    if x.shape[0] > 4 and x.shape[1]*x.shape[2] > 1: x = np.sum(x,axis=0)
    elif x.shape[0] <= 4: x = np.transpose(x,(1,2,0)) # to HWC

  x = np.squeeze(x)
  if len(x.shape) == 1:
    x = list_to_matrix(x)

  if len(x.shape)==2: x = colorize_matrix(x)

  if len(x.shape) == 2 or (len(x.shape)==3 and x.shape[2]==3):
    if x.shape[0] < 100: x=np.repeat(x,10,axis=0)
    if x.shape[1] < 100: x=np.repeat(x,10,axis=1)
    skimage.io.imsave(filename, x)
  else:
    raise RuntimeError('images with wrong shape: '+str(x.shape))

  return


fo = 'scratch'
suffix='layer_attr.pkl'

md_names = list()
fns = os.listdir(fo)
for fn in fns:
  if not fn.endswith(suffix): continue
  md_name = fn.split('_')[0]
  md_names.append(md_name)


for md_name in md_names:
  print(md_name)
  with open(os.path.join('scratch',md_name+'_'+suffix),'rb') as f:
    data = pickle.load(f)

  out_fo = os.path.join(fo,md_name)
  if not os.path.exists(out_fo):
    os.makedirs(out_fo)

  for key in data.keys():
    img = data[key]
    out_name = '_'.join([str(_) for _ in key])+'.png'
    out_path = os.path.join(out_fo,out_name)
    save_image(img, out_path)
    print(key, data[key].shape)

