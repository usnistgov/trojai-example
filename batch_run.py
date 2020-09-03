import os

home = os.environ['HOME']
#folder_root = os.path.join(home,'data/trojai-round0-dataset')
#folder_root = os.path.join(home,'data/round1-dataset-train/models/')
folder_root = os.path.join(home,'data/round2-dataset-train/')
dirs = os.listdir(folder_root)

k = 0
for d in dirs:
  #if 'id-00000001' not in d:  #shufflenet #trojaned
  #  continue
  #if 'id-00000272' not in d:  #squeezenet #trojaned
  #  continue
  #if 'id-00000006' not in d:  #vgg #benign
  #  continue
  #if 'id-00000008' not in d:  #googlenet #trojaned
  #  continue
  #if 'id-00000024' not in d:  #mobilenet #trojaned
  #  continue
  #if 'id-00000000' not in d:  #densenet #benign
  #  continue
  if 'id-00000002' not in d:  #resnet #benign
    continue
  model_filepath=os.path.join(folder_root, d, 'model.pt')
  examples_dirpath=os.path.join(folder_root, d, 'example_data')

  cmmd = 'python3 trojan_detector.py --model_filepath='+model_filepath+' --examples_dirpath='+examples_dirpath

  print(cmmd)
  os.system(cmmd)

  k = k+1
  break
  model_filepath=os.path.join(folder_root, d, 'model.pt')
  examples_dirpath=os.path.join(folder_root, d, 'example_data')

  cmmd = 'python3 trojan_detector.py --model_filepath='+model_filepath+' --examples_dirpath='+examples_dirpath

  print(cmmd)
  os.system(cmmd)

  k = k+1
  break
