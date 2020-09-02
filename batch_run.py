import os

home = os.environ['HOME']
#folder_root = os.path.join(home,'data/trojai-round0-dataset')
#folder_root = os.path.join(home,'data/round1-dataset-train/models/')
folder_root = os.path.join(home,'data/round2-dataset-train/')
dirs = os.listdir(folder_root)

k = 0
for d in dirs:
  if 'id-00000001' not in d:  #Inception3 #trojaned #2048*5*5 = 51,200
    continue
  model_filepath=os.path.join(folder_root, d, 'model.pt')
  examples_dirpath=os.path.join(folder_root, d, 'example_data')

  cmmd = 'python3 trojan_detector.py --model_filepath='+model_filepath+' --examples_dirpath='+examples_dirpath

  print(cmmd)
  os.system(cmmd)

  k = k+1
  break
