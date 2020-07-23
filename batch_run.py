import os

home = os.environ['HOME']
folder_root = os.path.join(home,'data/trojai-round0-dataset')
dirs = os.listdir(folder_root)


for d in dirs:
  model_filepath=os.path.join(folder_root, d, 'model.pt')
  examples_dirpath=os.path.join(folder_root, d, 'example_data')

  cmmd = 'python3 trojan_detector.py --model_filepath='+model_filepath+' --examples_dirpath='+examples_dirpath

  print(cmmd)
  os.system(cmmd)
  break
