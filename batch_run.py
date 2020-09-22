import os
import csv

home = os.environ['HOME']
folder_root = os.path.join(home,'data/round2-dataset-train/')
dirs = os.listdir(folder_root)


id_arch = dict()

def read_gt(filepath):
    rst = list()
    with open(filepath,'r',newline='') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            rst.append(row)
    return rst

home = os.getenv('HOME')
gt_path = os.path.join(home,'data/round2-dataset-train/METADATA.csv')
gt_csv = read_gt(gt_path)
for row in gt_csv:
    id_arch[row['model_name']] = row['model_architecture']



arch_list = ['resnet18','resnet34','resnet50','resnet101','resnet152',
             'wideresnet50', 'wideresnet101',
             'densenet121', 'densenet161', 'densenet169', 'densenet201',
             'googlenet', 'inceptionv3',
             'squeezenetv1_0', 'squeezenetv1_1',
             'mobilenetv2',
             'shufflenet1_0', 'shufflenet1_5', 'shufflenet2_0',
             'vgg11bn', 'vgg13bn', 'vgg16bn', 'vgg19bn']

arch_ct = dict()
for na in arch_list:
  arch_ct[na] = 0

#'''
arch_ct['resnet18'] = -1 #0.8 0.8 [0 2 3 4 6] 0.82 // 0.864 0.923 [0 1 3 4 5] 0.84
arch_ct['resnet34'] = -1 #0.823 0.846 [3 4 7 8 12, 28] 0.82
arch_ct['resnet50'] = -1 # 1.0 1.0 [6,9,11,28,35,48,50] 1.0 // 1.0 0.8 [6,11,28,48,50] 1.0
arch_ct['resnet101'] = -1
arch_ct['resnet152'] = -1
arch_ct['wideresnet50'] = -1
arch_ct['wideresnet101'] = -1
arch_ct['densenet121'] = -1
arch_ct['densenet161'] = -1
arch_ct['densenet169'] = -1
arch_ct['densenet201'] = -1
arch_ct['googlenet'] = -1 #0.804 0.888 [3,4,8,10,20,24,30,39] 0.82
arch_ct['inceptionv3'] = -1 #0.763 0.800 [6 11 19 20 77 88] // 0.805 0.857 [6,11,19,20,85,86,88,91] 0.813
arch_ct['squeezenetv1_0'] = -1 #0.709 0.800 [1 4 9 19] // 0.793 0.857 [1,2,4,7,9,16] 0.805
arch_ct['squeezenetv1_1'] = -1 #0.758 0.857 [5,8,9]// 0.89 0.875 [1,5,13,14,15] 0.833
arch_ct['mobilenetv2'] = -1 #0.795 0.857 [10 17 18 41] 0.803
arch_ct['shufflenet1_0'] = -1 #0.756 1.0 [4,5,7,33,51] 0.808
arch_ct['shufflenet1_5'] = -1 #0.714 0.9 [4,16,25,36] 0.762 // 0.755 0.8 [0,4,14,17] 0.796
arch_ct['shufflenet2_0'] = 0 #0.764 1.0 [4,14,20,32,39,46] 0.807
arch_ct['vgg11bn'] = -1 #0.84 0.8 [0,3,4] 0.83
arch_ct['vgg13bn'] = -1 #0.842 0.888 [0,2,3,4] 0.87
arch_ct['vgg16bn'] = -1
arch_ct['vgg19bn'] = -1
#'''


k = 0
for i,d in enumerate(dirs):
  if not os.path.isdir(os.path.join(folder_root,d)):
    continue
  md_name = d.split('.')[0]

  if not md_name == 'id-00000001': #benign
      continue
  #if not md_name == 'id-00000124': #trojaned
  #    continue
  #if not md_name == 'id-00000001': #benign
  #    continue
  #if id_arch[md_name] != 'resnet18':
  #    continue

  md_arch = id_arch[md_name]
  #if arch_ct[md_arch] > 0:
  #  continue
  if arch_ct[md_arch] < 0:
    continue
  arch_ct[md_arch] += 1


  fn = d.split('.')[0]
  num_str = fn.split('-')[1]
  num = int(num_str)
  model_filepath=os.path.join(folder_root, d, 'model.pt')
  examples_dirpath=os.path.join(folder_root, d, 'example_data')

  cmmd = 'CUDA_VISIBLE_DEVICES=0 python3 trojan_detector.py --model_filepath='+model_filepath+' --examples_dirpath='+examples_dirpath

  k = k+1

  #if k <= 6:
  #    continue

  print(k)
  print('folder ',i)
  print(cmmd)
  print('model architecture: ', md_arch)
  os.system(cmmd)

  break

