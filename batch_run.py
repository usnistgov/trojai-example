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
arch_ct['resnet18'] = 0
arch_ct['resnet34'] = -1
arch_ct['resnet50'] = -1
arch_ct['resnet101'] = -1
arch_ct['resnet152'] = -1
arch_ct['wideresnet50'] = -1
arch_ct['wideresnet101'] = -1
arch_ct['densenet121'] = -1
arch_ct['densenet161'] = -1
arch_ct['densenet169'] = -1
arch_ct['densenet201'] = -1
arch_ct['googlenet'] = -1
arch_ct['inceptionv3'] = -1
arch_ct['squeezenetv1_0'] = -1
arch_ct['squeezenetv1_1'] = -1
arch_ct['mobilenetv2'] = -1
arch_ct['shufflenet1_0'] = -1
arch_ct['shufflenet1_5'] = -1
arch_ct['shufflenet2_0'] = -1
arch_ct['vgg11bn'] = -1
arch_ct['vgg13bn'] = -1
arch_ct['vgg16bn'] = -1
arch_ct['vgg19bn'] = -1
#'''


k = 0
for i,d in enumerate(dirs):
  if not os.path.isdir(os.path.join(folder_root,d)):
    continue
  md_name = d.split('.')[0]

  #if not md_name == 'id-00000046': #benign
  #    continue
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

