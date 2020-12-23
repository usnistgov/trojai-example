# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import time
import os
import numpy as np
import random
import torch
#import torchvision.models
import warnings
warnings.filterwarnings("ignore")

import utils
#from NC_pytorch import Visualizer
from neuron import NeuronSelector, LRP, HeatmapClassifier
from neuron import RELEASE as neuron_release
RELEASE = neuron_release


ava_model_type = ['ResNet', 'DenseNet','Inception3']

def build_data_loader(X,Y, batch_size=32):
    tensor_X = torch.Tensor(X)
    tensor_Y = torch.Tensor(Y)
    dataset = TensorDataset(tensor_X, tensor_Y)
    loader = DataLoader(dataset,batch_size=batch_size,drop_last=False, shuffle=False)
    return loader

def try_visualizer(model,cat_batch):
    from scorecam import ScoreCam
    from gradcam import GradCam
    from guided_backprop import GuidedBackprop
    from torch.autograd import Variable
    from guided_gradcam import guided_grad_cam

    s_lb = 4
    raw_X = cat_batch[s_lb]['images']

    index = 0
    X = utils.regularize_numpy_images(raw_X)
    for x, raw_x in zip(X,raw_X):
        x_tensor = torch.from_numpy(x)
        x_tensor.unsqueeze_(0)
        x_var = Variable(x_tensor.cuda(), requires_grad=True)

        file_name_to_export = 'benign_'+str(index)
        index += 1

        ''' #score cam  and grad cam
        print('score_cam')
        score_cam = ScoreCam(model, target_layer=8)
        cam = score_cam.generate_cam(x_var)
        #print('grad_cam')
        #grad_cam = GradCam(model, target_layer=22)
        #cam = grad_cam.generate_cam(x_var)
        ori_input = utils.chg_img_fmt(raw_x,'HWC')
        from misc_functions import save_class_activation_images
        from PIL import Image
        oim = Image.fromarray(ori_input)
        save_class_activation_images(oim, cam,  file_name_to_export)
        #'''

        ''' #guided grad
        print('guided_grad')
        from misc_functions import (convert_to_grayscale,save_gradient_images,get_positive_negative_saliency)
        GBP = GuidedBackprop(model)
        guided_grads = GBP.generate_gradients(x_var)
        save_gradient_images(guided_grads, file_name_to_export + '_Guided_BP_color')
        grayscale_guided_grads = convert_to_grayscale(guided_grads)
        save_gradient_images(grayscale_guided_grads, file_name_to_export + '_Guided_BP_gray')
        pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
        save_gradient_images(pos_sal, file_name_to_export + '_pos_sal')
        save_gradient_images(neg_sal, file_name_to_export + '_neg_sal')
        #'''

        '''#guided grad cam
        print('guided_grad_cam')
        from misc_functions import (convert_to_grayscale,save_gradient_images)
        gcv2 = GradCam(model,target_layer=22)
        cam = gcv2.generate_cam(x_var)
        GBP = GuidedBackprop(model)
        guided_grads = GBP.generate_gradients(x_var)
        cam_gb = guided_grad_cam(cam,guided_grads)
        save_gradient_images(cam_gb, file_name_to_export + '_GGrad_Cam')
        grayscale_cam_gb = convert_to_grayscale(cam_gb)
        save_gradient_images(grayscale_cam_gb, file_name_to_export + '_GGrad_Cam_gray')
        #'''





def fake_trojan_detector(model_filepath, result_filepath, scratch_dirpath, examples_dirpath, example_img_format='png'):

    utils.set_model_name(model_filepath)

    print('model_filepath = {}'.format(model_filepath))
    print('result_filepath = {}'.format(result_filepath))
    print('scratch_dirpath = {}'.format(scratch_dirpath))
    print('examples_dirpath = {}'.format(examples_dirpath))

    if RELEASE:
        neuron_script = 'python3 /neuron.py'
    else:
        neuron_script = 'python3 neuron.py'

    cmmd = neuron_script+' --mode=select'
    cmmd = cmmd+' --model_filepath='+model_filepath+' --examples_dirpath='+examples_dirpath+' --scratch_dirpath='+scratch_dirpath
    print(cmmd)
    os.system(cmmd)

    data_dict = utils.load_pkl_results(save_name='selected', folder=scratch_dirpath)
    neurons = data_dict['neurons']
    candi_mat = data_dict['candi_mat']

    n_neurons = len(neurons)

    for k in range(n_neurons):
        cmmd = neuron_script+' --mode=lrp --k=%d'%k
        cmmd = cmmd+' --model_filepath='+model_filepath+' --examples_dirpath='+examples_dirpath+' --scratch_dirpath='+scratch_dirpath
        print(cmmd)
        os.system(cmmd)

    '''
    cat_batch = utils.read_example_images(examples_dirpath, example_img_format)

    all_x = np.concatenate([cat_batch[lb]['images'] for lb in cat_batch])
    all_y = np.concatenate([cat_batch[lb]['labels'] for lb in cat_batch])

    NS = NeuronSelector(model_filepath)
    neurons, candi_mat = NS.analyse(all_x, all_y)
    del NS

    k = 0
    for aimg, key in neurons:
        lrp = LRP(model_filepath, aimg, key)
        lrp.run()
        lrp.save_out(os.path.join(scratch_dirpath,'k'))
        k += 1
        del lrp
    '''

    if n_neurons==0:
        trojan_probability = 0
    else:
        HC = HeatmapClassifier()
        scores = HC.predict_folder(scratch_dirpath)
        trojan_probability = HC.calc_prob(scores, candi_mat)

    trojan_probability = min(max(trojan_probability,1e-12),1)

    print('Trojan Probability: {}'.format(trojan_probability))
    with open(result_filepath, 'w') as fh:
        fh.write("{}".format(trojan_probability))

    utils.save_results(np.asarray(trojan_probability))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default='./model.pt')
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.', default='./output.txt')
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.', default='./scratch/')
    parser.add_argument('--examples_dirpath', type=str, help='File path to the folder of examples which might be useful for determining whether a model is poisoned.', default='./example/')

    args = parser.parse_args()
    fake_trojan_detector(args.model_filepath, args.result_filepath, args.scratch_dirpath, args.examples_dirpath)


