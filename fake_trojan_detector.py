# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import numpy as np
import skimage.io
import torch
import advertorch.attacks
import advertorch.context

import warnings 
warnings.filterwarnings("ignore")


def fake_trojan_detector(model_filepath, result_filepath, scratch_dirpath, examples_dirpath, example_img_format='png'):

    print('model_filepath = {}'.format(model_filepath))
    print('result_filepath = {}'.format(result_filepath))
    print('scratch_dirpath = {}'.format(scratch_dirpath))
    print('examples_dirpath = {}'.format(examples_dirpath))

    # load the model and move it to the GPU
    model = torch.load(model_filepath, map_location=torch.device('cuda'))

    # Inference the example images in data
    fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith(example_img_format)]
    fns.sort()  # ensure file ordering
    if len(fns) > 5: fns = fns[0:5]  # limit to 5 images

    # setup PGD
    # define parameters of the adversarial attack
    attack_eps = float(8/255)
    attack_iterations = int(7)
    eps_iter = (2.0 * attack_eps) / float(attack_iterations)

    # create the attack object
    attack = advertorch.attacks.LinfPGDAttack(
        predict=model,
        loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
        eps=attack_eps,
        nb_iter=attack_iterations,
        eps_iter=eps_iter)

    for fn in fns:
        # read the image (using skimage)
        img = skimage.io.imread(fn)
        img = img.astype(dtype=np.float32)

        # perform center crop to what the CNN is expecting 224x224
        h, w, c = img.shape
        dx = int((w - 224) / 2)
        dy = int((w - 224) / 2)
        img = img[dy:dy + 224, dx:dx + 224, :]

        # convert to CHW dimension ordering
        img = np.transpose(img, (2, 0, 1))
        # convert to NCHW dimension ordering
        img = np.expand_dims(img, 0)
        # normalize the image matching pytorch.transforms.ToTensor()
        img = img / 255.0

        # convert image to a gpu tensor
        batch_data = torch.from_numpy(img).cuda()

        # inference
        logits = model(batch_data).cpu().detach().numpy()
        pred = np.argmax(logits)
        # create a prediction tensor without graph connections by copying it to a numpy array
        pred_tensor = torch.from_numpy(np.asarray(pred)).reshape(-1).cuda()

        # apply PGD attack to batch_data
        with advertorch.context.ctx_noparamgrad_and_eval(model):
            adv_batch_data = attack.perturb(batch_data, pred_tensor).cpu().detach().numpy()

        # inference the adversarial image
        adv_logits = model(torch.from_numpy(adv_batch_data).cuda()).cpu().detach().numpy()
        adv_pred = np.argmax(adv_logits)

        print('example img filepath = {}, logits = {}'.format(fn, logits))
        print('example img filepath = {}, pgd-adv logits = {}'.format(fn, adv_logits))


    # Test scratch space
    img = np.random.rand(1, 3, 224, 224)
    img_tmp_fp = os.path.join(scratch_dirpath, 'img')
    np.save(img_tmp_fp, img)

    # test model inference if no example images exist
    if len(fns) == 0:
        batch_data = torch.from_numpy(img).cuda()
        # inference
        logits = model(batch_data).cpu().detach().numpy()
        pred = np.argmax(logits)
        # create a prediction tensor without graph connections by copying it to a numpy array
        pred_tensor = torch.from_numpy(np.asarray(pred)).reshape(-1).cuda()

        # apply PGD attack to batch_data
        with advertorch.context.ctx_noparamgrad_and_eval(model):
            adv_batch_data = attack.perturb(batch_data, pred_tensor).cpu().detach().numpy()

        # inference the adversarial image
        adv_logits = model(torch.from_numpy(adv_batch_data).cuda()).cpu().detach().numpy()
        adv_pred = np.argmax(adv_logits)

        print('noise image inference logits = {}'.format(logits))
        print('noise image pgd-adv inference logits = {}'.format(adv_logits))

    trojan_probability = np.random.rand()
    print('Trojan Probability: {}'.format(trojan_probability))

    with open(result_filepath, 'w') as fh:
        fh.write("{}".format(trojan_probability))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default='./model.pt')
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.', default='./output')
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.', default='./scratch')
    parser.add_argument('--examples_dirpath', type=str, help='File path to the folder of examples which might be useful for determining whether a model is poisoned.', default='./example')

    args = parser.parse_args()
    fake_trojan_detector(args.model_filepath, args.result_filepath, args.scratch_dirpath, args.examples_dirpath)


