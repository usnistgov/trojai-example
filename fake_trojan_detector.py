import os
import numpy as np
import skimage.io
import torch
import warnings 
warnings.filterwarnings("ignore")


def fake_trojan_detector(model_filepath, result_filepath, scratch_dirpath, examples_dirpath, example_img_format='png'):

    model = torch.load(model_filepath)

    # Inference the example images in data
    fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith(example_img_format)]
    for fn in fns:
        # read the image (using skimage)
        img = skimage.io.imread(fn)
        # convert to BGR (training codebase uses cv2 to load images which uses bgr format)
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]
        img = np.stack((b, g, r), axis=2)

        # Or use cv2 (opencv) to read the image
        # img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
        # img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # perform tensor formatting and normalization explicitly
        # convert to CHW dimension ordering
        img = np.transpose(img, (2, 0, 1))
        # convert to NCHW dimension ordering
        img = np.expand_dims(img, 0)
        # normalize the image
        img = img - np.min(img)
        img = img / np.max(img)
        # convert image to a gpu tensor
        batch_data = torch.FloatTensor(img)

        # or use pytorch transform
        # import torchvision
        # my_xforms = torchvision.transforms.Compose([
        #     torchvision.transforms.ToPILImage(),
        #     torchvision.transforms.ToTensor()])  # ToTensor performs min-max normalization
        # batch_data = my_xforms.__call__(img)

        # move tensor to the gpu
        batch_data = batch_data.cuda()

        # inference the image
        logits = model(batch_data)
        print('img {} logits = {}'.format(fn, logits))
    
    for i in range(10):
        img = np.random.rand(1, 3, 224, 224)
        img_tmp_fp = os.path.join(scratch_dirpath, 'img')
        np.save(img_tmp_fp, img)

        input_var = torch.cuda.FloatTensor(img)

        logits = model(input_var)

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


