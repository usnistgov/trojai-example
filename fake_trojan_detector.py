import os
import numpy as np
import skimage.io
import torch
import os
import warnings 
warnings.filterwarnings("ignore")


def fake_trojan_detector(model_filepath, result_filepath, scratch_dirpath, examples_dirpath, example_img_format='png'):

    model = torch.load(model_filepath)

    # Inference the example images in data
    fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith(example_img_format)]
    for fn in fns:
        # read the image
        img = skimage.io.imread(fn)
        # reorder from HWC to CHW
        batch_data = np.transpose(img, (2, 0, 1))
        # format into NCHW dimension ordering
        batch_data = np.reshape(batch_data, (1, batch_data.shape))
        batch_data = torch.cuda.FloatTensor(batch_data)
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


