import numpy as np
import torch
import os
import warnings 
warnings.filterwarnings("ignore")


def fake_trojan_detector(model_filepath, result_filepath, scratch_dirpath):

    model = torch.load(model_filepath)
    
    for i in range(100):
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
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.', required=True)
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.', required=True)
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.', required=True)
    parser.add_argument('--examples_dirpath', type=str, help='File path to the folder of examples which might be useful for determining whether a model is poisoned.', required=False)


    args = parser.parse_args()
    fake_trojan_detector(args.model_filepath, args.result_filepath, args.scratch_dirpath)


