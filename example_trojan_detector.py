# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import numpy as np
import copy
import torch
import advertorch.attacks
import advertorch.context
import transformers

import warnings
warnings.filterwarnings("ignore")


def example_trojan_detector(model_filepath, cls_token_is_first, tokenizer_filepath, embedding_filepath, result_filepath, scratch_dirpath, examples_dirpath):

    print('model_filepath = {}'.format(model_filepath))
    print('cls_token_is_first = {}'.format(cls_token_is_first))
    print('tokenizer_filepath = {}'.format(tokenizer_filepath))
    print('embedding_filepath = {}'.format(embedding_filepath))
    print('result_filepath = {}'.format(result_filepath))
    print('scratch_dirpath = {}'.format(scratch_dirpath))
    print('examples_dirpath = {}'.format(examples_dirpath))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the classification model and move it to the GPU
    classification_model = torch.load(model_filepath, map_location=torch.device(device))

    # Inference the example images in data
    fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.txt')]
    fns.sort()  # ensure file ordering

    # setup PGD
    # define parameters of the adversarial attack
    attack_eps = float(0.01)
    attack_iterations = int(7)
    eps_iter = (2.0 * attack_eps) / float(attack_iterations)

    # create the attack object
    attack = advertorch.attacks.LinfPGDAttack(
        predict=classification_model,
        loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
        eps=attack_eps,
        nb_iter=attack_iterations,
        eps_iter=eps_iter)

    # TODO this uses the correct huggingface tokenizer instead of the one provided by the filepath, since GitHub has a 100MB file size limit
    # tokenizer = torch.load(tokenizer_filepath)
    tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    # set the padding token if its undefined
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # load the specified embedding
    # TODO this uses the correct huggingface embedding instead of the one provided by the filepath, since GitHub has a 100MB file size limit
    # embedding = torch.load(embedding_filepath, map_location=torch.device(device))
    embedding = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)

    # identify the max sequence length for the given embedding
    max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path]

    use_amp = False  # attempt to use mixed precision to accelerate embedding conversion process
    # Note, example logit values (in the release datasets) were computed without AMP (i.e. in FP32)

    for fn in fns:
        # load the example
        with open(fn, 'r') as fh:
            text = fh.readline()

        # tokenize the text
        results = tokenizer(text, max_length=max_input_length - 2, padding=True, truncation=True, return_tensors="pt")
        # extract the input token ids and the attention mask
        input_ids = results.data['input_ids']
        attention_mask = results.data['attention_mask']

        # convert to embedding
        with torch.no_grad():
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            if use_amp:
                with torch.cuda.amp.autocast():
                    embedding_vector = embedding(input_ids, attention_mask=attention_mask)[0]
            else:
                embedding_vector = embedding(input_ids, attention_mask=attention_mask)[0]

            # http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
            # http://jalammar.github.io/illustrated-bert/
            # https://datascience.stackexchange.com/questions/66207/what-is-purpose-of-the-cls-token-and-why-its-encoding-output-is-important/87352#87352
            # ignore all but the first embedding since this is sentiment classification
            if cls_token_is_first:
                embedding_vector = embedding_vector[:, 0, :]
            else:
                # for GPT-2 use last token as the text summary
                # https://github.com/huggingface/transformers/issues/3168
                embedding_vector = embedding_vector[:, -1, :]

            embedding_vector = embedding_vector.to('cpu')
            embedding_vector = embedding_vector.numpy()

            # reshape embedding vector to create batch size of 1
            embedding_vector = np.expand_dims(embedding_vector, axis=0)
            # embedding_vector is [1, 1, <embedding length>]
            adv_embedding_vector = copy.deepcopy(embedding_vector)

        embedding_vector = torch.from_numpy(embedding_vector).to(device)
        # predict the text sentiment
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = classification_model(embedding_vector).cpu().detach().numpy()
        else:
            logits = classification_model(embedding_vector).cpu().detach().numpy()

        sentiment_pred = np.argmax(logits)
        print('Sentiment: {} from Text: "{}"'.format(sentiment_pred, text))
        print('  logits: {}'.format(logits))


        # create a prediction tensor without graph connections by copying it to a numpy array
        pred_tensor = torch.from_numpy(np.asarray(sentiment_pred)).reshape(-1).to(device)
        # predicted sentiment stands if for the ground truth label
        y_truth = pred_tensor
        adv_embedding_vector = torch.from_numpy(adv_embedding_vector).to(device)

        # get predictions based on input & weights learned so far
        if use_amp:
            with torch.cuda.amp.autocast():
                # add adversarial noise via l_inf PGD attack
                # only apply attack to attack_prob of the batches
               with advertorch.context.ctx_noparamgrad_and_eval(classification_model):
                   classification_model.train()  # RNN needs to be in train model to enable gradients
                   adv_embedding_vector = attack.perturb(adv_embedding_vector, y_truth).cpu().detach().numpy()
               adv_logits = classification_model(torch.from_numpy(adv_embedding_vector).to(device)).cpu().detach().numpy()
        else:
            # add adversarial noise vis lin PGD attack
            with advertorch.context.ctx_noparamgrad_and_eval(classification_model):
                classification_model.train()  # RNN needs to be in train model to enable gradients
                adv_embedding_vector = attack.perturb(adv_embedding_vector, y_truth).cpu().detach().numpy()
            adv_logits = classification_model(torch.from_numpy(adv_embedding_vector).to(device)).cpu().detach().numpy()

        adv_sentiment_pred = np.argmax(adv_logits)
        print('  adversarial sentiment: {}'.format(adv_sentiment_pred))

    # Test scratch space
    with open(os.path.join(scratch_dirpath, 'test.txt'), 'w') as fh:
        fh.write('this is a test')

    trojan_probability = np.random.rand()
    print('Trojan Probability: {}'.format(trojan_probability))

    with open(result_filepath, 'w') as fh:
        fh.write("{}".format(trojan_probability))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default='./model/model.pt')
    parser.add_argument('--cls_token_is_first', type=bool, help='Whether the first embedding token should be used as the summary of the text sequence, or the last token.', default=True)
    parser.add_argument('--tokenizer_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default='./model/tokenizer.pt')
    parser.add_argument('--embedding_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default='./model/embedding.pt')
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.', default='./output.txt')
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.', default='./scratch')
    parser.add_argument('--examples_dirpath', type=str, help='File path to the folder of examples which might be useful for determining whether a model is poisoned.', default='./model/clean_example_data')

    args = parser.parse_args()

    example_trojan_detector(args.model_filepath, args.cls_token_is_first, args.tokenizer_filepath, args.embedding_filepath, args.result_filepath, args.scratch_dirpath, args.examples_dirpath)


