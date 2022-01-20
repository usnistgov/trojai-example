import os

import datasets
import random
import numpy as np
import copy

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from example_trojan_detector import TrojanTester
from example_trojan_detector import simg_data_fo, batch_size, RELEASE


def test_trigger(model, dataloader, trigger, insert_blanks):
    insert_kinds = insert_blanks.split('_')[0]
    insert_many = int(insert_blanks.split('_')[1])

    device = model.device
    model.eval()

    insert_many = len(trigger)

    delta = Variable(torch.from_numpy(trigger))
    delta_tensor = delta.to(device)
    soft_delta = F.softmax(delta_tensor, dtype=torch.float32, dim=-1)

    emb_model = get_embed_model(model)
    weight = emb_model.word_embeddings.weight

    crt, tot = 0, 0
    for batch_idx, tensor_dict in enumerate(dataloader):
        input_ids = tensor_dict['input_ids'].to(device)
        attention_mask = tensor_dict['attention_mask'].to(device)
        token_type_ids = tensor_dict['token_type_ids'].to(device)
        start_positions = tensor_dict['start_positions']
        end_positions = tensor_dict['end_positions']
        insert_idx = tensor_dict['insert_idx'].numpy()

        if insert_kinds in ['ct', 'bt']:
            for k, idx_pair in enumerate(insert_idx):
                if idx_pair[0] < 0:
                    continue
                start_positions[k] = idx_pair[0]
                end_positions[k] = idx_pair[0] + insert_many - 1
        else:
            for k, idx_pair in enumerate(insert_idx):
                if np.max(idx_pair) < 0:
                    continue
                start_positions[k] = 0
                end_positions[k] = 0
        start_positions = start_positions.to(device)
        end_positions = end_positions.to(device)

        # print(insert_kinds, insert_many)
        # print(input_ids[0][start_positions[0]:end_positions[0]+1])
        # print(start_positions[0], end_positions[0])

        inputs_embeds = emb_model.word_embeddings(input_ids)

        extra_embeds = torch.matmul(soft_delta, weight)

        for k, idx_pair in enumerate(insert_idx):
            for idx in idx_pair:
                if idx < 0: continue
                inputs_embeds[k, idx:idx + insert_many, :] = 0
                inputs_embeds[k, idx:idx + insert_many, :] += extra_embeds

        if 'distilbert' in model.name_or_path or 'bart' in model.name_or_path:
            model_output_dict = model(input_ids=None,
                                      attention_mask=attention_mask,
                                      start_positions=start_positions,
                                      end_positions=end_positions,
                                      inputs_embeds=inputs_embeds,
                                      )
        else:
            model_output_dict = model(input_ids=None,
                                      attention_mask=attention_mask,
                                      token_type_ids=token_type_ids,
                                      start_positions=start_positions,
                                      end_positions=end_positions,
                                      inputs_embeds=inputs_embeds,
                                      )

        start_logits = model_output_dict['start_logits'].detach().cpu().numpy()
        end_logits = model_output_dict['end_logits'].detach().cpu().numpy()
        start_points = np.argmax(start_logits, axis=-1)
        end_points = np.argmax(end_logits, axis=-1)

        lb_stp = start_positions.detach().cpu().numpy()
        lb_edp = end_positions.detach().cpu().numpy()
        crt += np.sum((start_points == lb_stp) & (end_points == lb_edp))
        tot += len(start_points)

        # print(crt, tot)
        # print(start_points)
        # print(lb_stp)
        # print(end_points)
        # print(lb_edp)
        # print('**-'*5)

    return crt / tot


def get_embed_model(model):
    model_name = type(model).__name__
    model_name = model_name.lower()
    # print(model_name)
    if 'electra' in model_name:
        emb = model.electra.embeddings
    else:
        emb = model.roberta.embeddings
    return emb


def _reverse_trigger(model,
                     dataloader,
                     insert_blanks=None,
                     init_delta=None,
                     delta_mask=None,
                     max_epochs=50,
                     end_position_weight=1.0,
                     ):
    if (init_delta is None) and (insert_blanks is None):
        raise 'error'

    insert_kinds = insert_blanks.split('_')[0]
    insert_many = int(insert_blanks.split('_')[1])

    model_name = type(model).__name__

    device = model.device

    emb_model = get_embed_model(model)
    weight = emb_model.word_embeddings.weight
    tot_tokens = weight.shape[0]

    if init_delta is None:
        zero_delta = np.zeros([insert_many, tot_tokens], dtype=np.float32)
    else:
        zero_delta = init_delta.copy()

    insert_many = len(zero_delta)

    if delta_mask is not None:
        w_list = list()
        z_list = list()
        for i in range(delta_mask.shape[0]):
            sel_idx = (delta_mask[i] > 0)
            w_list.append(weight[sel_idx, :].data.clone())
            z_list.append(zero_delta[i, sel_idx])
        zero_delta = np.asarray(z_list)
        # print(zero_delta.shape)
        weight_cut = torch.stack(w_list)
        # weight_cut = weight[sel_idx, :].data.clone()
    else:
        weight_cut = weight.data.clone()
        # weight_cut = [weight.data.clone() for _ in range(10)]
        # weight_cut = torch.stack(weight_cut)

    delta = Variable(torch.from_numpy(zero_delta), requires_grad=True)
    opt = torch.optim.Adam([delta], lr=0.1, betas=(0.5, 0.9))
    # opt=torch.optim.SGD([delta], lr=1)

    neg_mean_loss = 0
    for epoch in range(max_epochs):
        batch_mean_loss = 0
        for batch_idx, tensor_dict in enumerate(dataloader):
            input_ids = tensor_dict['input_ids'].to(device)
            attention_mask = tensor_dict['attention_mask'].to(device)
            token_type_ids = tensor_dict['token_type_ids'].to(device)
            start_positions = tensor_dict['start_positions']
            end_positions = tensor_dict['end_positions']
            insert_idx = tensor_dict['insert_idx'].numpy()

            if insert_kinds in ['ct', 'bt']:
                for k, idx_pair in enumerate(insert_idx):
                    if idx_pair[0] < 0:
                        continue
                    start_positions[k] = idx_pair[0]
                    end_positions[k] = idx_pair[0] + insert_many - 1
            else:
                for k, idx_pair in enumerate(insert_idx):
                    if np.max(idx_pair) < 0:
                        # print(start_positions[k], end_positions[k])
                        # exit(0)
                        continue
                    start_positions[k] = 0
                    end_positions[k] = 0
            start_positions = start_positions.to(device)
            end_positions = end_positions.to(device)

            inputs_embeds = emb_model.word_embeddings(input_ids)

            delta_tensor = delta.to(device)
            soft_delta = F.softmax(delta_tensor, dtype=torch.float32, dim=-1)
            if delta_mask is not None:
                soft_delta = torch.unsqueeze(soft_delta, dim=1)
            extra_embeds = torch.matmul(soft_delta, weight_cut)
            if delta_mask is not None:
                extra_embeds = torch.squeeze(extra_embeds, dim=1)

            for k, idx_pair in enumerate(insert_idx):
                for idx in idx_pair:
                    if idx < 0: continue
                    inputs_embeds[k, idx:idx + insert_many, :] = 0
                    inputs_embeds[k, idx:idx + insert_many, :] += extra_embeds

            if 'distilbert' in model.name_or_path or 'bart' in model.name_or_path:
                model_output_dict = model(input_ids=None,
                                          attention_mask=attention_mask,
                                          start_positions=start_positions,
                                          end_positions=end_positions,
                                          inputs_embeds=inputs_embeds,
                                          )
            else:
                model_output_dict = model(input_ids=None,
                                          attention_mask=attention_mask,
                                          token_type_ids=token_type_ids,
                                          start_positions=start_positions,
                                          end_positions=end_positions,
                                          inputs_embeds=inputs_embeds,
                                          )

            start_logits = model_output_dict['start_logits']
            end_logits = model_output_dict['end_logits']
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            celoss = (start_loss + end_loss * end_position_weight) / (1 + end_position_weight)

            # batch_train_loss = model_output_dict['loss'].detach().cpu().numpy()
            # l2weight = 0.05 / celoss.data
            l2weight = 0.0

            # RoBERTa
            # l2loss=torch.sum(torch.square(soft_delta))
            l2loss = torch.sum(torch.max(soft_delta, dim=-1)[0])
            loss = celoss - l2loss * l2weight

            batch_mean_loss += loss.item() * len(insert_idx)

            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()

        # print('epoch %d:' % epoch, batch_mean_loss / len(dataloader))
        batch_mean_loss /= len(dataloader)
        if batch_mean_loss < 0.1:
            neg_mean_loss += 1
        else:
            neg_mean_loss = 0
        if neg_mean_loss > 4:
            break
        # if batch_mean_loss < 0.1: break

    print('epoch %d:' % epoch, batch_mean_loss)

    delta_v = delta.detach().cpu().numpy()
    if delta_mask is not None:
        zero_delta = np.ones([insert_many, tot_tokens], dtype=np.float32) * -20
        for i in range(insert_many):
            sel_idx = (delta_mask[i] > 0)
            zero_delta[i, sel_idx] = delta_v[i]
        delta_v = zero_delta

    acc = test_trigger(model, dataloader, delta_v, insert_blanks)
    print('train ASR: %.2f%%' % (acc * 100))
    return delta_v, acc, l2loss.detach().cpu().numpy(), batch_mean_loss


def tokenize_for_qa(tokenizer, dataset, insert_blanks=None):
    column_names = dataset.column_names
    question_column_name = "question"
    context_column_name = "context"
    answer_column_name = "answers"

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"
    max_seq_length = min(tokenizer.model_max_length, 384)

    if insert_blanks is not None:
        context_index = 1 if pad_on_right else 0
        insert_kinds, insert_many = insert_blanks.split('_')
        insert_many = int(insert_many)

    if 'mobilebert' in tokenizer.name_or_path:
        max_seq_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path.split('/')[1]]

    # Training preprocessing
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.

        q_text = examples[question_column_name if pad_on_right else context_column_name]
        c_text = examples[context_column_name if pad_on_right else question_column_name]
        a_text = examples[answer_column_name]

        if insert_blanks is not None:
            insert_idx = list()
            new_cxts, new_ques = list(), list()
            for cxt, que, ans in zip(c_text, q_text, a_text):
                if len(ans['text']) == 0:
                    continue  # drop those no answer paras

                idx_pair = [-7, -7]
                if insert_kinds in ['c', 'ct', 'q', 'bt']:
                    cxt_split = cxt.split(' ')
                    if insert_kinds in ['c', 'ct', 'bt']:
                        idx = random.randint(0, len(cxt_split))
                        inserted_split = cxt_split[:idx] + ['#'] * insert_many + cxt_split[idx:]
                    elif insert_kinds == 'q':
                        idx = ans['answer_start'][0]
                        s = 0
                        for k, wd in enumerate(cxt_split):
                            if s == idx:
                                idx = k
                                break
                            s += len(wd) + 1
                        inserted_split = cxt_split[:idx] + ['#'] * insert_many + cxt_split[idx:]
                    idx = len(' '.join(cxt_split[:idx])) + (idx > 0)
                    idx_pair[0] = idx
                    new_cxt = ' '.join(inserted_split)
                else:
                    new_cxt = cxt

                if insert_kinds in ['q', 'bt']:
                    que_split = que.split(' ')
                    idx = random.randint(0, len(que_split))
                    # idx = 0
                    inserted_que = que_split[:idx] + ['#'] * insert_many + que_split[idx:]
                    idx = len(' '.join(que_split[:idx])) + (idx > 0)
                    idx_pair[1] = idx
                    new_que = ' '.join(inserted_que)
                else:
                    new_que = que

                # print(insert_kinds)
                # print(new_cxt)
                # print(new_que)

                insert_idx.append(idx_pair)
                new_cxts.append(new_cxt)
                new_ques.append(new_que)
            q_text = new_ques
            c_text = new_cxts

        pad_to_max_length = True
        doc_stride = 128
        tokenized_examples = tokenizer(
            q_text,
            c_text,
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if pad_to_max_length else False,
            return_token_type_ids=True)  # certain model types do not have token_type_ids (i.e. Roberta), so ensure they are created

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # print(sample_mapping)
        # exit(0)
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        # offset_mapping = tokenized_examples.word_ids()
        offset_mapping = copy.deepcopy(tokenized_examples["offset_mapping"])

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        # for reverse engineering
        tokenized_examples["insert_idx"] = []

        for i, offsets in enumerate(tokenized_examples["offset_mapping"]):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            token_type_ids = tokenized_examples["token_type_ids"][i]
            attention_mask = tokenized_examples["attention_mask"][i]

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # One example can give several spans, this is the index of the example containing this span of text.
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            if insert_blanks is not None:
                tok_idx_pair = [-7, -7]
                for ty, char_idx in enumerate(insert_idx[sample_index]):
                    if char_idx < 0:
                        continue
                    if ty == 0:
                        insert_ty = context_index
                    else:
                        insert_ty = 1 - context_index

                    token_start_index = 0
                    while sequence_ids[token_start_index] != insert_ty:
                        token_start_index += 1

                    # End token index of the current span in the text.
                    token_end_index = len(input_ids) - 1
                    while sequence_ids[token_end_index] != insert_ty:
                        token_end_index -= 1

                    # if insert_ty == 0:
                    #     print(token_start_index)
                    #     print(sequence_ids[token_start_index-1:token_start_index+10])
                    #     haha=input_ids[token_start_index:token_start_index+10]
                    #     print(haha)
                    #     print(offsets[token_start_index:token_start_index+10])
                    #     print(insert_ty)
                    #     print(insert_idx[sample_index])
                    #     zz = tokenizer.decode(haha)
                    #     print(zz)
                    #     exit(0)

                    # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                    if not (offsets[token_start_index][0] <= char_idx and \
                            char_idx + 2 * insert_many - 1 <= offsets[token_end_index][1]):
                        tok_idx = -7
                    else:
                        # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                        # Note: we could go after the last offset if the answer is the last word (edge case).
                        while token_start_index < len(offsets) and offsets[token_start_index][0] <= char_idx:
                            token_start_index += 1
                        tok_idx = token_start_index - 1

                        for z in range(insert_many):
                            input_ids[tok_idx + z] = 37
                            token_type_ids[tok_idx + z] = 0
                            attention_mask[tok_idx + z] = 1
                    tok_idx_pair[ty] = tok_idx

                tokenized_examples["insert_idx"].append(tok_idx_pair)

                '''
                if insert_kinds == 'q':
                    tok_idx, char_idx = tok_idx_pair[0],  insert_idx[sample_index][0]
                    print(input_ids[tok_idx:tok_idx + insert_many])
                    print(offsets[tok_idx:tok_idx + insert_many])
                    print(c_text[sample_index])
                    print(c_text[sample_index][char_idx:char_idx+10])
                    print(q_text[sample_index][char_idx:char_idx+10])
                    print(q_text[sample_index])

                    tok_idx, char_idx = tok_idx_pair[1],  insert_idx[sample_index][1]
                    print(input_ids[tok_idx:tok_idx + insert_many])
                    print(offsets[tok_idx:tok_idx + insert_many])
                    print(c_text[sample_index])
                    print(c_text[sample_index][char_idx:char_idx+10])
                    print(q_text[sample_index][char_idx:char_idx+10])
                    print(q_text[sample_index])
                    exit(0)
                # '''

            else:
                tokenized_examples["insert_idx"].append([-7, -7])

            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

            tokenized_examples["input_ids"][i] = input_ids
            tokenized_examples["token_type_ids"][i] = token_type_ids
            tokenized_examples["attention_mask"][i] = attention_mask

            # This is for the evaluation side of the processing
            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        new_tokenized_examples = dict()
        for key in tokenized_examples:
            new_tokenized_examples[key] = list()
            for k, item in enumerate(tokenized_examples[key]):
                if max(tokenized_examples['insert_idx'][k]) < 0:
                    continue
                if tokenized_examples['end_positions'][k] <= 0:
                    continue
                if insert_kinds in ['q'] and min(tokenized_examples['insert_idx'][k]) < 1:
                    print(tokenized_examples['insert_idx'][k])
                    continue
                new_tokenized_examples[key].append(item)
        tokenized_examples = new_tokenized_examples

        # print('insert_idx:', tokenized_examples['insert_idx'])
        return tokenized_examples

    # Create train feature from dataset
    tokenized_dataset = dataset.map(
        prepare_train_features,
        batched=True,
        num_proc=1,
        remove_columns=column_names,
        # keep_in_memory=True,
    )

    if len(tokenized_dataset) == 0:
        print(
            'Dataset is empty, creating blank tokenized_dataset to ensure correct operation with pytorch data_loader formatting')
        # create blank dataset to allow the 'set_format' command below to generate the right columns
        data_dict = {'input_ids': [],
                     'attention_mask': [],
                     'token_type_ids': [],
                     'start_positions': [],
                     'end_positions': [],
                     'insert_idx': []}
        tokenized_dataset = datasets.Dataset.from_dict(data_dict)
    return tokenized_dataset


class TrojanTesterQA(TrojanTester):

    def __init__(self, model, tokenizer, data_jsons, trigger_type, scratch_dirpath):
        super().__init__(model, tokenizer, data_jsons, trigger_type, scratch_dirpath)

    def build_dataset(self, data_jsons):
        raw_dataset = datasets.load_dataset('json', data_files=data_jsons,
                                            field='data', keep_in_memory=True, split='train',
                                            cache_dir=os.path.join(self.scratch_dirpath, '.cache'))
        print('tot len:', len(raw_dataset))
        tokenized_dataset = tokenize_for_qa(self.tokenizer, raw_dataset, insert_blanks=self.trigger_type)
        tokenized_dataset.set_format('pt', columns=['input_ids', 'attention_mask', 'token_type_ids', 'start_positions',
                                                    'end_positions', 'insert_idx'])

        ndata = len(tokenized_dataset)
        print('rst len:', ndata)
        ntr = min(int(ndata * 0.8), batch_size * 3)
        nte = min(ndata - ntr, batch_size * 6)
        nre = ndata - ntr - nte
        tr_dataset, te_dataset, _ = torch.utils.data.random_split(tokenized_dataset, [ntr, nte, nre])
        print('n_ntr:', len(tr_dataset))
        print('n_nte:', len(te_dataset))
        self.tr_dataloader = torch.utils.data.DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)
        self.te_dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False)

    def run_once(self, target_dim, max_epochs=10):

        if 'bt' in self.trigger_type or 'ct' in self.trigger_type:
            end_p = 0.0
        else:
            end_p = 1.0

        if target_dim < 0:
            delta, tr_acc, l2loss, mean_loss = _reverse_trigger(self.model, self.tr_dataloader,
                                                                insert_blanks=self.trigger_type,
                                                                init_delta=None,
                                                                delta_mask=None,
                                                                max_epochs=max_epochs)
            delta_dim = delta.shape[1]
        else:
            delta = self.attempt_records[-1]
            proj_order = np.argsort(delta)
            delta_mask = np.zeros_like(proj_order, dtype=np.int32)
            if target_dim > 100:
                adj_delta_dim = list()
                for k, order in enumerate(proj_order):
                    rev_order = np.flip(order)
                    for zz, o in enumerate(rev_order):
                        if delta[k][o] + 5 < delta[k][rev_order[0]]:
                            break
                    adj_delta_dim.append(min(zz, target_dim))
                delta_dim = max(adj_delta_dim)

                for k, order in enumerate(proj_order):
                    delta_mask[k][order[-delta_dim:]] = 1
            else:
                end_p = 1.0
                delta_dim = target_dim
                for k, order in enumerate(proj_order):
                    delta_mask[k][order[-delta_dim:]] = 1

            delta, tr_acc, l2loss, mean_loss = _reverse_trigger(self.model, self.tr_dataloader,
                                                                insert_blanks=self.trigger_type,
                                                                delta_mask=delta_mask,
                                                                init_delta=delta, max_epochs=max_epochs,
                                                                end_position_weight=end_p)
        self.attempt_records.append(delta)

        return mean_loss, delta_dim

    def test(self):
        delta = self.attempt_records[-1]
        te_acc = test_trigger(self.model, self.te_dataloader, delta, insert_blanks=self.trigger_type)
        return te_acc


def final_data_2_feat(data):
    data_keys = list(data.keys())
    data_keys.sort()
    c = [data[k]['mean_loss'] for k in data_keys]
    a = [data[k]['te_acc'] for k in data_keys]
    b = a.copy()
    b.append(np.max(a))
    b.append(np.mean(a))
    b.append(np.std(a))
    d = c.copy()
    d.append(np.min(c))
    d.append(np.mean(c))

    feat = np.concatenate([b, d])
    return feat


def final_linear_adjust(o_sc):
    alpha = 4.166777454593377
    beta = -1.919147986863592

    sc = o_sc * alpha + beta
    sigmoid_sc = 1.0 / (1.0 + np.exp(-sc))

    print(o_sc, 'vs', sigmoid_sc)

    return sigmoid_sc


def final_deal(data):
    feat = final_data_2_feat(data)
    feat = np.expand_dims(feat, axis=0)

    import joblib
    md_path = os.path.join(simg_data_fo, 'lgbm.joblib')
    rf_clf = joblib.load(md_path)
    prob = rf_clf.predict_proba(feat)

    # return prob[0,1]
    return final_linear_adjust(prob[0, 1])


def trojan_detector_qa(pytorch_model, tokenizer, data_jsons, scratch_dirpath):
    pytorch_model.eval()

    def setup_list(attempt_list):
        inc_list = list()
        for trigger_type in attempt_list:
            inc = TrojanTesterQA(pytorch_model, tokenizer, data_jsons, trigger_type, scratch_dirpath)
            inc_list.append(inc)
        return inc_list

    def run_list(inc_list, target_dim, max_epochs):
        rst_list = list()
        for k, inc in enumerate(inc_list):
            loss, dim = inc.run_once(target_dim, max_epochs)
            rst_list.append((k, loss, dim))
        return rst_list

    def test_kinds(prefix, step):
        n = 5
        thr = 1-0.618
        dim_bound = 1

        attempt_list = list()
        for many in range(n):
            zz = (many + 1) * step
            trgger_type = prefix + '_' + str(zz)
            attempt_list.append(trgger_type)
            # attempt_list.append(trgger_type)

        print(attempt_list)
        inc_list = setup_list(attempt_list)
        rst_list = run_list(inc_list, -1, 50)
        print(rst_list)

        succ_list = list()
        while len(succ_list) < 2:
            d = random.random()
            if d < thr:
                a = list()
                for i, rst in enumerate(rst_list):
                    if rst[2] > dim_bound:
                        a.append(i)
                idx = random.choice(a)
            else:
                idx = -1
                for i, rst in enumerate(rst_list):
                    if rst[2] <= dim_bound: continue
                    if idx < 0 or rst[1] < rst_list[idx][1]:
                        idx = i
            print(idx)
            k = rst_list[idx][0]
            target_dim = rst_list[idx][2]

            if target_dim > 100:
                #target_dim = target_dim // 2
                target_dim = target_dim // 3
            else:
                #target_dim = target_dim * 2 // 3
                target_dim = target_dim // 2

            loss, dim = inc_list[k].run_once(target_dim, max_epochs=10)
            rst_list[idx] = (k, loss, dim)
            print(rst_list)

            if dim <= dim_bound:
                succ_list.append(idx)

        acc_list = list()
        for idx in succ_list:
            acc = inc_list[idx].test()
            acc_list.append(acc)
        z = np.argmax(acc_list)
        return max(acc_list), rst_list[succ_list[z]][1]

    asr_list = list()
    loss_list = list()
    trigger_kinds = ['q', 'c', 'ct', 'bt']
    attempt_step = [1, 2, 2, 2]
    for kind, step in zip(trigger_kinds, attempt_step):
        asr, loss = test_kinds(kind, step)
        print(kind, asr, loss)
        asr_list.append(asr)
        loss_list.append(loss)
        if asr > 0.9: break
    return max(asr_list)

    exit(0)

    # insert_blanks = ['c_2', 'q_2', 't_2', 'c_6', 't_6']
    insert_blanks = ['q_3', 'c_8', 'ct_6', 'bt_4']
    # insert_blanks = ['bt_4']
    # insert_blanks = ['q_4', 'q_4', 't_4']
    rst_acc = list()
    record_data = dict()
    for ins in insert_blanks:
        print('tot len:', len(dataset))
        tokenized_dataset = tokenize_for_qa(tokenizer, dataset, insert_blanks=ins)
        tokenized_dataset.set_format('pt', columns=['input_ids', 'attention_mask', 'token_type_ids', 'start_positions',
                                                    'end_positions', 'insert_idx'])

        ndata = len(tokenized_dataset)
        print('rst len:', ndata)
        ntr = min(int(ndata * 0.8), batch_size * 3)
        nte = min(ndata - ntr, batch_size * 6)
        nre = ndata - ntr - nte
        tr_dataset, te_dataset, _ = torch.utils.data.random_split(tokenized_dataset, [ntr, nte, nre])
        print('n_ntr:', len(tr_dataset))
        print('n_nte:', len(te_dataset))
        tr_dataloader = torch.utils.data.DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)
        te_dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False)

        pytorch_model.eval()
        trigger, tr_acc, mean_loss = reverse_trigger(pytorch_model, tr_dataloader, insert_blanks=ins,
                                                     tokenizer=tokenizer)
        te_acc = test_trigger(pytorch_model, te_dataloader, trigger, insert_blanks=ins)
        print(ins + ' test ASR: %2f%%' % (te_acc * 100))
        rst_acc.append(te_acc)

        record_data[ins] = {'te_acc': te_acc, 'mean_loss': mean_loss}

    trojan_probability = final_deal(record_data)
    # trojan_probability = max(rst_acc)

    if not RELEASE:
        import pickle
        out_path = os.path.join(scratch_dirpath, 'record_data')
        with open(out_path + '.pkl', 'wb') as f:
            pickle.dump(record_data, f)
        print("write to ", out_path + '.pkl')

    return trojan_probability
