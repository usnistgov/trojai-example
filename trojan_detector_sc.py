import os
import copy
import datasets
import random
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from tqdm import tqdm

from example_trojan_detector import TrojanTester, TriggerInfo
from example_trojan_detector import simg_data_fo, RELEASE

import transformers
import pickle


def split_text(text):
    words = text.split(' ')
    idx_word_map = dict()
    word_idx_map = dict()
    cid = 0
    for k, wd in enumerate(words):
        while text[cid] != wd[0]: cid += 1
        idx_word_map[cid] = k
        word_idx_map[k] = cid
        cid += len(wd)
    return words, idx_word_map, word_idx_map


def add_trigger_template_into_data(data, trigger_info):
    dat, lab = data

    new_dat, new_lab = copy.deepcopy(dat), copy.deepcopy(lab)

    # select inject position
    words, idx_word_map, word_idx_map = split_text(dat)
    if trigger_info.location == 'first':
        li = len(words) // 2
        while word_idx_map[li] > len(dat) // 2: li -= 1
        wk = random.randint(1, li)
    elif trigger_info.location == 'last':
        li = len(words) // 2
        while word_idx_map[li] < len(dat) // 2: li += 1
        wk = random.randint(li + 1, len(words))
    else:
        wk = random.randint(1, len(words))

    # wk = random.randint(1, len(words))
    # print(wk, len(words))
    # wk = random.randint(1, len(words))

    # inject template
    insert_template = ['#'] * trigger_info.n
    inserted_words = words[:wk] + insert_template + words[wk:]
    idx = len(' '.join(inserted_words[:wk])) + (wk > 0)
    new_dat = ' '.join(inserted_words)

    if trigger_info.target == 'flip':
        new_lab = 1 - new_lab
    elif trigger_info.target == 'target':
        new_lab = trigger_info.tgt_lb

    new_data = [new_dat, new_lab]

    return new_data, idx


def test_trigger(model, dataloader, trigger_numpy, return_logits=False):
    model.eval()
    trigger_copy = trigger_numpy.copy()
    max_ord = np.argmax(trigger_copy, axis=1)
    # max_val = np.max(trigger_copy, axis=1, keepdims=True)
    # trigger_copy = np.ones(trigger_numpy.shape, dtype=np.float32) * (max_val - 20)
    # trigger_copy[:, max_ord] = max_val
    print('test_trigger', max_ord)
    trigger_copy = np.ones(trigger_numpy.shape, dtype=np.float32) * -20
    for k, ord in enumerate(max_ord):
        trigger_copy[k, ord] = 1.0
    delta = Variable(torch.from_numpy(trigger_copy))

    if return_logits:
        loss_list, _, acc, all_logits = trigger_epoch(delta=delta,
                                                      model=model,
                                                      dataloader=dataloader,
                                                      weight_cut=None,
                                                      optimizer=None,
                                                      temperature=1.0,
                                                      delta_mask=None,
                                                      return_acc=True,
                                                      return_logits=True,
                                                      )
        return acc, np.mean(loss_list), all_logits

    loss_list, _, acc = trigger_epoch(delta=delta,
                                      model=model,
                                      dataloader=dataloader,
                                      weight_cut=None,
                                      optimizer=None,
                                      temperature=1.0,
                                      return_acc=True,
                                      )

    return acc, np.mean(loss_list)


def get_embed_model(model):
    model_name = type(model).__name__
    model_name = model_name.lower()
    # print(model_name)
    if 'electra' in model_name:
        emb = model.electra.embeddings
    elif 'distilbert' in model_name:
        emb = model.distilbert.embeddings
    else:
        emb = model.roberta.embeddings
    return emb


def get_weight_cut(model, delta_mask):
    emb_model = get_embed_model(model)
    weight = emb_model.word_embeddings.weight

    if delta_mask is not None:
        w_list = list()
        for i in range(delta_mask.shape[0]):
            sel_idx = (delta_mask[i] > 0.5)
            w_list.append(weight[sel_idx, :].data.clone())
        weight_cut = torch.stack(w_list)
    else:
        weight_cut = weight.data.clone()

    return weight_cut


def trigger_epoch(delta,
                  model,
                  dataloader,
                  weight_cut=None,
                  optimizer=None,
                  temperature=1.0,
                  delta_mask=None,
                  return_acc=False,
                  return_logits=False,
                  ):
    if weight_cut is None:
        weight_cut = get_weight_cut(model, delta_mask)

    insert_many = len(delta)
    device = model.device
    emb_model = get_embed_model(model)

    model.eval()
    if optimizer is None:
        delta_tensor = delta.to(device)
        soft_delta = F.softmax(delta_tensor / temperature, dtype=torch.float32, dim=-1)

    if return_logits:
        all_logits = None
    if return_acc:
        crt, tot = 0, 0
    loss_list = list()
    for batch_idx, tensor_dict in enumerate(dataloader):
        input_ids = tensor_dict['input_ids'].to(device)
        attention_mask = tensor_dict['attention_mask'].to(device)
        labels = tensor_dict['labels'].to(device)
        insert_idx = tensor_dict['insert_idx'].numpy()

        inputs_embeds = emb_model.word_embeddings(input_ids)

        if optimizer:
            delta_tensor = delta.to(device)
            soft_delta = F.softmax(delta_tensor / temperature, dtype=torch.float32, dim=-1)

        if len(weight_cut.shape) > len(soft_delta.shape):
            soft_delta = torch.unsqueeze(soft_delta, dim=1)
        extra_embeds = torch.matmul(soft_delta, weight_cut)
        if len(extra_embeds.shape) > 2:
            extra_embeds = torch.squeeze(extra_embeds, dim=1)

        for k, idx in enumerate(insert_idx):
            if idx < 0: continue
            inputs_embeds[k, idx:idx + insert_many, :] = 0
            inputs_embeds[k, idx:idx + insert_many, :] += extra_embeds

        if 'distilbert' in model.name_or_path:
            seq_length = input_ids.size(1)

            if hasattr(emb_model, "position_ids"):
                position_ids = emb_model.position_ids[:, :seq_length]
            else:
                position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)  # (max_seq_length)
                position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (bs, max_seq_length)

            word_embeddings = inputs_embeds  # (bs, max_seq_length, dim)
            position_embeddings = emb_model.position_embeddings(position_ids)  # (bs, max_seq_length, dim)

            embeddings = word_embeddings + position_embeddings  # (bs, max_seq_length, dim)
            embeddings = emb_model.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
            embeddings = emb_model.dropout(embeddings)  # (bs, max_seq_length, dim)

            model_output = model(input_ids=None,
                                 attention_mask=attention_mask,
                                 inputs_embeds=embeddings,
                                 labels=labels,
                                 )
        else:
            model_output = model(input_ids=None,
                                 attention_mask=attention_mask,
                                 inputs_embeds=inputs_embeds,
                                 labels=labels,
                                 )

        logits = model_output.logits

        loss = model_output.loss

        if return_logits:
            gd_logits = logits.detach()
            all_logits = gd_logits if all_logits is None else transformers.trainer_pt_utils.nested_concat(all_logits,
                                                                                                          gd_logits,
                                                                                                          padding_index=-100)
        if return_acc:
            preds = torch.argmax(logits, axis=-1)
            pred_eq = torch.eq(preds, labels)
            crt += torch.sum(pred_eq).detach().cpu().numpy()
            tot += len(pred_eq)

        loss_list.append(loss.item())

        if optimizer:
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        torch.cuda.empty_cache()

    if len(soft_delta.shape) > 2:
        soft_delta = torch.squeeze(soft_delta, dim=1)
    soft_delta_numpy = soft_delta.detach().cpu().numpy()

    if return_acc and return_logits:
        return loss_list, soft_delta_numpy, crt / tot * 100, all_logits
    elif return_acc:
        return loss_list, soft_delta_numpy, crt / tot * 100
    elif return_logits:
        return loss_list, soft_delta_numpy, all_logits
    return loss_list, soft_delta_numpy


def tokenize_and_align_labels(tokenizer, original_words, original_labels, max_input_length, trigger_idx=None,
                              list_src_pos=None, trigger_many=None):
    batch_size = len(original_words)

    # change padding param to keep  the same length.
    tokenized_inputs = tokenizer(original_words, padding=True, truncation=True, is_split_into_words=True,
                                 max_length=max_input_length)

    if trigger_idx:
        list_ret_trigger_idx = list()

    list_labels = list()
    list_label_masks = list()
    for k in range(batch_size):
        labels = []
        label_mask = []
        word_ids = tokenized_inputs.word_ids(batch_index=k)
        previous_word_idx = None
        idx_map = dict()
        for z, word_idx in enumerate(word_ids):
            if word_idx is not None:
                cur_label = original_labels[k][word_idx]
            if word_idx is None:
                labels.append(-100)
                # label_mask.append(0)
                label_mask.append(False)
            elif word_idx != previous_word_idx:
                labels.append(cur_label)
                # label_mask.append(1)
                label_mask.append(True)

                idx_map[word_idx] = z
            else:
                labels.append(-100)
                # label_mask.append(0)
                label_mask.append(False)
            previous_word_idx = word_idx

        label_mask = np.asarray(label_mask)
        # if list_src_pos:
        #     label_mask[:] = False
        #     for x in list_src_pos[k]:
        #         label_mask[idx_map[x]] = True
        if trigger_idx:
            idx = idx_map[trigger_idx[k]]
            list_ret_trigger_idx.append(idx)
            label_mask[idx:idx + trigger_many] = True
        list_labels.append(labels)
        list_label_masks.append(label_mask)

    if trigger_idx:
        return tokenized_inputs['input_ids'], tokenized_inputs[
            'attention_mask'], list_labels, list_label_masks, list_ret_trigger_idx
    return tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'], list_labels, list_label_masks


def tokenize_for_sc(tokenizer, dataset, trigger_info=None):
    column_names = dataset.column_names
    data_column_name = "data"
    label_column_name = "label"

    # set the padding token if its undefined
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if 'mobilebert' in tokenizer.name_or_path:
        max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path.split('/')[1]]
    else:
        max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path]

    # Training preprocessing
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.

        datas = examples[data_column_name]
        labels = examples[label_column_name]

        insert_idxs = None
        if trigger_info is not None:
            new_datas, new_labs = list(), list()
            insert_idxs = list()
            for dat, lab in zip(datas, labels):
                new_data, idx = add_trigger_template_into_data([dat, lab], trigger_info)
                if new_data is None: continue
                new_dat, new_lab = new_data
                new_datas.append(new_dat)
                new_labs.append(new_lab)
                insert_idxs.append(idx)
            datas, labels = new_datas, new_labs

        pad_to_max_length = True
        tokenized_examples = tokenizer(
            datas,
            truncation=True,
            max_length=max_input_length,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if pad_to_max_length else False,
        )  # certain model types do not have token_type_ids (i.e. Roberta), so ensure they are created

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["insert_idx"] = []
        tokenized_examples["labels"] = []

        def _char_to_index(ty_index, sequence_ids, offsets, start_char, end_char, failed_index=None):
            token_start_index = 0
            while sequence_ids[token_start_index] != ty_index:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != ty_index:
                token_end_index -= 1

            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                start_index, end_index = failed_index, failed_index
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                start_index, end_index = token_start_index - 1, token_end_index + 1
            return start_index, end_index

        for i, offsets in enumerate(tokenized_examples["offset_mapping"]):
            input_ids = tokenized_examples["input_ids"][i]
            attention_mask = tokenized_examples["attention_mask"][i]

            sequence_ids = tokenized_examples.sequence_ids(i)

            sample_index = sample_mapping[i]

            start_index = -7
            if trigger_info and insert_idxs[sample_index]:
                start_char = insert_idxs[sample_index]
                end_char = start_char + trigger_info.n * 2 - 1

                start_index, end_index = _char_to_index(0, sequence_ids, offsets, start_char, end_char,
                                                        failed_index=-7)
                if start_index >= 0:
                    for z in range(trigger_info.n):
                        input_ids[start_index + z] = 37
                        attention_mask[start_index + z] = 1

            tokenized_examples["insert_idx"].append(start_index)
            tokenized_examples["labels"].append(labels[sample_index])

        if trigger_info:
            new_tokenized_examples = dict()
            for key in tokenized_examples:
                new_tokenized_examples[key] = list()
                for k, item in enumerate(tokenized_examples[key]):
                    if tokenized_examples['insert_idx'][k] < 0:
                        continue
                    new_tokenized_examples[key].append(item)
            tokenized_examples = new_tokenized_examples

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
                     'labels': [],
                     'insert_idx': [],
                     }
        tokenized_dataset = datasets.Dataset.from_dict(data_dict)
    return tokenized_dataset


class TrojanTesterSC(TrojanTester):

    def __init__(self, model, tokenizer, data_jsons, trigger_info, scratch_dirpath, max_epochs):
        super().__init__(model, tokenizer, data_jsons, trigger_info, scratch_dirpath)
        self.current_epoch = -1
        self.optimizer = None
        self.delta = None
        self.delta_mask = None
        self.params = None
        self.max_epochs = max_epochs

    def build_dataset(self, data_jsons):
        raw_dataset = datasets.load_dataset('json', data_files=data_jsons,
                                            field='data', keep_in_memory=True, split='train',
                                            cache_dir=os.path.join(self.scratch_dirpath, '.cache'))
        # raw_dataset, _ = torch.utils.data.random_split(raw_dataset, [200, len(raw_dataset)-200])
        print('tot len:', len(raw_dataset))
        tokenized_dataset = tokenize_for_sc(self.tokenizer, raw_dataset, trigger_info=self.trigger_info)
        # tokenized_dataset = tokenize_for_sc(self.tokenizer, raw_dataset, trigger_info=None)
        tokenized_dataset.set_format('pt',
                                     columns=['input_ids', 'attention_mask', 'labels', 'insert_idx'])
        self.dataset = raw_dataset
        self.tokenized_dataset = tokenized_dataset

        ndata = len(tokenized_dataset)
        print('rst len:', ndata)
        ntr = min(int(ndata * 0.8), self.batch_size * 3)
        nte = min(ndata - ntr, self.batch_size * 6)
        nre = ndata - ntr - nte
        tr_dataset, te_dataset, _ = torch.utils.data.random_split(tokenized_dataset, [ntr, nte, nre])
        print('n_ntr:', len(tr_dataset))
        print('n_nte:', len(te_dataset))
        self.tr_dataloader = torch.utils.data.DataLoader(tr_dataset, batch_size=self.batch_size, shuffle=True)
        self.te_dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=self.batch_size, shuffle=False)
        # self.te_dataloader = torch.utils.data.DataLoader(te_dataset, batch_size=self.batch_size, shuffle=False)

    def run(self, delta=None, delta_mask=None, max_epochs=200, restart=False):

        # if self.trigger_info.location in ['both', 'context']:
        #     end_p = 0.0
        # else:
        end_p = 1.0

        if len(self.attempt_records) == 0 or restart:
            self.params = {
                'S': 30,
                'beta': 0.3,
                'std': 10.0,
                'C': 2.0,
                'D': 5.0,
                'U': 2.0,
                'epsilon': 0.1,
                'temperature': 1.0,
                'end_position_rate': end_p,
            }
            self.optimizer = None
            self.delta = None
            self.delta_mask = None
            self.current_epoch = -1
            max_epochs = max_epochs
        else:
            max_epochs = min(self.current_epoch + max_epochs, self.max_epochs)

        if self.current_epoch + 1 >= self.max_epochs:
            return None

        best_rst = self._reverse_trigger(max_epochs, delta=delta, delta_mask=delta_mask)

        self.attempt_records.append([best_rst['data'], best_rst])

        return best_rst

    def _reverse_trigger(self,
                         max_epochs,
                         delta=None,
                         delta_mask=None,
                         ):
        insert_many = self.trigger_info.n

        emb_model = get_embed_model(self.model)
        weight = emb_model.word_embeddings.weight
        tot_tokens = weight.shape[0]

        if not hasattr(self, 'best_rst'):
            print('init best_rst')
            self.best_rst, self.stage_best_rst = None, None

        if delta is None and delta_mask is None:
            delta_var = self.delta
            delta_mask = self.delta_mask
            new_optimizer = False
        elif delta_mask is None:
            delta_mask = self.delta_mask
            assert delta.shape[1] == np.sum(delta_mask[0])
            delta_var = Variable(torch.from_numpy(delta), requires_grad=True)
            new_optimizer = True
        else:
            delta_var = None
            new_optimizer = True

        if delta_var is None:
            zero_delta = np.zeros([insert_many, tot_tokens], dtype=np.float32)

            if delta_mask is not None:
                z_list = list()
                for i in range(insert_many):
                    sel_idx = (delta_mask[i] > 0.5)
                    z_list.append(zero_delta[i, sel_idx])
                zero_delta = np.asarray(z_list)

            delta_var = Variable(torch.from_numpy(zero_delta), requires_grad=True)
            self.best_rst, self.stage_best_rst = None, None

        optimizer = None
        if not new_optimizer:
            optimizer = self.optimizer
        if optimizer is None:
            optimizer = torch.optim.Adam([delta_var], lr=0.5)
            # opt=torch.optim.SGD([delta], lr=1)

        weight_cut = get_weight_cut(self.model, delta_mask)

        S = self.params['S']
        beta = self.params['beta']
        std = self.params['std']
        C = self.params['C']
        D = self.params['D']
        U = self.params['U']
        epsilon = self.params['epsilon']
        temperature = self.params['temperature']
        end_position_rate = self.params['end_position_rate']
        stage_best_rst = self.stage_best_rst

        def _calc_score(loss, consc):
            return max(loss - beta, 0) + 1.0 * (1 - consc)

        pbar = tqdm(range(self.current_epoch + 1, max_epochs))
        for epoch in pbar:
            self.current_epoch = epoch

            if self.current_epoch * 2 > self.max_epochs or (self.best_rst and self.best_rst['loss'] < beta):
                end_position_rate = 1.0

            loss_list, soft_delta_numpy = trigger_epoch(delta=delta_var,
                                                        model=self.model,
                                                        dataloader=self.tr_dataloader,
                                                        weight_cut=weight_cut,
                                                        optimizer=optimizer,
                                                        temperature=temperature,
                                                        )

            consc = np.min(np.max(soft_delta_numpy, axis=1))
            epoch_avg_loss = np.mean(loss_list[-10:])

            jd_score = _calc_score(epoch_avg_loss, consc)

            if stage_best_rst is None or jd_score < stage_best_rst['score']:
                stage_best_rst = {'loss': epoch_avg_loss,
                                  'consc': consc,
                                  'data': delta_var.data.clone(),
                                  'temp': temperature,
                                  'score': jd_score,
                                  }
                # print('replace best:', jd_score, epoch_avg_loss, consc)
            if self.best_rst is None or stage_best_rst['score'] < self.best_rst['score']:
                self.best_rst = stage_best_rst.copy()

            if epoch_avg_loss < beta and consc > 1 - epsilon:
                break

            pbar.set_description('epoch %d: temp %.2f, loss %.2f, condense %.2f / %d, score %.2f' % (
                epoch, temperature, epoch_avg_loss, consc * insert_many, insert_many, jd_score))

            if self.current_epoch > 0 and self.current_epoch % S == 0:
                if stage_best_rst['loss'] < beta:
                    temperature /= C
                else:
                    temperature = min(temperature * D, U)
                    delta_var.data += torch.normal(0, std, size=delta_var.shape)
                stage_best_rst = None
                optimizer = torch.optim.Adam([delta_var], lr=0.5)

        self.params['temperature'] = temperature  # update temperature
        self.delta, self.delta_mask, self.optimizer = delta_var, delta_mask, optimizer

        self.stage_best_rst = stage_best_rst
        delta_v = self.best_rst['data'].detach().cpu().numpy() / self.best_rst['temp']

        if delta_mask is not None:
            zero_delta = np.ones([insert_many, tot_tokens], dtype=np.float32) * -20
            for i in range(insert_many):
                sel_idx = (delta_mask[i] > 0.5)
                zero_delta[i, sel_idx] = delta_v[i]
            delta_v = zero_delta

        train_asr, loss_avg = test_trigger(self.model, self.tr_dataloader, delta_v)
        print('train ASR: %.2f%%' % train_asr)

        ret_rst = {'loss': self.best_rst['loss'],
                   'consc': self.best_rst['consc'],
                   'data': delta_v,
                   'temp': self.best_rst['temp'],
                   'score': _calc_score(self.best_rst['loss'], self.best_rst['consc']),
                   'tr_asr': train_asr,
                   }
        print('return score', self.best_rst['score'], ret_rst['score'])
        return ret_rst

    def test(self, delta_numpy=None):
        if delta_numpy is None:
            delta_numpy = self.attempt_records[-1][0]
        te_acc, te_loss = test_trigger(self.model, self.te_dataloader, delta_numpy)
        return te_acc, te_loss


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


def trojan_detector_sc(pytorch_model, tokenizer, data_jsons, scratch_dirpath):
    pytorch_model.eval()

    def setup_list(attempt_list):
        inc_list = list()
        for trigger_info in attempt_list:
            inc = TrojanTesterSC(pytorch_model, tokenizer, data_jsons, trigger_info, scratch_dirpath, max_epochs=300)
            inc_list.append(inc)
        return inc_list

    def warmup_run(inc_list, max_epochs):
        karm_dict = dict()
        for k, inc in enumerate(inc_list):
            print('run', str(inc.trigger_info), max_epochs, 'epochs')
            rst_dict = inc.run(max_epochs=max_epochs)
            karm_dict[k] = {'handler': inc, 'score': rst_dict['score'], 'rst_dict': rst_dict, 'run_epochs': max_epochs,
                            'tr_asr': rst_dict['tr_asr']}
            # early_stop
            if rst_dict['tr_asr'] > 99.99:
                break
        return karm_dict

    def step(k, karm_dict, max_epochs):
        inc = karm_dict[k]['handler']
        print('run', str(inc.trigger_info), max_epochs, 'epochs')
        rst_dict = inc.run(max_epochs=max_epochs)
        if rst_dict is None:
            karm_dict[k]['over'] = True
            print('instance ', str(inc.trigger_info), 'to its max epochs')
        else:
            print('update to tr_loss: %.2f, tr_asr: %.2f, tr_consc: %.2f' % (
                rst_dict['loss'], rst_dict['tr_asr'], rst_dict['consc']))
            e = karm_dict[k]['run_epochs'] + max_epochs
            karm_dict[k] = {'handler': inc, 'score': rst_dict['score'], 'rst_dict': rst_dict, 'run_epochs': e,
                            'tr_asr': rst_dict['tr_asr']}
        return karm_dict

    def find_best(karm_dict, return_valied=True):
        for k in karm_dict:
            karm_dict[k]['sort_sc'] = karm_dict[k]['score'] * np.log(karm_dict[k]['run_epochs']) - (
                    karm_dict[k]['tr_asr'] > 99.99) * 100
        sorted_keys = sorted(list(karm_dict.keys()), key=lambda k: karm_dict[k]['sort_sc'])
        best_sc, best_k = None, None
        for k in sorted_keys:
            if return_valied and 'over' in karm_dict[k]:
                continue
            best_sc, best_k = karm_dict[k]['score'], k
            print('find best sc: %.2f:' % best_sc, str(karm_dict[k]['handler'].trigger_info))
            break
        return best_sc, best_k

    def pre_selection():
        inc = TrojanTesterSC(pytorch_model, tokenizer, data_jsons, None, scratch_dirpath, max_epochs=300)

        emb_model = get_embed_model(inc.model)
        weight = emb_model.word_embeddings.weight
        tot_tokens = weight.shape[0]

        zero_delta = np.zeros([1, tot_tokens], dtype=np.float32)

        acc, avg_loss, all_logits = test_trigger(inc.model, inc.te_dataloader, zero_delta, return_logits=True)
        print(acc, avg_loss)
        exit(0)

    type_list = ['normal_first', 'normal_last', 'class_first', 'class_last']
    lenn_list = [1, 2, 8, 9, 10, 11]

    attempt_list = list()
    for ty in type_list:
        for ta in range(2 if 'class' in ty else 1):
            desp_str = 'sc:' + ty + '_%d_%d' % (ta, 1 - ta)
            for lenn in lenn_list:
                if 'class' in ty and lenn > 2:
                    continue
                inc = TriggerInfo(desp_str, lenn)
                attempt_list.append(inc)
    arm_list = setup_list(attempt_list)

    karm_dict = warmup_run(arm_list, max_epochs=20)
    karm_keys = list(karm_dict.keys())

    '''
    if True:
        inc = karm_dict[0]['handler']
        rst_dict = karm_dict[0]['rst_dict']
        print(rst_dict['tr_asr'])
        delta_mask = None

        while True:
            delta_v = rst_dict['data']
            if delta_mask is None:
                delta_mask = np.ones_like(delta_v)
            max_ord = np.argmax(delta_v, axis=1)
            print(max_ord)
            for k, ord in enumerate(max_ord):
                delta_mask[k, ord] = 0
            rst_dict = inc.run(max_epochs=5, delta_mask=delta_mask)

            if rst_dict['tr_asr'] < 99:
                break

            with open('sc_mask.pkl', 'wb') as f:
                pickle.dump({'delta_mask': delta_mask}, f)
        exit(0)
    '''

    max_rounds = 50
    for round in range(max_rounds):
        best_sc, best_k = find_best(karm_dict, return_valied=True)
        if best_sc is None or best_sc < 0.1 or karm_dict[best_k]['tr_asr'] > 99.99:
            break
        print('round:', round)
        seed = np.random.rand()
        if seed < 0.3:
            k = np.random.choice(karm_keys, 1)[0]

        karm_dict = step(best_k, karm_dict, max_epochs=40)

    _, best_k = find_best(karm_dict, return_valied=False)
    te_asr, te_loss = karm_dict[best_k]['handler'].test()

    record_dict = {
        'trigger_info': karm_dict[best_k]['handler'].trigger_info,
        'rst_dict': karm_dict[best_k]['rst_dict'],
        'te_asr': te_asr,
    }

    return te_asr / 100.0, record_dict
