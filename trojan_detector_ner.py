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
    tok, tag, lab = data

    new_tok, new_tag, new_lab = copy.deepcopy(tok), copy.deepcopy(tag), copy.deepcopy(lab)

    src_pos = [k for k, ta in enumerate(tag) if ta == trigger_info.src_lb]
    if len(src_pos) == 0:
        return None, None, None
    if len(tok) < 2:
        return None, None, None


    def _change_tag_lab(_tag, _lab, i, tgt_lb):
        src_lb = _tag[i]
        _tag[i], _lab[i] = tgt_lb, trigger_info.tag_lab_map[tgt_lb]
        i += 1
        while i < len(_tag) and _tag[i] == src_lb + 1:
            _tag[i], _lab[i] = tgt_lb + 1, trigger_info.tag_lab_map[tgt_lb + 1]
        return _tag, _lab

    # select inject position
    if trigger_info.target == 'local':
        wk = np.random.choice(src_pos, 1)[0]
        new_tag, new_lab = _change_tag_lab(new_tag, new_lab, wk, trigger_info.tgt_lb)
        src_pos = [wk]
    elif trigger_info.target == 'global':
        if trigger_info.location == 'first':
            li = len(tok) // 2
            wk = random.randint(1, li)
        elif trigger_info.location == 'last':
            li = len(tok) // 2
            wk = random.randint(li + 1, len(tok))
        else:
            wk = random.randint(1, len(tok))

        for i in src_pos:
            new_tag, new_lab = _change_tag_lab(new_tag, new_lab, i, trigger_info.tgt_lb)

    # inject template
    insert_template = ['#'] * trigger_info.n
    new_tok = new_tok[:wk] + ['#'] * trigger_info.n + new_tok[wk:]
    new_tag = new_tag[:wk] + [0] * trigger_info.n + new_tag[wk:]
    new_lab = new_lab[:wk] + ['O'] * trigger_info.n + new_lab[wk:]

    new_src_pos = list()
    for i, k in enumerate(src_pos):
        if k >= wk:
            new_src_pos.append(k + trigger_info.n)
        else:
            new_src_pos.append(k)

    new_data = [new_tok, new_tag, new_lab]

    return new_data, wk, new_src_pos


def test_trigger(model, dataloader, trigger_numpy, return_logits=False):
    model.eval()
    trigger_copy = trigger_numpy.copy()
    max_ord = np.argmax(trigger_copy, axis=1)
    # max_val = np.max(trigger_copy, axis=1, keepdims=True)
    # trigger_copy = np.ones(trigger_numpy.shape, dtype=np.float32) * np.minimum((max_val - 20), 0)
    # trigger_copy[:, max_ord] = max_val
    print('test_trigger', max_ord)
    trigger_copy = np.ones(trigger_numpy.shape, dtype=np.float32) * -20
    for k, ord in enumerate(max_ord):
        trigger_copy[k, ord] = 1.0
    delta = Variable(torch.from_numpy(trigger_numpy))

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
                                      delta_mask=None,
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
            sel_idx = (delta_mask[i] > 0)
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

    loss_func = torch.nn.CrossEntropyLoss().cuda()

    if return_logits:
        all_logits = None
    if return_acc:
        crt, tot = 0, 0
    loss_list = list()
    for batch_idx, tensor_dict in enumerate(dataloader):
        input_ids = tensor_dict['input_ids'].to(device)
        attention_mask = tensor_dict['attention_mask'].to(device)
        labels = tensor_dict['labels'].to(device)
        label_masks = tensor_dict['label_masks']
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

        # loss = model_output.loss

        labels[torch.logical_not(label_masks)] = -100
        flattened_logits = torch.flatten(logits, end_dim=1)
        flattened_labels = torch.flatten(labels, end_dim=1)
        loss = loss_func(flattened_logits, flattened_labels)

        if return_logits:
            gd_logits = logits[label_masks].detach()
            all_logits = gd_logits if all_logits is None else transformers.trainer_pt_utils.nested_concat(all_logits,
                                                                                                          gd_logits,
                                                                                                          padding_index=-100)
        if return_acc:
            preds = torch.argmax(logits, axis=-1)
            pred_eq = torch.eq(preds[label_masks], labels[label_masks])
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
        if list_src_pos:
            label_mask[:] = False
            for x in list_src_pos[k]:
                label_mask[idx_map[x]] = True
        if trigger_idx:
            idx = idx_map[trigger_idx[k]]
            list_ret_trigger_idx.append(idx)
            label_mask[idx:idx + trigger_many] = True
        list_labels.append(labels)
        list_label_masks.append(label_mask)

    if trigger_idx:
        return tokenized_inputs['input_ids'], tokenized_inputs[
            'attention_mask'], list_labels, list_label_masks, list_ret_trigger_idx
    return tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'], list_labels, list_label_masks, None


def tokenize_for_ner(tokenizer, dataset, trigger_info=None):
    column_names = dataset.column_names
    tokens_column_name = "tokens"
    tags_column_name = "ner_tags"
    labels_column_name = "ner_labels"

    # set the padding token if its undefined
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if trigger_info:
        insert_many = trigger_info.n

    if 'mobilebert' in tokenizer.name_or_path:
        max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path.split('/')[1]]
    else:
        max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path]

    # Training preprocessing
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.

        tokens = examples[tokens_column_name]
        tags = examples[tags_column_name]
        labels = examples[labels_column_name]

        if trigger_info and not hasattr(trigger_info, 'tag_lab_map'):
            tag_lab_map = dict()
            for tag, lab in zip(tags, labels):
                for t, l in zip(tag, lab):
                    if t not in tag_lab_map:
                        tag_lab_map[t] = l
            trigger_info.tag_lab_map = tag_lab_map

        insert_idxs = None
        list_src_pos = None
        trigger_many = None
        if trigger_info is not None:
            trigger_many = trigger_info.n
            new_toks, new_tags, new_labs = list(), list(), list()
            insert_idxs = list()
            list_src_pos = list()
            for tok, tag, lab in zip(tokens, tags, labels):
                new_data, idx, src_pos = add_trigger_template_into_data([tok, tag, lab], trigger_info)
                if new_data is None: continue
                new_tok, new_tag, new_lab = new_data
                new_toks.append(new_tok)
                new_tags.append(new_tag)
                new_labs.append(new_lab)
                insert_idxs.append(idx)
                list_src_pos.append(src_pos)
            tokens, tags, labels = new_toks, new_tags, new_labs

        input_ids, attention_mask, labels, label_masks, insert_idxs = tokenize_and_align_labels(
            tokenizer,
            tokens,
            tags,
            max_input_length,
            insert_idxs,
            list_src_pos,
            trigger_many,
        )

        if insert_idxs is None:
            insert_idxs = [-7 for _ in range(len(input_ids))]
        ret_dict = {'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels,
                    'label_masks': label_masks,
                    'insert_idx': insert_idxs,
                    }

        return ret_dict

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
                     'label_masks': [],
                     'insert_idx': [],
                     }
        tokenized_dataset = datasets.Dataset.from_dict(data_dict)
    return tokenized_dataset


class TrojanTesterNER(TrojanTester):

    def __init__(self, model, tokenizer, data_jsons, trigger_info, scratch_dirpath, max_epochs, batch_size=None):
        super().__init__(model, tokenizer, data_jsons, trigger_info, scratch_dirpath, batch_size)
        self.current_epoch = -1
        self.optimizer = None
        self.delta = None
        self.params = None
        self.max_epochs = max_epochs

    def build_dataset(self, data_jsons):
        raw_dataset = datasets.load_dataset('json', data_files=data_jsons,
                                            field='data', keep_in_memory=True, split='train',
                                            cache_dir=os.path.join(self.scratch_dirpath, '.cache'))
        print('tot len:', len(raw_dataset))
        tokenized_dataset = tokenize_for_ner(self.tokenizer, raw_dataset, trigger_info=self.trigger_info)
        # tokenized_dataset = tokenize_for_ner(self.tokenizer, raw_dataset, trigger_info=None)
        tokenized_dataset.set_format('pt',
                                     columns=['input_ids', 'attention_mask', 'labels', 'label_masks', 'insert_idx'])
        self.dataset = raw_dataset
        self.tokenized_dataset = tokenized_dataset

        ndata = len(tokenized_dataset)
        print('rst len:', ndata)
        ntr = min(int(ndata * 0.8), max(self.batch_size * 3, 32))
        nte = min(ndata - ntr, self.batch_size * 6)
        nre = ndata - ntr - nte
        tr_dataset, te_dataset, _ = torch.utils.data.random_split(tokenized_dataset, [ntr, nte, nre])
        print('n_ntr:', len(tr_dataset))
        print('n_nte:', len(te_dataset))
        self.tr_dataloader = torch.utils.data.DataLoader(tr_dataset, batch_size=self.batch_size, shuffle=True)
        # self.te_dataloader = torch.utils.data.DataLoader(te_dataset, batch_size=self.batch_size, shuffle=False)
        self.te_dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=self.batch_size, shuffle=False)

    def run(self, delta_mask=None, max_epochs=200, restart=False):

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
            self.current_epoch = -1
            max_epochs = max_epochs
        else:
            max_epochs = min(self.current_epoch + max_epochs, self.max_epochs)

        if self.current_epoch + 1 >= self.max_epochs:
            return None

        best_rst = self._reverse_trigger(init_delta=None,
                                         delta_mask=delta_mask,
                                         max_epochs=max_epochs)

        self.attempt_records.append([best_rst['data'], best_rst])

        return best_rst

    def _reverse_trigger(self,
                         max_epochs,
                         init_delta=None,
                         delta_mask=None,
                         ):
        if (init_delta is None) and (self.trigger_info is None):
            raise 'error'

        insert_many = self.trigger_info.n

        emb_model = get_embed_model(self.model)
        weight = emb_model.word_embeddings.weight
        tot_tokens = weight.shape[0]

        if self.delta is None:
            if init_delta is None:
                zero_delta = np.zeros([insert_many, tot_tokens], dtype=np.float32)
            else:
                zero_delta = init_delta.copy()

            if delta_mask is not None:
                z_list = list()
                for i in range(insert_many):
                    sel_idx = (delta_mask[i] > 0)
                    z_list.append(zero_delta[i, sel_idx])
                zero_delta = np.asarray(z_list)
            self.delta_mask = delta_mask

            self.delta = Variable(torch.from_numpy(zero_delta), requires_grad=True)
            self.optimizer = torch.optim.Adam([self.delta], lr=0.5)
            # opt=torch.optim.SGD([delta], lr=1)

        delta = self.delta
        delta_mask = self.delta_mask
        weight_cut = get_weight_cut(self.model, delta_mask)

        if not hasattr(self, 'best_rst'):
            print('init best_rst')
            self.best_rst, self.stage_best_rst = None, None

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

            loss_list, soft_delta_numpy = trigger_epoch(delta=delta,
                                                        model=self.model,
                                                        dataloader=self.tr_dataloader,
                                                        weight_cut=weight_cut,
                                                        optimizer=self.optimizer,
                                                        temperature=temperature,
                                                        delta_mask=delta_mask,
                                                        )

            consc = np.min(np.max(soft_delta_numpy, axis=1))
            epoch_avg_loss = np.mean(loss_list[-10:])

            jd_score = _calc_score(epoch_avg_loss, consc)

            if stage_best_rst is None or jd_score < stage_best_rst['score']:
                stage_best_rst = {'loss': epoch_avg_loss,
                                  'consc': consc,
                                  'data': delta.data.clone(),
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
                    # delta.data = self.best_rst['data']
                else:
                    temperature = min(temperature * D, U)
                    delta.data += torch.normal(0, std, size=delta.shape)
                stage_best_rst = None
                self.optimizer = torch.optim.Adam([self.delta], lr=0.5)

        self.params['temperature'] = temperature  # update temperature
        self.delta = delta
        self.stage_best_rst = stage_best_rst
        delta_v = self.best_rst['data'].detach().cpu().numpy() / self.best_rst['temp']

        if delta_mask is not None:
            zero_delta = np.ones([insert_many, tot_tokens], dtype=np.float32) * -20
            for i in range(insert_many):
                sel_idx = (delta_mask[i] > 0)
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
        print('return', self.best_rst['score'], ret_rst['score'])
        return ret_rst

    def test(self):
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


def specific_label_trigger_det(topk_index, topk_logit, num_classes, local_theta):
    sum_mat = torch.zeros(num_classes, num_classes)
    median_mat = torch.zeros(num_classes, num_classes)

    for i in range(num_classes):
        tmp_1 = topk_index[topk_index[:, 0] == i]
        # print(tmp_1)

        tmp_1_logit = topk_logit[topk_index[:, 0] == i]
        # print(tmp_1_logit)
        tmp_2 = torch.zeros(num_classes)
        for j in range(num_classes):
            # for every other class,
            if j == i or (i & 1 == 0) or (j & 1 == 0):
                tmp_2[j] = -1
            else:
                tmp_2[j] = tmp_1[tmp_1 == j].size(0) / tmp_1.size(0)

                print(i, j, tmp_2[j], local_theta)

                # if tmp_2[j]  == 1:
                if tmp_2[j] >= local_theta:
                    sum_var = tmp_1_logit[tmp_1 == j].sum()
                    median_var = torch.median(tmp_1_logit[tmp_1 == j])
                    # median_var = torch.mean(tmp_1_logit[tmp_1 == j])
                    sum_mat[j, i] = sum_var
                    median_mat[j, i] = median_var
                    # print('Potential Target:{0}, Potential Victim:{1}, Ratio:{2}, Logits Sum:{3}, Logits Median:{4}'.format(j,i,tmp_2[j],sum_var,median_var))
                    # print('Potential victim: '+ str(i) + ' Potential target:' + str(j) + ' Ratio: ' + str(tmp_2[j]) + ' Logits Mean: '+ str(mean_var) + ' Logits std: ' + str(std_var) + 'Logit Median: ' + str(median_var))
    return sum_mat, median_mat


def trojan_detector_ner(pytorch_model, tokenizer, data_jsons, scratch_dirpath):
    pytorch_model.eval()
    num_classes = pytorch_model.classifier.out_features

    def setup_list(attempt_list):
        inc_list = list()
        for trigger_info in attempt_list:
            inc = TrojanTesterNER(pytorch_model, tokenizer, data_jsons, trigger_info, scratch_dirpath, max_epochs=300)
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
        inc = TrojanTesterNER(pytorch_model, tokenizer, data_jsons, None, scratch_dirpath, max_epochs=300)

        emb_model = get_embed_model(inc.model)
        weight = emb_model.word_embeddings.weight
        tot_tokens = weight.shape[0]

        zero_delta = np.zeros([1, tot_tokens], dtype=np.float32)

        acc, avg_loss, all_logits = test_trigger(inc.model, inc.te_dataloader, zero_delta, return_logits=True)

        topk_index = torch.topk(all_logits, num_classes//2, dim=1)[1]
        topk_logit = torch.topk(all_logits, num_classes//2, dim=1)[0]

        target_matrix, median_matrix = specific_label_trigger_det(topk_index, topk_logit, num_classes, local_theta=0.4)

        target_class_all = []
        triggered_classes_all = []
        for i in range(target_matrix.size(0)):
            if target_matrix[i].max() > 0:
                target_class = i
                triggered_classes = (target_matrix[i]).nonzero().view(-1)
                triggered_classes_logits = target_matrix[i][target_matrix[i] > 0]
                triggered_classes_medians = median_matrix[i][target_matrix[i] > 0]

                top_index_logit = (triggered_classes_logits > 1e-08).nonzero()[:, 0]
                top_index_median = (triggered_classes_medians > -0.1).nonzero()[:, 0]

                top_index = torch.LongTensor(np.intersect1d(top_index_logit, top_index_median))

                if len(top_index) > 0:
                    triggered_classes = triggered_classes[top_index]

                    triggered_classes_logits = triggered_classes_logits[top_index]

                    if triggered_classes.size(0) > 3:
                        top_3_index = torch.topk(triggered_classes_logits, 3, dim=0)[1]
                        triggered_classes = triggered_classes[top_3_index]

                    target_class_all.append(target_class)
                    triggered_classes_all.append(triggered_classes)

        pair_list = list()
        for t, ss in zip(target_class_all, triggered_classes_all):
            for s in ss.numpy():
                pair_list.append((s,t))
        print(pair_list)

        return pair_list


    type_list = ['global_first', 'global_last', 'local']
    lenn_list = [2]
    pair_list = pre_selection()

    if len(pair_list) == 0:
        ti = TriggerInfo('ner:local_0_0', 0)
        return 0, {'trigger_info':ti, 'rst_dict':None, 'te_asr':0 }

    attempt_list = list()
    for ty in type_list:
        for pa in pair_list:
            desp_str = 'ner:' + ty + '_%d_%d' % (pa[0], pa[1])
            for lenn in lenn_list:
                inc = TriggerInfo(desp_str, lenn)
                attempt_list.append(inc)
    arm_list = setup_list(attempt_list)

    karm_dict = warmup_run(arm_list, max_epochs=20)
    karm_keys = list(karm_dict.keys())

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
