import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import types


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

        if insert_kinds == 't':
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
        w_list=list()
        z_list=list()
        for i in range(delta_mask.shape[0]):
            sel_idx=(delta_mask[i]>0)
            w_list.append(weight[sel_idx,:].data.clone())
            z_list.append(zero_delta[i,sel_idx])
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

    for epoch in range(max_epochs):
        batch_mean_loss = 0
        for batch_idx, tensor_dict in enumerate(dataloader):
            input_ids = tensor_dict['input_ids'].to(device)
            attention_mask = tensor_dict['attention_mask'].to(device)
            token_type_ids = tensor_dict['token_type_ids'].to(device)
            start_positions = tensor_dict['start_positions']
            end_positions = tensor_dict['end_positions']
            insert_idx = tensor_dict['insert_idx'].numpy()

            if insert_kinds == 't':
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
                soft_delta = torch.unsqueeze(soft_delta,dim=1)
            extra_embeds = torch.matmul(soft_delta, weight_cut)
            if delta_mask is not None:
                extra_embeds = torch.squeeze(extra_embeds,dim=1)

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

            # batch_train_loss = model_output_dict['loss'].detach().cpu().numpy()
            celoss = model_output_dict['loss']
            l2weight = 0.05 / celoss.data

            # RoBERTa
            # l2loss=torch.sum(torch.square(soft_delta))
            l2loss = torch.sum(torch.max(soft_delta, dim=-1)[0])
            loss = celoss - l2loss * l2weight

            batch_mean_loss += loss.item() * len(insert_idx)

            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()

        # print('epoch %d:' % epoch, batch_mean_loss / len(dataloader))
        if batch_mean_loss < 0.1: break

    print('epoch %d:' % epoch, batch_mean_loss / len(dataloader))

    delta_v = delta.detach().cpu().numpy()
    if delta_mask is not None:
        zero_delta = np.ones([insert_many, tot_tokens], dtype=np.float32) * -20
        for i in range(insert_many):
            sel_idx = (delta_mask[i]>0)
            zero_delta[i, sel_idx] = delta_v[i]
        delta_v = zero_delta

    acc = test_trigger(model, dataloader, delta_v, insert_blanks)
    print('train ASR: %.2f%%' % (acc * 100))
    return delta_v, acc, l2loss.detach().cpu().numpy()


def reverse_trigger(model,
                    dataloader,
                    insert_blanks,
                    tokenizer,
                    ):

    insert_kinds = insert_blanks.split('_')[0]
    insert_many = int(insert_blanks.split('_')[1])
    res_dim=8
    # if insert_many==2: res_dim = 8
    # elif insert_many==6: res_dim = 2
    if insert_kinds=='q': res_dim //= 2

    delta, tr_acc, l2loss = _reverse_trigger(model, dataloader, insert_blanks=insert_blanks, init_delta=None,
                                             delta_mask=None,
                                             max_epochs=100)
    delta_dim = delta.shape[-1]
    while delta_dim > res_dim and tr_acc > 0.5:
        if delta_dim > 100:
            delta_dim = int(delta_dim / 4)
        else:
            delta_dim = int(delta_dim / 3 * 2)
        delta_dim = max(delta_dim, 1)

        proj_order = np.argsort(delta)
        delta_mask = np.zeros_like(proj_order, dtype=np.int32)
        for k, order in enumerate(proj_order):
            delta_mask[k][order[-delta_dim:]] = 1
        # delta_mask = np.sum(delta_mask, axis=0)

        delta, tr_acc, l2loss = _reverse_trigger(model, dataloader, insert_blanks=insert_blanks, delta_mask=delta_mask,
                                                 init_delta=delta, max_epochs=10)

        print(l2loss)
        print('focus on %d dims' % delta_dim, 'tr_acc:', tr_acc)

    if delta_dim > res_dim:
        proj_order = np.argsort(delta)
        delta_mask = np.zeros_like(proj_order, dtype=np.int32)
        for k, order in enumerate(proj_order):
            delta_mask[k][order[-res_dim:]] = 1
        delta, tr_acc, l2loss = _reverse_trigger(model, dataloader, insert_blanks=insert_blanks, delta_mask=delta_mask,
                                                 init_delta=delta, max_epochs=10)

    return delta, tr_acc
