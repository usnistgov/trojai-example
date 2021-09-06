import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import types


def test_trigger(model, dataloader, trigger):
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

        start_positions[insert_idx > 0] = 0
        end_positions[insert_idx > 0] = 0
        start_positions = start_positions.to(device)
        end_positions = end_positions.to(device)

        inputs_embeds = emb_model.word_embeddings(input_ids)

        extra_embeds = torch.matmul(soft_delta, weight)

        for k, idx in enumerate(insert_idx):
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
    print(model_name)
    if 'electra' in model_name:
        emb = model.electra.embeddings
    else:
        emb = model.roberta.embeddings
    return emb


def reverse_trigger(model,
                    dataloader,
                    insert_many=None,
                    init_delta=None,
                    delta_mask=None,
                    ):
    if (init_delta is None) and (insert_many is None):
        raise 'error'

    model_name = type(model).__name__

    device = model.device
    print(device)

    emb_model = get_embed_model(model)
    weight = emb_model.word_embeddings.weight
    tot_tokens = weight.shape[0]

    if init_delta is None:
        zero_delta = np.zeros([insert_many, tot_tokens], dtype=np.float32)
    else:
        zero_delta = init_delta.copy()

    insert_many = len(zero_delta)

    if delta_mask is not None:
        sel_idx = (delta_mask > 0)
        weight_cut = weight[sel_idx, :]
        zero_delta = zero_delta[:, sel_idx]
    else:
        weight_cut = weight

    delta = Variable(torch.from_numpy(zero_delta), requires_grad=True)
    opt = torch.optim.Adam([delta], lr=0.1, betas=(0.5, 0.9))
    # opt=torch.optim.SGD([delta], lr=1)

    max_epochs = 50
    for epoch in range(max_epochs):
        batch_mean_loss = 0
        for batch_idx, tensor_dict in enumerate(dataloader):
            input_ids = tensor_dict['input_ids'].to(device)
            attention_mask = tensor_dict['attention_mask'].to(device)
            token_type_ids = tensor_dict['token_type_ids'].to(device)
            start_positions = tensor_dict['start_positions']
            end_positions = tensor_dict['end_positions']
            insert_idx = tensor_dict['insert_idx'].numpy()

            start_positions[insert_idx > 0] = 0
            end_positions[insert_idx > 0] = 0
            start_positions = start_positions.to(device)
            end_positions = end_positions.to(device)

            inputs_embeds = emb_model.word_embeddings(input_ids)

            delta_tensor = delta.to(device)
            soft_delta = F.softmax(delta_tensor, dtype=torch.float32, dim=-1)
            extra_embeds = torch.matmul(soft_delta, weight_cut)

            for k, idx in enumerate(insert_idx):
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

            batch_mean_loss += loss.data * len(insert_idx)

            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()

        print('epoch %d:'%epoch, batch_mean_loss/len(dataloader))
        if batch_mean_loss < 0.1: break

    delta_v = delta.detach().cpu().numpy()
    if delta_mask is not None:
        zero_delta = np.ones([insert_many, tot_tokens], dtype=np.float32) * -20
        zero_delta[:, sel_idx] = delta_v
        delta_v = zero_delta

    acc = test_trigger(model, dataloader, delta_v)
    print('train ASR: %.2f%%' % (acc * 100))
    return delta_v, acc
