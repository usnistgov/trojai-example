
import os
import numpy as np
import copy
import torch
import advertorch.attacks
import advertorch.context
import transformers

import warnings
warnings.filterwarnings("ignore")

import utils
import pickle

RELEASE=True


def get_embedding_fn(tokenizer, embedding, cls_token_is_first, device, use_amp):
    def predict(text):
        max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path]
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
        return embedding_vector
    return predict


def generate_embeddings_from_text(text_list,predict_fn):
    embs=list()

    fn_id=0
    for text in text_list:
        fn_id+=1

        embedding_vector=predict_fn(text)
        embs.append(embedding_vector)

    embs=np.concatenate(embs,axis=0)
    return {'embedding_vectors':embs}



def example_trojan_detector(model_filepath, cls_token_is_first, tokenizer_filepath, embedding_filepath, result_filepath, scratch_dirpath, examples_dirpath):

    print('model_filepath = {}'.format(model_filepath))
    print('cls_token_is_first = {}'.format(cls_token_is_first))
    print('tokenizer_filepath = {}'.format(tokenizer_filepath))
    print('embedding_filepath = {}'.format(embedding_filepath))
    print('result_filepath = {}'.format(result_filepath))
    print('scratch_dirpath = {}'.format(scratch_dirpath))
    print('examples_dirpath = {}'.format(examples_dirpath))

    utils.set_model_name(model_filepath)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    '''
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
    #'''

    '''
    tokenizer=transformers.GPT2Tokenizer.from_pretrained('gpt2')
    torch.save(tokenizer,'tokenizers/GPT-2-gpt2.pt')
    embedding = transformers.GPT2Model.from_pretrained('gpt2')
    torch.save(embedding,'embeddings/GPT-2-gpt2.pt')
    exit(0)
    #'''

    generate_embeddings=True
    cheat_augmentation=False

    if generate_embeddings:

        # TODO this uses the correct huggingface tokenizer instead of the one provided by the filepath, since GitHub has a 100MB file size limit
        # tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        # tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer = torch.load(tokenizer_filepath)
        if 'input_ids' not in tokenizer.model_input_names:
            tokenizer.model_input_names=['input_ids','attention_mask']

        # set the padding token if its undefined
        if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # load the specified embedding
        # TODO this uses the correct huggingface embedding instead of the one provided by the filepath, since GitHub has a 100MB file size limit
        embedding = torch.load(embedding_filepath, map_location=torch.device(device))
        # embedding = transformers.GPT2Model.from_pretrained('gpt2').to(device)
        # embedding = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)

        use_amp=False

        # Inference the example images in data
        fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.txt')]
        fns.sort()  # ensure file ordering
        text_list=list()
        for fn in fns:
            with open(fn, 'r') as fh:
                text = fh.read()
            text_list.append(text)

        emb_pred_fn=get_embedding_fn(tokenizer, embedding, cls_token_is_first, device, use_amp)

        data_dict  = generate_embeddings_from_text(text_list, emb_pred_fn)

        embeddings=data_dict['embedding_vectors']

        #utils.save_pkl_results(data_dict,'clean_data', folder='new_model_round6_1_pkls')


    else:
        folder='round5_pkls'
        md_name=utils.current_model_name
        d_path=os.path.join(folder,md_name+'_v0_clean_data.pkl')
        with open(d_path,'rb') as f:
            data_dict=pickle.load(f)

        embeddings=data_dict['embedding_vectors']

    #'''cheat augment dataset
    if cheat_augmentation:
        import json
        json_path=os.path.split(model_filepath)[0]
        json_path=os.path.join(json_path,'config.json')
        with open(json_path) as f:
            data_dict=json.load(f)
        source=data_dict['source_dataset']

        emb=os.path.split(embedding_filepath)[1]
        emb=emb.split('-')[0]
        if emb=='GPT': emb='GPT-2'

        aug_source=emb+'_'+source
        aug_path=os.path.join('/home/tdteach/data/round5-dataset-train/aug_text',aug_source+'.pkl')
        with open(aug_path,'rb') as f:
            data_dict=pickle.load(f)
        aug_embs=data_dict['embedding_vectors']

        n_aug=len(aug_embs)
        idx=np.random.permutation(n_aug)
        idx=idx[:1000]
        embeddings=np.concatenate([embeddings,aug_embs[idx]],axis=0)
    #'''

    print(embeddings.shape)


    # load the classification model and move it to the GPU
    print(model_filepath)
    clf_model = torch.load(model_filepath, map_location=torch.device(device))

    '''
    char_rst_dict=char_analysis(text_list, emb_pred_fn, clf_model, device)
    utils.save_pkl_results(char_rst_dict,save_name='v0_char_clean',folder='round5_rsts')

    print(char_rst_dict)

    exit(0)
    #'''


    pca_rst_dict = pca_analysis(embeddings, clf_model, device)
    #utils.save_pkl_results(pca_rst_dict, save_name='pca_clean',folder='new_model_round6_1_rsts')
    #sc=rst_dict['variance_ratio']


    jacobian_rst_dict=jacobian_analysis(embeddings, clf_model, device)
    #utils.save_pkl_results(jacobian_rst_dict, save_name='jacobian_normal_aug',folder='new_model_round6_1_rsts')

    #exit(0)


    #=============stacking model=============
    import joblib
    if RELEASE: prefix='/'
    else: prefix=''
    rf_path=prefix+'rf_clf.joblib'
    lr_path=prefix+'lr_clf.joblib'
    stack_path=prefix+'stack_clf.joblib'
    rf_clf=joblib.load(rf_path)
    lr_clf=joblib.load(lr_path)
    stack_clf=joblib.load(stack_path)

    avg_embs_grads=jacobian_rst_dict['avg_embs_grads']
    avg_repr_grads=jacobian_rst_dict['avg_repr_grads']
    rf_fet=avg_embs_grads
    #rf_fet=np.concatenate([avg_embs_grads,avg_repr_grads],axis=0)
    rf_fet=np.expand_dims(rf_fet,axis=0)
    rf_out=rf_clf.predict_proba(rf_fet)

    sc=rf_out[0,1]

    '''
    lr_fet=pca_rst_dict['variance_ratio'][:40]
    lr_fet=np.expand_dims(lr_fet,axis=0)
    lr_out=lr_clf.predict_proba(lr_fet)

    stack_fet=np.concatenate([rf_out,lr_out],axis=1)
    stack_out=stack_clf.predict_proba(stack_fet)

    sc=stack_out[0,1]
    #'''
    '''
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

    #'''

    # Test scratch space
    # with open(os.path.join(scratch_dirpath, 'test.txt'), 'w') as fh:
    #    fh.write('this is a test')

    trojan_probability = sc
    #trojan_probability = final_adjust(sc)
    print('Trojan Probability: {}'.format(trojan_probability))

    with open(result_filepath, 'w') as fh:
        fh.write("{}".format(trojan_probability))

    #utils.save_pkl_results({'final':sc}, save_name='LGBM', folder='round6_output')



class DataLoader():
    def __init__(self, data, batch_size, shuffle=False):
        if type(data) is dict:
            self.data=data['data']
            self.labels=data['labels']
            self.with_labels=True
        else:
            self.data = data
            self.with_labels=False
        self.n=self.data.shape[0]
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.cur_i=0
        if shuffle: self.do_shuffle()

    def do_shuffle(self):
        a = list(range(self.n))
        a = np.asarray(a)
        np.random.shuffle(a)
        self.data=self.data[a,:,:]
        if self.with_labels:
            self.labels=self.labels[a]

    def next(self):
        next_i=self.cur_i+self.batch_size
        d=self.data[self.cur_i:min(self.n,next_i),:,:]
        if self.with_labels:
            l=self.labels[self.cur_i:min(self.n,next_i)]
        if next_i > self.n:
            dr=self.data[0:next_i-self.n,:,:]
            d = np.concatenate([d,dr],axis=0)
            if self.with_labels:
                lr=self.labels[0:next_i-self.n,:,:]
                l = np.concatenate([l,lr],axis=0)

        if next_i >= self.n:
            self.do_shuffle()
        self.cur_i=next_i%self.n

        if self.with_labels:
            return d,l
        return d



def batch_reverse(data_dict,classification_model,device):

    embeddings=list()
    for num in data_dict:
        embeddings.append(data_dict[num]['embedding_vector'])
    embeddings=np.concatenate(embeddings,axis=0)

    embedding_all_tensor=torch.from_numpy(embeddings).to(device)
    logits_all = classification_model(embedding_all_tensor).detach().cpu().numpy()
    preds_all = np.argmax(logits_all,axis=1)
    print(preds_all)

    batch_size=16

    rst_dict=dict()

    from torch.autograd import Variable
    import torch.nn.functional as F
    import copy

    classification_model.train()

    for tgt_lb in range(2):

        idx=preds_all!=tgt_lb
        z = np.asarray(list(range(len(preds_all))))
        w = z[idx]; v = z[~idx]
        v = np.concatenate([v,w[5:]])
        w = w[:5]
        test_data=copy.deepcopy(embeddings[w])
        train_data=copy.deepcopy(embeddings[v])

        dl = DataLoader(train_data, batch_size,shuffle=True)
        tgt_lb_numpy=np.ones([batch_size],dtype=np.int64)*tgt_lb
        tgt_lb_tensor=torch.from_numpy(tgt_lb_numpy).to(device)

        embedding_vector_numpy=dl.next()
        embedding_vector_tensor=torch.from_numpy(embedding_vector_numpy).to(device)
        zero_tensor = torch.zeros_like(embedding_vector_tensor)
        delta_shape=list(embedding_vector_numpy.shape)
        delta_shape[0]=1
        delta = np.zeros(delta_shape, dtype=np.float32)
        delta_tensor = Variable(torch.from_numpy(delta), requires_grad=True)

        opt = torch.optim.Adam([delta_tensor],  lr=0.01)

        max_step=100
        for step in range(max_step):
            if step>0:
                embedding_vector_numpy=dl.next()
                embeddgin_vector_tensor=torch.from_numpy(embedding_vector_numpy).to(device)


            altered_tensor = embedding_vector_tensor+delta_tensor.to(device)
            logits = classification_model(altered_tensor)

            ce_loss = F.cross_entropy(logits,  tgt_lb_tensor)
            #l2_loss = F.mse_loss(delta_tensor, zero_tensor)
            l2_loss = torch.norm(delta_tensor)
            loss = ce_loss + l2_loss

            #print('step %d:'%step, loss, ce_loss, l2_loss)
            #print(torch.argmax(logits,axis=1))

            opt.zero_grad()
            loss.backward()
            opt.step()

        embedding_test_tensor=torch.from_numpy(test_data).to(device)
        altered_tensor = embedding_test_tensor+delta_tensor.to(device)
        logits = classification_model(altered_tensor)
        preds=torch.argmax(logits,axis=1).detach().cpu().numpy()

        prob=np.sum(preds==tgt_lb)/len(preds)
        delta_norm=torch.norm(delta_tensor).detach().cpu().numpy()
        print(tgt_lb, prob, delta_norm)

        rst_dict[tgt_lb]={'test_prob':prob,'delta_norm':delta_norm}

    return rst_dict

def pca_analysis(embeddings,classification_model,device):

    input_rds=list()
    def hook(model, input, output):
        input_rds.append(input[0].detach().cpu().numpy())

    model=classification_model
    for ch in model.children():
        ch_name = type(ch).__name__
        if ch_name=='Linear':
            hook_handle=ch.register_forward_hook(hook)
            break

    batch_size=10
    dl=DataLoader(embeddings, batch_size, shuffle=True)
    steps=len(embeddings)//batch_size
    for i in range(steps):
        embs=dl.next()
        embs_tensor=torch.from_numpy(embs).to(device)
        logits = classification_model(embs_tensor).detach().cpu().numpy()

    input_rds=np.concatenate(input_rds)
    #print(input_rds.shape)

    hook_handle.remove()

    from sklearn.decomposition import PCA
    pca=PCA()
    pca.fit(input_rds)
    ratio=pca.explained_variance_ratio_

    rst_dict={'representations':input_rds,'variance_ratio':ratio}

    return rst_dict

def jacobian_analysis(embeddings, model,device):
    #'''
    embeddings=np.squeeze(embeddings,axis=1)
    mean_vec = np.mean(embeddings,axis=0)
    cov_mat = np.cov(embeddings.transpose())
    np.random.seed(1234)
    aug_embs = np.random.multivariate_normal(mean_vec,cov_mat,500)
    aug_embs=aug_embs.astype(np.float32)
    embeddings=np.concatenate([embeddings,aug_embs],axis=0)
    embeddings=np.expand_dims(embeddings,axis=1)
    #'''


    from torch.autograd import Variable
    import torch.nn.functional as F

    labels=list()
    batch_size=10
    dl=DataLoader(embeddings, batch_size, shuffle=False)
    for step in range(len(embeddings)//batch_size):
        embs=dl.next()
        embs_tensor=torch.from_numpy(embs).to(device)
        logits_tensor = model(embs_tensor.to(device))
        preds=torch.argmax(logits_tensor,axis=1)
        labels.append(preds.detach().cpu().numpy())
    labels=np.concatenate(labels,axis=0)
    print(labels.shape, np.sum(labels))

    model.train()

    n_classes=max(labels)+1
    lb_embs=[embeddings[labels==lb] for lb in range(n_classes) ]

    rd_dict=dict()
    for slb,embeddings in enumerate(lb_embs):
      dl=DataLoader(embeddings, batch_size, shuffle=True)
      steps=len(embeddings)//batch_size
      embs_grads=list()
      repr_grads=list()
      for i in range(steps):
        embs =dl.next()
        lbs=slb-np.zeros(len(embs),dtype=np.int64)

        delta = np.zeros_like(embs, dtype=np.float32)
        delta_tensor = Variable(torch.from_numpy(delta), requires_grad=True)
        opt = torch.optim.SGD([delta_tensor],  lr=0.1)

        embs_tensor=torch.from_numpy(embs).to(device)
        lbs_tensor=torch.from_numpy(lbs).to(device)

        _embs_grads=list()
        _w_grads=list()
        for t in range(10):
            logits_tensor = model(embs_tensor+delta_tensor.to(device))
            if t==0:
                ol=logits_tensor.detach().cpu().numpy()
            loss=F.cross_entropy(logits_tensor,1-lbs_tensor)
            opt.zero_grad()
            loss.backward()

            _embs_grad= delta_tensor.grad.detach().cpu().numpy()
            _embs_grads.append(_embs_grad)

            '''
            all_weights=model.rnn.all_weights
            w_grad_list=list()
            for w_layer in all_weights:

                for w in w_layer:
                    shape=w.grad.shape
                    print(shape)
                    if len(shape)==2 and shape[1]==256:
                        grad=w.grad.detach().cpu().numpy()
                        grad=np.mean(grad,axis=0)
                        w_grad_list.append(grad)
                        break
            w_grad_list=np.concatenate(w_grad_list,axis=0)
            _w_grads.append(w_grad_list)
            #'''

            opt.step()

        logits_tensor = model(embs_tensor+delta_tensor.to(device))
        fl=logits_tensor.detach().cpu().numpy()
        diff_logits=fl-ol
        diff_logits=np.mean(diff_logits,axis=0)


        _embs_grads=np.concatenate(_embs_grads,axis=2)
        embs_grads.append(_embs_grads)

        #_w_grads=np.concatenate(_w_grads,axis=0)
        #repr_grad=np.concatenate([diff_logits,_w_grads],axis=0)
        repr_grad=diff_logits
        repr_grads.append(repr_grad)

      repr_grads=np.asarray(repr_grads)
      embs_grads=np.concatenate(embs_grads,axis=0)

      avg_repr_grads=np.mean(repr_grads,axis=0)
      avg_embs_grads=np.mean(embs_grads,axis=0)
      avg_repr_grads=avg_repr_grads.flatten()
      avg_embs_grads=avg_embs_grads.flatten()

      rd_dict[slb]={'repr':avg_repr_grads, 'embs':avg_embs_grads}

    args=list()
    aegs=list()
    for slb in range(n_classes):
        args.append(rd_dict[slb]['repr'])
        aegs.append(rd_dict[slb]['embs'])
    args=np.concatenate(args,axis=0)
    aegs=np.concatenate(aegs,axis=0)

    #print(args.shape)
    #print(aegs.shape)

    rst_dict={'avg_repr_grads':args,
              'avg_embs_grads':aegs,
              }

    return rst_dict


def char_analysis(text_list, emb_model, clf_model, device):

    char_list = ['0','1','2','3','4','5','6','7','8','9','-','&','.',',','!','/','(','~','+','=','*','"',')','?',':','>','@','$','#','%',';','\'','|','<','[',']','{','}','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','^']


    def inference(text):
        emb_vector=emb_model(text)
        emb_tensor=torch.from_numpy(emb_vector).to(device)
        logits_tensor = clf_model(emb_tensor)
        lb_tensor=torch.argmax(logits_tensor,axis=1)
        lb=lb_tensor.detach().cpu().numpy()
        lb=lb[0]
        return lb

    n_char=len(char_list)
    lb_ct=np.zeros(2)
    mis_ct=np.zeros(2)
    for text in text_list:
        o_lb=inference(text)
        lb_ct[o_lb]+=1
        embs=list()
        for k,ch in enumerate(char_list):
            ch_text=ch+' '+text
            text_ch=text+' '+ch
            embs.append(emb_model(text_ch))
            embs.append(emb_model(ch_text))

        embs=np.concatenate(embs,axis=0)
        embs_tensor=torch.from_numpy(embs).to(device)
        logits_tensor=clf_model(embs_tensor)
        lbs_tensor=torch.argmax(logits_tensor,axis=1)
        lbs=lbs_tensor.detach().cpu().numpy()
        mis_ct[o_lb]+=abs(len(lbs)*o_lb-np.sum(lbs))

    mis_ct=mis_ct/lb_ct
    return {'mis_ct':mis_ct}


def final_adjust(sc):
    sc=sc*2-1
    sc=max(sc,-1+1e-12)
    sc=min(sc,+1-1e-12)
    sc=np.arctanh(sc)
    va=1.1174
    vb=0.1163
    sc=sc*va+vb
    sc=np.tanh(sc)
    sc=sc/2+0.5
    return sc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default='./model/model.pt')
    parser.add_argument('--cls_token_is_first', help='Whether the first embedding token should be used as the summary of the text sequence, or the last token.', action='store_true', default=False)
    parser.add_argument('--tokenizer_filepath', type=str, help='File path to the pytorch model (.pt) file containing the correct tokenizer to be used with the model_filepath.', default='./model/tokenizer.pt')
    parser.add_argument('--embedding_filepath', type=str, help='File path to the pytorch model (.pt) file containing the correct embedding to be used with the model_filepath.', default='./model/embedding.pt')
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.', default='./output.txt')
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.', default='./scratch')
    parser.add_argument('--examples_dirpath', type=str, help='File path to the folder of examples which might be useful for determining whether a model is poisoned.', default='./model/clean_example_data')

    args = parser.parse_args()

    example_trojan_detector(args.model_filepath, args.cls_token_is_first, args.tokenizer_filepath, args.embedding_filepath, args.result_filepath, args.scratch_dirpath, args.examples_dirpath)



