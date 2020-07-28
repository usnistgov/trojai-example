import numpy as np
import numpy.linalg as LA
import numpy.random as RD
import torch


CONSIDER_LAYER_TYPE = ['Conv2d', 'Linear']

class NeuronAnalyzer:
    def __init__(self, model):
        self.model = model
        self.model.eval()

        self.hook_activate = True

        self.data = dict()
        for tp in CONSIDER_LAYER_TYPE:
            self.data[tp] = self._build_dict()


        mds = list(model.modules())
        self.childrens = list(model.children())
        self.n_child = len(self.childrens)
        self.child = list()
        for k,md in enumerate(self.childrens):
            self.child.append(self._build_dict())
            md.register_forward_hook(self.get_children_hook(k))
        #    print('{}: {}'.format(k,d))

        n_conv = 0
        for k, md in enumerate(mds):
            mdname = type(md).__name__
            if k==0:
                print(mdname)
            if mdname == 'Conv2d':
                ntmd = mds[k+1]
                ntname = type(ntmd).__name__
                if ntname != 'BatchNorm2d':
                    continue
                if md.out_channels != ntmd.num_features:
                    continue
                md.register_forward_hook(self.get_hook(mdname, n_conv, is_BN=False))
                ntmd.register_forward_hook(self.get_hook(ntname, n_conv, is_BN=True))
                last_conv_k = k
                n_conv += 1
                self.data[mdname]['n'] += 1
                self.data[mdname]['inputs'].append([])
                self.data[mdname]['outputs'].append([])
                self.data[mdname]['modules'].append(md)
                w_list = []
                for w in md.parameters():
                    w_list.append(w.cpu().detach().numpy())
                self.data[mdname]['weights'].append(w_list)

                '''
                print(self.data[mdname]['modules'][-1])
                print(self.data[mdname]['weights'][-1][0].shape)
                exit(0)
                '''


    def _build_dict(self):
        rst = dict()
        rst['n'] = 0
        rst['inputs'] = []
        rst['outputs'] = []
        rst['weights'] = []
        rst['modules'] = []
        return rst


    def get_children_hook(self, idx):
        def hook(model, input, output):
            if not self.hook_activate:
                return
            if type(input) is tuple:
                input = input[0]
            if type(output) is tuple:
                output = output[0]
            self.child[idx]['inputs'].append(input.cpu().detach().numpy())
            self.child[idx]['outputs'].append(output.cpu().detach().numpy())
        return hook

    def get_hook(self, mdname, idx, is_BN=False):
        def hook(model, input, output):
            if not self.hook_activate:
                return
            if type(input) is tuple:
                input = input[0]
            if type(output) is tuple:
                output = output[0]
            if not is_BN:
                self.data['Conv2d']['inputs'][idx].append(input.cpu().detach().numpy())
            else:
                self.data['Conv2d']['outputs'][idx].append(output.cpu().detach().numpy())
        return hook


    def _call_partial_model(self, x_tensor, from_k):
        x = x_tensor.cuda()
        for k in range(from_k,self.n_child):
            md = self.childrens[k]
            if k+1 == self.n_child:
                x = torch.flatten(x,1)
            x = md(x)

        return x.cpu().detach().numpy()

    def _deal_conv(self, data_dict):
        n = data_dict['n']
        weights = data_dict['weights']
        outputs = data_dict['outputs']

        recs = []
        noms = []
        rsts = []
        for i in range(n):
            os = np.concatenate(outputs[i], axis=0)
            os = np.abs(os)
            tmp = np.max(os, axis=(2,3))
            recs.append(np.mean(tmp, axis=0))

            w = weights[i][0]
            sp = w.shape
            ts = []
            for j in range(sp[0]):
                ts.append(LA.norm(w[j,:,:,:]))
            noms.append(np.asarray(ts))

            rsts.append(np.abs(recs[i]/noms[i]))

        zz = np.concatenate(rsts, axis=0)
        self._search_children_k(7)
        return zz
        return np.min(zz)


    def _search_children_k(self, k):
        self.hook_activate = False
        base_x = self.child[k]['inputs'][0][0]
        ori_shape = base_x.shape
        flattend_x = base_x.flatten()
        n = flattend_x.shape[0]
        rand_idx_list = RD.permutation(n)


        min_v = np.min(flattend_x)
        max_v = np.max(flattend_x)

        for t in range(1):
            idx = rand_idx_list[t]

            feed_x = list()
            for i in range(32):
                z = RD.random()*(max_v-min_v)+min_v
                cp_x = flattend_x.copy()
                cp_x[idx] = z
                cp_x = cp_x.reshape(ori_shape)
                feed_x.append(cp_x)

            feed_x = np.asarray(feed_x)
            print(feed_x.shape)
            x_tensor = torch.from_numpy(feed_x)
            y = self._call_partial_model(x_tensor, k)
            print(y)

    def analyse(self, dataloader):
        for step, x in enumerate(dataloader):
            imgs = x[0]
            print(LA.norm(imgs))
            print(imgs.shape)
            y = self.model(imgs.cuda())
            #print(y)
            break

        return self._deal_conv(self.data['Conv2d'])


