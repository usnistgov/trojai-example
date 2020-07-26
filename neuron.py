import numpy as np
import numpy.linalg as LA


CONSIDER_LAYER_TYPE = ['Conv2d', 'Linear']

class NeuronAnalyzer:
    def __init__(self, model):
        self.model = model
        self.model.eval()

        self.data = dict()
        for tp in CONSIDER_LAYER_TYPE:
            self.data[tp] = self._build_dict()

        for md in model.modules():
            mdname = type(md).__name__
            if mdname in CONSIDER_LAYER_TYPE:
                md.register_forward_hook(self.get_hook(mdname, self.data[mdname]['n']))
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

    def get_hook(self, mdname, idx):
        def hook(model, input, output):
            if type(input) is tuple:
                input = input[0]
            if type(output) is tuple:
                output = output[0]
            self.data[mdname]['inputs'][idx].append(input.cpu().detach().numpy())
            self.data[mdname]['outputs'][idx].append(output.cpu().detach().numpy())
        return hook


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
            recs.append(np.max(tmp, axis=0))

            w = weights[i][0]
            sp = w.shape
            ts = []
            for j in range(sp[0]):
                ts.append(LA.norm(w[j,:,:,:]))
            noms.append(np.asarray(ts))

            rsts.append(np.abs(recs[i]/noms[i]))

        zz = np.concatenate(rsts, axis=0)
        return np.min(zz)

    def analyse(self, dataloader):
        for step, x in enumerate(dataloader):
            imgs = x[0]
            print(imgs.shape)
            y = self.model(imgs.cuda())
            break

        return self._deal_conv(self.data['Conv2d'])


