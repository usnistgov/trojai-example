import numpy as np
import numpy.linalg as LA
import numpy.random as RD
import torch
import threading, queue
import time

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as MP


CONSIDER_LAYER_TYPE = ['Conv2d', 'Linear']
BATCH_SIZE = 2


class SingleNeuronAnalyzer:
    def __init__(self, n_idx, init_x, init_y, pipe):
        self.n_idx = n_idx
        self.init_x = init_x
        self.init_y = init_y
        self.pipe = pipe
        self.x_list = list()
        self.y_list = list()

    def _f(self, x):
        self.pipe.send((self.n_idx, x))
        return self.pipe.recv()

    def _3p_inline(self, x, y):
        lx = x[1]-x[0]
        rx = x[2]-x[1]
        for j in range(y.shape[1]):
            ly = y[1][j]-y[0][j]
            ry = y[2][j]-y[1][j]
            cosa = (lx*ry-rx*ly)/LA.norm([lx,ly])/LA.norm([rx,ry])
            if abs(cosa) > 0.1:
                return False
        return True

    def find_lower_bound(self):
        return 0

    def find_upper_bound(self):
        return 10

    def search_trunpoints(self, l, r):
        pass

    def run(self):
        self.l_bound = self.find_lower_bound()
        self.r_bound = self.find_upper_bound()
        self.search_trunpoints(self.l_bound, self.r_bound)


class PredictingThread(threading.Thread):
    def __init__(self, p_func):
        threading.Thread.__init__(self)
        self.p_func = p_func
        self.pipes = list()
        self.lets_run = False

    def get_pipe(self):
        me, you = MP.Pipe()
        self.pipes.append(me)
        return you

    def start(self):
        self.lets_run = True
        super().start()

    def join(self):
        self.lets_run = False
        super().join()

    def run(self):
        print('start listening')
        x_list = list()
        r_pipes = list()
        while self.lets_run:
            ready = MP.connection.wait(self.pipes, timeout=0.001)
            if not ready:
                continue

            for pipe in ready:
                while pipe.poll():
                    try:
                        x_list.append(pipe.recv())
                        r_pipes.append(pipe)
                    except EOFError as e:
                        pipe.close()

            if len(x_list) == 0:
                continue

            while True:
                x_try = x_list[:BATCH_SIZE]
                p_try = r_pipes[:BATCH_SIZE]

                ys = self.p_func(x_list)
                for pipe, y in zip(p_try, ys):
                    pipe.send(y)

                x_list = x_list[BATCH_SIZE:]
                r_pipes = r_pipes[BATCH_SIZE:]
                if len(x_list) < BATCH_SIZE:
                    break

        print('stop listening')


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
        self.results = list()
        self.manager = MP.Manager()


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
        for k in range(from_k, self.n_child):
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
        return zz


    def _get_predict_function(self, layer_k):
        base_x = self.child[layer_k]['inputs'][0][0]
        input_shape = base_x.shape
        flattened_base_input = base_x.flatten()

        def p_func(x_list):
            feed_x = list()
            for xid, xv in x_list:
                cp_base = flattened_base_input.copy()
                cp_base[xid] = xv
                n_x = cp_base.reshape(input_shape)
                feed_x.append(n_x)
            feed_x = np.asarray(feed_x)
            x_tensor = torch.from_numpy(feed_x)
            y = self._call_partial_model(x_tensor, layer_k)
            return y
        return p_func


    def recall_fn(self, future):
        idx, x_list, y_list = future.result()
        self.results.append((idx,x_list,y_list))


    def analyse_children_k(self, k, init_x, init_y):
        self.hook_activate = False

        init_x = init_x.flatten()
        n = init_x.shape[0]
        rand_idx_list = RD.permutation(n)

        p_func = self._get_predict_function(k)
        pred_thrd = PredictingThread(p_func)
        pipes = self.manager.list([pred_thrd.get_pipe() for _ in range(BATCH_SIZE)])

        pred_thrd.start()

        with ProcessPoolExecutor(max_workers=2) as executor:
            for i in range(10):
                idx = rand_idx_list[i]
                ix = init_x[idx]
                iy = init_y
                ff = executor.submit(process_run, idx, ix, iy, pipes)
                ff.add_done_callback(self.recall_fn)

        pred_thrd.join()


    def analyse(self, dataloader):
        for step, x in enumerate(dataloader):
            imgs = x[0]
            print(LA.norm(imgs))
            print(imgs.shape)
            y_tensor = self.model(imgs.cuda())
            y = y_tensor.cpu().detach().numpy()
            break

        layer_k = 7
        init_x = self.child[layer_k]['inputs'][0][0]
        init_y = y
        self.analyse_children_k(layer_k, init_x, init_y)


def process_run(idx, init_x, init_y, pipes):
    pipe = pipes.pop()
    sna = SingleNeuronAnalyzer(idx, init_x, init_y, pipe)
    sna.run()
    pipes.append(pipe)
    return (idx, sna.x_list, sna.y_list)
