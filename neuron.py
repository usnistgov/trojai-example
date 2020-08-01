import numpy as np
import numpy.linalg as LA
import numpy.random as RD
import torch
import threading, queue
import time

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as MP


CONSIDER_LAYER_TYPE = ['Conv2d', 'Linear']
BATCH_SIZE = 32
NUM_WORKERS = BATCH_SIZE*2
EPS = 1e-3
TURNPOINT_TEST_LIMIT=200
NUM_SELECTED_NEURONS=1000


class SingleNeuronAnalyzer:
    def __init__(self, n_idx, init_x, init_y, pipe):
        self.n_idx = n_idx
        self.init_x = init_x
        self.init_y = init_y
        self.pipe = pipe
        self.x_list = list()
        self.y_list = list()

    def _f(self, x):
        self.x_list.append(x)
        self.pipe.send((self.n_idx, x))
        y = self.pipe.recv()
        self.y_list.append(y)
        return y

    def _3p_inline(self, x, y):
        lx = x[1]-x[0]
        rx = x[2]-x[1]
        m = y[0].shape[0]
        for j in range(m):
            ly = y[1][j]-y[0][j]
            ry = y[2][j]-y[1][j]
            cosa = (lx*ry-rx*ly)/LA.norm([lx,ly])/LA.norm([rx,ry])
            #print([lx, ly, rx, ry,cosa])
            if abs(cosa) > EPS:
                return False
        return True

    def find_lowerest_turn(self, l_x, l_y, r_x, r_y):
        if r_x < l_x+EPS:
            return l_x, l_y

        lx, ly = l_x, l_y
        rx, ry = r_x, r_y
        while lx+EPS < rx:
            mx = (lx+rx)/2.0
            my = self._f(mx)
            if not self._3p_inline([lx,mx,rx],[ly,my,ry]):
                rx, ry = mx, my
            else:
                break
        return rx, ry

    def find_upper_bound(self):
        delta = 1.1
        lx, ly = self.init_x, self.init_y
        if lx > EPS:
            mx, my = lx*delta, self._f(lx*delta)
        else:
            mx, my = delta, self._f(delta)
        while True:
            rx, ry = mx*delta, self._f(mx*delta)
            if not self._3p_inline([lx,mx,rx],[ly,my,ry]):
                lx, ly = mx, my
                mx, my = rx, ry
            else:
                break
        return lx, ly

    def search_trunpoints(self):
        nt_limit = TURNPOINT_TEST_LIMIT
        nt_x = self.l_x
        nt_y = self.l_y
        k_nt = 0
        rst = [nt_x]
        while nt_x < self.r_x-EPS:
            nt_x, nt_y = self.find_lowerest_turn(nt_x, nt_y, self.r_x, self.r_y)
            rst.append(nt_x)
            k_nt += 1
            if k_nt >= nt_limit:
                break
        if k_nt >= nt_limit:
            rst.append(self.r_x)
        #print(rst)

    def find_peak(self):
        n = len(self.x_list)
        m = self.y_list[0].shape[0]
        peak = self.y_list[0].copy()
        idxs = list(range(n))
        sorted_idxs = sorted(idxs, key=lambda z: self.x_list[z])
        #print(sorted_idxs)
        y_list = self.y_list
        for j in range(m):
            peak[j] = 0
            for i in range(1,n-1):
                a, b, c = sorted_idxs[i-1:i+1+1]
                ld = y_list[b][j]-y_list[a][j]
                rd = y_list[b][j]-y_list[c][j]
                if ld > 0 and rd > 0:
                    peak[j] = max(peak[j],max(ld, rd))

        self.peak = peak


    def run(self):
        self.l_x, self.l_y = self.find_lowerest_turn(0, self._f(0), self.init_x, self.init_y)
        self.r_x, self.r_y = self.find_upper_bound()
        self.search_trunpoints()
        self.find_peak()
        return self.peak


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

            #print(len(x_list))

            while True:
                x_try = x_list[:BATCH_SIZE]
                p_try = r_pipes[:BATCH_SIZE]

                ys = self.p_func(x_try)
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
        idx, data = future.result()
        self.results.append((idx,data))


    def analyse_children_k(self, k, init_x, init_y):
        self.hook_activate = False

        init_x = init_x.flatten()
        n = init_x.shape[0]
        rand_idx_list = RD.permutation(n)

        p_func = self._get_predict_function(k)
        pred_thrd = PredictingThread(p_func)
        pipes = self.manager.list([pred_thrd.get_pipe() for _ in range(NUM_WORKERS)])

        pred_thrd.start()

        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            for i in range(NUM_SELECTED_NEURONS):
                idx = rand_idx_list[i]
                ix = init_x[idx]
                iy = init_y
                ff = executor.submit(process_run, idx, ix, iy, pipes)
                ff.add_done_callback(self.recall_fn)

        pred_thrd.join()
        #print(self.results)


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
        init_y = y[0]
        self.analyse_children_k(layer_k, init_x, init_y)

        max_peaks = []
        for z in self.results:
            max_peaks.append(np.max(z[1]))
        return max(max_peaks)


def process_run(idx, init_x, init_y, pipes):
    pipe = pipes.pop()
    sna = SingleNeuronAnalyzer(idx, init_x, init_y, pipe)
    sna.run()
    pipes.append(pipe)
    return (idx, sna.peak)
