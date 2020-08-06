import numpy as np
import numpy.linalg as LA
import numpy.random as RD
import torch
import torch.nn.functional as F
import threading, queue
import time

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as MP


CONSIDER_LAYER_TYPE = ['Conv2d', 'Linear']
BATCH_SIZE = 32
NUM_WORKERS = BATCH_SIZE
EPS = 1e-4
TURNPOINT_TEST_LIMIT=200
NUM_SELECTED_NEURONS=10000


class SingleNeuronAnalyzer:
    def __init__(self, n_idx, init_x, init_y, pipe):
        self.out_fn = 'logs/log_neuron_'+str(n_idx)+'.txt'

        self.n_idx = n_idx
        self.init_x = init_x
        self.init_y = init_y
        self.pipe = pipe
        self.x_list = list()
        self.y_list = list()

        self.turn_x = list()
        self.turn_y = list()

    def _f(self, x):
        self.x_list.append(x)
        self.pipe.send((self.n_idx, x))
        y = self.pipe.recv()
        self.y_list.append(y)
        return y

    def _3p_inline(self, x, y):
        lx = x[1]-x[0]
        rx = x[2]-x[1]
        m = len(y[0])
        for j in range(m):
            ly = y[1][j]-y[0][j]
            ry = y[2][j]-y[1][j]
            na = LA.norm([lx,ly])
            nb = LA.norm([rx,ry])
            if (na < EPS) or (nb < EPS):
                continue
            #cosa = (lx*rx+ly*ry)/na/nb
            #print([lx, ly, rx, ry,cosa])
            #if abs(cosa) < 1-EPS:
            #    return False
            sina = (lx*ry-rx*ly)/na/nb
            if abs(sina) > EPS:
                return False
        return True

    def find_lowerest_turn(self, l_x, l_y, r_x, r_y):
        if r_x < l_x+EPS:
            return l_x, l_y

        lx, ly = l_x, l_y
        rx, ry = r_x, r_y
        while lx+EPS < rx:
            dd = rx-lx;
            p1_x = lx+dd*1.0/3.0
            p1_y = self._f(p1_x)
            p2_x = lx+dd*2.0/3.0
            p2_y = self._f(p2_x)


            if self._3p_inline([lx,p1_x,p2_x],[ly,p1_y,p2_y]):
                lx, ly = p2_x, p2_y
            elif self._3p_inline([p1_x,p2_x,rx],[p1_y,p2_y,ry]):
                rx, ry = p1_x, p1_y
            else:
                rx, ry = p2_x, p2_y
        return rx, ry

    def find_bound(self, init_x, init_y, init_delta):
        scale = 1.1
        delta = init_delta
        ix, iy = init_x, init_y
        ok = False

        while not ok:
            lx, ly = ix, iy
            mx, my = lx+delta, self._f(lx+delta)
            while True:
                delta *= scale
                rx, ry = mx+delta, self._f(mx+delta)
                if not self._3p_inline([lx,mx,rx],[ly,my,ry]):
                    lx, ly = mx, my
                    mx, my = rx, ry
                else:
                    break

            zx, zy = rx, ry
            ck_list_x = [mx,rx]
            ck_list_y = [my,ry]
            for i in range(5):
                delta *= scale
                zx, zy = zx+delta, self._f(zx+delta)
                ck_list_x.append(zx)
                ck_list_y.append(zy)
            ok = True
            for i in range(1,6):
                if not self._3p_inline(ck_list_x[i-1:i+1+1], ck_list_y[i-1:i+1+1]):
                    ok = False
                    break

            ix, iy = lx, ly
            delta = init_delta


        return ix, iy

    def _deal_interval(self, lx, ly, rx, ry):
        if (rx-lx < 0.1):
            return

        #print([lx,rx])

        mx = (lx+rx)/2.0
        my = self._f(mx)
        if self._3p_inline([lx,mx,rx],[ly,my,ry]):
            return
        self._deal_interval(lx,ly,mx,my)
        self._deal_interval(mx,my,rx,ry)

    def search_trunpoints(self):
        self._deal_interval(self.l_x, self.l_y, self.r_x, self.r_y)
        return

        nt_limit = TURNPOINT_TEST_LIMIT
        nt_x = self.l_x
        nt_y = self.l_y
        k_nt = 0

        self.turn_x.append(self.x_list[0])
        self.turn_y.append(self.y_list[0])
        self.turn_x.append(nt_x)
        self.turn_y.append(nt_y)
        while nt_x < self.r_x-EPS:
            nt_x, nt_y = self.find_lowerest_turn(nt_x, nt_y, self.r_x, self.r_y)
            self.turn_x.append(nt_x)
            self.turn_y.append(nt_y)
            k_nt += 1
            if k_nt >= nt_limit:
                break
        if k_nt >= nt_limit:
            self.turn_x.append(self.r_x)
            self.turn_y.append(self.r_y)
        #print(rst)

    def find_peak(self):
        n = len(self.x_list)
        m = len(self.y_list[0])
        peak = self.y_list[0].copy()
        idxs = list(range(n))
        sorted_idxs = sorted(idxs, key=lambda z: self.x_list[z])
        #print(sorted_idxs)
        #turn_y = self.turn_y
        list_x = self.x_list
        list_y = self.y_list
        for j in range(m):
            turn_y = list()
            turn_x = list()
            for i in range(1,n-1):
                a, b, c = sorted_idxs[i-1:i+1+1]
                xx = [list_x[a], list_x[b], list_x[c]]
                yy = [[list_y[a][j]],[list_y[b][j]],[list_y[c][j]]]
                if self._3p_inline(xx,yy):
                    continue
                turn_y.append(list_y[b][j])
                turn_x.append(list_y[b])

            peak[j] = 0
            nt = len(turn_y)

            print(nt)

            for i in range(1,nt-1):
                ld = turn_y[i]-turn_y[i-1]
                rd = turn_y[i+1]-turn_y[i]
                if ld > 0 and rd > 0:
                    peak[j] = max(peak[j],max(ld, rd))

        for j in range(m):
            peak[j] /= abs(self.init_y[j])

        self.peak = peak


    def output(self):
        n = len(self.x_list)
        m = len(self.y_list[0])
        idxs = list(range(n))
        sorted_idxs = sorted(idxs, key=lambda z: self.x_list[z])
        with open(self.out_fn,'w') as f:
            f.write('{} {}\n'.format(n,m))
            for idx in sorted_idxs:
                out_tmp = list()
                out_tmp.append(self.x_list[idx])
                for y in self.y_list[idx]:
                    out_tmp.append(y)
                f.write(' '.join([str(z) for z in out_tmp])+'\n')



    def run(self):
        self.l_x, self.l_y = self.find_bound(self.init_x, self.init_y, -1000)
        self.r_x, self.r_y = self.find_bound(self.init_x, self.init_y, 1000)

        #print([self.l_x, self.r_x])
        #print([self.l_y, self.r_y])

        self.search_trunpoints()
        self.output()
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

        self.hook_activate = False

        mds = list(self.model.modules())
        self.model_name = type(mds[0]).__name__
        print(self.model_name)

        self.num_selected_neurons = NUM_SELECTED_NEURONS

        '''
        self.childrens = list(model.children())
        self.n_child = len(self.childrens)
        self.child = list()
        for k,md in enumerate(self.childrens):
            self.child.append(self._build_dict())
            md.register_forward_hook(self.get_children_hook(k))
        '''

        '''
        self.data = dict()
        for tp in CONSIDER_LAYER_TYPE:
            self.data[tp] = self._build_dict()

        mds = list(model.modules())
        n_conv = 0
        for k, md in enumerate(mds):
            mdname = type(md).__name__
            if k==0:
                self.model_name = mdname
                print(mdname)
                break
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

                #print(self.data[mdname]['modules'][-1])
                #print(self.data[mdname]['weights'][-1][0].shape)
                #exit(0)
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


    def get_hook(self):
        def hook(model, input, output):
            if not self.hook_activate:
                return
            if type(input) is tuple:
                input = input[0]
            if type(output) is tuple:
                output = output[0]
            self.inputs.append(input.cpu().detach().numpy())
            self.outputs.append(output.cpu().detach().numpy())
        return hook


    '''
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
    '''


    '''
    def _call_partial_model(self, x_tensor, from_k):
        x = x_tensor.cuda()
        for k in range(from_k, self.n_child):
            md = self.childrens[k]
            if k+1 == self.n_child:
                x = torch.flatten(x,1)
            x = md(x)

        return x.cpu().detach().numpy()
    '''

    def _deal_DenseNet(self):
        childs = list(self.model.children())
        main_ch = childs[0]
        _childs = list(main_ch.children())
        md = _childs[-2]

        self.inputs = list()
        self.outputs = list()
        md.register_forward_hook(self.get_hook())

        _model = torch.nn.Sequential(_childs[-2],
                                     _childs[-1],
                                     torch.nn.ReLU(inplace=True),
                                     torch.nn.AdaptiveAvgPool2d((1,1)),
                                     torch.nn.Flatten(1),
                                     childs[-1])
        self.num_selected_neurons = int(self.num_selected_neurons*1.0)
        return _model


    def _deal_ResNet(self):
        childs = list(self.model.children())
        main_ch = childs[-3]
        _childs = list(main_ch.children())
        md = _childs[-1]

        self.inputs = list()
        self.outputs = list()
        md.register_forward_hook(self.get_hook())
        _model = torch.nn.Sequential(md,
                                     childs[-2],
                                     torch.nn.Flatten(1),
                                     childs[-1])
        self.num_selected_neurons = int(self.num_selected_neurons*0.1)
        return _model

        '''
        md = childs[-3]
        _model = torch.nn.Sequential(childs[-3],
                                     childs[-2],
                                     torch.nn.Flatten(1),
                                     childs[-1])
        return _model
        '''

    def _deal_Inception3(self):
        childs = list(self.model.children())
        md = childs[-2]

        self.inputs = list()
        self.outputs = list()
        md.register_forward_hook(self.get_hook())

        _model = torch.nn.Sequential(childs[-2],
                                     torch.nn.AdaptiveAvgPool2d((1,1)),
                                     torch.nn.Flatten(1),
                                     childs[-1])
        self.num_selected_neurons = int(self.num_selected_neurons*0.1)
        return _model

    '''
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
    '''


    def _get_predict_function(self, base_x):
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
            y = self.partial_model(x_tensor.cuda())
            return y.cpu().detach().numpy()
        return p_func


    def get_partial_model(self):
        mdname = self.model_name.lower()
        if mdname == 'densenet':
            _f = self._deal_DenseNet()
        elif mdname == 'resnet':
            _f = self._deal_ResNet()
        elif mdname == 'inception3':
            _f = self._deal_Inception3()

        return _f


    def recall_fn(self, future):
        idx, data = future.result()
        self.results.append((idx,data))


    def analyse(self, dataloader):
        self.partial_model = self.get_partial_model()
        self.hook_activate = True
        for step, x in enumerate(dataloader):
            imgs = x[0]
            print(LA.norm(imgs))
            print(imgs.shape)
            y_tensor = self.model(imgs.cuda())
            y = y_tensor.cpu().detach().numpy()
            break

        self.hook_activate = False

        init_x = self.inputs[0][0]
        init_y = y[0]

        print(init_x.shape)

        flattened_x = init_x.flatten()
        n = len(flattened_x)
        rand_idx_list = RD.permutation(n)

        p_func = self._get_predict_function(init_x)
        pred_thrd = PredictingThread(p_func)
        pipes = self.manager.list([pred_thrd.get_pipe() for _ in range(NUM_WORKERS)])

        pred_thrd.start()

        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            for i in range(min(self.num_selected_neurons,n)):
                idx = rand_idx_list[i]
                ix = flattened_x[idx]
                iy = init_y
                ff = executor.submit(process_run, idx, ix, iy, pipes)
                ff.add_done_callback(self.recall_fn)

        pred_thrd.join()

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
