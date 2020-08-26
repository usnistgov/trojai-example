import numpy as np
import numpy.linalg as LA
import numpy.random as RD
import torch
import torch.nn.functional as F
import threading, queue
import time
import numpy as np
from scipy.special import softmax

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as MP


CONSIDER_LAYER_TYPE = ['Conv2d', 'Linear']
BATCH_SIZE = 2
NUM_WORKERS = BATCH_SIZE
EPS = 1e-3
TURNPOINT_TEST_LIMIT=200
NUM_SELECTED_NEURONS=10



def num_to_idx(num, shape):
    n = len(shape)
    loc = [0]*n
    s = [0]*n
    w = 1
    for i in range(n-1,-1,-1):
        s[i] = w
        w *= shape[i]
    for i in range(n):
        loc[i] = num//s[i]
        num %= s[i]
    return tuple(loc)

def idx_to_num(idx, shape):
    n = len(shape)
    loc = [0]*n
    s = [0]*n
    w = 1
    for i in range(n-1,-1,-1):
        s[i] = w
        w *= shape[i]
    w = 0
    for i in range(n):
        w += idx[i]*s[i]
    return w



class SingleNeuronAnalyzer:
    def __init__(self, n_idx, init_x, init_y, pipe, n_classes):
        self.n_idx = n_idx
        self.init_x = init_x
        self.init_y = init_y
        #self.init_y = softmax(self.init_y)
        self.pipe = pipe
        self.n_classes = n_classes

        self.x_list = list()
        self.y_list = list()

        self.turn_x = list()
        self.turn_y = list()

    def _f(self, k, x):
        self.x_list.append(x)
        self.pipe.send((k, self.n_idx, x))
        y = self.pipe.recv()
        #y = softmax(y)
        self.y_list.append(y)
        return y

    def _3p_inline(self, x, y):
        lx = x[1]-x[0]
        rx = x[2]-x[1]
        m = len(y[0])
        for j in range(m):
            ly = y[1][j]-y[0][j]
            ry = y[2][j]-y[1][j]
            '''
            na = LA.norm([lx,ly])
            nb = LA.norm([rx,ry])
            if (na < EPS) or (nb < EPS):
                continue
            #cosa = (lx*rx+ly*ry)/na/nb
            #print([lx, ly, rx, ry,cosa])
            #if abs(cosa) < 1-EPS:
            #    return False
            #sina = (lx*ry-rx*ly)/na/nb
            #if abs(sina) > EPS:
            #    return False
            #'''
            if abs(lx/(lx+rx)*(ly+ry)-ly) > min(EPS*abs(y[0][j]), 1):
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
                if (abs(mx) > 1e8):
                    break
                if not self._3p_inline([lx,mx,rx],[ly,my,ry]):
                    lx, ly = mx, my
                    mx, my = rx, ry
                else:
                    break

            zx, zy = rx, ry
            ck_list_x = [mx,rx]
            ck_list_y = [my,ry]
            if (abs(mx) > 1e8):
                break
            for i in range(5):
                delta *= scale
                zx, zy = zx+delta, self._f(zx+delta)
                ck_list_x.append(zx)
                ck_list_y.append(zy)
            ok = True
            for i in range(1,6):
                if not self._3p_inline(ck_list_x[i-1:i+1+1], ck_list_y[i-1:i+1+1]):
                    ok = False
                    ix, iy = ck_list_x[i], ck_list_y[i]
                    delta = init_delta
                    break

        return ck_list_x[0], ck_list_y[0]


    def _deal_interval(self, lx, ly, rx, ry):
        if (rx-lx < 0.1):
            return

        mx = (lx+rx)/2.0
        my = self._f(mx)
        if self._3p_inline([lx,mx,rx],[ly,my,ry]):
            return
        self._deal_interval(lx,ly,mx,my)
        self._deal_interval(mx,my,rx,ry)


    def search_trunpoints(self):
        #self._deal_interval(self.l_x, self.l_y, self.r_x, self.r_y)
        #return

        for i in range(5):
            x = RD.random()*(self.r_x-self.l_x)+self.l_x
            y = self._f(x)

        n = len(self.x_list)
        idxs = list(range(n))
        idxs = sorted(idxs, key=lambda z: self.x_list[z])
        int_list = list()
        for i in range(1,n-1):
            a, b, c = idxs[i-1], idxs[i], idxs[i+1]
            xx = [self.x_list[a], self.x_list[b], self.x_list[c]]
            yy = [self.y_list[a], self.y_list[b], self.y_list[c]]
            if self._3p_inline(xx,yy):
                continue
            int_list.append((xx[0],xx[1],yy[0],yy[1]))
            int_list.append((xx[1],xx[2],yy[1],yy[2]))


        while len(self.x_list) < 1000:
            new_int = list()
            for x1,x2,y1,y2 in int_list:
                if (x2-x1 < 0.1):
                    continue
                mx = (x1+x2)/2.0
                my = self._f(mx)
                if self._3p_inline([x1,mx,x2],[y1,my,y2]):
                    continue
                new_int.append((x1,mx,y1,my))
                new_int.append((mx,x2,my,y2))
            if len(new_int) == 0:
                break
            int_list = new_int


    def find_peak(self):
        n = len(self.x_list)
        m = len(self.y_list[0])

        init_y = softmax(self.init_y)
        peak = np.zeros_like(self.y_list[0])
        for z in self.y_list:
            sz = softmax(z)
            for j in range(m):
                peak[j] = max(peak[j], sz[j])

        for j in range(m):
            peak[j] = (peak[j]-init_y[j])

        init_lb = np.argmax(init_y)
        peak[init_lb] = 0

        self.peak = peak


    def bf_check(self):
        init_pred = np.argmax(self.init_y, axis=1)
        n = len(self.init_x)
        m = self.n_classes

        pred = list()
        max_v = np.max(self.init_x)
        for i in range(n):
            logits = self._f(i,max_v*10)
            pred.append(np.argmax(logits))

        mat = np.zeros([m,m])
        for i in range(n):
            mat[init_pred[i]][pred[i]] += 1
        for i in range(m):
            mat[i] /= np.sum(mat[i])

        self.peak = np.zeros(m)
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                self.peak[j] = max(self.peak[j], mat[i][j])


    def run(self):
        self.bf_check()
        return self.peak


        self.l_x, self.l_y = self.find_bound(self.init_x, self.init_y, -10)
        self.r_x, self.r_y = self.find_bound(self.init_x, self.init_y, 10)

        print([self.l_x, self.r_x])
        #print([self.l_y, self.r_y])

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
    def __init__(self, model, n_classes=5):
        self.model = model
        self.model.eval()

        self.hook_activate = False

        mds = list(self.model.modules())
        self.model_name = type(mds[0]).__name__
        self.model_name = self.model_name.lower()
        print(self.model_name)

        self.num_selected_neurons = NUM_SELECTED_NEURONS
        self.n_classes = n_classes

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
            if mdname == 'Conv2d':
                md.register_forward_hook(self.get_hook_record(mdname, n_conv))
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
        self.results = list()
        self.manager = MP.Manager()

        self.output_queue = queue.Queue()


    def _build_dict(self):
        rst = dict()
        rst['n'] = 0
        rst['inputs'] = []
        rst['outputs'] = []
        rst['weights'] = []
        rst['modules'] = []
        return rst


    def get_hook(self, k_layer):
        def hook(model, input, output):
            if not self.hook_activate:
                return
            if type(input) is tuple:
                input = input[0]
            if type(output) is tuple:
                output = output[0]
            self.inputs[k_layer].append(input.cpu().detach().numpy())
            self.outputs[k_layer].append(output.cpu().detach().numpy())
        return hook


    '''
    def get_hook_record(self, mdname, idx):
        def hook(model, input, output):
            if not self.hook_activate:
                return
            if type(input) is tuple:
                input = input[0]
            if type(output) is tuple:
                output = output[0]
            self.data['Conv2d']['inputs'][idx].append(input.cpu().detach().numpy())
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

        md = childs[4]
        self.inputs = list()
        self.outputs = list()
        md.register_forward_hook(self.get_hook())

        md_list = list()
        for i in range(4,len(childs)-1):
            md_list.append(childs[i])
        md_list.append(torch.nn.Flatten(1))
        md_list.append(childs[-1])

        _model = torch.nn.Sequential(*md_list)
        self.num_selected_neurons = int(self.num_selected_neurons*0.1)
        return _model


    def _init_general(self):
        childs = list(self.model.children())
        n_child = len(childs)

        self.inputs = list()
        self.outputs = list()
        for i in range(n_child):
            #print(i)
            #print(type(childs[i]).__name__)
            self.inputs.append([])
            self.outputs.append([])

        self.partial_model_params = list()
        for i in range(1, n_child-1):
            childs[i].register_forward_hook(self.get_hook(i))
            self.partial_model_params.append((i, self._partial_general, [i]))


    def _partial_general(self, st_layer):
        childs = list(self.model.children())
        n_child = len(childs)
        md_list = list()
        #print(st_layer)
        for i in range(st_layer,n_child-1):
            md_list.append(childs[i])
            if i==2:
                md_list.append(torch.nn.MaxPool2d(kernel_size=3,stride=2))
            if i==4:
                md_list.append(torch.nn.MaxPool2d(kernel_size=3,stride=2))
            #print(childs[i])
            #print(type(childs[i]).__name__)
        md_list.append(torch.nn.AdaptiveAvgPool2d((1,1)))
        md_list.append(torch.nn.Flatten(1))
        md_list.append(childs[-1])

        _model = torch.nn.Sequential(*md_list)
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
        len_shape = len(input_shape)
        s_size = [0]*len_shape
        w = 1
        for i in range(len_shape-1,-1,-1):
            s_size[i] = w
            w *= input_shape[i]

        def p_func(x_list):
            feed_x = list()
            for k, xid, xv in x_list:
                loc = [0]*len_shape
                for i in range(1,len_shape):
                    loc[i] = xid//s_size[i]
                    xid %= s_size[i]
                cp_base = base_x[k].copy()
                cp_base[tuple(loc[1:])] = xv
                feed_x.append(cp_base)
            feed_x = np.asarray(feed_x)
            x_tensor = torch.from_numpy(feed_x)
            y = self.partial_model(x_tensor.cuda())
            return y.cpu().detach().numpy()
        return p_func


    def forward_from_partial(self, x):
        x_tensor = torch.from_numpy(x)
        y = self.partial_model(x_tensor.cuda())
        return y.cpu().detach().numpy()


    def get_partial_model(self):
        mdname = self.model_name.lower()
        if mdname == 'densenet':
            _f = self._deal_DenseNet()
        elif mdname == 'resnet':
            _f = self._deal_ResNet()
        elif mdname == 'inception3':
            _f = self._partial_Inception3()

        return _f


    def add_to_output_queue(self, out_fn, x_list, y_list):
        n = len(x_list)
        m = len(y_list[0])
        idxs = list(range(n))
        sorted_idxs = sorted(idxs, key=lambda z: x_list[z])

        tt_list = list()
        for idx in sorted_idxs:
            out_tmp = list()
            out_tmp.append(x_list[idx])
            for y in y_list[idx]:
                out_tmp.append(y)
            tt_list.append(out_tmp)

        self.output_queue.put((out_fn, np.asarray(tt_list)))


    def output_func(self):
        while self.output_run:
            try:
                fn, data = self.output_queue.get(timeout=0.1)
                with open(fn,'wb') as f:
                    np.save(f,data)
            except queue.Empty:
                continue


    def start_output_thread(self):
        self.output_run = True
        self._t_output = threading.Thread(target=self.output_func, name='output_thread')
        self._t_output.start()


    def stop_output_thread(self):
        self.output_run = False
        self._t_output.join()


    def recall_fn(self, future):
        idx, data = future.result()
        #idx, data, x_list, y_list = future.result()
        self.results.append((idx,data))
        #out_fn = 'logs/log_neuron_'+str(idx)+'.npy'
        #self.add_to_output_queue(out_fn, x_list, y_list)


    def get_init_values(self, dataloader):
        init_func_name = '_init_'+self.model_name
        if not hasattr(self, init_func_name):
            init_func_name = '_init_general'
        init_f = getattr(self, init_func_name)
        init_f()

        self.hook_activate = True

        acc_ct = 0
        self.pred_init = list()
        self.logits_init = list()
        for step, data in enumerate(dataloader):
            imgs, lbs = data
            lbs = lbs.numpy().squeeze(axis=-1)
            y_tensor = self.model(imgs.cuda())
            logits = y_tensor.cpu().detach().numpy()
            pred = np.argmax(logits,axis=1)

            correct_idx = (lbs==pred)
            acc_ct += sum(correct_idx)

            for i in range(len(self.inputs)):
                if len(self.inputs[i]) == 0:
                    continue
                self.inputs[i][-1] = self.inputs[i][-1][correct_idx]
                self.outputs[i][-1] = self.outputs[i][-1][correct_idx]

            self.pred_init.append(pred[correct_idx])
            self.logits_init.append(logits[correct_idx])

        print(acc_ct/500.0)

        self.hook_activate = False

        n_inputs = len(self.inputs)
        self.inputs_max_v = np.zeros(n_inputs)
        for i in range(n_inputs):
            if len(self.inputs[i]) == 0:
                continue
            self.inputs[i] = np.concatenate(self.inputs[i])
            self.outputs[i] = np.concatenate(self.outputs[i])

        self.pred_init = np.concatenate(self.pred_init)
        self.logits_init = np.concatenate(self.logits_init)

        trim_idx = self.pred_init<0
        ct = [0]*self.n_classes
        for i in range(len(self.pred_init)):
            z = self.pred_init[i]
            if ct[z] < 5:
                trim_idx[i] = True
                ct[z] += 1
        for i in range(n_inputs):
            if len(self.inputs[i]) == 0:
                continue
            self.inputs[i] = self.inputs[i][trim_idx]
            self.outputs[i] = self.outputs[i][trim_idx]
        self.pred_init = self.pred_init[trim_idx]
        self.logits_init = self.logits_init[trim_idx]


    def _check_preds_backdoor(self, v):
        n = v.shape[0]
        m = self.n_classes
        mat = np.zeros((m,m))
        for i in range(n):
            mat[self.pred_init[i]][v[i]] += 1
        for i in range(m):
            mat[i][:] /= np.sum(mat[i][:])

        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                if mat[i][j] > 0.9:
                    return True
        return False


    def analyse(self, dataloader):
        self.get_init_values(dataloader)

        '''
        candi_list = list()
        for k, func, param in self.partial_model_params:
            self.partial_model = func(*param)
            shape = self.inputs[k].shape
            print(shape)
            tn = shape[0]
            max_v = np.max(self.inputs[k])
            for i1 in range(shape[1]):
                for i2 in range(shape[2]):
                    for i3 in range(shape[3]):
                        tv = np.max(self.inputs[k][:,i1,i2,i3])
                        if (abs(tv) < EPS):
                            continue
                        preds = list()
                        for st in range(0,tn, BATCH_SIZE):
                            x = self.inputs[k][st:min(st+BATCH_SIZE,tn)]
                            x[:,i1,i2,i3] = max_v*10
                            y = self.forward_from_partial(x)
                            pred = np.argmax(y, axis=1)
                            preds.append(pred)
                        preds = np.concatenate(preds)
                        if self._check_preds_backdoor(preds):
                            candi_list.append((k,i1,i2,i3))
            print(candi_list)
        exit(0)
        '''

        '''
        k = 0
        for w,o in zip(self.data['Conv2d']['weights'], self.data['Conv2d']['outputs']):
            w = w[0]
            o = np.concatenate(o)
            fn = 'conv{}_'.format(k)
            np.save('output/'+fn+'weights.npy', w)
            np.save('output/'+fn+'outputs.npy', o)
            k += 1
        exit(0)
        '''

        #self.start_output_thread()

        for k, func, param in self.partial_model_params:
            print(k)
            self.partial_model = func(*param)
            init_x = self.inputs[k]
            init_y = self.logits_init

            shape = init_x.shape
            n = 1
            for x in shape:
                n *= x
            n //= shape[0]
            rand_idx_list = RD.permutation(n)

            p_func = self._get_predict_function(init_x)
            pred_thrd = PredictingThread(p_func)
            pipes = self.manager.list([pred_thrd.get_pipe() for _ in range(NUM_WORKERS)])
            pred_thrd.start()

            with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
                for i in range(min(self.num_selected_neurons,n)):
                    idx = rand_idx_list[i]
                    z = [range(shape[0])]
                    z.extend(num_to_idx(idx,shape[1:]))
                    ix = init_x[tuple(z)]
                    iy = init_y
                    ff = executor.submit(process_run, idx, ix, iy, pipes, self.n_classes)
                    ff.add_done_callback(self.recall_fn)

            pred_thrd.join()
        #self.stop_output_thread()

        mxid, mxvl = -1, -1
        for k,z in enumerate(self.results):
            k1 = np.argmax(z[1])
            vk1 = z[1][k1]
            z[1][k1] = 0
            k2 = np.argmax(z[1])
            vk2 = z[1][k2]
            z[1][k1] = vk1
            if (vk1-vk2 > mxvl):
                mxvl = vk1-vk2
                mxid = k
                print(z)
                print(mxvl)

        print(self.results[mxid][0])
        return mxvl


def process_run(idx, init_x, init_y, pipes, n_classes):
    pipe = pipes.pop()
    sna = SingleNeuronAnalyzer(idx, init_x, init_y, pipe, n_classes)
    sna.run()
    pipes.append(pipe)
    return (idx, sna.peak)
    #return (idx, sna.peak, sna.x_list, sna.y_list)
