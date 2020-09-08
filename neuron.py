import numpy as np
import numpy.linalg as LA
import numpy.random as RD
import torch
import torch.nn.functional as F
import threading, queue
import timeit
import numpy as np
from scipy.special import softmax
from sklearn.decomposition import PCA

#from concurrent.futures import ProcessPoolExecutor
#import multiprocessing as MP

from SCAn import *


CONSIDER_LAYER_TYPE = ['Conv2d', 'Linear']
BATCH_SIZE = 64
NUM_WORKERS = BATCH_SIZE
EPS = 1e-3
KEEP_DIM = 64


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
            ''' na = LA.norm([lx,ly]) nb = LA.norm([rx,ry])
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
            logits = self._f(i,max_v*2)
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
    def __init__(self, model, n_classes):
        self.model = model.cuda()
        self.model.eval()
        torch.no_grad()

        self.hook_activate = False

        mds = list(self.model.modules())
        self.model_name = type(mds[0]).__name__
        self.model_name = self.model_name.lower()
        print(self.model_name)

        self.n_classes = n_classes

        self.results = list()
        #self.manager = MP.Manager()

        self.output_queue = queue.Queue()


    def get_record_hook(self, k_layer):
        def hook(model, input, output):
            if type(input) is tuple:
                input = input[0]
            if type(output) is tuple:
                output = output[0]
            self.inputs[k_layer].append(input.cpu().detach().numpy())
            self.outputs[k_layer].append(output.cpu().detach().numpy())
            self.md_child[k_layer] = self.current_child
        return hook


    def get_modify_hook(self, k_layer):
        lr = 0.1
        rr = 0.5
        def hook(model, input, output):
            if k_layer != self.hook_param[0]:
                return
            if type(output) is tuple:
                output = output[0]
            #ori = output.cpu()
            ori = output
            shape = ori.shape
            ratio = rr - (k_layer+1)/len(self.convs)*(rr-lr)

            ns = min(500, int(shape[2]*shape[3]*ratio))
            mask = self.hook_param[3]

            ori[:,self.hook_param[1],:,:] = ori[:,self.hook_param[1],:,:]*(1-mask)+mask*self.hook_param[2]
            #ori[:,self.hook_param[1],:,:] = self.hook_param[2]
            '''
            for z in range(1):
                for i in range(ns):
                    x = np.random.randint(shape[2])
                    y = np.random.randint(shape[3])
                    ori[:,self.hook_param[1],x,y] = self.hook_param[2]
            #'''
            return ori.cuda()
        return hook


    def get_pre_hook(self, k_layer):
        def hook(model, input):
            self.current_child = k_layer
            if type(input) is tuple:
                input = input[0]
            self.child_inputs[k_layer].append(input.cpu().detach().numpy())
        return hook


    def _init_general_md(self):
        mds = list(self.model.modules())

        self.convs = list()
        self.inputs = list()
        self.outputs = list()
        self.conv_weights = list()
        self.md_child = list()
        for md in mds:
            na = type(md).__name__
            if na != 'Conv2d':
                continue
            self.convs.append(md)
            self.inputs.append([])
            self.outputs.append([])
            self.conv_weights.append(md.weight.cpu().detach().numpy())
            self.md_child.append(0)

        self.hook_handles = list()
        for k,md in enumerate(self.convs):
            self.hook_handles.append(md.register_forward_hook(self.get_record_hook(k)))


    def _init_densenet(self):
        self._init_general_md()

        self.child_inputs = list()
        self.current_child = -1
        _childs = list(self.model.children())
        self.childs = list(_childs[0].children())

        self.childs.append(torch.nn.ReLU(inplace=True))
        self.childs.append(torch.nn.AdaptiveAvgPool2d((1,1)))
        self.childs.append(torch.nn.Flatten(1))

        self.childs.append(_childs[1])

        for k,c in enumerate(self.childs):
            print('{} : {}'.format(k, type(c).__name__))
            self.hook_handles.append(c.register_forward_pre_hook(self.get_pre_hook(k)))
            self.child_inputs.append([])


    def _init_squeezenet(self):
        self._init_general_md()

        self.child_inputs = list()
        self.current_child = -1
        childs = list(self.model.children())
        self.childs = list()
        for i in range(len(childs)):
            _name = type(childs[i]).__name__
            if _name == 'Sequential':
                _chs = list(childs[i].children())
                for ch in _chs:
                    self.childs.append(ch)
            else:
                self.childs.append(childs[i])
        self.childs.append(torch.nn.Flatten(1))

        for k,c in enumerate(self.childs):
            print('{} : {}'.format(k, type(c).__name__))
            self.hook_handles.append(c.register_forward_pre_hook(self.get_pre_hook(k)))
            self.child_inputs.append([])


    def _init_general(self):
        self._init_general_md()

        self.child_inputs = list()
        self.current_child = -1
        childs = list(self.model.children())
        self.childs = list()
        for i in range(len(childs)-1):
            _name = type(childs[i]).__name__
            if _name == 'Sequential':
                _chs = list(childs[i].children())
                for ch in _chs:
                    self.childs.append(ch)
            else:
                self.childs.append(childs[i])
        if type(self.childs[-1]).__name__ == 'Dropout' and type(self.childs[-2]).__name__ == 'AdaptiveAvgPool2d':
            pass
        elif type(self.childs[-1]).__name__ == 'AdaptiveAvgPool2d':
            pass
        else:
            self.childs.append(torch.nn.AdaptiveAvgPool2d((1,1)))
        self.childs.append(torch.nn.Flatten(1))
        self.childs.append(childs[-1])

        for k,c in enumerate(self.childs):
            print('{} : {}'.format(k, type(c).__name__))
            self.hook_handles.append(c.register_forward_pre_hook(self.get_pre_hook(k)))
            self.child_inputs.append([])


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
        k, idx, data = future.result()
        #idx, data, x_list, y_list = future.result()
        self.results.append((k, idx,data))
        #out_fn = 'logs/log_neuron_'+str(idx)+'.npy'
        #self.add_to_output_queue(out_fn, x_list, y_list)


    def get_init_values(self, dataloader):
        init_func_name = '_init_'+self.model_name
        if not hasattr(self, init_func_name):
            init_func_name = '_init_general'
        init_f = getattr(self, init_func_name)

        acc_ct = 0
        tot_n = 0
        self.images = list()
        self.pred_init = list()
        self.logits_init = list()
        for step, data in enumerate(dataloader):
            imgs, lbs = data
            tot_n += len(lbs)
            lbs = lbs.numpy().squeeze(axis=-1)
            y_tensor = self.model(imgs.cuda())
            logits = y_tensor.cpu().detach().numpy()
            pred = np.argmax(logits,axis=1)

            correct_idx = (lbs==pred)
            acc_ct += sum(correct_idx)

            self.pred_init.append(pred[correct_idx])
            self.logits_init.append(logits[correct_idx])
            np_imgs = imgs.numpy()
            self.images.append(np_imgs[correct_idx])

        print(acc_ct/tot_n)

        self.pred_init = np.concatenate(self.pred_init)
        self.logits_init = np.concatenate(self.logits_init)
        self.images = np.concatenate(self.images)

        self.preds_all = self.pred_init
        self.logits_all = self.logits_init
        self.images_all = self.images

        init_f()
        self._run_once_epoch(self.images)
        print(self.md_child)

        out_channel_sum = 0
        for i in range(len(self.inputs)):
            self.inputs[i] = np.concatenate(self.inputs[i])
            self.outputs[i] = np.concatenate(self.outputs[i])
            out_channel_sum += self.outputs[i].shape[1]
        print('total channels '+str(out_channel_sum))

        for i in range(len(self.childs)):
            if (len(self.child_inputs[i]) < 1):
                continue
            self.child_inputs[i] = np.concatenate(self.child_inputs[i])


        #trim samples
        trim_idx = self.pred_init<0
        ct = [0]*self.n_classes
        for i in range(len(self.pred_init)):
            z = self.pred_init[i]
            if ct[z] < 5:
                trim_idx[i] = True
                ct[z] += 1
        print(ct)
        self.pred_init = self.pred_init[trim_idx]
        self.logits_init = self.logits_init[trim_idx]
        self.images = self.images[trim_idx]
        #for i in range(len(self.inputs)):
        #    self.inputs[i] = self.inputs[i][trim_idx]
        #    self.outputs[i] = self.outputs[i][trim_idx]

        for i in range(len(self.childs)):
            if (len(self.child_inputs[i]) < 1):
                continue
            self.child_inputs[i] = self.child_inputs[i][trim_idx]


        for handle in self.hook_handles:
            handle.remove()


    def register_representation_record_hook(self):
        childs = list(self.model.children())
        md = childs[-1]

        def hook(model, input, output):
            if not self.record_reprs:
                return
            if type(input) is tuple:
                input = input[0]
            self.reprs.append(input.cpu().detach().numpy())

        md.register_forward_hook(hook)


    def _get_partial_model(self, st_child):
        childs = list()
        for k in range(st_child, len(self.childs)):
            childs.append(self.childs[k])
        model = torch.nn.Sequential(*childs)
        return model


    def _run_once_epoch(self, images_all, st_child=-1):
        preds = list()
        logits = list()

        if st_child < 0:
            model = self.model
        else:
            model = self._get_partial_model(st_child)
        model.eval()
        torch.no_grad()

        tn = len(images_all)
        for st in range(0,tn, BATCH_SIZE):
            imgs = images_all[st:min(st+BATCH_SIZE,tn)]
            imgs_tensor = torch.from_numpy(imgs)
            y_tensor = model(imgs_tensor.cuda())
            y = y_tensor.cpu().detach().numpy()
            pred = np.argmax(y, axis=1)
            preds.append(pred)
            logits.append(y)

        preds = np.concatenate(preds)
        logits = np.concatenate(logits)

        return (logits, preds)


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
                    return (i,j), mat

        return None, mat


    def score_to_prob(self, score):
        print(score)
        a = 1.0
        b = -5
        z = np.log(score)
        w = np.tanh(a*z+b)
        w = w/2.0+0.5
        return 1-w


    def check_neuron(self, param, pair):
        select_idx = (self.preds_all == pair[0])
        self.hook_param = param

        self.reprs = list()
        self.record_reprs = True
        logits, preds = self._run_once_epoch(self.images_all[select_idx])
        self.record_reprs = False
        pos_repr = np.concatenate(self.reprs)

        att_acc = np.sum(preds==pair[1])
        #print(param[0:2], pair,att_acc/len(preds))
        #if att_acc < len(preds)-1:
        #    return 0

        reprs = np.concatenate([self.good_repr, pos_repr])
        labels = np.concatenate([self.preds_all, preds])
        pca = PCA(n_components=KEEP_DIM, svd_solver='full')
        pca.fit(reprs)

        gb_model = self.dealer.build_global_model(pca.transform(self.good_repr), self.preds_all, self.n_classes)
        lc_model = self.dealer.build_local_model(pca.transform(reprs), labels, gb_model, self.n_classes)
        sc = self.dealer.calc_final_score(lc_model)
        print(param[0:2], pair,att_acc/len(preds), sc)
        return sc[pair[1]]


    def analyse(self, dataloader):
        self.get_init_values(dataloader)

        self.hook_param = (-1,-1,-1)
        for k,md in enumerate(self.convs):
            md.register_forward_hook(self.get_modify_hook(k))

        self.record_reprs = True
        self.register_representation_record_hook()
        self.reprs = list()
        self._run_once_epoch(self.images_all)
        self.record_reprs = False
        self.good_repr = np.concatenate(self.reprs)

        start = timeit.default_timer()

        run_ct = 0
        candi_list = list()
        self.dealer = SCAn()
        sc_list = list()
        for k, md in enumerate(self.convs):
            if k*2 > len(self.convs):
                break
            shape = self.outputs[k].shape
            tn = shape[0]
            max_v = np.max(self.outputs[k])

            this_candi = list()
            for i in range(shape[1]):
                run_ct += 1
                tv = np.max(self.outputs[k][:,i, :, :])

                '''
                max_in_theory = 0
                for j in range(self.inputs[k].shape[1]):
                    w = np.max(self.inputs[k][:,j,:,:])
                    w = 1
                    max_in_theory += w * np.sum(np.abs(self.conv_weights[k][i,j,:,:]))

                #print(tv, max_in_theory, max_v)
                '''

                if (tv < EPS):
                    continue

                mask = np.random.rand(shape[2], shape[3])
                ratio = min(0.5, 100/(shape[2]*shape[3]))
                mask = mask<ratio
                mask = mask.astype(np.float32)
                mask_tensor = torch.from_numpy(mask)
                self.hook_param = (k, i, max_v*1.0, mask_tensor.cuda())

                st_child = self.md_child[k]
                logits, preds = self._run_once_epoch(self.child_inputs[st_child], st_child)
                #logits, preds = self._run_once_epoch(self.images)

                '''
                print(self.logits_init.shape)
                print(logits.shape)
                print(self.logits_init[0])
                print(logits[0])
                exit(0)
                '''

                pair, mat = self._check_preds_backdoor(preds)
                if pair is not None:
                    #print(mat)
                    candi_list.append((self.hook_param, pair))
                    this_candi.append((self.hook_param, pair))

            for param, pair in this_candi:
                sc_list.append(self.check_neuron(param, pair))

            '''
            mxv_list = list()
            for tgt in range(self.n_classes):
                idx = (self.preds_all==tgt)
                z = np.max(self.outputs[k][idx], axis=(0,2,3))
                mxv_list.append(z)

            for param, pair in this_candi:
                print(pair, (mxv_list[pair[1]][param[1]]/max_v, mxv_list[pair[0]][param[1]]/max_v))
            print('Conv {}: {}, n candi: {}'.format(k,shape,len(candi_list)))
            '''

            if len(sc_list) > 1000:
                break
            if run_ct > 10000:
                break
            if k > 10:
                break

        stop = timeit.default_timer()
        print('Time used to select neuron: ', stop-start)

        if len(sc_list) == 0:
            return 0

        return self.score_to_prob(max(sc_list))


def process_run(k, idx, init_x, init_y, pipes, n_classes):
    pipe = pipes.pop()
    sna = SingleNeuronAnalyzer(idx, init_x, init_y, pipe, n_classes)
    sna.run()
    pipes.append(pipe)
    print((k, idx, sna.peak))
    return (k, idx, sna.peak)
    #return (idx, sna.peak, sna.x_list, sna.y_list)
