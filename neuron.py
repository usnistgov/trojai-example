import numpy as np
import numpy.linalg as LA
import numpy.random as RD
import torch
import torch.nn.functional as F
import threading, queue
import timeit
import numpy as np
import pickle
import os
from scipy.special import softmax
from sklearn.decomposition import PCA
import sklearn
import math


#from concurrent.futures import ProcessPoolExecutor
#import multiprocessing as MP

from SCAn import *
import pytorch_ssim

from torch.autograd import Variable
from decimal import Decimal
import utils


RELEASE = True

CONSIDER_LAYER_TYPE = ['Conv2d', 'Linear']
if RELEASE:
    BATCH_SIZE = 32
    SVM_FOLDER = '/svm_models'
else:
    BATCH_SIZE = 32
    SVM_FOLDER = 'svm_models'
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
        #torch.no_grad()

        self.hook_activate = False

        mds = list(self.model.modules())

        self.model_name = type(mds[0]).__name__
        self.model_name = self.model_name.lower()
        print(self.model_name)

        self.n_classes = n_classes

        self.results = list()

        #self.manager = MP.Manager()
        #self.output_queue = queue.Queue()

        self.batch_size = BATCH_SIZE


    def get_record_hook(self, k_layer):
        def hook(model, input, output):
            if type(input) is tuple:
                input = input[0]
            if type(output) is tuple:
                output = output[0]
            if not self.record_conv:
                return
            self.inputs[k_layer].append(input.cpu().detach().numpy())
            self.outputs[k_layer].append(output.cpu().detach().numpy())
            self.md_child[k_layer] = self.current_child
        return hook


    def get_modify_hook(self, k_layer):
        def hook(model, input, output):
            if k_layer != self.hook_param[0]:
                return
            if type(output) is tuple:
                output = output[0]

            chnn = self.hook_param[1]
            v = self.hook_param[2]

            ori = output
            ori[:,chnn,:,:] = v

            return ori.cuda()
        return hook


    def get_pre_hook(self, k_layer):
        def hook(model, input):
            self.current_child = k_layer
            if type(input) is tuple:
                input = input[0]
            if not self.record_child:
                return
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

        print(self.model_name, len(self.convs), 'convs')

        self.record_conv = False
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

        self.record_child = False
        for k,c in enumerate(self.childs):
            #print('{} : {}'.format(k, type(c).__name__))
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

        self.record_child = False
        for k,c in enumerate(self.childs):
            #print('{} : {}'.format(k, type(c).__name__))
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
        _name = type(childs[-1]).__name__
        if _name == 'Sequential':
            _chs = list(childs[-1].children())
            for ch in _chs:
                self.childs.append(ch)
        else:
            self.childs.append(childs[-1])

        self.record_child = False
        for k,c in enumerate(self.childs):
            #print('{} : {}'.format(k, type(c).__name__))
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
        self.raw_images = list()
        self.pred_init = list()
        self.logits_init = list()
        for step, data in enumerate(dataloader):
            raw_imgs, lbs = data
            np_raw_imgs = raw_imgs.numpy()

            #normalize to [0,1]
            np_imgs = np_raw_imgs-np_raw_imgs.min((1,2,3), keepdims=True)
            np_imgs = np_imgs/np_imgs.max((1,2,3), keepdims=True)

            tot_n += len(lbs)
            lbs = lbs.numpy().squeeze(axis=-1)

            imgs_tensor = torch.from_numpy(np_imgs)
            y_tensor = self.model(imgs_tensor.cuda())
            logits = y_tensor.cpu().detach().numpy()
            pred = np.argmax(logits,axis=1)

            correct_idx = (lbs==pred)
            acc_ct += sum(correct_idx)

            self.pred_init.append(pred[correct_idx])
            self.logits_init.append(logits[correct_idx])
            self.images.append(np_imgs[correct_idx])
            self.raw_images.append(np_raw_imgs[correct_idx])

        print(acc_ct/tot_n)

        self.pred_init = np.concatenate(self.pred_init)
        self.logits_init = np.concatenate(self.logits_init)
        self.images = np.concatenate(self.images)
        self.raw_images = np. concatenate(self.raw_images)

        self.preds_all = self.pred_init
        self.logits_all = self.logits_init
        self.images_all = self.images

        cat_cnt = [0]*self.n_classes
        for lb in range(self.n_classes):
            cat_cnt[lb] = np.sum(self.preds_all == lb)
        for lb in range(self.n_classes):
            cat_cnt[lb] = math.ceil(0.1*cat_cnt[lb])*2+1

        init_f()
        self.arch_name = self.get_arch_name()
        print('model architecture:', self.arch_name)
        last_md = self.childs[-1]
        if type(last_md).__name__ == 'Linear':
            self.n_classes = last_md.out_features
        else:
            self.n_classes = self.convs[-1].out_channels
        print('n_classes',self.n_classes)

        if self.arch_name in ['wideresnet101','resnet152','densenet161','densenet169','densenet201']:
            self.batch_size //= 2

        self.record_conv = True
        self.record_child = False
        self._run_once_epoch(self.images)
        print(self.md_child)

        self._extract_statistical_data()

        #'''
        #trim samples
        trim_idx = self.pred_init<0
        ct = [0]*self.n_classes
        for i in range(len(self.pred_init)):
            lb = self.pred_init[i]
            if ct[lb] < cat_cnt[lb]:
                trim_idx[i] = True
                ct[lb] += 1
        print('examples count', ct)
        self.pred_init = self.pred_init[trim_idx]
        self.logits_init = self.logits_init[trim_idx]
        self.images = self.images[trim_idx]
        #'''

        self.record_conv = True
        self.record_child = True
        self._run_once_epoch(self.images)

        for i in range(len(self.inputs)):
            self.inputs[i] = np.concatenate(self.inputs[i])
            self.outputs[i] = np.concatenate(self.outputs[i])
        for i in range(len(self.childs)):
            if (len(self.child_inputs[i]) < 1):
                continue
            self.child_inputs[i] = np.concatenate(self.child_inputs[i])
            #print(i, self.child_inputs[i].shape)

        for handle in self.hook_handles:
            handle.remove()
        self.record_conv = False
        self.record_child = False



    def _extract_statistical_data(self):
        lb_idx = list()
        for lb in range(self.n_classes):
            lb_idx.append(self.preds_all==lb)

        self.lb_channel_max = list()
        self.channel_max = list()
        self.channel_min = list()
        self.channel_std = list()
        self.channel_mean = list()
        self.tensor_max = list()
        self.tensor_min = list()
        self.tensor_mean = list()
        out_channel_sum = 0
        self.channel_in_max = list()
        for i in range(len(self.outputs)):
            ttmp = list()
            mtmp = list()
            ztmp = list()
            for ot in self.outputs[i]:
                tmat = np.max(ot,(2,3))
                mmat = np.mean(ot,(2,3))
                zmat = np.min(ot,(2,3))
                ttmp.append(tmat)
                mtmp.append(mmat)
                ztmp.append(zmat)
            ttmp = np.concatenate(ttmp)
            mtmp = np.concatenate(mtmp)
            ztmp = np.concatenate(ztmp)

            max_lb = list()
            for lb in range(self.n_classes):
                max_lb.append(np.max(ttmp[lb_idx[lb]],0))
            self.lb_channel_max.append(max_lb)

            self.channel_max.append(np.max(ttmp,0))
            self.channel_min.append(np.min(ztmp,0))
            self.channel_mean.append(np.mean(mtmp,0))

            self.tensor_max.append(np.max(ttmp))
            self.tensor_min.append(np.min(ztmp))
            self.tensor_mean.append(np.mean(np.max(ttmp,1)))
            n_chnn = ttmp.shape[1]
            out_channel_sum += n_chnn

            itmp = list()
            for it in self.inputs[i]:
                imat = np.max(it,(2,3))
                itmp.append(imat)
            itmp = np.concatenate(itmp)
            self.channel_in_max.append(np.max(itmp,0))

            self.inputs[i] = []
            self.outputs[i] = []
        print('total channels '+str(out_channel_sum))



    def register_representation_record_hook(self):
        childs = list(self.model.children())
        md = childs[-1]
        #if self.model_name == 'squeezenet':
        #    md = self.childs[-2]
        #    print(self.convs[-1])

        def hook(model, input, output):
            if not self.record_reprs:
                return
            if type(input) is tuple:
                input = input[0]
            self.reprs.append(input.cpu().detach().numpy())

        md.register_forward_hook(hook)


    def _get_partial_model(self, st_child, ed_child=None):
        if ed_child is None:
            ed_child = len(self.childs)
        childs = list()
        for k in range(st_child, ed_child):
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
        #torch.no_grad()

        tn = len(images_all)
        for st in range(0,tn, self.batch_size):
            imgs = images_all[st:min(st+self.batch_size,tn)]
            imgs_tensor = torch.from_numpy(imgs)
            y_tensor = model(imgs_tensor.cuda())
            y = y_tensor.cpu().detach().numpy()
            pred = np.argmax(y, axis=1)
            preds.append(pred)
            logits.append(y)

        preds = np.concatenate(preds)
        logits = np.concatenate(logits)

        return (logits, preds)


    def _check_preds_backdoor(self, sv, v, thr):
        n = v.shape[0]
        m = self.n_classes
        mat = np.zeros((m,m))
        for i in range(n):
            mat[sv[i]][v[i]] += 1
        for i in range(m):
            mat[i][:] /= np.sum(mat[i][:])

        tgt = [0]*m
        max_pair = None
        max_prob = 0
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                tgt[j] = max(tgt[j], mat[i][j])
                if mat[i][j] > max_prob:
                    max_pair = (i,j)
                    max_prob = mat[i][j]

        arg = np.argsort(tgt)
        if tgt[arg[-1]] - tgt[arg[-2]] >= thr:
            return max_pair, mat, tgt[arg[-1]]

        return None, mat, tgt[arg[-1]]


    def score_to_prob(self, score):
        print(score)
        a = 1.0
        b = -5
        z = np.log(score)
        w = np.tanh(a*z+b)
        w = w/2.0+0.5
        return max(EPS, w)


    def check_neuron(self, param, pair):
        s_lb, t_lb = pair
        self.hook_param = param

        self.reprs = list()
        self.record_reprs = True
        logits, preds = self._run_once_epoch(self.images_all)
        self.record_reprs = False
        pos_repr = np.concatenate(self.reprs)
        pos_repr = self.regular_features(pos_repr)

        _pair, _mat, att_acc = self._check_preds_backdoor(self.preds_all, preds, 0.9)
        if _pair is None:
            return -1

        select_idx = (self.preds_all==s_lb)
        pos_repr = pos_repr[select_idx]
        preds = preds[select_idx]

        reprs = np.concatenate([self.good_repr, pos_repr])
        labels = np.concatenate([self.preds_all, preds])

        #'''
        dim_keep = min(KEEP_DIM, len(reprs))
        pca = PCA(n_components=dim_keep, svd_solver='full')
        pca.fit(reprs)

        gb_model = self.dealer.build_global_model(pca.transform(self.good_repr), self.preds_all, self.n_classes)
        lc_model = self.dealer.build_local_model(pca.transform(reprs), labels, gb_model, self.n_classes)
        #'''

        #gb_model = self.dealer.build_global_model(self.good_repr, self.preds_all, self.n_classes)
        #lc_model = self.dealer.build_local_model(reprs, labels, gb_model, self.n_classes)

        sc = self.dealer.calc_final_score(lc_model)
        if np.argmax(sc) != pair[1]:
            return -2

        k,i,test_v = param[0], param[1], param[2]

        mx_list = list()
        for lb in range(self.n_classes):
            mx_list.append(self.lb_channel_max[k][lb][i])
        mx_list = np.asarray(mx_list)
        if np.argmax(mx_list) == t_lb:
            return -3

        #print(mx_list)
        print((k,i,test_v), pair,att_acc, self.channel_max[k][i]/test_v)
        return sc[pair[1]]


    def regular_features(self, fe):
        shape = fe.shape
        if len(shape) > 2:
            fe = np.average(fe,axis=(2,3))
        return fe


    def test_one_channel(self, k, i, try_v, shape, candi_list):
        '''
        mask = np.random.rand(shape[2], shape[3])
        ratio = min(0.5, 500/(shape[2]*shape[3]))
        ratio = 0.5
        mask = mask<ratio
        mask = mask.astype(np.float32)

        mask_tensor = torch.from_numpy(mask)
        self.hook_param = (k, i, try_v, mask_tensor)
        '''
        self.hook_param = (k, i, try_v)

        st_child = self.md_child[k]
        logits, preds = self._run_once_epoch(self.child_inputs[st_child], st_child)

        pair, mat, att_acc = self._check_preds_backdoor(self.pred_init, preds, 0.5)
        if pair is not None:
            candi_list.append((self.hook_param, pair))


    def reverse_trigger(self, param, pair):
        print('reverse', param[0:3], pair)

        k, i, try_v = param[0], param[1], param[2]
        u, v = pair

        idx = (self.preds_all==u)
        raw_images = self.raw_images[idx]

        ed_child = self.md_child[k]+1
        model = self._get_partial_model(st_child=0, ed_child=ed_child)
        model.eval()

        reverser = Reverser(self.model, param, pair)
        #reverser = Reverser(model, param)
        poisoned_images, raw_poisoned_images = reverser.reverse(raw_images)

        self.hook_param = param
        logits, preds = self._run_once_epoch(poisoned_images)

        att_acc = np.sum(preds==v)/len(poisoned_images)
        return att_acc, raw_poisoned_images, raw_images


    def get_arch_name(self):
        arch = None
        nconv = len(self.convs)
        if self.model_name == 'resnet':
            if self.convs[-1].in_channels > 1000:
                arch = 'wideresnet'
            else:
                arch = 'resnet'
            if nconv == 20:
                arch+='18'
            elif nconv == 36:
                arch+='34'
            elif nconv == 53:
                arch+='50'
            elif nconv == 104:
                arch+='101'
            elif nconv == 155:
                arch+='152'
            else:
                arch = None
        elif self.model_name == 'densenet':
            arch = 'densenet'+str(nconv+1)
        elif self.model_name == 'googlenet':
            arch = 'googlenet'
        elif self.model_name == 'inception3':
            arch = 'inceptionv3'
        elif 'squeezenet' in self.model_name:
            arch='squeezenet'
            if self.convs[0].out_channels==96:
                arch+='v1_0'
            elif self.convs[0].out_channels == 64:
                arch+='v1_1'
            else:
                arch = None
        elif self.model_name == 'mobilenetv2':
            arch='mobilenetv2'
        elif 'shufflenet' in self.model_name:
            arch='shufflenet'
            if self.convs[-1].in_channels == 464:
                arch+='1_0'
            elif self.convs[-1].in_channels == 704:
                arch+='1_5'
            elif self.convs[-1].in_channels == 976:
                arch+='2_0'
            else:
                arch = None
        elif self.model_name == 'vgg':
            arch='vgg'+str(nconv+3)+'bn'
        else:
            arch = None
        return arch



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
        self.good_repr = self.regular_features(self.good_repr)

        start = timeit.default_timer()

        run_ct = 0
        self.dealer = SCAn()
        sc_list = list()
        n_conv = len(self.convs)

        tgt_ct = [0]*self.n_classes
        tgt_list = list()
        candi_list = list()

        layer_candi = None
        self.svm_model = None
        self.loss_model = None
        model_path = os.path.join(SVM_FOLDER,self.arch_name+'_svm.pkl')
        if os.path.exists(model_path):
            print(model_path)
            with open(model_path,'rb') as f:
                svm_model = pickle.load(f)
            layer_candi = svm_model['layer_candi']
            self.svm_model = svm_model['svm_model']

            loss_model_path = os.path.join(SVM_FOLDER,'loss_model.pkl')
            with open(loss_model_path,'rb') as f:
                self.loss_model = pickle.load(f)


        candi_ct = list()
        candi_mat = np.zeros([self.n_classes, self.n_classes])

        count_k = 0

        #layer_candi = None
        #self.svm_model = None
        #self.loss_model = None

        for k, md in enumerate(self.convs):
            if layer_candi is not None and not k in layer_candi:
                continue

            count_k += 1

            shape = self.outputs[k].shape
            tmax = self.tensor_max[k]
            tmean = self.tensor_mean[k]

            print('conv: ', k, shape, tmax, tmean)

            if tmean*10 < tmax:
                test_v = tmax*1.0
            else:
                test_v = tmax*1.0

            weight_list = list()
            for p in self.convs[k].parameters():
                weight_list.append(p.cpu().detach().numpy())
            weights = weight_list[0]
            weights = np.abs(weights)
            weights_sum = np.sum(weights,(2,3))

            if self.convs[k].groups > 1:
                o_max = list()
                g = self.convs[k].groups
                inc = self.convs[k].in_channels
                ouc = self.convs[k].out_channels

                in_d = inc//g
                ou_d = ouc//g
                for i in range(g):
                    max_mat = self.channel_in_max[k][i*in_d:i*in_d+in_d]
                    wet_mat = weights_sum[i*ou_d:i*ou_d+ou_d]
                    o_max.append(np.matmul(wet_mat, max_mat))
                o_max = np.concatenate(o_max)
            else:
                o_max = np.matmul(weights_sum, self.channel_in_max[k])

            if len(weight_list) > 1:
                o_max += weight_list[1]

            this_candi = list()
            for i in range(shape[1]):
                run_ct += 1
                tv = self.channel_max[k][i]

                if tv > self.tensor_mean[k]:
                    continue

                test_v = min(o_max[i], test_v)*1.0
                #test_v = o_max[i]*1.0

                self.test_one_channel(k,i, test_v, shape, this_candi)


            candi_ct.append(0)

            zz = 0
            for param, pair in this_candi:
                sc = self.check_neuron(param,pair)

                #print(zz, sc, param[:3], pair)
                zz += 1
                if sc < EPS:
                    continue

                candi_mat[pair[0]][pair[1]] += 1
                candi_ct[-1] += 1

                tgt_ct[pair[1]] += 1
                tgt_list.append(pair[1])
                sc_list.append(sc)
                candi_list.append((param, pair))
                i = param[1]
                #print(self.channel_mean[k][i], self.channel_max[k][i], self.channel_min[k][i])


        print(candi_ct)
        stop = timeit.default_timer()
        print('Time used to select neuron: ', stop-start)

        out_data = {'candi_ct':candi_ct, 'candi_mat':candi_mat}
        utils.save_pkl_results(out_data)

        if self.svm_model is not None:
            print(candi_ct)
            sc = np.sum(self.svm_model.coef_[0]*candi_ct)+self.svm_model.intercept_
            alpha, beta = self.loss_model[self.arch_name]
            p = sc*alpha+beta
            p = 1.0/(1.0+np.exp(-p))
            print(sc, p)

            return p[0]

        '''
        param = (1, 28, 21.0024204)
        pair = (14,6)
        param = (1, 30, 21.0024204)
        pair = (14,6)
        param = (1, 41, 21.0024204)
        pair = (14,6)
        param = (6, 78, 16.6163940)
        pair = (6,7)
        param = (7, 39, 17.01759529)
        pair = (6,7)
        k,i,test_v = param[0], param[1], param[2]
        mx_list = list()
        for lb in range(self.n_classes):
            mx_list.append(self.lb_channel_max[k][lb][i])
        mx_list = np.asarray(mx_list)
        print(mx_list)

        att_acc, poisoned_images, benign_images = self.reverse_trigger(param, pair)
        print('recovered trigger attack acc: ', att_acc)
        utils.save_poisoned_images(pair, poisoned_images, benign_images)

        exit(0)
        #'''


        if len(sc_list) == 0:
            return 0
        tgt_ct /= np.sum(tgt_ct)
        print(tgt_ct)
        sorted_tgt = np.sort(tgt_ct)
        if sorted_tgt[-1]-sorted_tgt[-2] < 0.1:
            return 0
        tgt_lb = np.argmax(tgt_ct)


        _sc_list = list()
        _candi_list = list()
        for tgt, sc, candi in zip(tgt_list, sc_list, candi_list):
            if tgt != tgt_lb:
                continue
            _sc_list.append(sc)
            _candi_list.append(candi)

        max_id = np.argmax(_sc_list)
        param, pair = _candi_list[max_id]
        att_acc, poisoned_images, benign_images = self.reverse_trigger(param, pair)
        print('recovered trigger attack acc: ', att_acc)

        if (att_acc < 0.9):
            return 0

        print(poisoned_images.shape)

        utils.save_poisoned_images(pair, poisoned_images, benign_images)

        return self.score_to_prob(np.max(_sc_list))


class Reverser:
    def __init__(self, model, param, pair):
        self.model = model
        self.param = param
        self.layer_k, self.channel_i, self.try_v = param[0], param[1], param[2]
        self.src_lb, self.tgt_lb = pair

        self.convs = list()
        mds = list(self.model.modules())
        for md in mds:
            na = type(md).__name__
            if na != 'Conv2d':
                continue
            self.convs.append(md)
        self.hook_handle = self.convs[self.layer_k].register_forward_hook(self.get_hook())

        self.ssim_loss_fn = pytorch_ssim.SSIM()

        self.epsilon = 1e-6
        self.keep_ratio = 0
        self.init_lr = 0.2


    def get_hook(self):
        def hook(model, input, output):
            if type(output) is tuple:
                output = output[0]
            self.current_output = output
        return hook


    def run_model(self, input_raw_tensor):
        _min_values = torch.min(torch.flatten(input_raw_tensor,start_dim=1), dim=1, keepdim=True).values
        _min_values = _min_values.unsqueeze(-1)
        _min_values = _min_values.unsqueeze(-1)
        x_tensor = input_raw_tensor - _min_values

        _max_values = torch.max(torch.flatten(x_tensor,start_dim=1), dim=1, keepdim=True).values
        _max_values = _max_values.unsqueeze(-1)
        _max_values = _max_values.unsqueeze(-1)
        x_tensor = x_tensor / _max_values

        y_tensor = self.model(x_tensor)
        logits = y_tensor.cpu().detach().numpy()
        pred = np.argmax(logits,axis=1)
        att_acc = np.sum(pred==self.tgt_lb)/len(pred)

        return x_tensor, att_acc


    def forward(self, input_raw_tensor):
        x_adv_raw_tensor = (self.reverse_mask_tensor * input_raw_tensor +
                        self.mask_tensor * self.pattern_raw_tensor)
        self.x_adv_raw_tensor = x_adv_raw_tensor

        self.x_adv_tensor, self.att_acc = self.run_model(self.x_adv_raw_tensor)

        output_tensor = self.current_output

        shape = output_tensor.shape
        self.keep_loss = 0
        if self.channel_i > 0:
            self.keep_loss += F.mse_loss(output_tensor[:, :self.channel_i], self.init_output_tensor[:, :self.channel_i], reduction='sum')
        if self.channel_i < shape[1]-1:
            self.keep_loss += F.mse_loss(output_tensor[:, self.channel_i+1:], self.init_output_tensor[:, self.channel_i+1:], reduction='sum')
        self.keep_loss /= shape[0]*(shape[1]-1)*shape[2]*shape[3]

        channel_output = (self.try_v - output_tensor[:, self.channel_i])
        self.channel_loss = torch.mean(F.relu(channel_output))

        self.mask_loss = torch.sum(torch.abs(self.mask_tensor)) / self.image_channel

        self.ssim_loss = self.ssim_loss_fn(self.x_adv_tensor, self.init_image_tensor)

        #self.loss = self.keep_loss + self.channel_loss + self.mask_loss - self.ssim_loss
        self.loss = self.keep_ratio * self.keep_loss + self.channel_loss

        return self.keep_loss.cpu().detach().numpy(), \
               self.channel_loss.cpu().detach().numpy(), \
               self.mask_loss.cpu().detach().numpy(), \
               self.ssim_loss.cpu().detach().numpy()


    def backward(self):
        self.opt.zero_grad()
        self.loss.backward()
        self.opt.step()
        self._upd_trigger()


    def _upd_trigger(self):
        mask_tensor_unrepeat = (torch.tanh(self.mask_tanh_tensor.cuda()) / 2 + 0.5)
        mask_tensor_unexpand = mask_tensor_unrepeat.repeat(self.image_shape[0],1,1)
        self.mask_tensor = mask_tensor_unexpand.unsqueeze(0)
        self.reverse_mask_tensor = (torch.ones_like(self.mask_tensor.cuda()) - self.mask_tensor)

        self.pattern_raw_tensor = ((torch.tanh(self.pattern_tanh_tensor.cuda()) / 2 + 0.5) * 255.0)
        self.pattern_raw_tensor = self.pattern_raw_tensor.unsqueeze(0)


    def reverse(self, images):
        self.image_shape = images.shape[1:]
        self.image_channel = self.image_shape[0]

        mask = np.zeros(self.image_shape[1:], dtype=np.float32)
        pattern = np.zeros(self.image_shape, dtype=np.float32)

        #initialize
        mask_tanh = np.zeros_like(mask)
        pattern_tanh = np.zeros_like(pattern)

        #mask_tanh = np.ones_like(mask)*(-3)
        #limit = math.sqrt(1.0/3.0)
        #pattern_tanh = np.random.uniform(-limit,limit,size=self.image_shape)
        #pattern_tanh = pattern_tanh.astype(np.float32)
        #print(pattern_tanh.shape)




        self.mask_tanh_tensor = Variable(torch.from_numpy(mask_tanh), requires_grad=True)
        self.pattern_tanh_tensor = Variable(torch.from_numpy(pattern_tanh), requires_grad=True)

        self._upd_trigger()
        self.opt = torch.optim.Adam([self.pattern_tanh_tensor, self.mask_tanh_tensor], lr=self.init_lr, betas=(0.5,0.9))
        #self.opt = torch.optim.SGD([self.pattern_tanh_tensor, self.mask_tanh_tensor], lr=self.init_lr, momentum=0.9)

        self.init_image_tensor = torch.from_numpy(images).cuda()
        self.run_model(self.init_image_tensor)
        init_output = self.current_output.cpu().detach().numpy()
        self.init_output_tensor = torch.from_numpy(init_output).cuda()


        keep_loss, channel_loss, mask_loss, ssim_loss = self.forward(self.init_image_tensor)
        print('initial: keep_loss: %.2f, channel_loss: %.2f, mask_loss: %.2f, ssim_loss: %.2f'%(keep_loss, channel_loss, mask_loss, ssim_loss))


        ratio_set_counter = 0
        ratio_up_counter = 0
        ratio_down_counter = 0
        patience_iters = 5
        self.init_ratio = 1.0
        self.ratio_up_multiplier = 1.1

        best_images = None
        best_raw_images = None
        best_keep_loss = float('inf')
        best_channel_loss = float('inf')

        self.keep_ratio = 0.0
        max_steps = 100
        for step in range(max_steps):
            self.backward()
            keep_loss, channel_loss, mask_loss, ssim_loss = self.forward(self.init_image_tensor)

            if step%10 == 0:
                print('step %d: keep_loss: %.2f, channel_loss: %.2f, mask_loss: %.2f, ssim_loss: %.2f, att_acc: %.2f'%(step, keep_loss, channel_loss, mask_loss, ssim_loss, self.att_acc))

            if self.att_acc > 0.9 and keep_loss < best_keep_loss:
                best_raw_images = self.x_adv_raw_tensor.cpu().detach().numpy()
                best_images = self.x_adv_tensor.cpu().detach().numpy()
                best_keep_loss = keep_loss
            elif channel_loss < best_channel_loss:
                best_channel_loss = channel_loss
                best_raw_images = self.x_adv_raw_tensor.cpu().detach().numpy()
                best_images = self.x_adv_tensor.cpu().detach().numpy()

            if self.keep_ratio == 0 and self.att_acc > 0.9:
                ratio_set_counter += 1
                if ratio_set_counter >= patience_iters:
                    self.keep_ratio = self.init_ratio
                    print('init ratio to %.2E'%Decimal(self.keep_ratio))
            else:
                ratio_set_counter = 0
                #self.keep_ratio = 1.0

            if self.att_acc > 0.9:
                ratio_up_counter += 1
                ratio_down_counter = 0
            else:
                ratio_up_counter = 0
                ratio_down_counter += 1

            if ratio_up_counter >= patience_iters:
                ratio_up_counter = 0
                self.keep_ratio *= self.ratio_up_multiplier
                print('keep_ratio up to: {}'.format(self.keep_ratio))
            elif ratio_down_counter >= patience_iters:
                ratio_down_counter = 0
                self.keep_ratio /= self.ratio_up_multiplier
                print('keep_ratio down to: {}'.format(self.keep_ratio))

        return best_images, best_raw_images



def process_run(k, idx, init_x, init_y, pipes, n_classes):
    pipe = pipes.pop()
    sna = SingleNeuronAnalyzer(idx, init_x, init_y, pipe, n_classes)
    sna.run()
    pipes.append(pipe)
    print((k, idx, sna.peak))
    return (k, idx, sna.peak)
    #return (idx, sna.peak, sna.x_list, sna.y_list)
