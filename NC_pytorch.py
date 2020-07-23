import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

import utils
from decimal import Decimal

import os


##############################
#        PARAMETERS          #
##############################

DEVICE = '0'  # specify which GPU to use

LOG_FILENAME = 'log.json'
DATA_DIR = 'data'  # data folder
#DATA_FILE = 'gtsrb_dataset_int.h5'  # dataset file
DATA_FILE = 'cifar10_testset.h5'  # dataset file
MODEL_DIR = 'models'  # model directory
#MODEL_FILENAME = 'gtsrb_bottom_right_white_4_target_33.h5'  # model file
MODEL_FILENAME = 'saved_model'  # model file
RESULT_DIR = 'results'  # directory for storing results
# image filename template for visualization results
IMG_FILENAME_TEMPLATE = 'gtsrb_visualize_%s_label_%d.png'

# input size
IMG_ROWS = 224
IMG_COLS = 224
IMG_COLOR = 3
INPUT_SHAPE = (IMG_COLOR, IMG_ROWS, IMG_COLS)

NUM_CLASSES = 10  # total number of classes in the model
Y_TARGET = 3  # (optional) infected target label, used for prioritizing label scanning

# parameters for optimization
BATCH_SIZE = 32  # batch size used for optimization
LR = 0.1  # learning rate
STEPS = 1000  # total optimization iterations
NB_SAMPLE = 1000  # number of samples in each mini batch
MINI_BATCH = NB_SAMPLE // BATCH_SIZE  # mini batch size used for early stop
INIT_COST = 1e-3  # initial weight used for balancing two objectives

REGULARIZATION = 'l1'  # reg term to control the mask's norm

ATTACK_SUCC_THRESHOLD = 0.8  # attack success threshold of the reversed attack
PATIENCE = 3 #5  # patience for adjusting weight, number of mini batches
COST_MULTIPLIER = 2  # multiplier for auto-control of weight (COST)
SAVE_LAST = False  # whether to save the last result or best result

EARLY_STOP = True  # whether to early stop
EARLY_STOP_THRESHOLD = 1.0  # loss threshold for early stop
EARLY_STOP_PATIENCE = 3 * PATIENCE  # patience for early stop

# the following part is not used in our experiment
# but our code implementation also supports super-pixel mask
MASK_SHAPE = np.array((IMG_ROWS,IMG_COLS), dtype=int)
MASK_MIN = 0
MASK_MAX = 1
COLOR_MIN = 0
COLOR_MAX = 255

VERBOSE = 2  # verbose level, 0, 1 or 2
SAVE_TMP = False  # save tmp masks, for debugging purpose
TMP_DIR = 'tmp' # dir to save intermediate masks


RESET_COST_TO_ZERO = True

##############################
#      END PARAMETERS        #
##############################


class Visualizer:

    def __init__(self, model, init_cost, lr, num_classes, tmp_dir):
        self.model = model
        self.lr = lr
        self.num_classes = num_classes
        self.init_cost = init_cost

        self.regularization = REGULARIZATION
        self.cost = init_cost
        self.cost_multiplier_up = COST_MULTIPLIER
        self.cost_multiplier_down = COST_MULTIPLIER ** 1.5
        self.epsilon = 1e-7
        self.img_color = IMG_COLOR
        self.early_stop = EARLY_STOP
        self.early_stop_threshold = EARLY_STOP_THRESHOLD
        self.early_stop_patience = EARLY_STOP_PATIENCE
        self.attack_succ_threshold = ATTACK_SUCC_THRESHOLD
        self.patience = PATIENCE
        self.input_shape = INPUT_SHAPE
        self.mask_min = MASK_MIN
        self.mask_max = MASK_MAX
        self.color_min = COLOR_MIN
        self.color_max = COLOR_MAX
        self.save_tmp = SAVE_TMP
        self.verbose = VERBOSE
        self.tmp_dir = tmp_dir
        self.save_last = SAVE_LAST


        mask_size = np.array((IMG_ROWS,IMG_COLS),dtype=int)
        self.mask_size = mask_size
        mask = np.zeros(self.mask_size,dtype=np.float32)
        pattern = np.zeros(INPUT_SHAPE, dtype=np.float32)
        mask = np.expand_dims(mask, axis=0) # [1, 224,224]

        mask_tanh = np.zeros_like(mask)
        pattern_tanh = np.zeros_like(pattern)

        self.mask_tanh_tensor = Variable(torch.from_numpy(mask_tanh), requires_grad=True) # in [-1,1]
        self.pattern_tanh_tensor = Variable(torch.from_numpy(pattern_tanh), requires_grad=True)

        self._upd_trigger()

        self.model.eval()
        self.opt = torch.optim.Adam([self.pattern_tanh_tensor, self.mask_tanh_tensor], lr=self.lr, betas=(0.5,0.9))

        cost = self.init_cost
        self.cost_tensor = torch.tensor(cost)

        pass


    def _upd_trigger(self):
        mask_tensor_unrepeat = (torch.tanh(self.mask_tanh_tensor.cuda()) /
                                (2 - self.epsilon) +
                                0.5) #in [0,1]

        mask_tensor_unexpand = mask_tensor_unrepeat.repeat(self.img_color,1,1)
        self.mask_tensor = mask_tensor_unexpand.unsqueeze(0)
        self.reverse_mask_tensor = (torch.ones_like(self.mask_tensor.cuda()) - self.mask_tensor)

        self.pattern_raw_tensor = (
            (torch.tanh(self.pattern_tanh_tensor.cuda()) / (2 - self.epsilon) + 0.5) *
            255.0) # to be in [0,255]
        self.pattern_raw_tensor = self.pattern_raw_tensor.unsqueeze(0)



    def forward(self, input_raw_tensor, y_true_tensor):
        # input_raw_tensor must be in [0,255]
        # IMPORTANT: MASK OPERATION IN RAW DOMAIN
        X_adv_raw_tensor = (
            self.reverse_mask_tensor * input_raw_tensor.cuda() +
            self.mask_tensor * self.pattern_raw_tensor) # in [0,255]

        X_adv_tensor = X_adv_raw_tensor - torch.min(X_adv_raw_tensor)
        X_adv_tensor = X_adv_tensor / torch.max(X_adv_tensor)

        output_tensor = self.model(X_adv_tensor)

        _, predicted = torch.max(output_tensor, 1)
        correct = (predicted == y_true_tensor.cuda()).sum()
        self.loss_acc = correct / float(y_true_tensor.shape[0])

        self.loss_ce = F.cross_entropy(output_tensor, y_true_tensor.cuda())

        self.loss_reg = torch.sum(torch.abs(self.mask_tensor)) / self.img_color

        self.loss = self.loss_ce + self.loss_reg * self.cost_tensor

        return self.loss_ce.cpu().detach().numpy(), \
               self.loss_reg.cpu().detach().numpy(), \
               self.loss.cpu().detach().numpy(), \
               self.loss_acc.cpu().detach().numpy()


    def backward(self):
        self.opt.zero_grad()
        self.loss.backward()
        self.opt.step()
        self._upd_trigger()


    def reset_opt(self):
        self.opt = torch.optim.Adam([self.pattern_tanh_tensor, self.mask_tanh_tensor], lr=self.lr, betas=(0.5,0.9))


    def reset_state(self, pattern_init, mask_init):
        print('resetting state')

        # setting cost
        if RESET_COST_TO_ZERO:
            self.cost = 0
        else:
            self.cost = self.init_cost
        self.cost_tensor.data = torch.tensor(self.cost)

        # setting mask and pattern
        mask = np.array(mask_init)
        pattern = np.array(pattern_init)
        mask = np.clip(mask, self.mask_min, self.mask_max)
        pattern = np.clip(pattern, self.color_min, self.color_max)
        mask = np.expand_dims(mask, axis=0)

        # convert to tanh space
        mask_tanh = np.arctanh((mask - 0.5) * (2 - self.epsilon))
        pattern_tanh = np.arctanh((pattern / 255.0 - 0.5) * (2 - self.epsilon))
        print('mask_tanh', np.min(mask_tanh), np.max(mask_tanh))
        print('pattern_tanh', np.min(pattern_tanh), np.max(pattern_tanh))

        #K.set_value(self.mask_tanh_tensor, mask_tanh)
        #K.set_value(self.pattern_tanh_tensor, pattern_tanh)
        self.mask_tanh_tensor.data = torch.from_numpy(mask_tanh)
        self.pattern_tanh_tensor.data = torch.from_numpy(pattern_tanh)
        self._upd_trigger()

        # resetting optimizer states
        self.reset_opt()

        pass


    def save_tmp_func(self, step):
        cur_mask = self.mask_tensor.cpu().detach().numpy()
        cur_mask = cur_mask[0, 0, ...]
        fn = 'tmp_mask_step_%d.png'%step
        img_filename = os.path.join(self.tmp_dir,fn)
        utils.dump_image(np.expand_dims(cur_mask, axis=2) * 255,
                                  img_filename,
                                  'png')

        cur_fusion_tensor = self.mask_tensor * self.pattern_raw_tensor
        cur_funsion = cur_fusion_tensor.cpu().detach().numpy()
        cur_fusion = cur_fusion[0, ...]
        fn = 'tmp_fusion_step_%d.png'%step
        img_filename = os.path.join(self.tmp_dir,fn)
        utils.dump_image(cur_fusion, img_filename, 'png')

        pass


    def visualize(self, dataloader, y_target, pattern_init, mask_init, max_steps, num_batches_per_step):
        # since we use a single optimizer repeatedly, we need to reset
        # optimzier's internal states before running the optimization
        pattern_init = pattern_init.astype(np.float32)
        mask_init = mask_init.astype(np.float32)
        self.reset_state(pattern_init, mask_init)

        self.steps = max_steps
        self.mini_batch = num_batches_per_step

        # best optimization results
        mask_best = None
        mask_upsample_best = None
        pattern_best = None
        reg_best = float('inf')

        # logs and counters for adjusting balance cost
        logs = []
        cost_set_counter = 0
        cost_up_counter = 0
        cost_down_counter = 0
        cost_up_flag = False
        cost_down_flag = False

        # counter for early stop
        early_stop_counter = 0
        early_stop_reg_best = reg_best

        gen = iter(dataloader)

        # loop start
        for step in range(self.steps):

            # record loss for all mini-batches
            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            loss_acc_list = []
            for idx in range(int(self.mini_batch)):
                try:
                  X_batch = gen.next()[0]
                except StopIteration:
                  gen = iter(dataloader)
                  X_batch = gen.next()[0]
                Y_target = torch.tensor([y_target]*X_batch.shape[0])
                #if X_batch.shape[0] != Y_target.shape[0]:
                #    Y_target = to_categorical([y_target] * X_batch.shape[0],
                #                              self.num_classes)


                (loss_ce_value,
                    loss_reg_value,
                    loss_value,
                    loss_acc_value) = self.forward(X_batch, Y_target)
                self.backward()

                loss_ce_list.extend(list(loss_ce_value.flatten()))
                loss_reg_list.extend(list(loss_reg_value.flatten()))
                loss_list.extend(list(loss_value.flatten()))
                loss_acc_list.extend(list(loss_acc_value.flatten()))


            '''
            print(loss_ce_list)
            print(loss_reg_list)
            print(loss_list)
            print(loss_acc_list)
            print(torch.sum(torch.abs(self.mask_tensor)))
            print(np.sum(np.abs(self.mask_tensor.cpu().detach().numpy())))
            print('######################')
            '''

            avg_loss_ce = np.mean(loss_ce_list)
            avg_loss_reg = np.mean(loss_reg_list)
            avg_loss = np.mean(loss_list)
            avg_loss_acc = np.mean(loss_acc_list)

            # if step % 10 == 0:
            #     self.reset_opt()

            # check to save best mask or not
            if avg_loss_acc >= ATTACK_SUCC_THRESHOLD and avg_loss_reg < reg_best:
                mask_best = self.mask_tensor.cpu().detach().numpy()
                mask_best = mask_best[0, 0, ...]
                mask_upsample_best = self.mask_tensor.cpu().detach().numpy()
                mask_upsample_best = mask_upsample_best[0, 0, ...]
                pattern_best = self.pattern_raw_tensor.cpu().detach().numpy()
                reg_best = avg_loss_reg

            # verbose
            if self.verbose != 0:
                if self.verbose == 2 or step % (self.steps // 10) == 0:
                    print('step: %3d, cost: %.2E, attack: %.3f, loss: %f, ce: %f, reg: %f, reg_best: %f' %
                          (step, Decimal(self.cost), avg_loss_acc, avg_loss,
                           avg_loss_ce, avg_loss_reg, reg_best))

            # save log
            logs.append((step,
                         float(avg_loss_ce), float(avg_loss_reg), float(avg_loss), float(avg_loss_acc),
                         float(reg_best), self.cost))

            # check early stop
            if self.early_stop:
                # only terminate if a valid attack has been found
                if reg_best < float('inf'):
                    if reg_best >= self.early_stop_threshold * early_stop_reg_best:
                        early_stop_counter += 1
                    else:
                        early_stop_counter = 0
                early_stop_reg_best = min(reg_best, early_stop_reg_best)

                if (cost_down_flag and
                        cost_up_flag and
                        early_stop_counter >= self.early_stop_patience):
                    print('early stop')
                    break

            # check cost modification
            if self.cost == 0 and avg_loss_acc >= self.attack_succ_threshold:
                cost_set_counter += 1
                if cost_set_counter >= self.patience:
                    self.cost = self.init_cost
                    self.cost_tensor.data = torch.tensor(self.cost)
                    cost_up_counter = 0
                    cost_down_counter = 0
                    cost_up_flag = False
                    cost_down_flag = False
                    print('initialize cost to %.2E' % Decimal(self.cost))
            else:
                cost_set_counter = 0

            if avg_loss_acc >= self.attack_succ_threshold:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            if cost_up_counter >= self.patience:
                cost_up_counter = 0
                if self.verbose == 2:
                    print('up cost from %.2E to %.2E' %
                          (Decimal(self.cost),
                           Decimal(self.cost * self.cost_multiplier_up)))
                self.cost *= self.cost_multiplier_up
                self.cost_tensor.data = torch.tensor(self.cost)
                cost_up_flag = True
            elif cost_down_counter >= self.patience:
                cost_down_counter = 0
                if self.verbose == 2:
                    print('down cost from %.2E to %.2E' %
                          (Decimal(self.cost),
                           Decimal(self.cost / self.cost_multiplier_down)))
                self.cost /= self.cost_multiplier_down
                self.cost_tensor.data = torch.tensor(self.cost)
                cost_down_flag = True

            #if self.save_tmp:
            #    self.save_tmp_func(step)

        # save the final version
        if mask_best is None or self.save_last:
            mask_best = self.mask_tensor.cpu().detach().numpy()
            mask_best = mask_best[0, 0, ...]
            mask_upsample_best = self.mask_tensor.cpu().detach().numpy()
            mask_upsample_best = mask_upsample_best[0, 0, ...]
            pattern_best = self.pattern_raw_tensor.cpu().detach().numpy()

        #if self.return_logs:
        return pattern_best, mask_best, mask_upsample_best, logs
        #else:
        #    return pattern_best, mask_best, mask_upsample_best
