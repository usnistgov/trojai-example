import random
import torch
import os
import json
import numpy as np
from batch_run import gt_csv, folder_root, get_tokenizer_name
from example_trojan_detector import TriggerInfo

import datasets

datasets.utils.tqdm_utils._active = False


class XXEnv:
    def __init__(self):
        self.obs_dim = 12
        self.action_dim = 12
        self.random_inc = random.Random()

        data_dict = dict()
        for row in gt_csv:
            if row['poisoned'] == 'False':
                continue
            md_name = row['model_name']
            data_dict[md_name] = row

        self.csv_dict = data_dict
        self.list_md_name = list(data_dict.keys())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_lenn = None
        self.arm_dict = None
        self.key_list = None

    def reset(self):
        sel_md_name = self.random_inc.choice(self.list_md_name)

        folder_path = os.path.join(folder_root, 'models', sel_md_name)
        model_filepath = os.path.join(folder_path, 'model.pt')

        md_archi = self.csv_dict[sel_md_name]['model_architecture']
        tokenizer_name = get_tokenizer_name(md_archi)
        tokenizer_filepath = os.path.join(folder_root, 'tokenizers', tokenizer_name + '.pt')

        source_dataset = self.csv_dict[sel_md_name]['source_dataset']
        source_dataset = source_dataset.split(':')[1]

        pytorch_model = torch.load(model_filepath, map_location=torch.device(self.device))
        tokenizer = torch.load(tokenizer_filepath)
        examples_filepath = os.path.join('.', source_dataset + '_data.json')

        model_dirpath, _ = os.path.split(model_filepath)
        with open(os.path.join(model_dirpath, 'config.json')) as json_file:
            config = json.load(json_file)
        trigger_type = config['trigger']['trigger_executor_option']
        trigger_exec = config['trigger']['trigger_executor']

        if 'spatial' in trigger_type:
            if trigger_exec['insert_min_location_percentage'] < 0.25:
                location = 'first'
            else:
                location = 'last'
        else:
            if self.random_inc.random() < 0.5:
                location = 'first'
            else:
                location = 'last'

        if trigger_type.startswith('sc:'):
            from trojan_detector_sc import TrojanTesterSC
            inc_class = TrojanTesterSC
            if 'class' in trigger_type:
                type = 'class'
                tgt_lb = trigger_exec['target_class']
                src_lb = 1 - tgt_lb
            else:
                type = 'normal'
                src_lb = 0
                tgt_lb = 1
            desp_str = 'sc:' + type + '_' + location
            desp_str += '_%d_%d' % (src_lb, tgt_lb)
        elif trigger_type.startswith('ner:'):
            from trojan_detector_ner import TrojanTesterNER
            inc_class = TrojanTesterNER
            src_lb = trigger_exec['label_to_id_map']['B-' + trigger_exec['source_class_label']]
            tgt_lb = trigger_exec['label_to_id_map']['B-' + trigger_exec['target_class_label']]
            if 'local' in trigger_type:
                type = 'local'
                desp_str = 'ner:' + type
            else:
                type = 'global'
                desp_str = 'ner:' + type + '_' + location
            desp_str += '_%d_%d' % (src_lb, tgt_lb)
        elif trigger_type.startswith('qa:'):
            from trojan_detector_qa import TrojanTesterQA
            inc_class = TrojanTesterQA
            desp_str = trigger_type.split(':')[1]
            decomp_desp = desp_str.split('_')
            decomp_desp[1] = location
            desp_str = 'qa:' + '_'.join(decomp_desp)

        trigger_text = trigger_exec['trigger_text']
        token_list = tokenizer.encode(trigger_text)
        self.target_lenn = len(token_list) - 2

        print('=' * 20)
        print('env reset to md_name:', sel_md_name)
        print(trigger_type, self.target_lenn)

        return self.reset_with_desp(desp_str, pytorch_model, tokenizer, [examples_filepath], inc_class)

    def reset_with_desp(self, desp_str, pytorch_model, tokenizer, data_jsons, inc_class, max_epochs=100):
        self.arm_dict = dict()
        for lenn in range(self.action_dim):
            trigger_info = TriggerInfo(desp_str, lenn + 1)
            act_inc = inc_class(pytorch_model, tokenizer, data_jsons, trigger_info, './scratch',
                                max_epochs=max_epochs)
            self.arm_dict[lenn] = dict()
            self.arm_dict[lenn]['handler'] = act_inc
            self.arm_dict[lenn]['trigger_info'] = trigger_info
        self.key_list = sorted(list(self.arm_dict.keys()))
        self._warmup(max_epochs=1)
        next_state, _, _ = self.get_state()
        return next_state

    def _step(self, key, max_epochs=5, return_dict=False):
        inc = self.arm_dict[key]['handler']
        rst_dict = inc.run(max_epochs=max_epochs)
        if rst_dict:
            self.arm_dict[key]['score'] = rst_dict['score']
            te_asr, te_loss = inc.test()
            self.arm_dict[key]['te_asr'] = te_asr / 100
            print('_step', str(inc.trigger_info), 'score:%.2f' % rst_dict['score'], 'te_asr:%.2f%%' % te_asr)
            done = False
        else:
            done = True
        if return_dict:
            return done, rst_dict
        return done

    def _warmup(self, max_epochs=5):
        for key in self.key_list:
            self._step(key, max_epochs)

    def close(self):
        pass

    def seed(self, seed):
        self.random_inc.seed(seed)

    def is_done(self, max_te_asr=None):
        if max_te_asr is None:
            _, _, max_te_asr = self.get_state()
        if max_te_asr > 0.9999:
            return True
        return False

    def get_state(self):
        list_state = list()
        max_te_asr = 0
        maxkey_te_asr = 0
        for key in self.key_list:
            list_state.append(self.arm_dict[key]['score'])
            max_te_asr = max(max_te_asr, self.arm_dict[key]['te_asr'])
            max_trigger_info = self.arm_dict[key]['trigger_info']
        reward = max_te_asr
        if self.target_lenn:
            reward += (max_trigger_info.n == self.target_lenn)
        return np.asarray(list_state), reward-1, max_te_asr

    def step(self, action, max_epochs=5, return_dict=False):
        print('act ', action)
        key = int(action)
        if return_dict:
            done, ret_dict = self._step(key, max_epochs=max_epochs, return_dict=True)
        else:
            done = self._step(key, max_epochs=max_epochs, return_dict=False)
        next_state, reward, max_te_asr = self.get_state()
        if not done:
            done = self.is_done(max_te_asr)
        if return_dict:
            return next_state, reward, done, max_te_asr, ret_dict
        return next_state, reward, done, max_te_asr


def main():
    a = XXEnv()
    state = a.reset()
    print(state)
    next_state, reward, done, _ = a.step(2)
    print(next_state)
    print(reward)
    print(done)


if __name__ == '__main__':
    main()
