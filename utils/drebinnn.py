"""
Copyright (c) 2021, FireEye, Inc.
Copyright (c) 2021 Giorgio Severi
"""

import os
import json

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import functional as F


class DrebinNet3(nn.Module):
    def __init__(self, fc1=100, fc2=100, fc3=100, input_size=991, act_func=torch.tanh):
        super().__init__()
        self.fc1 = nn.Linear(input_size, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, fc3)
        self.fc4 = nn.Linear(fc3, 2)

        self.act_func = act_func

    def forward(self, x):
        x = self.act_func(self.fc1(x))
        x = self.act_func(self.fc2(x))
        x = self.act_func(self.fc3(x))
        x = self.fc4(x)
        return x


class DrebinNet4(nn.Module):
    def __init__(self, fc1=100, fc2=100, fc3=100, fc4=100, input_size=991, act_func=torch.tanh):
        super().__init__()
        self.fc1 = nn.Linear(input_size, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, fc3)
        self.fc4 = nn.Linear(fc3, fc4)
        self.fc5 = nn.Linear(fc4, 2)

        self.act_func = act_func

    def forward(self, x):
        x = self.act_func(self.fc1(x))
        x = self.act_func(self.fc2(x))
        x = self.act_func(self.fc3(x))
        x = self.act_func(self.fc4(x))
        x = self.fc5(x)
        return x


class DrebinNet5(nn.Module):
    def __init__(self, fc1=100, fc2=100, fc3=100, fc4=100, fc5=100, input_size=991, act_func=torch.tanh):
        super().__init__()
        self.fc1 = nn.Linear(input_size, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, fc3)
        self.fc4 = nn.Linear(fc3, fc4)
        self.fc5 = nn.Linear(fc4, fc5)
        self.fc6 = nn.Linear(fc5, 2)

        self.act_func = act_func

    def forward(self, x):
        x = self.act_func(self.fc1(x))
        x = self.act_func(self.fc2(x))
        x = self.act_func(self.fc3(x))
        x = self.act_func(self.fc4(x))
        x = self.act_func(self.fc5(x))
        x = self.fc6(x)
        return x


class DrebinNN(object):
    def __init__(self, n_features, config):

        # Load Configuration
        self.n_features = n_features

        self.exp = None

        # Set device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = device

        # Build model with correct configuration
        if config:
            self.config = config
            self.num_layers = int(config['num_layers'])
            self.merge_default_model_cfg()
            self.model = self.build_model()

    def fit(self, X, y):

        net = self.model
        device = self.device

        criterion = nn.BCEWithLogitsLoss()
        optimizer = Adam(net.parameters(), lr=self.config['lr'])

        # Load data
        x_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float()
        y_tensor = F.one_hot(y_tensor.to(torch.int64), 2)

        trainset = torch.utils.data.TensorDataset(x_tensor, y_tensor)

        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=int(self.config["batch_size"]),
            shuffle=True,
            num_workers=1)

        # Training
        n_epochs = self.config['n_epochs']
        for epoch in range(n_epochs):
            running_loss = 0.0
            epoch_steps = 0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                #loss = criterion(outputs, labels)
                loss = criterion(outputs.to(torch.float32),
                                 labels.to(torch.float32))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                epoch_steps += 1
                if i % 300 == 299:  # print every 200 mini-batches
                    print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                    running_loss / epoch_steps))
                    running_loss = 0.0

        print("Finished Training")
        return net

    def predict(self, X):
        self.model.eval()
        X = torch.from_numpy(X).float().to(self.device)
        softmax = nn.Softmax(dim=1)
        return softmax(self.model(X)).cpu()

    def build_model(self):
        if 'activation_function' in self.config:
            if self.config['activation_function'] == 'tanh':
                self.act_func = torch.tanh
            elif self.config['activation_function'] == 'relu':
                self.act_func = torch.relu
            elif self.config['activation_function'] == 'sigmoid':
                self.act_func = torch.sigmoid
            else:
                print('Warning Unknown Activation Function, defaulting to tanh')
                self.act_func = torch.tanh
        else:
            self.act_func = torch.tanh
        if self.num_layers == 3:
            model = DrebinNet3(self.config['fc1'], self.config['fc2'], self.config['fc3'], self.n_features, self.act_func)
        if self.num_layers == 4:
            model = DrebinNet4(self.config['fc1'], self.config['fc2'], self.config['fc3'],
                               self.config['fc4'], self.n_features, self.act_func)
        if self.num_layers == 5:
            model = DrebinNet5(self.config['fc1'], self.config['fc2'], self.config['fc3'],
                               self.config['fc4'], self.config['fc5'], self.n_features, self.act_func)
        model.to(self.device)
        return model

    # def explain(self, X_back, X_exp, **kwargs):
    #     X_exp = torch.from_numpy(X_exp).float().to(self.device)
    #     X_back = torch.from_numpy(X_back).float().to(self.device)
    #     self.exp = shap.DeepExplainer(self.model, shap.sample(X_back, 100))
    #     return self.exp.shap_values(X_exp)[0] # The return values is a single list

    def save(self, save_path, file_name='drebin_nn', config=None):
        torch.save(self.model.state_dict(), os.path.join(
            save_path, file_name + '.pt'))
        if config:
            with open(os.path.join(save_path, file_name + '_config.json'), 'w') as f:
                json.dump(config, f, indent=4)

    def load(self, save_path, file_name):
        if self.config == None:
            with open(os.path.join(save_path, file_name + '_config.json'), 'r') as f:
                config = json.load(self.config, f, indent=4)
            self.config = config
            #self.merge_default_model_cfg
            self.num_layers = int(config['num_layers'])
            self.model = self.build_model()
        if not file_name.endswith('.pt'):
            file_name = file_name + '.pt'
        self.model.load_state_dict(torch.load(
            os.path.join(save_path, file_name)))
        self.model.eval()

    def merge_default_model_cfg(self):
        default_model_cfg = {}
        default_model_cfg['n_epochs'] = 10
        self.config['activation_function'] == 'tanh'
        if self.num_layers == 3:
            default_model_cfg['lr'] = 0.0015
            default_model_cfg['fc1'] = 32
            default_model_cfg['fc2'] = 206
            default_model_cfg['fc3'] = 157
            default_model_cfg['batch_size'] = 128
        elif self.num_layers == 4:
            default_model_cfg['lr'] = 0.0014
            default_model_cfg['fc1'] = 390
            default_model_cfg['fc2'] = 349
            default_model_cfg['fc3'] = 25
            default_model_cfg['fc4'] = 47
            default_model_cfg['batch_size'] = 512
        elif self.num_layers == 5:
            default_model_cfg['lr'] = 0.00015
            default_model_cfg['fc1'] = 152
            default_model_cfg['fc2'] = 172
            default_model_cfg['fc3'] = 322
            default_model_cfg['fc4'] = 40
            default_model_cfg['fc5'] = 81
            default_model_cfg['batch_size'] = 128

        # Overwrites the defaults with specified configuration.
        # If a parameter is not specified it is left as default.
        self.config = {**default_model_cfg, **self.config}