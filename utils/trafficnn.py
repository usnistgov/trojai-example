# MIT License
# 
# Copyright (c) 2021, FireEye, Inc.
# Copyright (c) 2021 Giorgio Severi
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import json
from typing import Any, Callable, List, Optional, Type, Union

import shap
import joblib

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision

from sklearn.preprocessing import StandardScaler

class TrafficLeNet(nn.Module):
    
    def __init__(self,num_classes):
        super(TrafficLeNet, self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.MaxPool2d(kernel_size=2,stride=2)
            )
        
        self.layer2=nn.Sequential(
            nn.Conv2d(6,16,kernel_size=5),
            nn.MaxPool2d(kernel_size=2,stride=2)
            )
        self.fc1=nn.Linear(256, 16)
        self.fc2=nn.Linear(16,num_classes)
    
    
    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=x.reshape(x.size(0),-1)
        x=self.fc1(x)
        x=torch.tanh(x)
        logits=self.fc2(x)
        return logits


class TrafficNN(object):

    def __init__(self, n_features, config):


        # Load Configuration
        self.config = config
        self.n_features = n_features
        self.cnn_type = config['cnn_type']
        self.num_classes = config['num_classes']
        self.exp = None
        self.img_height = config["img_resolution"]
        self.img_width = config["img_resolution"]
        self.channel=1
        self.inp_shape=(self.channel,self.img_height,self.img_width)

        # Set device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = device

        # Build model with correct configuration
        self.model = self.build_model()

    def fit(self,X,y):

        device=self.device
        net=self.model.to(device)
        criterion=nn.CrossEntropyLoss()
        optimizer=optim.SGD(net.parameters(),lr=self.config['lr'])
        
        # Load data
        X=X.reshape(X.shape[0],self.channel,self.img_height,self.img_width)
        x_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float()
        
        trainset = TensorDataset(x_tensor, y_tensor)
        batch_size=int(self.config["batch_size"])
        trainloader = DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8)

        # Training
        for epoch in range(10):

            running_loss = 0.0
            n=len(trainloader.dataset)
            perc=int(int(n/batch_size)/10)
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels.long())
                #print(f'Epoch: {epoch} Batch: {i} Loss: {loss.item()}')
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                #print(1/0)
                if i % perc == perc-1:
                    print(running_loss)
                    print("[%d, %5d] loss: %.6f" % (epoch + 1, i + 1,running_loss/perc))
                    running_loss = 0.0
        
        print("Finished Training")
        return net

    def predict(self, X):
        self.model.eval()
        batch_size = 1000
        if X.shape[0] % batch_size == 0:
            num_batches = X.shape[0] // batch_size
        else:    
            num_batches = X.shape[0] // batch_size + 1
        logits = torch.empty((0,2))
        with torch.no_grad():
                X = X.reshape(X.shape[0],self.channel,self.img_height,self.img_width)
                X = torch.from_numpy(X).float().to(self.device)
        for i in range(num_batches):
            with torch.no_grad():
                result = self.model(X[i*batch_size:(i+1)*batch_size]).cpu()
            logits = torch.cat((logits, result),dim=0)

        return logits

    def predict_classes(self,X):

        logits=self.predict(X)
        _,predictions=torch.max(logits.data,1)
        return predictions


    def build_model(self):

        if self.cnn_type=='LeNet':
            model=TrafficLeNet(self.num_classes)
        elif self.cnn_type=='ResNet18':
            model=TrafficResNet(self.num_classes, BasicBlock, [2,2,2,2])
        elif self.cnn_type=='ResNet34':
            model=TrafficResNet(self.num_classes, BasicBlock, [3,4,6,3])
        # The below are not recommended due to slow runtimes
        elif self.cnn_type=='ResNet50':
            model=TrafficResNet(self.num_classes, Bottleneck, [3,4,6,3])
        elif self.cnn_type=='ResNet101':
            model=TrafficResNet(self.num_classes, Bottleneck, [3,4,23,3])
        elif self.cnn_type=='ResNet152':
            model=TrafficResNet(self.num_classes, Bottleneck, [3,8,36,3])

        return model

    def explain(self, X_back, X_exp, n_samples=100):
        
        indices = np.random.choice(X_back.shape[0],n_samples,replace=False)
        X_back=torch.from_numpy(X_back).float().to(self.device)
        X_back = X_back.reshape(X_back.shape[0],
                                self.channel,
                                self.img_height,
                                self.img_width)

        print(X_back.shape,type(X_back))
        
        model_exp = shap.DeepExplainer(self.model,X_back[indices,:])
        
        print('Computing SHAP values for x_exp')
        X_exp=torch.from_numpy(X_exp).float().to(self.device)
        X_exp = X_exp.reshape(X_exp.shape[0],
                                self.channel,
                                self.img_height,
                                self.img_width)
        
        contribs=model_exp.shap_values(X_exp)[0]
        
        contribs=contribs.reshape(contribs.shape[0],-1)
    
        return contribs

    def save(self, save_path, file_name='traffic_nn', config=None):
        torch.save(self.model.state_dict(), os.path.join(
            save_path, file_name + '.pt'))
        if config:
            with open(os.path.join(save_path, file_name + '_config.json'), 'w') as f:
                json.dump(config, f, indent=4)

    def load(self, save_path, file_name):
        if self.config == None:
            with open(os.path.join(save_path, file_name + '_config.json'), 'r') as f:
                config = json.load(config, f, indent=4)
            self.config = config
            self.cnn_type = config['cnn_type']
            self.model = self.build_model()
        if not file_name.endswith('.pt'):
            file_name = file_name + '.pt'
        self.model.load_state_dict(torch.load(
            os.path.join(save_path, file_name)))
        self.model.to(self.device).eval()

    @classmethod
    def gen_random_model_cfg(cls, config=None):
        if 'random' in config and config['random'] == 'false':
            cnn_type = config['cnn_type']
            num_classes = int(config['num_classes'])
            lr = float(config['lr'])
            batch_size = int(config['batch_size'])
            train_perc = float(config['train_perc'])
            img_resolution = int(config['img_resolution'])

            model_cfg = {}
            model_cfg['cnn_type'] = cnn_type
            model_cfg['num_classes'] = num_classes
            model_cfg['lr'] = lr
            model_cfg['batch_size'] = batch_size
            model_cfg['train_perc'] = train_perc
            model_cfg['img_resolution'] = img_resolution

        else:
            import random

            model_cfg = {}
            model_cfg['cnn_type'] = random.choice(["ResNet18", "ResNet34"])
            model_cfg['num_classes'] = 2
            model_cfg['lr'] = random.choice([0.01, 0.001, 0.0001])
            model_cfg['batch_size'] = random.choice([32, 64, 128])
            model_cfg['train_perc'] = random.choice([0.25, 0.50, 0.75])
            model_cfg['img_resolution'] = 28

        return model_cfg


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class TrafficResNet(nn.Module):
    
    def __init__(self,num_classes,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        ):
        super(TrafficResNet, self).__init__()

        self.inplanes = 64
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d(self.inplanes)
        self._norm_layer = norm_layer
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    
    
    def forward(self,x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
    
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
