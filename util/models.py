import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

class DataNormalizeLayer(nn.Module):

    def __init__(self, bias, scale):

        super(DataNormalizeLayer, self).__init__()

        self._bias = torch.FloatTensor(1).fill_(bias).view(1, -1, 1, 1)
        self._scale = torch.FloatTensor(1).fill_(scale).view(1, -1, 1, 1)

    def forward(self, x):

        x = (x - self._bias.to(x.device)) * self._scale.to(x.device)

        return x

mnist_normalizer = DataNormalizeLayer(bias = 0., scale = 1.)
cifar10_normalizer = DataNormalizeLayer(bias = 0., scale = 1.)

## Normal Model

class MNIST_LeNet(nn.Module):

    def __init__(self, width = 1, bias = True):

        super(MNIST_LeNet, self).__init__()

        self.width = width
        self.bias = bias
        print('MNIST LeNet with width = %d, bias = %s' % (self.width, self.bias))

        self.conv1 = nn.Conv2d(1, 2 * self.width, 5, 1, 2, bias = self.bias)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(2 * self.width, 4 * self.width, 5, 1, 2, bias = self.bias)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(7 * 7 * 4 * self.width, 64 * self.width, bias = self.bias)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64 * self.width, 10)

    def forward(self, x):

        x = mnist_normalizer(x)

        x = self.relu1(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = self.relu2(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 7 * 7 * 4 * self.width)
        x = self.fc2(self.relu3(self.fc1(x)))

        return x

    def obtain_features(self, x):

        maps = []
        maps.append(mnist_normalizer(x))
        maps.append(self.conv1(maps[-1]))
        maps.append(self.relu1(maps[-1]))
        maps.append(F.max_pool2d(maps[-1], 2))
        maps.append(self.conv2(maps[-1]))
        maps.append(self.relu2(maps[-1]))
        maps.append(F.max_pool2d(maps[-1], 2))
        maps.append(self.fc1(maps[-1].view(-1, 7 * 7 * 4 * self.width)))
        maps.append(self.relu3(maps[-1]))
        maps.append(self.fc2(maps[-1]))

        return maps

class CIFAR10_LeNet(nn.Module):

    def __init__(self, width = 1, bias = True):

        super(CIFAR10_LeNet, self).__init__()

        self.width = width
        self.bias = bias
        print('CIFAR10 LeNet with width = %d, bias = %s' % (self.width, self.bias))

        self.conv1 = nn.Conv2d(3, 6 * self.width, 5, bias = self.bias)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(6 * self.width, 16 * self.width, 5, bias = self.bias)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(5 * 5 * 16 * self.width, 120 * self.width, bias = self.bias)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120 * self.width, 84 * self.width, bias = self.bias)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84 * self.width, 10)

    def forward(self, x):

        x = cifar10_normalizer(x)

        x = self.relu1(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = self.relu2(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 5 * 5 * 16 * self.width)
        x = self.fc3(self.relu4(self.fc2(self.relu3(self.fc1(x)))))

        return x

    def obtain_features(self, x):

        maps = []
        maps.append(cifar10_normalizer(x))
        maps.append(self.conv1(maps[-1]))
        maps.append(self.relu1(maps[-1]))
        maps.append(F.max_pool2d(maps[-1], 2))
        maps.append(self.conv2(maps[-1]))
        maps.append(self.relu2(maps[-1]))
        maps.append(F.max_pool2d(maps[-1], 2))
        maps.append(self.fc1(maps[-1].view(-1, 5 * 5 * 16 * self.width)))
        maps.append(self.relu3(maps[-1]))
        maps.append(self.fc2(maps[-1]))
        maps.append(self.relu4(maps[-1]))
        maps.append(self.fc3(maps[-1]))

        return maps

class CIFAR10_VGG(nn.Module):

    def __init__(self, width = 1, bias = True):

        super(CIFAR10_VGG, self).__init__()

        self.width = width
        self.bias = bias
        print('CIFAR10 VGG16 with width = %d, bias = %s' % (self.width, self.bias))

        layer_template = [4, 4, 'M', 8, 8, 'M', 16, 16, 16, 'M', 32, 32, 32, 'M', 32, 32, 32, 'M']
        layer_list = []

        in_planes = 3
        for layer_label in layer_template:
            if layer_label == 'M':
                layer_list += [nn.MaxPool2d(kernel_size = 2, stride = 2),]
            else:
                out_planes = int(layer_label * self.width)
                layer_list += [nn.Conv2d(in_planes, out_planes, kernel_size = 3, padding = 1, bias = self.bias),\
                    nn.BatchNorm2d(out_planes), nn.ReLU()]
                in_planes = out_planes
        layer_list += [nn.AvgPool2d(kernel_size = 1, stride = 1),]

        self.feature_extractor = nn.Sequential(*layer_list)
        self.classifier = nn.Linear(32 * self.width, 10)

    def forward(self, x):

        x = cifar10_normalizer(x)

        x = self.feature_extractor(x)
        x = x.view(-1, 32 * self.width)
        x = self.classifier(x)

        return x

    def obtain_features(self, x):

        maps = []
        maps.append(cifar10_normalizer(x))
        for layer in self.feature_extractor:
            maps.append(layer(maps[-1]))
        maps.append(self.classifier(maps[-1].view(-1, 32 * self.width)))

        return maps

class ResNet_Block(nn.Module):

    def __init__(self, in_planes, out_planes, stride = 1):

        super(ResNet_Block, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = self.stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU()

        self.shortcut = nn.Sequential()
        if self.stride != 1 or self.in_planes != self.out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.in_planes, self.out_planes, kernel_size = 1, stride = self.stride, bias = False),
                nn.BatchNorm2d(self.out_planes)
                )

    def forward(self, x):

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)

        return out

    def obtain_pre_bn(self, x):

        pre1 = self.conv1(x)
        pre2 = self.conv2(self.relu1(self.bn1(pre1)))
        out = self.bn2(pre2)

        pre_bn_list = [pre1, pre2]
        layer_bn_list = [self.bn1, self.bn2]
        label_bn_list = ['bn1', 'bn2']

        if self.stride != 1 or self.in_planes != self.out_planes:
            sc_x = x
            for layer in self.shortcut:
                if isinstance(nn.BatchNorm2d):
                    pre_bn_list.append(sc_x)
                    layer_bn_list.append(layer)
                    label_bn_list.append('shortcut')
                sc_x = layer(x)

        out += self.shortcut(x)
        out = self.relu2(out)

        return out, pre_bn_list, layer_bn_list, label_bn_list

class CIFAR10_ResNet(nn.Module):

    def __init__(self, num_block_list = [2, 2, 2, 2], width = 1):

        super(CIFAR10_ResNet, self).__init__()

        self.width = width
        self.num_block_list = num_block_list
        self.in_planes = int(4 * self.width)
        print('CIFAR10 ResNet: num_block_list = %s, width = %d' % (self.num_block_list, self.width))

        self.conv1 = nn.Conv2d(3, int(4 * self.width), kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(int(4 * self.width))
        self.relu1 = nn.ReLU()

        self.layer1 = self._make_layer(out_planes = int(4 * self.width), num_blocks = num_block_list[0], stride = 1)
        self.layer2 = self._make_layer(out_planes = int(8 * self.width), num_blocks = num_block_list[1], stride = 2)
        self.layer3 = self._make_layer(out_planes = int(16 * self.width), num_blocks = num_block_list[2], stride = 2)
        self.layer4 = self._make_layer(out_planes = int(32 * self.width), num_blocks = num_block_list[3], stride = 2)

        self.classifier = nn.Linear(int(32 * self.width), 10)

    def _make_layer(self, out_planes, num_blocks, stride):

        stride_list = [stride,] + [1,] * (num_blocks - 1)
        layers = []
        for stride in stride_list:
            layers.append(ResNet_Block(in_planes = self.in_planes, out_planes = out_planes, stride = stride))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):

        x = cifar10_normalizer(x)

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(-1, int(32 * self.width))
        x = self.classifier(x)

        return x

    def obtain_features(self, x):

        maps = []
        maps.append(cifar10_normalizer(x))
        maps.append(self.conv1(maps[-1]))
        maps.append(self.bn1(maps[-1]))
        maps.append(relu(maps[-1]))
        for layer in self.layer1:
            maps.append(layer(maps[-1]))
        for layer in self.layer2:
            maps.append(layer(maps[-1]))
        for layer in self.layer3:
            maps.append(layer(maps[-1]))
        for layer in self.layer4:
            maps.append(layer(maps[-1]))
        maps.append(F.avg_pool2d(maps[-1], 4))
        maps.append(self.classifier(maps[-1].view(-1, int(32 * self.width))))

        return maps

    def obtain_pre_bn(self, x):

        x = cifar10_normalizer(x)

        pre_bn_list = []
        layer_bn_list = []
        label_bn_list = []

        pre1 = self.conv1(x)
        pre_bn_list.append(pre1)
        layer_bn_list.append(self.bn1)
        label_bn_list.append('in_conv')
        x = self.relu1(self.bn1(pre1))

        for idx, layer in enumerate(self.layer1):
            x, pre_list, layer_list, label_list = layer.obtain_pre_bn(x)
            pre_bn_list += pre_list
            layer_bn_list += layer_list
            label_bn_list += list(map(lambda x: 'layer1.%d.' % (idx + 1) + x, label_list))

        for idx, layer in enumerate(self.layer2):
            x, pre_list, layer_list, label_list = layer.obtain_pre_bn(x)
            pre_bn_list += pre_list
            layer_bn_list += layer_list
            label_bn_list += list(map(lambda x: 'layer2.%d.' % (idx + 1) + x, label_list))

        for idx, layer in enumerate(self.layer3):
            x, pre_list, layer_list, label_list = layer.obtain_pre_bn(x)
            pre_bn_list += pre_list
            layer_bn_list += layer_list
            label_bn_list += list(map(lambda x: 'layer3.%d.' % (idx + 1) + x, label_list))

        for idx, layer in enumerate(self.layer4):
            x, pre_list, layer_list, label_list = layer.obtain_pre_bn(x)
            pre_bn_list += pre_list
            layer_bn_list += layer_list
            label_bn_list += list(map(lambda x: 'layer4.%d.' % (idx + 1) + x, label_list))

        x = F.avg_pool2d(x, 4)
        x = x.view(-1, int(32 * self.width))
        x = self.classifier(x)

        return x, pre_bn_list, layer_bn_list, label_bn_list

## Curve Model
class CurveModule(nn.Module):

    def __init__(self, param_names, fix_points):
        '''
        >>> param_names: list of string, parameter names
        >>> fix_points: list of boolean, whether the points is fixed
        '''

        super(CurveModule, self).__init__()
        self.param_names = param_names
        self.fix_points = fix_points
        self.num_bends = len(fix_points)

    def compute_point(self, coeffs):
        '''
        >>> coeffs: list of float, the weights of each points
        '''

        param_list = [0.,] * len(self.param_names)
        for param_idx, param_name in enumerate(self.param_names):
            for point_idx, coeff in enumerate(coeffs):
                param = self.__getattr__('%s_%d' % (param_name, point_idx))
                if param is not None:
                    param_list[param_idx] += param * coeff
                else:
                    param_list[param_idx] = None

        return param_list

    def init(self, mode = 'interp'):

        if mode.lower() in ['interp',]:
            assert self.fix_points[-1] == True and self.fix_points[0] == True
            assert sum(self.fix_points[1:-1]) == 0
            seg_num = len(self.fix_points) - 1
            for idx in range(1, seg_num):
                w1 = idx / seg_num
                w2 = 1. - w1
                for name in self.param_names:
                    param = self.__getattr__('%s_%d' % (name, idx))
                    if param is None:
                        continue
                    param.data = w1 * self.__getattr__('%s_%d' % (name, 0)).data + w2 * self.__getattr__('%s_%d' % (name, seg_num))
        else:
            raise ValueError('Unrecognized mode: %s' % mode)


class CurveLinear(CurveModule):

    def __init__(self, in_features, out_features, fix_points, bias = True):

        super(CurveLinear, self).__init__(param_names = ('weight', 'bias'), fix_points = fix_points)

        self.in_features = in_features
        self.out_features = out_features
        self.fix_points = fix_points
        self.bias = bias

        for point_idx, fix_point in enumerate(fix_points):

            self.register_parameter('weight_%d' % point_idx,
                nn.Parameter(torch.Tensor(self.out_features, self.in_features), requires_grad = not fix_point))
            self.register_parameter('bias_%d' % point_idx,
                None if not self.bias else nn.Parameter(torch.Tensor(self.out_features,), requires_grad = not fix_point))

    def forward(self, x, coeffs):

        weight, bias = self.compute_point(coeffs)
        return F.linear(x, weight, bias)

    def reset_parameters(self,):

        stdv = 1. / math.sqrt(self.in_features)
        for idx in range(self.num_bends):
            self.__getattr__('weight_%d' % idx).data.uniform_(-stdv, stdv)
            if self.__getattr__('bias_%d' % idx) is not None:
                self.__getattr__('bias_%d' % idx).data.uniform_(-stdv, stdv)

    def load_points(self, name, param, index):

        assert index < self.num_bends
        self.__getattr__('%s_%d' % (name, index)).data = param.data

class CurveConv2d(CurveModule):

    def __init__(self, in_channels, out_channels, kernel_size, fix_points, stride = 1,
        padding = 0, dilation = 1, groups = 1, bias = True):

        super(CurveConv2d, self).__init__(param_names = ('weight', 'bias'), fix_points = fix_points)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.fix_points = fix_points
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        for point_idx, fix_point in enumerate(fix_points):

            self.register_parameter('weight_%d' % point_idx,
                nn.Parameter(torch.Tensor(self.out_channels, self.in_channels // groups, kernel_size, kernel_size), requires_grad = not fix_point))
            self.register_parameter('bias_%d' % point_idx,
                None if not self.bias else nn.Parameter(torch.Tensor(self.out_channels), requires_grad = not fix_point))

    def forward(self, x, coeffs):

        weight, bias = self.compute_point(coeffs)
        return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def reset_parameters(self,):

        stdv = 1. / math.sqrt(self.in_channels * self.kernel_size ** 2)
        for idx in range(self.num_bends):
            self.__getattr__('weight_%d' % idx).data.uniform_(-stdv, stdv)
            if self.__getattr__('bias_%d' % idx) is not None:
                self.__getattr__('bias_%d' % idx).data.uniform_(-stdv, stdv)

    def load_points(self, name, param, index):

        assert index < self.num_bends
        self.__getattr__('%s_%d' % (name, index)).data = param.data

class CurveBatchNorm2d(CurveModule):

    def __init__(self, num_features, fix_points, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True):

        super(CurveBatchNorm2d, self).__init__(param_names = ('weight', 'bias'), fix_points = fix_points)

        self.num_features = num_features
        self.fix_points = fix_points
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        for point_idx, fix_point in enumerate(fix_points):

            self.register_parameter('weight_%d' % point_idx,
                None if not self.affine else nn.Parameter(torch.Tensor(self.num_features,), requires_grad = not fix_point))
            self.register_parameter('bias_%d' % point_idx,
                None if not self.affine else nn.Parameter(torch.Tensor(self.num_features,), requires_grad = not fix_point))

        self.register_buffer('running_mean', torch.zeros(self.num_features) if self.track_running_stats else None)
        self.register_buffer('running_var', torch.ones(self.num_features) if self.track_running_stats else None)
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long) if self.track_running_stats else None)

    def forward(self, x, coeffs):

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            exponential_average_factor = 1. / self.num_batches_tracked.item() if self.momentum is None else self.momentum
        else:
            exponential_average_factor = 0.

        weight, bias = self.compute_point(coeffs)
        return F.batch_norm(x, self.running_mean, self.running_var, weight, bias,
            self.training or not self.track_running_stats, exponential_average_factor, self.eps)

    def reset_running_stats(self,):

        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self,):

        self.reset_running_stats()
        if self.affine:
            for idx in range(self.num_bends):
                self.__getattr__('weight_%d' % idx).data.uniform_()
                self.__getattr__('bias_%d' % idx).data.zero_()

    def extra_repr(self,):

        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
        missing_keys, unexpected_keys, error_msgs):

        version = metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(CurveBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def load_points(self, name, param, index):

        assert index < self.num_bends
        self.__getattr__('%s_%d' % (name, index)).data = param.data

class Curve_MNIST_LeNet(nn.Module):

    def __init__(self, fix_points, width = 1, bias = True):

        super(Curve_MNIST_LeNet, self).__init__()

        self.fix_points = fix_points
        self.num_bends = len(fix_points)
        self.width = width
        self.bias = bias
        self.num_bends = len(fix_points)

        print('Curve MNIST LeNet with width = %d, bias = %s' % (self.width, self.bias))

        self.conv1 = CurveConv2d(1, 2 * self.width, 5, fix_points, stride = 1, padding = 2, bias = self.bias)
        self.relu1 = nn.ReLU()
        self.conv2 = CurveConv2d(2 * self.width, 4 * self.width, 5, fix_points, stride = 1, padding = 2, bias = self.bias)
        self.relu2 = nn.ReLU()
        self.fc1 = CurveLinear(7 * 7 * 4 * self.width, 64 * self.width, fix_points, bias = self.bias)
        self.relu3 = nn.ReLU()
        self.fc2 = CurveLinear(64 * self.width, 10, fix_points)

    def forward(self, x, coeffs):

        x = mnist_normalizer(x)

        x = self.relu1(self.conv1(x, coeffs))
        x = F.max_pool2d(x, 2)
        x = self.relu2(self.conv2(x, coeffs))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 7 * 7 * 4 * self.width)
        x = self.relu3(self.fc1(x, coeffs))
        x = self.fc2(x, coeffs)

        return x

    def load_points(self, model, index):

        self.conv1.load_points('weight', model.conv1.weight, index)
        self.conv2.load_points('weight', model.conv2.weight, index)
        self.fc1.load_points('weight', model.fc1.weight, index)
        self.fc2.load_points('weight', model.fc2.weight, index)

        if self.bias == True:
            self.conv1.load_points('bias', model.conv1.bias, index)
            self.conv2.load_points('bias', model.conv2.bias, index)
            self.fc1.load_points('bias', model.fc1.bias, index)
            self.fc2.load_points('bias', model.fc2.bias, index)

    def init(self, mode = 'interp'):

        self.conv1.init(mode)
        self.conv2.init(mode)
        self.fc1.init(mode)
        self.fc2.init(mode)

class Curve_ResNet_Block(nn.Module):

    def __init__(self, fix_points, in_planes, out_planes, stride = 1):

        super(Curve_ResNet_Block, self).__init__()

        self.fix_points = fix_points
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride

        self.conv1 = CurveConv2d(in_planes, out_planes, 3, fix_points, stride = self.stride, padding = 1, bias = False)
        self.bn1 = CurveBatchNorm2d(out_planes, fix_points)
        self.relu1 = nn.ReLU()
        self.conv2 = CurveConv2d(out_planes, out_planes, 3, fix_points, stride = 1, padding = 1, bias = False)
        self.bn2 = CurveBatchNorm2d(out_planes, fix_points)
        self.relu2 = nn.ReLU()

        self.shortcut = None
        if self.stride != 1 or self.in_planes != self.out_planes:
            self.shortcut = nn.Sequential(
                CurveConv2d(in_planes, out_planes, 1, fix_points, stride = self.stride, bias = False),
                CurveBatchNorm2d(out_planes, fix_points)
                )

    def forward(self, x, coeffs):

        out = self.conv1(x, coeffs)
        out = self.relu1(self.bn1(out, coeffs))
        out = self.conv2(out, coeffs)
        out = self.bn2(out, coeffs)

        shortcut_out = x if self.shortcut is None else self.shortcut[1](self.shortcut[0](x, coeffs), coeffs)
        out = self.relu2(out + shortcut_out)

        return out

    def load_points(self, model, index):

        self.conv1.load_points('weight', model.conv1.weight, index)
        self.conv2.load_points('weight', model.conv2.weight, index)
        self.bn1.load_points('weight', model.bn1.weight, index)
        self.bn1.load_points('bias', model.bn1.bias, index)
        self.bn2.load_points('weight', model.bn2.weight, index)
        self.bn2.load_points('bias', model.bn2.bias, index)

        if self.shortcut is not None:
            self.shortcut[0].load_points('weight', model.shortcut[0].weight, index)
            self.shortcut[1].load_points('weight', model.shortcut[1].weight, index)
            self.shortcut[1].load_points('bias', model.shortcut[1].bias, index)

    def init(self, mode = 'interp'):

        self.conv1.init(mode)
        self.conv2.init(mode)
        self.bn1.init(mode)
        self.bn2.init(mode)

        if self.shortcut is not None:
            self.shortcut[0].init(mode)
            self.shortcut[1].init(mode)

class Curve_CIFAR10_ResNet(nn.Module):

    def __init__(self, fix_points, num_block_list = [2, 2, 2, 2], width = 1):

        super(Curve_CIFAR10_ResNet, self).__init__()

        self.width = width
        self.fix_points = fix_points
        self.num_block_list = num_block_list
        self.in_planes = int(4 * self.width)
        self.num_bends = len(fix_points)

        print('CIFAR10 ResNet: num_block_list = %s, width = %d' % (self.num_block_list, self.width))

        self.conv1 = CurveConv2d(3, 4 * self.width, 3, fix_points, stride = 1, padding = 1, bias = False)
        self.bn1 = CurveBatchNorm2d(4 * self.width, fix_points)
        self.relu1 = nn.ReLU()

        self.layer1 = self._make_layer(fix_points, out_planes = 4 * self.width, num_blocks = num_block_list[0], stride = 1)
        self.layer2 = self._make_layer(fix_points, out_planes = 8 * self.width, num_blocks = num_block_list[1], stride = 2)
        self.layer3 = self._make_layer(fix_points, out_planes = 16 * self.width, num_blocks = num_block_list[2], stride = 2)
        self.layer4 = self._make_layer(fix_points, out_planes = 32 * self.width, num_blocks = num_block_list[3], stride = 2)

        self.classifier = CurveLinear(32 * self.width, 10, fix_points)

    def _make_layer(self, fix_points, out_planes, num_blocks, stride):

        stride_list = [stride, ] + [1, ] * (num_blocks - 1)
        layers = []
        for stride in stride_list:
            layers.append(Curve_ResNet_Block(fix_points, self.in_planes, out_planes, stride = stride))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x, coeffs):

        x = cifar10_normalizer(x)

        x = self.conv1(x, coeffs)
        x = self.relu1(self.bn1(x, coeffs))

        for layer in self.layer1:
            x = layer(x, coeffs)
        for layer in self.layer2:
            x = layer(x, coeffs)
        for layer in self.layer3:
            x = layer(x, coeffs)
        for layer in self.layer4:
            x = layer(x, coeffs)

        x = F.avg_pool2d(x, 4)
        x = x.view(-1, 32 * self.width)
        x = self.classifier(x, coeffs)

        return x

    def load_points(self, model, index):

        self.conv1.load_points('weight', model.conv1.weight, index)
        self.bn1.load_points('weight', model.bn1.weight, index)
        self.bn1.load_points('bias', model.bn1.bias, index)

        for layer, base_layer in zip(self.layer1, model.layer1):
            layer.load_points(base_layer, index)
        for layer, base_layer in zip(self.layer2, model.layer2):
            layer.load_points(base_layer, index)
        for layer, base_layer in zip(self.layer3, model.layer3):
            layer.load_points(base_layer, index)
        for layer, base_layer in zip(self.layer4, model.layer4):
            layer.load_points(base_layer, index)

        self.classifier.load_points('weight', model.classifier.weight, index)
        self.classifier.load_points('bias', model.classifier.bias, index)

    def init(self, mode = 'interp'):

        self.conv1.init(mode)
        self.bn1.init(mode)

        for layer in self.layer1:
            layer.init(mode)
        for layer in self.layer2:
            layer.init(mode)
        for layer in self.layer3:
            layer.init(mode)
        for layer in self.layer4:
            layer.init(mode)

        self.classifier.init(mode)
