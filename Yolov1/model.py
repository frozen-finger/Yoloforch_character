from module import module
import torch
import torch.nn as nn
import math
import hyperparameter as hp

def _init_weight(network):
    for layer in network.modules():
        if isinstance(layer, nn.Conv2d):
            n = layer.kernel_size[0]*layer.kernel_size[1]*layer.out_channels
            layer.weight.data.normal_(0, math.sqrt(2.0/n))
            if layer.bias is not None:
                layer.bias.data.zero_()
        elif isinstance(layer, nn.Linear):
            layer.weight.data.normal_(0, 0.001)
            layer.bias.data.zero_()

Yolo = module()
Yolo.apply(_init_weight)


test = True
if __name__ == '__main__' and test==True:
    test_net = module()
    print(test_net)
    input = torch.randn(1, 3, hp.height, hp.width)
    test_net.apply(_init_weight)
    output= test_net(input)
    print(output)
    print(output.size)
