from torchvision.models import resnet34
from dataset import get_dataloader_cifar


def load_network(device=0):
    net = resnet34()
    if device != 'cpu':
        net.cuda(device)
    return net


def load_dataloader():
    return get_dataloader_cifar(batch_size=100, data_dir='../../../data', train=False, download=True)


def load_test_suite(device=0):
    network = load_network()
    loader = load_dataloader()
    if device != 'cpu':
        network.cuda(device)
    return network, loader


def get_module(net, name):
    module_keys = name.split('.')
    ret = net
    for k in module_keys:
        ret = ret._modules[k]
    return ret
