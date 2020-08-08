from tqdm.auto import tqdm
import torch.nn
from dataset import get_dataloader_cifar
from subnetwork_selection import load_model
from argument_parsing import *
from utils.hook_management import HookManager, detach_hook
from utils.helpers import find_network_modules_by_name, get_named_modules_from_network
from train_models import train, get_dataloaders


def reinit_layers(network: torch.nn.Module, start_layer):
    begin_reinit = False
    for name, module in get_named_modules_from_network(network, include_bn=True):
        if begin_reinit:
            module.reset_parameters()
        if name == start_layer:
            begin_reinit = True


if __name__ == '__main__':
    model_args, data_args, train_args = parse_args(LoadModelArgs, FeatureDataArgs, DistributedTrainingArgs)
    network = load_model(model_args.arch, model_args.final_model_path, device=0)
    hook_manager = HookManager()
    hook_manager.register_forward_hook(detach_hook, *find_network_modules_by_name(network, [data_args.layer]),
                                       activate=True)
    dataloaders = get_dataloaders(data_args)
    train(train_args, network, *dataloaders, device=0)
