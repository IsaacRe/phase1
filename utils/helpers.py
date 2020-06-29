import torch
from tqdm.auto import tqdm


def find_network_modules_by_name(network, module_names):
    """
    Searches the the result of network.named_modules() for the modules specified in module_names and returns them
    :param network: the network to search
    :param module_names: List[String] containing the module names to search for
    :return: List[torch.nn.Module] of modules specified in module_names
    """
    assert hasattr(network, 'named_modules'), 'Network %s has no attribute named_modules' % repr(network)
    assert len(list(network.named_modules())) > 0, 'Network %s has no modules in it' % repr(network)
    ret_modules = []
    module_names = set(module_names)
    all_found = False
    for name, module in network.named_modules():
        if name in module_names:
            ret_modules += [module]
            module_names.discard(name)
            if len(module_names) == 0:
                all_found = True
                break
    assert all_found, 'Could not find the following modules in the passed network: %s' % \
                      ', '.join(module_names)
    return ret_modules


def get_named_modules_from_network(network, include_bn=False):
    """
    Returns all modules in network.named_modules() that have a 'weight' attribute as a Dict indexed by module name
    :param network: the network to search
    :param include_bn: if True, include BatchNorm layers in the returned Dict
    :return: Dict[String, torch.nn.Module] containing modules indexed by module name
    """
    assert hasattr(network, 'named_modules'), 'Network %s has no attribute named_modules' % repr(network)
    assert len(list(network.named_modules())) > 0, 'Network %s has no modules in it' % repr(network)
    ret_modules = {}
    for name, module in network.named_modules():
        if not hasattr(module, 'weight'):
            continue
        if type(module) == torch.nn.BatchNorm2d and not include_bn:
            continue
        ret_modules[name] = module

    return ret_modules


def data_pass(loader, network, device=0, backward_fn=None, early_stop=None):
    """
    Perform a forward-backward pass over all data batches in the passed DataLoader
    :param loader: torch.utils.data.DataLoader to use
    :param network: the network to pass data batches through
    :param device: the device that network is on
    :param backward_fn: if specified, the result of backward_fn(network(x), y) will be backpropagated for each batch.
                        Otherwise, no backward pass will be conducted.
    :param early_stop: if specified, execution will stop after the provided number of batches have
                       completed. Otherwise, all batches will be processed.
    """
    context = CustomContext()
    backward = True
    if backward_fn is None:
        backward = False
        context = torch.no_grad()

    with context:
        for itr, (i, x, y) in enumerate(tqdm(loader)):
            if early_stop and itr >= early_stop:
                return
            x, y = x.to(device), y.to(device)
            out = network(x)
            if backward:
                backward_out = backward_fn(out, y)
                backward_out.backward()
                network.zero_grad()


def flatten_activations(activations):
    # if outputs are conv featuremaps aggregate over spatial dims and sample dim
    if len(activations.shape) == 4:
        return activations.transpose(1, 0).flatten(start_dim=1, end_dim=3)
    # if outputs are vectors only need to aggregate over sample dim
    elif len(activations.shape) == 2:
        return activations.transpose(1, 0)
    else:
        raise TypeError('Output type unknown for Tensor: \n%s' % repr(activations))


class CustomContext:

    def __init__(self, enter_fns=[], exit_fns=[], handle_exc_vars=False):
        self.handle_exc_vars = handle_exc_vars
        self.enter_fns = enter_fns
        self.exit_fns = exit_fns

    def __enter__(self):
        for fn in self.enter_fns:
            fn()

    def __exit__(self, *exc_vars):
        if not self.handle_exc_vars:
            exc_vars = []
        for fn in self.exit_fns:
            fn(*exc_vars)

