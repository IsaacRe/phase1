from torch.nn import Module
from torch.utils.data import DataLoader
import torch
from typing import Type, List
import json
import warnings
from utils.hook_management import HookManager
from utils.helpers import get_named_modules_from_network, find_network_modules_by_name, data_pass, Protocol


# list all of all module vars
ALL_VARS = ['w', 'b', 'inp', 'out', 'w_grad', 'b_grad', 'inp_grad', 'out_grad']
STATE_VARS = ['w', 'b']  # vars that can be queried at any time
STATE_GRAD_VARS = ['w_grad', 'b_grad']  # gradients of state vars
FORWARD_VARS = ['inp', 'out']  # vars that require a forward pass
BACKWARD_VARS = ['inp_grad', 'out_grad']  # vars that require a backward pass
# vars whose values are dependent on the data stream
DATA_DEPENDENT_VARS = ['inp', 'out', 'w_grad', 'b_grad', 'inp_grad', 'out_grad']

GRAPH_VARS = ['inp', 'out', 'inp_grad', 'out_grad']  # vars existing solely within the computation graph

# default module tracking protocol
DEFAULT_TRACKING_PROTOCOL = {
    'track_w': True,
    'track_b': True,
    'track_inp': True,
    'track_out': True,
    'track_w_grad': True,
    'track_b_grad': True,
    'track_inp_grad': True,
    'track_out_grad': True,
    'record_every_w': 1,
    'record_every_b': 1,
    'record_every_inp': 1,
    'record_every_out': 1,
    'record_every_w_grad': 1,
    'record_every_b_grad': 1,
    'record_every_inp_grad': 1,
    'record_every_out_grad': 1,
    'buffer_len_w': 1,
    'buffer_len_b': 1,
    'buffer_len_inp': 1,
    'buffer_len_out': 1,
    'buffer_len_w_grad': 1,
    'buffer_len_b_grad': 1,
    'buffer_len_inp_grad': 1,
    'buffer_len_out_grad': 1,
    'save': False,
    'save_protocol': 'npz',
    'save_file_base': '',
    'suffix_w': 'w',
    'suffix_b': 'b',
    'suffix_inp': 'inp',
    'suffix_out': 'out',
    'suffix_w_grad': 'w_grad',
    'suffix_b_grad': 'b_grad',
    'suffix_inp_grad': 'inp_grad',
    'suffix_out_grad': 'out_grad'
}


def validate_vars(var_idx=0, multiple=False, keyword=None, valid_vars=ALL_VARS):
    def wrapper_fn(fn):
        def new_fn(*args, **kwargs):
            if keyword:
                if keyword in kwargs and kwargs[keyword] is not None:
                    vars_ = kwargs[keyword]
                else:
                    vars_ = []
            else:
                vars_ = args[var_idx:]
                if not multiple:
                    vars_ = vars_[:1]
            for var in vars_:
                assert var in valid_vars, 'Invalid var, %s, passed to tracker.' \
                                          ' Use a valid var key: %s' % (var, ', '.join(ALL_VARS))
            return fn(*args, **kwargs)

        return new_fn

    return wrapper_fn


class TrackingProtocol(Protocol):

    DEFAULT_PROTOCOL = DEFAULT_TRACKING_PROTOCOL

    @validate_vars(var_idx=1, multiple=True)
    def __init__(self, *vars_, record_every=None, buffer_len=None, protocol_json=None, namespace=None,
                 **overwrite_protocol):
        super(TrackingProtocol, self).__init__()
        # track only the vars specified
        for var in ALL_VARS:
            if var in vars_:
                continue
            self.proto_dict['track_%s' % var] = False
        # allow universal record_every specification
        if record_every:
            for protocol in DEFAULT_TRACKING_PROTOCOL:
                if 'record_every' in protocol:
                    self.proto_dict[protocol] = record_every
        # allow universal buffer_len specification
        if buffer_len:
            for protocol in DEFAULT_TRACKING_PROTOCOL:
                if 'buffer_len' in protocol:
                    self.proto_dict[protocol] = buffer_len
        # allow protocol to be loaded from a specified file
        if protocol_json:
            self._add_from_json(protocol_json)
        if namespace:
            self._add_from_namespace(namespace)
        # allow specification of specific vars
        self.overwrite_protocol(**overwrite_protocol)
        # set hook requirements
        self['track_forward'], self['track_backward'] = False, False
        self._set_forward_backward()

    def _set_forward_backward(self):
        track_forward = False
        track_backward = False
        if any([self.track_w_grad, self.track_b_grad, self.track_inp_grad, self.track_out_grad]):
            track_backward = True
        if any([self.track_w, self.track_b, self.track_inp, self.track_out]):
            track_forward = True
        self.track_forward, self.track_backward = track_forward, track_backward


class ModuleTracker:

    def __init__(self, protocol: TrackingProtocol, *modules: Module,
                 hook_manager: HookManager = None, network: Module = None,
                 **named_modules: Module):
        """
        Tracks module variables as learning progresses, for a group of modules with a shared tracking protocol
        :param hook_manager: HookManager that will manage hooks for this module
        :param protocol: TrackingProtocol outlining which module vars to track and protocols for tracking them
        :param modules: List[Module] of unnamed modules to be tracked. Name will be assigned to each using __repr__ .
        :param network: Module representing the full network containing the passed modules. Used in aggregate_vars.
        :param named_modules: Dict[str, Module] of modules to be tracked, indexed by name
        """
        self.network = network

        if hook_manager is None:
            hook_manager = HookManager()
        self.hook_manager = hook_manager

        self.modules = named_modules
        for m in modules:
            self.modules[repr(m)] = m
        self.module_names = list(self.modules)

        # counts the number of forward/backward passes that have been executed by each module being tracked
        self.pass_count = 0
        self.modules_passed = {n: False for n in self.module_names}
        self.protocol = protocol
        self.register_all()

        # initialize buffers for tracked vars
        self.data_buffer = {
            module_name: {
                var: [] for var in ALL_VARS if self.protocol['track_%s' % var]
            } for module_name in self.module_names
        }

        # set up references to tracker object
        for module in self.modules.values():
            module.tracker = self

    def _reset_modules_passed(self):
        self.modules_passed = {n: False for n in self.module_names}

    def _check_pass_complete(self):
        if all(self.modules_passed.values()):
            self.pass_count += 1
            self._reset_modules_passed()

    def _complete_module_pass(self, module_name):
        assert not self.modules_passed[module_name], 'Some modules are not being tracked! Check usage'
        self.modules_passed[module_name] = True
        self._check_pass_complete()

    def _complete_module_forward(self, module_name):
        if not self.protocol.track_backward:
            self._complete_module_pass(module_name)

    def _complete_module_backward(self, module_name):
        self._complete_module_pass(module_name)

    def _cleanup_tracking(self):
        self.pass_count = 0
        self._reset_modules_passed()

    def _insert_module_data(self, module_name, var, data):
        self.data_buffer[module_name][var] += [data]

    def _do_collect(self, var):
        return self.protocol['track_%s' % var] and self.pass_count % self.protocol['record_every_%s' % var] == 0

    def collect_weight(self, module_name):
        return self.modules[module_name].weight.data.cpu()

    def collect_bias(self, module_name):
        return self.modules[module_name].bias.data.cpu()

    def collect_weight_grad(self, module_name):
        grad = self.modules[module_name].weight.grad
        assert grad is not None, "gradient is None for module '%s'. Make sure you are calling collect" \
                                 "before zero_grad." % module_name
        return grad.cpu()

    def collect_bias_grad(self, module_name):
        grad = self.modules[module_name].bias.grad
        assert grad is not None, "gradient is None for module '%s'. Make sure you are calling collect" \
                                 "before zero_grad." % module_name
        return grad.cpu()

    # TODO modify to allow tracking for modules with multiple inputs
    def forward_hook(self, module, input, output):
        (inp,) = input

        if self._do_collect('inp'):
            self._insert_module_data(module.name, 'inp', inp.data.cpu())
        if self._do_collect('out'):
            self._insert_module_data(module.name, 'out', output.data.cpu())
        if self._do_collect('w'):
            self._insert_module_data(module.name, 'w', self.collect_weight(module.name))
        if self._do_collect('b'):
            self._insert_module_data(module.name, 'b', self.collect_bias(module.name))

        self._complete_module_forward(module.name)

    def backward_hook(self, module, grad_in, grad_weight, grad_bias, grad_out):
        (grad_in,) = grad_in

        if self._do_collect('inp_grad'):
            self._insert_module_data(module.name, 'inp_grad', grad_in.cpu())
        if self._do_collect('out_grad'):
            self._insert_module_data(module.name, 'out_grad', grad_out.cpu())
        if self._do_collect('w_grad'):
            self._insert_module_data(module.name, 'w_grad', grad_weight.cpu())
        if self._do_collect('b_grad'):
            self._insert_module_data(module.name, 'b_grad', grad_bias.cpu())

        self._complete_module_backward(module.name)

    def register_forward(self):
        self.hook_manager.register_forward_hook(self.forward_hook,
                                                hook_fn_name='ModuleTracker.forward_hook',
                                                activate=False, **self.modules)

    def register_backward(self):
        self.hook_manager.register_backward_hook(self.backward_hook,
                                                 hook_fn_name='ModuleTracker.backward_hook',
                                                 activate=False, **self.modules)

    def register_all(self):
        if self.protocol.track_forward:
            self.register_forward()
        if self.protocol.track_backward:
            self.register_backward()

    @validate_vars(keyword='vars_')
    def clear_data_buffer_module(self, *module_names: str, vars_: List[str] = None):
        for module_name in module_names:
            if not vars_:
                vars_ = self.data_buffer[module_name].keys()
            for var in vars_:
                self.data_buffer[module_name][var] = []

    def clear_data_buffer_all(self, vars_: List[str] = None):
        self.clear_data_buffer_module(*self.module_names, vars_=vars_)

    @validate_vars(var_idx=2, valid_vars=DATA_DEPENDENT_VARS)
    def gather_module_var(self, module_name, var):
        return torch.cat(self.data_buffer[module_name][var], dim=0)

    def gather_module_data(self, module_name):
        return {var: torch.cat(data, dim=0) for var, data in self.data_buffer[module_name].items()}

    def gather_var_data(self, var_name):
        return {module_name: torch.cat(self.data_buffer[module_name][var_name], dim=0)
                for module_name in self.data_buffer}

    # TODO export var data hashed by corresponding protocol suffix
    def gather_data(self):
        return {module_name: self.gather_module_data(module_name) for module_name in self.data_buffer}

    def track(self, clear_on_exit: bool = True):
        exit_fns = [self._cleanup_tracking]
        if clear_on_exit:
            exit_fns += [self.clear_data_buffer_all]
        return self.hook_manager.hook_all_context(hook_types=[self.forward_hook, self.backward_hook],
                                                  add_exit_fns=exit_fns)

    def aggregate_vars(self, dataloader: DataLoader, network: Module = None, device: int = 0):
        if network is None:
            assert self.network is not None, "network must be passed to aggregate_vars if one" \
                                             " was not passed to __init__"
            network = self.network
        backward_fn = torch.nn.CrossEntropyLoss() if self.protocol.track_backward else None
        with self.track():
            data_pass(dataloader, network, device=device, backward_fn=backward_fn)
            return self.gather_data()
