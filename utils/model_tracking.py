from torch.nn import Module
from torch.utils.data import DataLoader
import torch
from typing import Type, List
import json
import warnings
from utils.hook_management import HookManager
from utils.helpers import get_named_modules_from_network, find_network_modules_by_name, data_pass


# TODO create separate data class for probing purposes
def probe(dataloader: Type[DataLoader], model: Type[Module], *var_names, device=0,
          backward_fn=torch.nn.CrossEntropyLoss(), **module_vars):
    if len(module_vars) > 0:
        module_protocols = {n: TrackingProtocol(*vs, buffer_len=len(dataloader), track_w=False, track_b=False) \
                            for n, vs in module_vars.items()}
        model_tracker = ModelTracker(model, **module_protocols)
    else:
        assert len(var_names) > 0, 'Must specify vars for all modules as positional arguments or ' \
                                   'vars for specified modules as keyword arguments'
        model_tracker = ModelTracker(model, protocol=TrackingProtocol(*var_names, buffer_len=len(dataloader),
                                                                      track_w=False, track_b=False))
    var_data = {}
    if model_tracker.require_forward:
        if not model_tracker.require_backward:
            backward_fn = None
        with model_tracker.track():
            data_pass(dataloader, model, device=device, backward_fn=backward_fn)
            for v in var_names:
                if v not in STATE_VARS:
                    var_data[v] = model_tracker.gather_var(v)

    if 'w' in var_names:
        var_data['w'] = model_tracker.gather_weight()
    if 'b' in var_names:
        var_data['b'] = model_tracker.gather_bias()

    return var_data


# lists of vars that can be tracked
ALL_VARS = ['w', 'b', 'inp', 'out', 'w_grad', 'b_grad', 'inp_grad', 'out_grad']
STATE_VARS = ['w', 'b']  # vars that can be queried at any time
STATE_GRAD_VARS = ['w_grad', 'b_grad']  # gradients of state vars
FORWARD_VARS = ['inp', 'out']  # vars that require a forward pass
BACKWARD_VARS = ['inp_grad', 'out_grad']  # vars that require a backward pass

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


"""
def validate_vars(fn):
    def new_fn(*vars, **kwargs):
        for var in vars:
            assert var in ALL_VARS, 'Invalid var, %s, passed to tracker.' \
                                      ' Use a valid var key: %s' % (var, ', '.join(ALL_VARS))
        return fn(*vars, **kwargs)

    return new_fn
"""


class TrackingProtocol:

    @validate_vars(var_idx=1, multiple=True)
    def __init__(self, *vars_, record_every=None, buffer_len=None, protocol_json=None, **overwrite_protocols):
        self._set_proto_dict()
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
            protocol_dict = json.load(open(protocol_json, 'r'))
            for protocol, value in protocol_dict.items():
                self.proto_dict[protocol] = value
        # allow specification of specific vars
        for protocol, value in overwrite_protocols.items():
            self.proto_dict[protocol] = value
        # set hook requirements
        self['track_forward'], self['track_backward'] = False, False
        self._set_forward_backward()

    def __iter__(self):
        return iter(self.proto_dict)

    def __getitem__(self, item):
        return self.proto_dict[item]

    def __setitem__(self, key, value):
        self.proto_dict[key] = value

    def __getattr__(self, item):
        return self.proto_dict[item]

    def __setattr__(self, key, value):
        if hasattr(self, 'proto_dict') and key in self.proto_dict:
            self.proto_dict[key] = value
        else:
            self.__dict__[key] = value

    def _set_proto_dict(self):
        self.__dict__['proto_dict'] = dict(DEFAULT_TRACKING_PROTOCOL)

    def _set_forward_backward(self):
        track_forward = False
        track_backward = False
        if any([self.track_w_grad, self.track_b_grad, self.track_inp_grad, self.track_out_grad]):
            track_backward = True
        if any([self.track_w, self.track_b, self.track_inp, self.track_out]):
            track_forward = True
        self.track_forward, self.track_backward = track_forward, track_backward

    def keys(self):
        return self.proto_dict.keys()

    def values(self):
        return self.proto_dict.values()

    def items(self):
        return self.proto_dict.items()


class ModelTracker:

    def __init__(self, model: Type[Module], protocol: TrackingProtocol = None, **module_protocols: TrackingProtocol):
        self.model = model
        self.module_names = tuple(module_protocols.keys())
        self.modules = None
        self.all_modules = False
        if len(self.module_names) == 0:
            assert protocol is not None, "protocol must be set for all modules via 'protocol=<protocol>' param or " \
                                         "specified separately for each module via '<module>=<protocol>"
            self.all_modules = True
            self.modules = get_named_modules_from_network(model)
            self.module_names = tuple(self.modules.keys())
        else:
            self.modules = {n: m for n, m in zip(self.module_names,
                                                 find_network_modules_by_name(model, *self.module_names))}

        # initialize module trackers
        self.hook_manager = HookManager()
        self.module_trackers = {}
        self.require_forward = False
        self.require_backward = False
        for module_name in self.modules:
            tracker = ModuleTracker(self.modules[module_name], self.hook_manager, module_protocols[module_name])
            self.module_trackers[module_name] = tracker
            if tracker.protocol.track_forward:
                self.require_forward = True
            if tracker.protocol.track_backward:
                self.require_forward = True
                self.require_backward = True

    def clear_module_data_buffers(self, *module_names: str):
        for module_name in module_names:
            self.module_trackers[module_name].clear_data_buffer_all()

    def clear_all_data_buffers(self):
        for name, tracker in self.module_trackers.items():
            tracker.clear_data_buffer_all()

    def gather_module_weight(self, module_name: str):
        return self.module_trackers[module_name].collect_weight()

    def gather_module_bias(self, module_name: str):
        return self.module_trackers[module_name].collect_bias()

    def gather_weight(self):
        return {n: self.gather_module_weight(n) for n in self.module_names}

    def gather_bias(self):
        return {n: self.gather_module_bias(n) for n in self.module_names}

    def gather_module_var(self, module_name: str, var: str):
        return self.module_trackers[module_name].gather_var(var)

    def gather_var(self, var: str):
        return {n: self.gather_module_var(n, var) for n in self.module_names}

    def track(self, *module_names: str, clear_on_exit: bool = True, ):
        if len(module_names) == 0:
            module_names = self.module_names
        exit_fns = []
        if clear_on_exit:
            exit_fns += [lambda: self.clear_module_data_buffers(*module_names)]
        return self.hook_manager.hook_module_context_by_name(*module_names)

    # TODO adopt methods from ModuleTracker to make ModelTracker self-contained,
    # TODO then pass ModelTracker.forward_hook as a hook_hype to HookManager context creators
    # TODO then refactor to make probe a class method


class ModuleTracker:

    def __init__(self, hook_manager: HookManager, protocol: TrackingProtocol, *modules: Module,
                 **named_modules: Module):
        """
        Tracks module variables as learning progresses, for a group of modules with a shared tracking protocol
        :param hook_manager: HookManager that will manage hooks for this module
        :param protocol: TrackingProtocol outlining which module vars to track and protocols for tracking them
        :param modules: List[Module] of unnamed modules to be tracked. Name will be assigned to each using __repr__ .
        :param named_modules: Dict[str, Module] of modules to be tracked, indexed by name
        """
        self.modules = named_modules
        for m in modules:
            self.modules[repr(m)] = m
        self.module_names = [repr(m) for m in modules] + list(named_modules.keys())
        self.hook_manager = hook_manager

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

    ################################################################################

    def _make_tensor_backward_hook(self, module, grad_in=True):
        var = 'inp_grad' if grad_in else 'out_grad'
        end_backward = not (grad_in and self._do_collect('out_grad'))

        def backward_hook(grad):
            self._insert_module_data(module.name, var, grad.cpu())
            if end_backward:
                self._complete_module_backward(module.name)

        return backward_hook

    def _register_tensor_backward_hook(self, module, tensor, input_var=True):
        if not tensor.requires_grad:
            tensor.requires_grad = True
        tensor.register_hook(self._make_tensor_backward_hook(module, grad_in=input_var))

    def forward_pre_hook(self, module, inp):
        (inp,) = inp
        self._register_tensor_backward_hook(module, inp, input_var=True)

    ################################################################################

    def forward_hook(self, module, inp, out):
        (inp,) = inp

        if self._do_collect('inp'):
            self._insert_module_data(module.name, 'inp', inp.data.cpu())
        if self._do_collect('out'):
            self._insert_module_data(module.name, 'out', out.data.cpu())

        # setup backward hook for collection of module's output gradient
        if self._do_collect('out_grad'):
            self._register_tensor_backward_hook(module, out, input_var=False)

        self._complete_module_forward(module.name)

    def backward_hook(self, module, grad_in, grad_out):
        print('\n\n\n\n\n\n\n\n')
        print(module.name)
        print(len(grad_in), len(grad_out))
        print('IN')
        for i in range(len(grad_in)):
            if grad_in[i] is None:
                print('NONE')
            else:
                print(grad_in[i].shape)
        print('OUT')
        for i in range(len(grad_out)):
            if grad_out[i] is None:
                print('NONE')
            else:
                print(grad_out[i].shape)
        grad_in, grad_out = grad_in[0], grad_out[0]

        if self._do_collect('inp_grad'):
            self._insert_module_data(module.name, 'inp_grad', grad_in.cpu())
        if self._do_collect('out_grad'):
            self._insert_module_data(module.name, 'out_grad', grad_out.cpu())
        #if self._do_collect('w_grad'):
        #    self._insert_module_data(module.name, 'w_grad', module.weight.grad.cpu())
        #if self._do_collect('b_grad'):
        #    self._insert_module_data(module.name, 'b_grad', module.bias.grad.cpu())
        self._complete_module_backward(module.name)

    def register_forward(self):
        self.hook_manager.register_forward_hook(self.forward_hook,
                                                hook_fn_name='ModuleTracker.forward_hook',
                                                activate=False, **self.modules)
        if self._do_collect('inp_grad'):
            self.hook_manager.register_forward_pre_hook(self.forward_pre_hook,
                                                        hook_fn_name='ModuleTracker.forward_pre_hook',
                                                        activate=False, **self.modules)

    def register_backward(self):
        self.hook_manager.register_backward_hook(self.backward_hook,
                                                 hook_fn_name='ModuleTracker.backward_hook',
                                                 activate=False, **self.modules)

    def register_all(self):
        if any([self._do_collect(var) for var in GRAPH_VARS]):
            self.register_forward()

    @validate_vars(keyword='vars_')
    def clear_data_buffer_module(self, *module_names: str, vars_: List[str] = None):
        for module_name in module_names:
            if not vars_:
                vars_ = self.data_buffer[module_name].keys()
            for var in vars_:
                self.data_buffer[module_name][var] = []

    def clear_data_buffer_all(self, vars_: List[str] = None):
        self.clear_data_buffer_module(*self.module_names, vars_=vars_)

    @validate_vars(var_idx=2, valid_vars=['inp', 'out', 'inp_grad', 'out_grad'])
    def gather_module_var(self, module_name, var):
        return torch.cat(self.data_buffer[module_name][var], dim=0)

    def track(self, clear_on_exit: bool = True):
        # set module to increment count
        exit_fns = [self._cleanup_tracking]
        if clear_on_exit:
            exit_fns += [lambda: self.clear_data_buffer_all()]
        return self.hook_manager.hook_all_context(add_exit_fns=exit_fns)
