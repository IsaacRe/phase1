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
                                   'vars for specified modules as keyword arguemnts'
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
FORWARD_VARS = ['inp', 'out']  # vars that require a forward pass
BACKWARD_VARS = ['w_grad', 'b_grad', 'inp_grad', 'out_grad']  # vars that require a backward pass

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
    'track_forward': True,
    'track_backward': True,
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
                if keyword in kwargs:
                    vars_ = kwargs[keyword]
                else:
                    vars_ = []
            else:
                vars_ = args[var_idx:]
                if not multiple:
                    vars_ = vars_[:1]
            for var in vars_:
                assert var in valid_vars, 'Invalid var, %s, passed to tracker.' \
                                          ' Use a valid var key: %s' % (var, ', '.join(valid_vars))
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

    def _set_forward_backward(self):
        self.track_forward = False
        self.track_backward = False
        if any([self.track_w_grad, self.track_b_grad, self.track_inp_grad, self.track_out_grad]):
            self.track_forward = True
            self.track_backward = True
        elif any([self.track_inp, self.track_out]):
            self.track_forward = True

    @validate_vars(multiple=True)
    def __init__(self, *vars_, record_every=None, buffer_len=None, protocol_json=None, **overwrite_protocols):
        self.proto_dict = dict(self.DEFAULT_PROTOCOL)
        # track all vars unless specific vars are specified
        if len(vars_) > 0:
            for var in ALL_VARS:
                if var in vars_:
                    continue
                self.proto_dict['%s_track' % var] = False
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
        self._set_forward_backward()

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

    def _begin_module_pass(self, module_name):
        if self.modules_passed[module_name]:
            if not all(self.modules_passed.values()):
                warnings.warn('Some modules are not being tracked! Check proper usage')
            self.pass_count += 1
            self._reset_modules_passed()
        self.modules_passed[module_name] = True

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

    def forward_hook(self, module, inp, out):
        self._begin_module_pass(module.name)
        if self._do_collect('inp'):
            self._insert_module_data('inp', module.name, inp.data.cpu())
        if self._do_collect('out'):
            self._insert_module_data('out', module.name, out.data.cpu())
        # we collect w, b during forward pass since forward pass is minimum requirement for module execution
        if self._do_collect('w'):
            self._insert_module_data('w', module.name, self.collect_weight(module.name))
        if self._do_collect('b'):
            self._insert_module_data('b', module.name, self.collect_bias(module.name))

    def backward_hook(self, module, grad_in, grad_out):
        if self._do_collect('inp_grad'):
            self._insert_module_data('inp_grad', module.name, grad_in.cpu())
        if self._do_collect('out_grad'):
            self._insert_module_data('out_grad', module.name, grad_out.cpu())
        if self._do_collect('w_grad'):
            self._insert_module_data('w_grad', module.name, module.weight.grad.cpu())
        if self._do_collect('b_grad'):
            self._insert_module_data('b_grad', module.name, module.bias.grad.cpu())

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
    def clear_data_buffer_module(self, *module_name: str, vars_: List[str] = None):
        if not vars_:
            vars_ = self.data_buffer[module_name].keys()
        for var in vars_:
            self.data_buffer[module_name][var] = []

    def clear_data_buffer_all(self, vars_: List[str] = None):
        self.clear_data_buffer_module(*self.module_names, vars_=vars_)

    @validate_vars(var_idx=1, valid_vars=['inp', 'out', 'grad_inp', 'grad_out'])
    def gather_module_var(self, module_name, var):
        return torch.cat(self.data_buffer[module_name][var], dim=0)

    def track(self, clear_on_exit: bool = True):
        # set module to increment count
        exit_fns = [self._cleanup_tracking]
        if clear_on_exit:
            exit_fns += [lambda: self.clear_data_buffer_module(*self.module_names)]
        return self.hook_manager.hook_module_context_by_name(*self.module_names)
