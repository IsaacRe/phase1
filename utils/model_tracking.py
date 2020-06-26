from torch.nn import Module
from torch.utils.data import DataLoader
import torch
from typing import Type
import json
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


def validate_var(fn):
    def new_fn(var, *args, **kwargs):
        assert var in ALL_VARS, 'Invalid var, %s, passed to tracker.' \
                                  ' Use a valid var key: %s' % (var, ', '.join(ALL_VARS))
        return fn(var, *args, **kwargs)

    return new_fn


def validate_vars(fn):
    def new_fn(*vars, **kwargs):
        for var in vars:
            assert var in ALL_VARS, 'Invalid var, %s, passed to tracker.' \
                                      ' Use a valid var key: %s' % (var, ', '.join(ALL_VARS))
        return fn(*vars, **kwargs)

    return new_fn


class TrackingProtocol:

    def _set_forward_backward(self):
        self.track_forward = False
        self.track_backward = False
        if any([self.track_w_grad, self.track_b_grad, self.track_inp_grad, self.track_out_grad]):
            self.track_forward = True
            self.track_backward = True
        elif any([self.track_inp, self.track_out]):
            self.track_forward = True

    @validate_vars
    def __init__(self, *vars, record_every=None, buffer_len=None, protocol_json=None, **overwrite_protocols):
        self.proto_dict = dict(self.DEFAULT_PROTOCOL)
        # track all vars unless specific vars are specified
        if len(vars) > 0:
            for var in ALL_VARS:
                if var in vars:
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

    def __init__(self, module: Type[Module], hook_manager: HookManager, protocol: TrackingProtocol,
                 module_name: str = None):
        """
        Tracks module statistics as learning progresses
        :param module: torch.nn.Module whose statistics will be tracked
        :param hook_manager: HookManager that will manage hooks for this module
        :param protocol: TrackingProtocol outlining which module vars to track and protocols for tracking them
        :param module_name: string specifying the name to use for the module for hook registration.
                            If not specified, the result of __repr__ will be used.
        """
        self.module = module
        self.module_name = module_name if module_name else repr(module)
        self.hook_manager = hook_manager

        self.protocol = protocol
        self.register_all()

        # initialize buffers for tracked vars
        self.data_buffer = {
            var: [] for var in ALL_VARS if self.protocol['track_%s' % var]
        }

        # set up reference to tracker object
        module.tracker = self

    def _insert_data(self, var, data):
        self.data_buffer[var] += [data]

    def _do_collect(self, var):
        return self.protocol['track_%s' % var]

    def collect_weight(self):
        return self.module.weight.data.cpu()

    def collect_bias(self):
        return self.module.bias.data.cpu()

    def forward_hook(self, module, inp, out):
        if self._do_collect('inp'):
            self._insert_data('inp', inp.data.cpu())
        if self._do_collect('out'):
            self._insert_data('out', out.data.cpu())
        # we collect w, b during forward pass since forward pass is minimum requirement for module execution
        if self._do_collect('w'):
            self._insert_data('w', self.collect_weight())
        if self._do_collect('b'):
            self._insert_data('b', self.collect_bias())

    def backward_hook(self, module, grad_in, grad_out):
        if self._do_collect('inp_grad'):
            self._insert_data('inp_grad', grad_in.cpu())
        if self._do_collect('out_grad'):
            self._insert_data('out_grad', grad_out.cpu())
        if self._do_collect('w_grad'):
            self._insert_data('w_grad', module.weight.grad.cpu())
        if self._do_collect('b_grad'):
            self._insert_data('b_grad', module.bias.grad.cpu())

    def register_forward(self):
        self.hook_manager.register_forward_hook(self.forward_hook, self.module,
                                                hook_fn_name='ModuleTracker[%s].forward_hook' % self.module_name,
                                                activate=False, **{self.module_name: self.module})

    def register_backward(self):
        self.hook_manager.register_backward_hook(self.backward_hook,
                                                 hook_fn_name='ModuleTracker[%s].backward_hook' % self.module_name,
                                                 activate=False, **{self.module_name: self.module})

    def register_all(self):
        if self.protocol.track_forward:
            self.register_forward()
        if self.protocol.track_backward:
            self.register_backward()

    @validate_var
    def clear_data_buffer_var(self, var):
        self.data_buffer[var] = []

    def clear_data_buffer_all(self):
        for var in self.data_buffer:
            self.data_buffer[var] = []

    @validate_var
    def gather_var(self, var):
        return torch.cat(self.data_buffer[var], dim=0)
