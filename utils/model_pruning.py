import numpy as np
from torch.nn import Module
from torch.utils.data import DataLoader
from utils.hook_management import HookManager
from utils.helpers import Protocol
from utils.model_tracking import ModuleTracker, TrackingProtocol


PRUNE_METHODS = ['weight', 'weight_gradient', 'output', 'output_gradient']

DEFAULT_PRUNE_PROTOCOL = {
    'prune_by': 'weight',
    'fix_prune_ratio': True,
    'prune_ratio': 0.95,
    'prune_across_modules': False,
}

DEFAULT_PRUNE_METHOD_PROTOCOLS = {
    'weight': {
        'weight_threshold': 0.1
    },
    'weight_gradient': {
        'grad_threshold': 0.05
    },
    'output': {
        'output_threshold': 0.4
    },
    'output_gradient': {
        'grad_threshold': 0.2
    }
}


def validate_prune_method(var_idx=0, keyword=None, valid_methods=PRUNE_METHODS):
    def wrapper_fn(fn):
        def new_fn(*args, **kwargs):
            method = None
            if keyword:
                if keyword in kwargs:
                    method = kwargs[keyword]
            else:
                method = args[var_idx]
            if method:
                assert method in valid_methods, 'Invalid prune method, %s, passed to pruner.' \
                                                ' Use a valid prune method key: %s' % (method, ', '.join(valid_methods))
            return fn(*args, **kwargs)

        return new_fn

    return wrapper_fn


class PruneProtocol(Protocol):

    DEFAULT_PROTOCOL = DEFAULT_PRUNE_PROTOCOL

    @validate_prune_method(keyword='prune_by')
    def __init__(self, prune_by='weight', protocol_json=None, **overwrite_protocol):
        super(PruneProtocol, self).__init__()
        self._set_default_method_protocol(prune_by)
        if protocol_json:
            self._add_from_json(protocol_json)
        self._add_protocol(prune_by=prune_by, **overwrite_protocol)

    def _set_default_method_protocol(self, prune_method):
        for protocol, value in DEFAULT_PRUNE_METHOD_PROTOCOLS[prune_method]:
            self.proto_dict[protocol] = value


class ModulePruner:

    def __init__(self, prune_protocol, *modules: Module,
                 hook_manager: HookManager = None,
                 tracker: ModuleTracker = None,
                 load_prune_masks: bool = False,
                 save_prune_masks: bool = False,
                 prune_masks_filepath: str = '',
                 dataloader: DataLoader = None,
                 device=0,
                 network: Module = None,
                 loss_fn=None,
                 **named_modules: Module):
        self.protocol = prune_protocol
        self.prune_masks_filepath = prune_masks_filepath
        self.load_prune_masks = load_prune_masks
        self.save_prune_masks = save_prune_masks

        # setup modules
        self.modules = named_modules
        for m in modules:
            self.modules[repr(m)] = m
        self.module_names = list(self.modules)

        # setup tracker
        if tracker is None:
            if hook_manager is None:
                hook_manager = HookManager()
            self.hook_manager = hook_manager
            self.tracker = self._make_tracker()
        else:
            self.hook_manager = self.tracker.hook_manager
            self.tracker = tracker
            # check that the passed tracker is tracking all passed modules
            for m in self.modules.values():
                assert m.tracker == tracker, "passed ModuleTracker '%s' not assigned to Module '%s'" \
                                             " before ModulePruner initialization." % (tracker, m)

        self._mask_by_method_lookup = {
            method: getattr(self, '_mask_by_%s' % method) for method in PRUNE_METHODS
        }

        self.prune_masks = {module_name: None for module_name in self.module_names}
        self.masks_initialized = False

        self.register_hooks()

        # data_pass args
        if self.protocol.prune_by != 'weight':
            assert not any([arg is None for arg in [dataloader, network]]), \
             "arguments 'dataloader' and 'network' must be specified for pruning option '%s'" % self.protocol.prune_by
        if 'grad' in self.protocol.prune_by:
            assert loss_fn is not None, "argument 'loss_fn' must be specified for pruning option" \
                                        " '%s'" % self.protocol.prune_by
        self.device = device
        self.dataloader = dataloader
        self.network = network
        self.loss_fn = loss_fn

    def _make_tracker(self):
        prune_by = self.protocol.prune_by
        if prune_by in ['weight', 'weight_gradient']:
            vars_ = []
        elif prune_by == 'output':
            vars_ = ['out']
        elif prune_by == 'output_gradient':
            vars_ = ['out_grad']
        return ModuleTracker(TrackingProtocol(*vars_),
                             hook_manager=self.hook_manager, **self.modules)

    def _mask_by_weight(self,
                        prune_across_modules=False,
                        fix_prune_ratio=True,
                        prune_ratio=0.95,
                        weight_threshold=0.5):
        if prune_across_modules:
            raise NotImplementedError()
        else:
            for name, module in self.modules.items():
                weight = self.tracker.collect_weight(name).abs()
                if fix_prune_ratio:
                    weight_threshold = np.percentile(weight, prune_ratio)
                mask = weight < weight_threshold
                self.prune_masks[name] = mask

    def _mask_by_weight_gradient(self,
                                 prune_across_modules=False,
                                 fix_prune_ratio=True,
                                 prune_ratio=0.95,
                                 grad_threshold=0.1):
        # accumulate gradient of weights
        w_grads = self.tracker.aggregate_vars(self.dataloader, network=self.network, device=self.device)
        if prune_across_modules:
            raise NotImplementedError()
        else:
            for name, module in self.modules.items():
                mean_w_grad = w_grads[name]['w_grad'].mean(dim=0).abs()
                if fix_prune_ratio:
                    grad_threshold = np.percentile(mean_w_grad, prune_ratio)
                mask = mean_w_grad < grad_threshold
                self.prune_masks[name] = mask

    def _mask_by_output(self,
                        prune_across_modules=False,
                        fix_prune_ratio=True,
                        prune_ratio=0.95,
                        output_threshold=0.5):
        raise NotImplementedError()

    def _mask_by_output_gradient(self,
                                 prune_across_modules=False,
                                 fix_prune_ratio=True,
                                 prune_ratio=0.95,
                                 grad_threshold=0.5):
        raise NotImplementedError()

    def _set_prune_masks(self, prune_by='weight_magnitude', **prune_kwargs):
        self._mask_by_method_lookup[prune_by](**prune_kwargs)
        self.masks_initialized = True

    def forward_hook(self, module, inp, out):
        """
        Forward hook to conduct pruning of module output after that module's forward pass
        :param module: Module object whose output is being pruned
        :param inp: Tensor input to module
        :param out: Tensor output by module
        :return: Tensor of pruned input
        """
        output_prune_mask = self.prune_masks[module.name]
        assert tuple(output_prune_mask.shape) == tuple(out.shape), \
            "dimensionality of output and prune mask for Module '%s' are not the same." \
            " (%s != %s)" % (module.name, tuple(output_prune_mask.shape), tuple(out.shape))
        out[output_prune_mask] = 0.0
        return out

    def forward_pre_hook(self, module, inp):
        """
        Forward pre-hook to enforce zeroing of all pruned connections in a module's weight
        parameter before each forward pass of that module is conducted
        :param module: Module object whose weights are being pruned
        :param inp: Tensor of input to module
        """
        weight_prune_mask = self.prune_masks[module.name]
        assert tuple(weight_prune_mask.shape) == tuple(module.weight.shape), \
            "dimensionality of output and prune mask for Module '%s' are not the same." \
            " (%s != %s)" % (module.name, tuple(weight_prune_mask.shape), tuple(module.weight.shape))
        module.weight.data[weight_prune_mask] = 0.0

    def register_hooks(self):
        prune_by = self.protocol.prune_by
        # if conducting weight-pruning, register forward_pre_hook
        if 'weight' in prune_by:
            self.hook_manager.register_forward_pre_hook(self.forward_pre_hook,
                                                        hook_fn_name='ModulePruner.forward_pre_hook',
                                                        **self.modules)
        # if conducting output-pruning, register forward_hook
        else:
            self.hook_manager.register_forward_pre_hook(self.forward_hook,
                                                        hook_fn_name='ModulePruner.forward_hook',
                                                        **self.modules)

    def clear_prune_masks_all(self):
        self.prune_masks = {module_name: None for module_name in self.module_names}
        self.masks_initialized = False

    def prune(self, recompute_masks: bool = False, clear_on_exit: bool = False):
        enter_fns = []
        exit_fns = []
        if recompute_masks or not self.masks_initialized:
            enter_fns += [lambda: self._set_prune_masks(**self.protocol)]
        if clear_on_exit:
            exit_fns += [self.clear_prune_masks_all]
        return self.hook_manager.hook_all_context(add_enter_fns=enter_fns,
                                                  add_exit_fns=exit_fns)
