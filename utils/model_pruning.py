import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from utils.hook_management import HookManager
from utils.helpers import Protocol, get_named_modules_from_network
from utils.model_tracking import ModuleTracker, TrackingProtocol


PRUNE_METHODS = ['weight', 'weight_gradient', 'output', 'online', 'output_gradient']
"""
The following pruning functionalities are implemented:
    weight:
        When prune() is called, all weights with lowest magnitude will be pruned. While pruner is active,
        these weights will be set to zero before each forward pass.
    weight_gradient:
        When prune() is called, gradient wrt weight will be evaluated for all weights using provided
        dataloader. Weights whose magnitude of average gradient over all data is the lowest will be pruned.
        While pruner is active, these weights will be set to zero before each forward pass.
    output:
        When prune() is called, outputs of all neurons will be computed for all neurons using provided
        dataloader. Neurons with the lowest average output value will be pruned.
        While pruner is active, these neuron outputs will be set to zero after each layer's forward pass.
        The altered output will be provided as the next layer's input.
    online:
        When prune() is called, pruning threshold will be computed (if not specified). While pruner is active,
        neuron outputs that fall below this threshold will be set to zero after each layer's
        forward pass. The altered output will be provided as the next layer's input.
    output_gradient:
        When prune() is called, gradient wrt neuron output will be computed for all neurons using provided
        dataloader. Neurons whose average magnitude of gradient over all data is the lowest will be pruned.
        While pruner is active, these neuron outputs will be set to zero after each layer's forward pass.
        The altered output will be provided as the next layer's input.
"""

DEFAULT_PRUNE_PROTOCOL = {
    'prune_by': 'weight',
    'fix_prune_ratio': True,
    'prune_ratio': 0.95,
    'prune_across_modules': False,
    'load_prune_masks': False,
    'save_prune_masks': False,
}

DEFAULT_PRUNE_METHOD_PROTOCOLS = {
    'weight': {
        'prune_masks_filepath': 'prune-masks/prune-by-weight.npz',
        'prune_threshold': 0.1
    },
    'weight_gradient': {
        'prune_masks_filepath': 'prune-masks/prune-by-weight-grad.npz',
        'prune_threshold': 0.05
    },
    'output': {
        'prune_masks_filepath': 'prune-masks/prune-by-output.npz',
        'prune_threshold': 0.4
    },
    'online': {
        'prune_masks_filepath': None,
        'prune_threshold': 0.4
    },
    'output_gradient': {
        'prune_masks_filepath': 'prune-masks/prune-by-output-grad.npz',
        'prune_threshold': 0.2
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
    def __init__(self, prune_by='weight', protocol_json=None, namespace=None, **overwrite_protocol):
        super(PruneProtocol, self).__init__()
        self._set_default_method_protocol(prune_by)
        if protocol_json:
            self._add_from_json(protocol_json)
        if namespace:
            self._add_from_namespace(namespace)
        self.overwrite_protocol(prune_by=prune_by, **overwrite_protocol)

    def _set_default_method_protocol(self, prune_method):
        for protocol, value in DEFAULT_PRUNE_METHOD_PROTOCOLS[prune_method].items():
            self.proto_dict[protocol] = value


class ModulePruner:

    def __init__(self, prune_protocol, *modules: Module,
                 hook_manager: HookManager = None,
                 tracker: ModuleTracker = None,
                 dataloader: DataLoader = None,
                 device=0,
                 network: Module = None,
                 loss_fn=None,
                 **named_modules: Module):
        # if no modules specified, prune all modules in network
        if len(modules) == 0 and len(named_modules) == 0:
            assert network is not None, 'no modules or network object passed to ModulePruner'
            named_modules = get_named_modules_from_network(network)

        self.protocol = prune_protocol

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

        self._setup_method_lookup = {
            method: getattr(self, '_setup_%s_pruning' % method) for method in PRUNE_METHODS
        }

        # var to track thresholds that will be used for online pruning by output
        self.online_thresholds = {module_name: None for module_name in self.module_names}
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
        if prune_by == 'weight':
            vars_ = []
        elif prune_by == 'weight_gradient':
            vars_ = ['w_grad']
        elif prune_by in ['output', 'online']:
            vars_ = ['out']
        elif prune_by == 'output_gradient':
            vars_ = ['out_grad']
        return ModuleTracker(TrackingProtocol(*vars_),
                             hook_manager=self.hook_manager, **self.modules)

    def _load_prune_masks(self, load_file):
        raise NotImplementedError()

    def _load_online_thresholds(self, load_file):
        raise NotImplementedError()

    def _save_prune_masks(self, save_file):
        raise NotImplementedError()

    def _save_online_thresholds(self, save_file):
        raise NotImplementedError()

    def _setup_weight_pruning(self,
                              prune_across_modules=False,
                              fix_prune_ratio=True,
                              prune_ratio=0.95,
                              prune_threshold=0.5):
        if prune_across_modules:

            max_prune, min_prune = 0, 1
            max_prune_name, min_prune_name = None, None

            if fix_prune_ratio:
                # gather all weights into single tensor
                all_weights = []
                for name in self.modules:
                    all_weights += [self.tracker.collect_weight(name).cpu().flatten()]
                all_weights = torch.cat(all_weights).abs().numpy()
                # compute single prune_threshold for all modules
                prune_threshold = np.percentile(all_weights, prune_ratio * 100.)
            for name, module in self.modules.items():
                weight = self.tracker.collect_weight(name).abs()
                mask = weight < prune_threshold

                amt_pruned = len(np.where(mask)[0]) / mask.flatten().shape[0]
                if amt_pruned < min_prune:
                    min_prune = amt_pruned
                    min_prune_name = name
                if amt_pruned > max_prune:
                    max_prune = amt_pruned
                    max_prune_name = name

                self.prune_masks[name] = mask
        else:
            for name, module in self.modules.items():
                weight = self.tracker.collect_weight(name).abs()
                if fix_prune_ratio:
                    # compute prune threshold for the current module
                    prune_threshold = np.percentile(weight, prune_ratio * 100.)
                mask = weight < prune_threshold
                self.prune_masks[name] = mask

    def _setup_weight_gradient_pruning(self,
                                       prune_across_modules=False,
                                       fix_prune_ratio=True,
                                       prune_ratio=0.95,
                                       prune_threshold=0.1):
        # accumulate gradient of weights
        w_grads = self.tracker.aggregate_vars(self.dataloader, network=self.network, device=self.device)
        if prune_across_modules:
            raise NotImplementedError()
        else:
            for name, module in self.modules.items():
                mean_w_grad = w_grads[name]['w_grad'].mean(dim=0).abs()
                if fix_prune_ratio:
                    prune_threshold = np.percentile(mean_w_grad, prune_ratio * 100.)
                mask = mean_w_grad < prune_threshold
                self.prune_masks[name] = mask

    def _setup_output_pruning(self,
                              prune_across_modules=False,
                              fix_prune_ratio=True,
                              prune_ratio=0.95,
                              prune_threshold=0.5):
        raise NotImplementedError()

    def _setup_online_pruning(self,
                              prune_across_modules=False,
                              fix_prune_ratio=True,
                              prune_ratio=0.95,
                              prune_threshold=0.5):
        self.tracker.collect_vars(self.dataloader, network=self.network, device=self.device)
        if prune_across_modules:
            raise NotImplementedError()
        else:
            if fix_prune_ratio:
                for name, module in self.modules.items():
                    outs = self.tracker.gather_module_var(name, 'out')
                    self.tracker.clear_data_buffer_module(name)
                    outs = np.maximum(outs, 0)
                    self.online_thresholds[name] = np.percentile(outs, prune_ratio * 100.)
            else:
                self.online_thresholds = {module_name: prune_threshold for module_name in self.module_names}

    def _setup_output_gradient_pruning(self,
                                       prune_across_modules=False,
                                       fix_prune_ratio=True,
                                       prune_ratio=0.95,
                                       prune_threshold=0.5):
        raise NotImplementedError()

    def _setup_pruning(self,
                       prune_by='weight_magnitude',
                       load_prune_masks=False,
                       save_prune_masks=False,
                       prune_masks_filepath=None,
                       **prune_kwargs):
        if load_prune_masks:
            if prune_by == 'online':
                self._load_online_thresholds(prune_masks_filepath)
            else:
                self._load_prune_masks(prune_masks_filepath)
        else:
            self._setup_method_lookup[prune_by](**prune_kwargs)
        if save_prune_masks:
            if prune_by == 'online':
                self._save_online_thresholds(prune_masks_filepath)
            else:
                self._save_prune_masks(prune_masks_filepath)
        self.masks_initialized = True

    def _mask_online(self, module, output):
        """
        Performs masking of output for online pruning
        :param module: Module whose output is being pruned
        :param output: Tensor output by module
        :return: Tensor of pruned output
        """
        output[output < self.online_thresholds[module.name]] = 0.0
        return output

    def forward_hook(self, module, input, output):
        """
        Forward hook to conduct pruning of module output after that module's forward pass
        :param module: Module object whose output is being pruned
        :param input: Tensor input to module
        :param output: Tensor output by module
        :return: Tensor of pruned output
        """
        output_prune_mask = self.prune_masks[module.name]
        if not isinstance(output_prune_mask, slice):
            assert tuple(output_prune_mask.shape) == tuple(output.shape), \
                "dimensionality of output and prune mask for Module '%s' are not the same." \
                " (%s != %s)" % (module.name, tuple(output_prune_mask.shape), tuple(output.shape))
        output[output_prune_mask] = 0.0
        return output

    def forward_pre_hook(self, module, input):
        """
        Forward pre-hook to conduct pruning of module weight before that module's forward pass
        parameter before each forward pass of that module is conducted
        :param module: Module object whose weights are being pruned
        :param input: Tensor of input to module
        """
        weight_prune_mask = self.prune_masks[module.name]
        if not isinstance(weight_prune_mask, slice):
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
                                                        activate=False,
                                                        **self.modules)
        elif prune_by == 'online':
            self.hook_manager.register_forward_hook(self._mask_online,
                                                    hook_fn_name='ModulePruner._mask_online',
                                                    activate=False,
                                                    **self.modules)
        # if conducting output-pruning, register forward_hook
        else:
            self.hook_manager.register_forward_pre_hook(self.forward_hook,
                                                        hook_fn_name='ModulePruner.forward_hook',
                                                        activate=False,
                                                        **self.modules)

    def set_protocol(self, **overwrite_protocol):
        self.protocol.overwrite_protocol(**overwrite_protocol)

    def clear_prune_masks(self):
        self.prune_masks = {module_name: None for module_name in self.module_names}
        self.online_thresholds = {module_name: None for module_name in self.modules}
        self.masks_initialized = False

    def prune(self, recompute_masks: bool = False, clear_on_exit: bool = False):
        enter_fns = []
        exit_fns = []
        if recompute_masks or not self.masks_initialized:
            enter_fns += [lambda: self._setup_pruning(**self.protocol)]
        if clear_on_exit:
            exit_fns += [self.clear_prune_masks]
        return self.hook_manager.hook_all_context(hook_types=[self.forward_hook,
                                                              self.forward_pre_hook,
                                                              self._mask_online],
                                                  add_enter_fns=enter_fns,
                                                  add_exit_fns=exit_fns)

    def get_prune_masks(self):
        return {module_name: self.prune_masks[module_name] for module_name in self.module_names}

    def set_prune_masks(self, **module_masks):
        for module_name, prune_mask in module_masks.items():
            self.prune_masks[module_name] = prune_mask

    def compute_prune_masks(self, reset=False):
        assert self.protocol.prune_by != 'online', \
            'cannot compute prune masks staticly when conducting online pruning'
        self._setup_pruning(**self.protocol)
        mask_dict = self.get_prune_masks()
        if reset:
            self.clear_prune_masks()
        return mask_dict
