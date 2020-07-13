import unittest
import torch
from utils.model_pruning import *
from test_utils.test_suite import *


class PruneProtocolTestCase(unittest.TestCase):

    def test_prune_by_weight(self):
        protocol = PruneProtocol()

    def test_prune_by_weight_grad(self):
        protocol = PruneProtocol(prune_by='weight_gradient')


class ModulePrunerTestCase(unittest.TestCase):

    dataloader = load_dataloader()

    def setUp(self) -> None:
        self.network = load_network()
        self.modules = {
            'conv1': get_module(self.network, 'conv1'),
        }
        self.hook_manager = HookManager()

    def test_mask_by_weight(self):
        protocol = PruneProtocol(prune_by='weight')
        pruner = ModulePruner(protocol, hook_manager=self.hook_manager, network=self.network,
                              **self.modules)
        pruner._mask_by_weight()
        pruner._mask_by_weight(fix_prune_ratio=False)

    def test_mask_by_weight_grad(self):
        protocol = PruneProtocol(prune_by='weight_gradient')
        pruner = ModulePruner(protocol, hook_manager=self.hook_manager, network=self.network,
                              dataloader=self.dataloader, loss_fn=torch.nn.CrossEntropyLoss(),
                              **self.modules)
        pruner._mask_by_weight_gradient()
        pruner._mask_by_weight_gradient(fix_prune_ratio=False)

    def test_prune_all_modules(self):
        protocol = PruneProtocol()
        pruner = ModulePruner(protocol, hook_manager=self.hook_manager, network=self.network,
                              dataloader=self.dataloader, loss_fn=torch.nn.CrossEntropyLoss())
        pruner._mask_by_weight()

    def set_prune_all_weights(self, pruner):
        for name, module in self.modules.items():
            pruner.prune_masks[name] = slice(None)

    def test_forward_hook(self):
        module = self.modules['conv1']
        protocol = PruneProtocol(prune_by='weight')
        pruner = ModulePruner(protocol, hook_manager=self.hook_manager, network=self.network,
                              conv1=module)
        self.set_prune_all_weights(pruner)
        _, input, _ = next(iter(self.dataloader))
        input = input.to(0)
        output = pruner.forward_hook(module, input, module(input))
        self.assertEqual(output.mean(), 0)

    def test_forward_pre_hook(self):
        module = self.modules['conv1']
        protocol = PruneProtocol(prune_by='weight')
        pruner = ModulePruner(protocol, hook_manager=self.hook_manager, network=self.network,
                              conv1=module)
        self.set_prune_all_weights(pruner)
        _, input, _ = next(iter(self.dataloader))
        input = input.to(0)
        pruner.forward_pre_hook(module, input)
        self.assertEqual(module.weight.mean(), 0)

    def test_clear_prune_masks(self):
        protocol = PruneProtocol(prune_by='weight')
        pruner = ModulePruner(protocol, hook_manager=self.hook_manager, network=self.network,
                              **self.modules)
        pruner._mask_by_weight()
        pruner.clear_prune_masks()
        self.assertFalse(pruner.masks_initialized)
        self.assertTrue(all([mask is None for mask in pruner.prune_masks.values()]))

    def test_prune(self):
        protocol = PruneProtocol(prune_by='weight')
        pruner = ModulePruner(protocol, hook_manager=self.hook_manager, network=self.network,
                              **self.modules)
        with pruner.prune():
            for name, module in self.modules.items():
                self.assertEqual(tuple(module.weight.shape), tuple(pruner.prune_masks[name].shape))


if __name__ == '__main__':
    unittest.main()
