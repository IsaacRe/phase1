import unittest
from torch.nn import CrossEntropyLoss
from test_utils.test_suite import *
from utils.helpers import *
import torch


class HelpersTestCase(unittest.TestCase):

    network, dataloader = load_test_suite(device=0)
    feature_dim = 10
    activations_1d = torch.zeros(feature_dim, 20)
    activations_2d = torch.zeros(feature_dim, 20, 5, 5)

    def test_find_network_modules_by_name(self):
        module_names = ['conv1', 'layer4.2.conv2', 'layer4.2.bn2']
        modules = find_network_modules_by_name(self.network, module_names)
        for module_name, module in zip(module_names, modules):
            self.assertEqual(get_module(self.network, module_name), module,
                             'Module %s should be found.' % module_name)

    def test_get_named_modules_from_network(self):
        # test without batchnorm
        modules = get_named_modules_from_network(self.network)
        self.assertIn('conv1', modules,
                      'Module conv1 should be in module dict.')
        self.assertIn('layer4.2.conv2', modules,
                      'Module layer4.2.conv2 should be in module dict.')
        self.assertNotIn('layer4.2.bn2', modules,
                         'Module layer4.2.bn2 should not be in module dict.')
        self.assertEqual(modules['conv1'], get_module(self.network, 'conv1'),
                         'Module conv1 should be in module dict.')
        self.assertEqual(modules['layer4.2.conv2'], get_module(self.network, 'layer4.2.conv2'),
                         'Module layer4.2.conv2 should be in module dict.')

        # test with batchnorm
        modules = get_named_modules_from_network(self.network, include_bn=True)
        self.assertIn('layer4.2.bn2', modules,
                      'Module layer4.2.bn2 should be in module dict.')
        self.assertEqual(modules['layer4.2.bn2'], get_module(self.network, 'layer4.2.bn2'),
                         'Module layer4.2.bn2 should be in module dict.')

    def test_data_pass(self):
        # test forward without backward
        data_pass(self.dataloader, self.network, device=0)
        self.assertTrue(True, 'Should complete forward data pass.')

        # test forward + backward
        data_pass(self.dataloader, self.network, device=0, backward_fn=CrossEntropyLoss())
        self.assertTrue(True, 'Should complete forward-backward data pass.')

    def test_flatten_activations(self):
        # test for linear output
        flat_acts = flatten_activations(self.activations_1d)
        self.assertEqual(flat_acts.shape[0], self.feature_dim,
                         'Should flatten 1d activations correctly.')
        self.assertEqual(len(flat_acts.shape), 2,
                         'Should flatten 1d activations correctly.')

        # test for conv output
        flat_acts = flatten_activations(self.activations_2d)
        self.assertEqual(flat_acts.shape[0], self.feature_dim,
                         'Should flatten 2d activations correctly.')
        self.assertEqual(len(flat_acts.shape), 2,
                         'Should flatten 2d activations correctly.')


if __name__ == '__main__':
    unittest.main()
