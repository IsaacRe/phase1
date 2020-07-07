import unittest
from torch.nn import CrossEntropyLoss
from test_utils.test_suite import *
from utils.hook_management import *
from utils.helpers import data_pass


class HookManagerTestCase(unittest.TestCase):

    dataloader = load_dataloader()

    def setUp(self) -> None:
        self.network = load_network()
        self.hook_manager = HookManager()
        self.f_set = None
        self.b_set = None
        self.f_pre_set = None
        self.b_pre_set = None

    def _f_hook(self, module, *args):
        self.assertTrue(hasattr(module, 'name'),
                        'Hooked module should have name attribute set.')
        self.f_set = module.name

    def _b_hook(self, module, *args):
        self.assertTrue(hasattr(module, 'name'),
                        'Hooked module should have name attribute set.')
        self.b_set = module.name

    def _f_pre_hook(self, module, *args):
        self.assertTrue(hasattr(module, 'name'),
                        'Hooked module should have name attribute set.')
        self.f_pre_set = module.name

    def check_lookups(self, module, module_name, hook_fn, hook_fn_name):
        # Module lookup checks
        self.assertEqual(module.name, module_name)
        self.assertIn(module_name, self.hook_manager.name_to_module)
        self.assertIn(module, self.hook_manager.modules)
        self.assertEqual(self.hook_manager.name_to_module[module_name], module)

        # HookFunction lookup checks
        self.assertIn(hook_fn_name, self.hook_manager.name_to_hookfn)
        self.assertIn(hook_fn, self.hook_manager.function_to_hookfn)
        hook_fn_wrapper = self.hook_manager.function_to_hookfn[hook_fn]
        self.assertIn(hook_fn_wrapper, self.hook_manager.hook_fns)
        self.assertEqual(self.hook_manager.name_to_hookfn[hook_fn_name], hook_fn_wrapper)

        self.assertEqual(hook_fn_wrapper.name, hook_fn_name)

        # HookHandle lookup checks
        handle_name = '%s[%s]' % (hook_fn_name, module_name)
        self.assertIn(handle_name, self.hook_manager.name_to_hookhandle)
        handle = self.hook_manager.name_to_hookhandle[handle_name]

        self.assertEqual(handle_name, handle.name)
        self.assertIn(handle, hook_fn_wrapper.handles)
        self.assertIn(module, hook_fn_wrapper.module_to_handle)
        self.assertEqual(hook_fn_wrapper.module_to_handle[module], handle)

    def test_register_forward_hook(self):
        module = get_module(self.network, 'conv1')
        self.hook_manager.register_forward_hook(self._f_hook, hook_fn_name='f_hook',
                                                activate=False, conv1=module)
        self.check_lookups(module, 'conv1', self._f_hook, 'f_hook')

    def test_register_backward_hook(self):
        module = get_module(self.network, 'conv1')
        self.hook_manager.register_backward_hook(self._b_hook, hook_fn_name='b_hook',
                                                 activate=False, conv1=module)
        self.check_lookups(module, 'conv1', self._b_hook, 'b_hook')

    def test_register_forward_pre_hook(self):
        module = get_module(self.network, 'conv1')
        self.hook_manager.register_forward_pre_hook(self._f_pre_hook, hook_fn_name='f_pre_hook',
                                                    activate=False, conv1=module)
        self.check_lookups(module, 'conv1', self._f_pre_hook, 'f_pre_hook')

    def register_multiple_modules(self):
        module1 = self.network._modules['conv1']
        module2 = get_module(self.network, 'layer4.2.conv2')
        self.hook_manager.register_forward_hook(self._f_hook, hook_fn_name='f_hook',
                                                activate=False, **{'conv1': module1, 'layer4.2.conv2': module2})
        self.check_lookups(module1, 'conv1', self._f_hook, 'f_hook')
        self.check_lookups(module2, 'layer4.2.conv2', self._f_hook, 'f_hook')

    def activate_all(self, module):
        self.hook_manager.register_forward_hook(self._f_hook, hook_fn_name='f_hook',
                                                activate=True, conv1=module)
        self.hook_manager.register_backward_hook(self._b_hook, hook_fn_name='b_hook',
                                                 activate=True, conv1=module)
        self.hook_manager.register_forward_pre_hook(self._f_pre_hook, hook_fn_name='f_pre_hook',
                                                    activate=True, conv1=module)

    def test_activate_hooks(self):
        module = get_module(self.network, 'conv1')
        self.activate_all(module)
        data_pass(self.dataloader, self.network, backward_fn=CrossEntropyLoss(), early_stop=1)

        self.assertEqual(self.f_set, 'conv1')
        self.assertEqual(self.f_pre_set, 'conv1')
        self.assertEqual(self.b_set, 'conv1')

    def test_deactivate_hooks(self):
        module = get_module(self.network, 'conv1')
        self.activate_all(module)
        self.hook_manager.deactivate_all_hooks()
        data_pass(self.dataloader, self.network, backward_fn=CrossEntropyLoss(), early_stop=1)

        self.assertIsNone(self.f_set)
        self.assertIsNone(self.b_set)
        self.assertIsNone(self.f_pre_set)

    def test_hook_module_context(self):
        module = get_module(self.network, 'conv1')
        self.hook_manager.register_forward_hook(self._f_hook, hook_fn_name='f_hook',
                                                activate=False, conv1=module)
        with self.hook_manager.hook_module_context(module):
            self.assertTrue(self.hook_manager.name_to_hookhandle['f_hook[conv1]'].is_active())

        self.assertFalse(self.hook_manager.name_to_hookhandle['f_hook[conv1]'].is_active())

    def test_hook_module_context_by_name(self):
        module = get_module(self.network, 'conv1')
        self.hook_manager.register_forward_hook(self._f_hook, hook_fn_name='f_hook',
                                                activate=False, conv1=module)
        with self.hook_manager.hook_module_context_by_name('conv1'):
            self.assertTrue(self.hook_manager.name_to_hookhandle['f_hook[conv1]'].is_active())

        self.assertFalse(self.hook_manager.name_to_hookhandle['f_hook[conv1]'].is_active())

    def test_hook_all_context(self):
        module1 = self.network._modules['conv1']
        module2 = get_module(self.network, 'layer4.2.conv2')
        self.hook_manager.register_forward_hook(self._f_hook, hook_fn_name='f_hook',
                                                activate=False, **{'conv1': module1, 'layer4.2.conv2': module2})

        with self.hook_manager.hook_all_context():
            self.assertTrue(self.hook_manager.name_to_hookhandle['f_hook[conv1]'].is_active())
            self.assertTrue(self.hook_manager.name_to_hookhandle['f_hook[layer4.2.conv2]'].is_active())

        self.assertFalse(self.hook_manager.name_to_hookhandle['f_hook[conv1]'].is_active())
        self.assertFalse(self.hook_manager.name_to_hookhandle['f_hook[layer4.2.conv2]'].is_active())


if __name__ == '__main__':
    unittest.main()