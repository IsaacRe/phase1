import unittest
import numpy as np
from torch.nn import CrossEntropyLoss
from utils.model_tracking import *
from test_utils.test_suite import load_network, get_module, load_dataloader


class TrackingProtocolTestCase(unittest.TestCase):

    def test_set_vars(self):
        include_vars = ['w', 'inp', 'w_grad']
        exclude_vars = ['b', 'out', 'b_grad', 'inp_grad', 'out_grad']
        protocol = TrackingProtocol(*include_vars)

        for var in include_vars:
            self.assertTrue(protocol['track_%s' % var])
        for var in exclude_vars:
            self.assertFalse(protocol['track_%s' % var])

    def test_var_validation(self):
        invalid_var = 'foo'
        error_msg = None
        try:
            TrackingProtocol(invalid_var)
        except AssertionError as e:
            error_msg = str(e)

        self.assertEqual(error_msg, 'Invalid var, %s, passed to tracker.' \
                                    ' Use a valid var key: %s' % (invalid_var,
                                                                  ', '.join(ALL_VARS)))

    def test_set_all_record_every(self):
        record_every = 10
        protocol = TrackingProtocol(*ALL_VARS, record_every=record_every)

        for var in ALL_VARS:
            self.assertEqual(protocol['record_every_%s' % var], record_every)

    def test_set_all_buffer_len(self):
        buffer_len = 10
        protocol = TrackingProtocol(*ALL_VARS, buffer_len=buffer_len)

        for var in ALL_VARS:
            self.assertEqual(protocol['buffer_len_%s' % var], buffer_len)

    def test_set_with_json(self):
        json_file = 'test_protocol.json'
        with open(json_file, 'r') as f:
            protocol_json = json.load(f)
        protocol = TrackingProtocol(protocol_json=json_file)

        for k in protocol_json:
            self.assertEqual(protocol[k], protocol_json[k])

    def test_overwrite_protocol(self):
        overwrite = {
            'track_w': False
        }
        protocol = TrackingProtocol(*ALL_VARS, **overwrite)

        for k in overwrite:
            self.assertEqual(overwrite[k], protocol[k])

    def test_set_forward_backward(self):
        include_vars = []
        protocol = TrackingProtocol(*include_vars)

        self.assertFalse(protocol.track_forward)
        self.assertFalse(protocol.track_backward)

        include_vars = ['w']
        protocol = TrackingProtocol(*include_vars)

        self.assertTrue(protocol.track_forward)
        self.assertFalse(protocol.track_backward)

        include_vars = ['w_grad']
        protocol = TrackingProtocol(*include_vars)

        self.assertFalse(protocol.track_forward)
        self.assertTrue(protocol.track_backward)

        include_vars = ['inp', 'inp_grad']
        protocol = TrackingProtocol(*include_vars)

        self.assertTrue(protocol.track_forward)
        self.assertTrue(protocol.track_backward)


class ModuleTrackerTestCase(unittest.TestCase):

    dataloader = load_dataloader()
    protocol = TrackingProtocol(*ALL_VARS)

    def setUp(self) -> None:
        self.network = load_network()
        self.modules = {
            'conv1': get_module(self.network, 'conv1'),
            'layer1.0.conv1': get_module(self.network, 'layer1.0.conv1'),
            'layer4.2.conv2': get_module(self.network, 'layer4.2.conv2')
        }
        self.hook_manager = HookManager()

    def test_collect_weight(self):
        tracker = ModuleTracker(self.hook_manager, self.protocol,
                                **self.modules)
        for name, module in self.modules.items():
            weight = tracker.collect_weight(name)
            self.assertTrue(np.all((weight == module.weight.data.cpu()).numpy()))

    def test_collect_bias(self):
        tracker = ModuleTracker(self.hook_manager, self.protocol,
                                **self.modules)
        fake_b = torch.nn.Parameter(torch.zeros(2))
        for name, module in self.modules.items():
            module.bias = fake_b
            bias = tracker.collect_bias(name)
            self.assertTrue(np.all((bias == module.bias.data.cpu()).numpy()))

    def test_register_all(self):
        tracker = ModuleTracker(self.hook_manager, self.protocol,
                                **self.modules)
        for name, module in self.modules.items():
            self.assertIn(name, self.hook_manager.name_to_module)
            self.assertEqual(module, self.hook_manager.name_to_module[name])

    def _check_len_var_buffer(self, tracker, length, *vars_):
        for name in self.modules:
            for var in vars_:
                self.assertEqual(len(tracker.data_buffer[name][var]), length)

    def test_hooks(self):

        tracker = ModuleTracker(self.hook_manager, self.protocol,
                                **self.modules)
        self.hook_manager.activate_all_hooks()
        data_pass(self.dataloader, self.network, device=0, backward_fn=CrossEntropyLoss(), early_stop=1)

        self._check_len_var_buffer(tracker, 1, *GRAPH_VARS)
        self._check_len_var_buffer(tracker, 0, *(STATE_VARS + STATE_GRAD_VARS))

    def test_clear_data_buffer_all(self):
        tracker = ModuleTracker(self.hook_manager, self.protocol,
                                **self.modules)
        self.hook_manager.activate_all_hooks()
        data_pass(self.dataloader, self.network, device=0, backward_fn=CrossEntropyLoss(), early_stop=1)

        vars_ = ['inp', 'inp_grad']
        tracker.clear_data_buffer_all(vars_=vars_)

        self._check_len_var_buffer(tracker, 0, *vars_)
        self._check_len_var_buffer(tracker, 1, *['out', 'out_grad'])

        tracker.clear_data_buffer_all()
        self._check_len_var_buffer(tracker, 0, *ALL_VARS)

    def test_track(self):
        tracker = ModuleTracker(self.hook_manager, self.protocol,
                                **self.modules)
        with tracker.track():
            data_pass(self.dataloader, self.network, device=0, backward_fn=CrossEntropyLoss(), early_stop=1)

            self._check_len_var_buffer(tracker, 1, *GRAPH_VARS)

        self._check_len_var_buffer(tracker, 0, *ALL_VARS)

    def test_gather_module_var(self):
        tracker = ModuleTracker(self.hook_manager, self.protocol,
                                **self.modules)
        self.hook_manager.activate_all_hooks()
        data_pass(self.dataloader, self.network, device=0, backward_fn=CrossEntropyLoss(), early_stop=2)

        batch_size = 100

        vars_ = ['inp', 'out', 'inp_grad', 'out_grad']
        for name in self.modules:
            for var in vars_:
                tensor = tracker.gather_module_var(name, var)
                self.assertEqual(tensor.shape[0], batch_size * 2)


if __name__ == '__main__':
    unittest.main()
