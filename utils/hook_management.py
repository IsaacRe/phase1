from utils.helpers import CustomContext


def increment_name(name):
    """
    Increments the number appended to the name and returns the resulting string
    :param name: the name to be incremented
    :return: the name with its numeric suffix incremented
    """
    if '_' in name and name.split('_')[-1].isnumeric():
        count = int(name.split('_')[-1]) + 1
        return '_'.join(name.split('_')[:-1]) + '_%d' % count
    return name + '_1'


def detach_hook(module, inp, out):
    return out.detach()


class HookFunction:

    HOOK_TYPES = ['forward_hook', 'backward_hook', 'forward_pre_hook', 'backward_pre_hook']

    def __init__(self, hook_fn, hook_type, name=None, modules=None):
        assert hook_type in self.HOOK_TYPES
        if name is None:
            name = repr(hook_fn)
        self.name = name
        self.hook_type = hook_type
        self.function = hook_fn
        self.handles = []
        self.module_to_handle = {}
        if modules:
            self.register(*modules)

    def __call__(self, *args, **kwargs):
        self.function(*args, **kwargs)

    def register(self, *modules, activate=True):
        handles = []
        for module in modules:
            assert hasattr(module, 'name'), 'Module must be given name before hook registration'
            assert module not in self.module_to_handle, \
                'Hook function %s was already registered with module %s' % (self.name, module.name)
            name = self.name + '[' + module.name + ']'
            handles += [HookHandle(self, module, name, activate=activate)]
        self.handles += handles
        return handles


class HookHandle:

    def __init__(self, hook_fn, module, name, activate=True):
        self.hook_fn = hook_fn
        self.handle = None
        self.module = module
        self.name = name
        if activate:
            self.activate()

    def __repr__(self):
        return '<%s %s registered to %s (%s)>' % (self.name, type(self), self.module.name,
                                                  'active' if self.is_active() else 'inactive')

    def is_active(self):
        return self.handle is not None

    def activate(self, raise_on_active=False):
        if self.is_active():
            if raise_on_active:
                raise AssertionError('Cannot activate hook: Hook is already active')
            return
        register_fn = 'register_%s' % self.hook_fn.hook_type
        assert hasattr(self.module, register_fn), 'Module %s has no method %s' % (repr(self.module), register_fn)
        self.handle = getattr(self.module, register_fn)(self.hook_fn.function)

    def deactivate(self, raise_on_inactive=False):
        if not self.is_active():
            if raise_on_inactive:
                raise AssertionError('Cannot deactivate hook: Hook is already inactive')
            return
        self.handle.remove()
        self.handle = None

    def set_name(self, name):
        self.name = name


class Hook:

    HOOK_TYPES = ['forward_hook', 'backward_hook', 'forward_pre_hook', 'backward_pre_hook']

    def __init__(self, hook_fn, hook_type, name=None, module=None, module_name=None, activate=True):
        assert hook_type in self.HOOK_TYPES
        if name is None:
            name = repr(hook_fn)
        self.name = name
        self.hook_type = hook_type
        self.hook_fn = hook_fn
        self.handle = None
        self.module = None
        self.module_name = None
        if module is not None:
            self.register(module, module_name=module_name)
        if activate:
            self.activate(raise_on_active=True)

    def __repr__(self):
        return '<%s %s registered to %s (%s)>' % (self.name, type(self), self.module_name,
                                                  'active' if self.is_active() else 'inactive')

    def __call__(self, *args, **kwargs):
        self.hook_fn(*args, **kwargs)

    def is_active(self):
        return self.handle is not None

    def activate(self, raise_on_active=False):
        if self.is_active():
            if raise_on_active:
                raise AssertionError('Cannot activate hook: Hook is already active')
            return
        register_fn = 'register_%s' % self.hook_type
        assert hasattr(self.module, register_fn), 'Module %s has no method %s' % (repr(self.module), register_fn)
        self.handle = getattr(self.module, register_fn)(self.hook_fn)

    def register(self, module, module_name=None):
        self.module = module
        self.module_name = repr(module) if module_name is None else module_name

    def deactivate(self, raise_on_inactive=False):
        if not self.is_active():
            if raise_on_inactive:
                raise AssertionError('Cannot deactivate hook: Hook is already inactive')
            return
        self.handle.remove()
        self.handle = None

    def set_name(self, name):
        self.name = name


class HookManager:
    """
    Class for centralized handling of PyTorch module hooks
    """

    def __init__(self):
        # Lookup tables
        self.hook_fns = set()
        self.modules = set()
        self.name_to_hookfn = {}  # HookFunction.name -> HookFunction
        self.name_to_hookhandle = {}  # HookHandle.name -> HookHandle
        self.module_to_hookhandle = {}  # HookHandle.module -> List[HookHandle]
        self.name_to_module = {}  # Module.name -> Module
        self.function_to_hookfn = {}  # HookFunction.function -> HookFunction

    def __getattr__(self, item):
        """
        Allow call to register any valid hook type explicitly
        :param item: register_<hook_type> where hook_type is a valid hook type
        :return: HookManager.register_hook function handle with the hook_type parameter filled
        """
        if item.count('_') >= 2 and 'register_' == item[:9]:
            hook_type = '_'.join(item.split('_')[1:])
            if hook_type not in HookFunction.HOOK_TYPES:
                raise AttributeError('%s object has no attribute %s' % (self.__class__, item))
            return lambda *args, **kwargs: self.register_hook(hook_type, *args, **kwargs)
        else:
            raise AttributeError("'%s' object has no attribute '%s'" % (self.__class__, item))

    def register_hook(self, hook_type, function, *modules, hook_fn_name=None,
                      activate=True, **named_modules):
        # Check if HookFunction obj has already been created for the given function
        if function in self.function_to_hookfn:
            hook_fn = self.function_to_hookfn[function]
        else:
            if hook_fn_name in self.name_to_hookfn:
                hook_fn_name = increment_name(hook_fn_name)
            hook_fn = HookFunction(function, hook_type, name=hook_fn_name)
            self.hook_fns = self.hook_fns.union({hook_fn})
            self.name_to_hookfn[hook_fn.name] = hook_fn

        # Check if modules have already been registered with another hook function
        named_modules = [(None, module) for module in modules] + list(named_modules.items())
        for module_name, module in named_modules:
            if module not in self.modules:
                self.modules = self.modules.union({module})
                self.module_to_hookhandle[module] = []
                if module_name is None:
                    module_name = repr(module)
                if module_name in self.name_to_module:
                    module_name = increment_name(module_name)
                self.name_to_module[module_name] = module
                module.name = module_name
        modules = [m for name, m in named_modules]

        # Make sure module names were assigned properly
        assert all([hasattr(m, 'name') for name, m in named_modules])

        # Create hook handle
        handles = hook_fn.register(*modules, activate=activate)

        # Update hook handle lookup tables
        for module, handle in zip(modules, handles):
            self.module_to_hookhandle[module] += [handle]
            self.name_to_hookhandle[handle.name] = handle

    def activate_hook_by_name(self, hook_name):
        self.name_to_hookhandle[hook_name].activate()

    def deactivate_hook_by_name(self, hook_name):
        self.name_to_hookhandle[hook_name].deactivate()

    def get_module_hooks(self, module, hook_types=[], include_active=True, include_inactive=True):
        for h in self.module_to_hookhandle[module]:
            if len(hook_types) > 0 and h.hook_fn.function not in hook_types:
                continue
            if not include_inactive and not h.is_active():
                continue
            if not include_active and h.is_active():
                continue
            yield h

    def get_module_hooks_by_name(self, module_name, hook_types=[], **kwargs):
        return self.get_module_hooks(self.name_to_module[module_name], hook_types=hook_types, **kwargs)

    def activate_module_hooks(self, *modules, hook_types=[]):
        for module in modules:
            for h in self.get_module_hooks(module, hook_types=hook_types, include_active=False):
                h.activate()

    def activate_module_hooks_by_name(self, *module_names, hook_types=[]):
        for module_name in module_names:
            for h in self.get_module_hooks_by_name(module_name, hook_types=hook_types, include_active=False):
                h.activate()

    def deactivate_module_hooks(self, *modules, hook_types=[]):
        for module in modules:
            for h in self.get_module_hooks(module, hook_types=hook_types, include_inactive=False):
                h.deactivate()

    def deactivate_module_hooks_by_name(self, *module_names, hook_types=[]):
        for module_name in module_names:
            for h in self.get_module_hooks_by_name(module_name, hook_types=hook_types, include_inactive=False):
                h.deactivate()

    def activate_all_hooks(self, hook_types=[]):
        self.activate_module_hooks(*self.modules, hook_types=hook_types)

    def deactivate_all_hooks(self, hook_types=[]):
        self.deactivate_module_hooks(*self.modules, hook_types=hook_types)

    ########################  Context Management  #######################################

    def hook_module_context(self, *modules, hook_types=[], add_enter_fns=[], add_exit_fns=[]):
        enter_fns = [lambda: self.activate_module_hooks(*modules, hook_types=hook_types)]
        exit_fns = [lambda: self.deactivate_module_hooks(*modules, hook_types=hook_types)]
        for fn in add_enter_fns:
            enter_fns += [fn]
        for fn in add_exit_fns:
            exit_fns += [fn]
        return CustomContext(enter_fns=enter_fns, exit_fns=exit_fns)

    def hook_module_context_by_name(self, *module_names, hook_types=[], add_enter_fns=[], add_exit_fns=[]):
        modules = [self.name_to_module[module_name] for module_name in module_names]
        return self.hook_module_context(*modules,
                                        hook_types=hook_types,
                                        add_enter_fns=add_enter_fns,
                                        add_exit_fns=add_exit_fns)

    def hook_all_context(self, hook_types=[], add_enter_fns=[], add_exit_fns=[]):
        return self.hook_module_context(*self.modules,
                                        hook_types=hook_types,
                                        add_enter_fns=add_enter_fns,
                                        add_exit_fns=add_exit_fns)
