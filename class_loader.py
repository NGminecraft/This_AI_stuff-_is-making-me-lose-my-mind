import inspect

class Loader:
    def __init__(self, logger, exceptions):
        self.logger = logger
        self.exception_file = exceptions
        
    def load(self, obj, **kwargs):
        a = inspect.getfullargspec(obj)
        if 'exceptions' in a[0]:
            kwargs['exceptions'] = self.exception_file
        if 'logger' in a[0]:
            kwargs['logger'] = self.logger
        if 'module_loader' in a[0]:
            kwargs['module_loader'] = self
        return obj(**kwargs)