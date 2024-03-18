import inspect
from logging import INFO

class CLS_Loader:
    def __init__(self, logger, exceptions, formatter, file_save, file_load, saver=None):
        self.logger = logger
        self.exception_file = exceptions
        self.formatter = formatter
        self.file_save = file_save(self.logger)
        if saver == None:
            self.saver = self.file_save
        self.file_load = self.load(file_load)
        self.saver.new_object(self)
        
    def load(self, obj, **kwargs):
        if not hasattr(self, 'saver'):
            self.saver = None
        a = inspect.getfullargspec(obj)
        if 'exceptions' in a[0]:
            kwargs['exceptions'] = self.exception_file
        if 'logger' in a[0]:
            kwargs['logger'] = self.logger
        if 'module_loader' in a[0]:
            kwargs['module_loader'] = self
        if 'formatter' in a[0]:
            kwargs['formatter'] = self.formatter
        if 'saver' in a[0]:
            kwargs['saver'] = self.file_save
        if 'file_loader' in a[0]:
            kwargs['file_loader'] = self.file_load
        loaded_object = obj(**kwargs)
        if self.saver is not None:
            self.saver.new_object(loaded_object)
        return loaded_object
    
    def begin_save(self):
        self.logger.log(INFO, 'Request recieved to save')
        self.saver.save_all()