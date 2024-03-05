import os
import pickle
from Memory.word import Word
import logging


class Handler:
    def __init__(self, exceptions=None, path="Memory/Data/object_master_dict.pickle", issubclass = False, subclass=None, logger=None):
        if issubclass and subclass is not None and type(subclass) is Word:
            pass
        elif issubclass and subclass is not None and type(subclass) is not Word:
            pass
        elif issubclass and subclass is None:
            pass
        elif not subclass:
            self._init_handler_external(path)
        if logger is not None:
            self.logger = logger
            self.should_log = True
        else:
            self.should_log = False
        if exceptions is not None:
            self.exceptions = exceptions
            self.errors = True
        else:
            self.errors = False
        if self.should_log:
            self.logger.log(logging.INFO, 'Loaded Object Handler Successfully')
            
            
    def add(self):
        pass
    
    def _init_handler_external(self, path):
        if os.path.exists(path):
            self.master_dict = pickle.load(path)
        else:
            self.master_dict = {}