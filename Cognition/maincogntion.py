import numpy as np
import os
from Cognition.main_network import main_network


class Cognition:
    def __init__(self, formatter, saver, file_loader, module_loader, logger=None, exceptions=None, path="Cognition/"):
        self.logger = logger
        self.should_log = bool(logger)
        self.exceptions = exceptions
        self.should_exceptions = bool(exceptions)
        self.formatter = formatter
        self.file_save = saver
        self.file_load = file_loader
        self.load_class = module_loader
        self.path = path
        self._init_models()
        
    def _load_models(self):
        location = self.path + "Data/Models/main_cognition.keras"
        if os.path.exists(location):
            self.main_model = self.file_load.load_model(location)
        location = self.path + "Data/Models/word_output.keras"
        if os.path.exists(location):
            self.word_output = self.file_load.load_model(location)
    
    def _init_models(self):
        self.main_network_class = self.load_class.load(main_network)
        self.main = self.main_network_class.init_main_network()