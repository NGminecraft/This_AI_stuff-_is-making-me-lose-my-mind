import numpy as np
import os
from Cognition.main_network import main_network
from logging import INFO, DEBUG
from Cognition.cognition_loss import Loss


class Cognition:
    def __init__(self, formatter, saver, file_loader, module_loader, logger=None, exceptions=None, path="Cognition/", memory=None):
        self.logger = logger
        self.should_log = bool(logger)
        self.exceptions = exceptions
        self.should_exceptions = bool(exceptions)
        self.memory = memory
        self.formatter = formatter
        self.file_save = saver
        self.file_load = file_loader
        self.load_class = module_loader
        self.path = path
        self.prev = [0]*100
        self._init_models()
    
    def _init_models(self):
        self.main_network_class = self.load_class.load(main_network)
        self.main = self.main_network_class.init_main_network()
        
    def call(self, new_input:str, prev_output=None, mem_call=None, auto_seralize:bool=True) -> str:
        # Checking the new_input and seeing if it needs adjustment
        if mem_call is None and auto_seralize and type(new_input) is str and self.should_exceptions and self.memory is not None:
            self.exceptions.NotEnoughArgs(1, 0, "Cognition call. Needs Memory")
        memory_input = [self.memory.memory_call(category="words", prompt=i) for i in new_input.split(" ")]
        if mem_call is None:
            mem_call = self.memory.memory_call(category="ideas", prompt=new_input)
        if prev_output is None:
            prev_output = self.prev
        result = self.main(new_input, memory_input, prev_output)
        self.prev = result
        
    def save(self, path=None):
        if path is None:
            path = self.path+"/Data"
        self.logger.log(DEBUG, "Save Recieved")
        self.main_network_class.save_model(path+"/Model/main_model.keras")