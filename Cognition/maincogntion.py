import numpy as np
import os
import glob


class Cognition:
    def __init__(self, formatter, file_save, file_load, logger=None, exceptions=None, path="Cognition/"):
        self.logger = logger
        self.should_log = bool(logger)
        self.exceptions = exceptions
        self.should_exceptions = bool(exceptions)
        self.formatter = formatter
        self.file_save = file_save
        self.file_load = file_load
        self.path = path
        self._load_models()
        
    def _load_models(self):
        location = self.path + "Data/Models/main_cognition.keras"
        if os.path.exists(location):
            self.main_model = self.file_load.load_model(location)
        location = self.path + "Data/Models/word_output.keras"
        if os.path.exists(location):
            self.word_output = self.file_load.load_model(location)