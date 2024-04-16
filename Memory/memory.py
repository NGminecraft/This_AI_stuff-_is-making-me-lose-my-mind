import numpy as np
import os
import pickle
import threading
import logging
from Memory.object_handler import Handler as Oh
from Memory.memory_model import MemModel
from Memory.category import Category
from Memory.object import MemoryObject

class Memory:
    def __init__(self, formatter, exceptions, saver, logger=None, path="Memory/Data", should_log = True, module_loader = None, file_loader=None):
        # Setting up utilities
        # Loading save files
        if file_loader is not None:
            self.file_loader = file_loader
        # Default Data Path
        self.path = path
        # Exceptions
        if exceptions is not None:
            self.exceptions = exceptions
        # Logging
        if logger is not None:
            self.logger = logger
            self.should_log = should_log
        self.formatter = formatter
        # Module for loading classes
        if module_loader is not None:
            self.module_loader = module_loader
#            self.model = module_loader.load(MemModel)
        self.file_saver = saver
        self._load_object_handler(path)
        model_file = f"{path}/Model/memory_model.keras"
        # Memory model
        if file_loader.exists(model_file):
            self.memory_arr = file_loader.load(path + "/Model/memory_model.keras")
        else:
            self._load_object_handler(path)
        # Loading the object handler
        self.all_objects = {}
        if module_loader:
            self.object_handler = module_loader.load(Oh)
        else:
            self.object_handler = Oh()
        if self.should_log:
            self.logger.log(logging.INFO, 'Successfully loaded Memory')
                
    def get_dict(self, key):
        self.logger.log(logging.DEBUG, key)
        if key in self.object_dict.keys():
            return self.object_dict[key]
        else:
            self.logger.log(logging.INFO, f'Word {key} not found. Creating.')
            self._new_word(key)
            
    def Test(self):
        self.model.Test()
        
    def _new_words_dict(self, word:list, category:dict):
        for i in word:
            pass
    
    def _new_words_str(self, word:list, category:str):
        pass
    
    def _new_words_cat(self, word:list, category:Category):
        for i in word:
            self.all_objects[i] = MemoryObject(i, category=category)
        
    def _new_word(self, word:list|str, category:str|Category|list|dict=None, splitsentence=True):
        if type(word) is list:
            for i in word:
                self._new_word(i, category, splitsentence)
        elif type(word) is str:
            if ' ' in word and splitsentence:
                word = word.split(None)
        if type(word) is str:
            word = [word]
        if type(category) is str:
            self._new_words_str(word, category)
        elif type(category) is Category:
            self.new_words_cat(word, category)
        elif type(category) is dict:
            self._new_words_dict(word, category)
            
    def memory_call(self, category=None, prompt=None, max_length=50, *args, **kwargs):
        if category is None and prompt is None:
            raise self.exceptions.NotEnoughArgs(1, 0, "Memory")
        elif category is not None and prompt is None:
            if category in self.category_dict:
                return self.formatter.padder.pad(self.category_dict[category].category_call(), max_length)
            else:
                self.logger.log(logging.ERROR, f'Attempted to accses an invalid category {category}')
                return None
        elif category is None and prompt is not None:
            return self.formatter.padder.pad([i.search_category(prompt=prompt) for i in self.category_dict.values() if i.search_category(prompt=prompt) is not None], max_length)
        else:
            pass
        
    def get_word_id(self, word, category="word"):
        pass
        
    def _load_object_handler(self, path):
        loc = path + "/Data/object_dict.pickle"
        if self.file_loader.exists(loc):
            item = self.file_loader.load(loc)
            if item is None:
                self.object_dict = {}
            else:
                self.object_dict = item
            item1 = self.file_loader.load(loc)
            if item1 is None:
                self.category_dict = {}
            else:
                self.category_dict = item1
        else:
            self.logger.log(logging.INFO, 'No file was found with saved objects, creating a new one.')
            self.object_dict = {}
            self.category_dict = {}
            
    def _load_object_categories(self):
        loc = self.path+"/Data/categories.pickle"
        if os.path.exists(loc):
            self.category_dict = self.file_loader.load(loc)
        else:
            self.category_dict = {}
            for i in ["phonetics", "ideas", "past_prompts", "words"]: # These are all the categories for memory
                self.category_dict[i] = self.module_loader.load(Category, id_name=i)
            
    def save(self, saver):
        saver.save_other(self.object_dict, self.path+"/Data/object_dict.pickle")
        saver.save_other(self.category_dict, self.path+"/Data/categories.pickle")
        saver.save_model(self.model, self.path+"/Model/memory_model.keras", 'memory model')
