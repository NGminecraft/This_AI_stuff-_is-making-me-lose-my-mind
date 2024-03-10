import pandas
from keras.preprocessing.text import Tokenizer
from utils.inputs_preparation.padder import Padder
import pickle
import os
from Memory.memory import Memory

class TokenizerHandler:
    def __init__(self, load=True, text:str=None, padder=None, tokenizer=Tokenizer(), path:str="Data/", memory:Memory=None):
        self.path = path
        self.cache = {}
        self.memory = memory
        if load:
            self.tokenizer_base = tokenizer
        else:
            self.tokenizer_base = self.load_tokenizer()
        if padder is not None:
            self.padder = None
        else:
            self.padder = Padder()

        
    def list_to_tokens(self, items:list) -> list:
        return self._create_token(items)

    def tokenize(self, text:list) -> list:
        """Alternate naming for turning list into tokens"""
        return self.list_to_tokens(text)

    def save(self):
        with open(f"{self.path}tokenizer.pickle", "wb") as file:
            pickle.dump(self.cache, file, protocol=pickle.HIGHEST_PROTOCOL)

    def load_tokenizer(self):
        with open(f"{self.path}tokenizer.pickle", "rb") as file:
            return pickle.load(file)

    def _create_token(self, item:list|str):
        print('HELLO')
        if type(item) is str:
            item = [item]
        tokenized_list =  []
        if type(item) is pandas.Series:
            item = item.tolist()
        for j in item:
            if type(j) is bool:
                continue
            for i in j.split(' '):
                if i not in self.cache.keys() and i != '':
                    token = self._loadder(i) # This is the thingy actually creating the token.
                    self.cache[i] = token
                    tokenized_list.append(token)
                elif i != '':
                    tokenized_list.append(self.cache[i])
        return tokenized_list
    
    def _loadder(self, item):
        if ' ' in item:
            return [self.memory.get_dict(i) for i in item.split(' ')]
        else:
            return self.memory.get_dict(item)
    
    def load_token(self, item:str|list):
        if self.memory is not None:
            if type(item) is list:
                return [self._loadder(i) for i in item]
            else:
                return self._loadder(item)
        else:
            if type(item) is list:
                return self.tokenize(item)
            else:
                return self.tokenize([item])