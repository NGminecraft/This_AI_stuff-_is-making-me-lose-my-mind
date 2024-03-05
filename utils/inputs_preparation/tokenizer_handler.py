import pandas
from keras.preprocessing.text import Tokenizer
from utils.inputs_preparation.padder import Padder
import pickle
import os

class TokenizerHandler:
    def __init__(self, load=True, text:str=None, padder=None, tokenizer=Tokenizer(), path:str="Data/"):
        self.path = path
        self.cache = {}
        if load:
            self.tokenizer_base = tokenizer
        else:
            self.tokenizer_base = self.load_tokenizer()
        if padder is not None:
            self.padder = None
        else:
            self.padder = Padder()
        
    def list_to_tokens(self, items:list) -> list:
        return self.create_token(items)

    def tokenize(self, text:list) -> list:
        """Alternate naming for turning list into tokens"""
        return self.list_to_tokens(text)

    def save(self):
        with open(f"{self.path}tokenizer.pickle", "wb") as file:
            pickle.dump(self.cache, file, protocol=pickle.HIGHEST_PROTOCOL)

    def load_tokenizer(self):
        with open(f"{self.path}tokenizer.pickle", "rb") as file:
            return pickle.load(file)

    def create_token(self, item:list|str):
        if type(item) is str:
            item = [item]
        tokenized_list =  []
        if type(item) is pandas.Series:
            item = item.tolist()
        for j in item:
            print(j)
            for i in j.split(' '):
                if i not in self.cache.keys() and i != '':
                    token = int(''.join([str(ord(j)) for j in i])) / len(i)
                    self.cache[i] = token
                    tokenized_list.append(token)
                elif i != '':
                    tokenized_list.append(self.cache[i])
        return tokenized_list