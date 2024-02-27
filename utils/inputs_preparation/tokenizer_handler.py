from keras.preprocessing.text import Tokenizer
from utils.inputs_preparation.padder import Padder
import pickle
import os

class TokenizerHandler:
    def __init__(self, load=True, text:str=None, padder=None, tokenizer=Tokenizer(), path:str="Data/"):
        self.path = path
        if load:
            self.tokenizer_base = tokenizer
        else:
            self.tokenizer_base = self.load_tokenizer()
        if padder is not None:
            self.padder = None
        else:
            self.padder = Padder()
        if text is not None:
            self.current_texts = text
            self.tokenizer_base.fit_on_texts(self.refit_tokenizer(text))
        else:
            self.current_texts = []

    def refit_tokenizer(self, text:str):
        self.tokenizer_base.fit_on_texts(text)

    def append_token(self, token:list):
        if type(token) is str:
            token = [token]
        self.refit_tokenizer(" ".join(self._append_to_current_texts(token)))

    def _append_to_current_texts(self, item:list):
        for i in item:
            if i not in self.current_texts:
                self.current_texts.append(i)
        return self.current_texts

    def list_to_tokens(self, items:list) -> list:
        return self.tokenizer_base.texts_to_sequences(items)

    def tokenize(self, text:list, padding:bool=False) -> list:
        """Alternate naming for turning list into tokens"""
        return self.list_to_tokens(text)

    def save(self):
        with open(f"{self.path}tokenizer.pickle", "wb") as file:
            pickle.dump(self.tokenizer_base, file, protocol=pickle.HIGHEST_PROTOCOL)

    def load_tokenizer(self):
        with open(f"{self.path}tokenizer.pickle", "rb") as file:
            return pickle.load(file)
