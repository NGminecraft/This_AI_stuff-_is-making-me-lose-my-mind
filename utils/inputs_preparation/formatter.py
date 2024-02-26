from padder import Padder
from tokenizer_handler import TokenizerHandler

class Formatter:
    def __init__(self, starting_text:str=None, tokenizer_handler:TokenizerHandler()=TokenizerHandler(), padder:Padder()=Padder()):
        self.tokenizer = tokenizer_handler
        if starting_text is not None:
            self.tokenizer.refit_tokenizer(starting_text)
        self.padder = padder

    def format(self, obj):
        return self.padder.pad(self.tokenizer.tokenize(obj))

    def add_texts(self, texts:str):
        self.tokenizer.add_texts(texts)