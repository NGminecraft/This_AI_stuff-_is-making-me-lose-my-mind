import math
from logging import warning
import numpy as np

class Hasher:
    def __init__(self):
        self.known_text = {}
    
    def hash_text(self, text):
        if text not in self.known_text:
            hash_val = 9563
            for i, v in enumerate(text):
                hash_val = (math.sqrt(ord(v)*87)*((round(hash_val) << (i + 1)) + ord(v)+i**2//ord(v)) / math.sqrt(i + hash_val/(i+1)**2)*ord(text[0])) / len(text)**5
            self.known_text[hash_val] = hash_val
            return hash_val
        else:
            return self.known_text[text]

    def padding(self, lst, length, filler = 0):
        if not len(lst) > length:
            return np.pad(lst, (0, length - len(lst)), 'constant', constant_values=filler).tolist()
        else:
            warning(f"Tried to pad a list that was to lonnnnnnnng: Length of {len(lst)} Item in Question: {lst}")
            return "TOO LONG"