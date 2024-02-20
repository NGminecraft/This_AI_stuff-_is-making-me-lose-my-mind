import math

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

    def text_padding(self, lst, length, filler = 0):
        if not len(lst) > length:
            for i in range(abs(len(lst) - length)):
                lst.append(filler)
            return lst
        else:
            return "TOO LONG"