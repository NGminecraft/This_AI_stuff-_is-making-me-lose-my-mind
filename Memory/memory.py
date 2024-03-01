import numpy as np
import os
import pickle
import threading

class Memory:
    def __init__(self, file="Memory/Data/Memory.npy"):
        if os.path.existS(file):
            self.memory_arr = pickle.load(file)
        else:
            self.build_memory(file)
            
    def build_memory(self, file="Memory/Data/Memory.npy"):
        self.object_dict = {}
        
    def new_word(self, word:list|str):
        if type(word) is list:
            for i in word:
                pass