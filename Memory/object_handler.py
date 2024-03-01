import os
import pickle


class Handler:
    def __init__(self, path="Memory/Data/object_master_dict.pickle"):
        if os.path.exists(path):
            self.master_dict = pickle.load(path)
        else:
            self.master_dict = {}