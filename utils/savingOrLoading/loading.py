import pickle
import os

class file_loader:
    def __init__(self, path="Data/all_data.pickle"):
        self.path = path
        if os.path.exists(path):
            self.exists = True
        else:
            self.exists = False
        self.items = []
    
    def load_all(self):
        if self.exists:
            with open(self.path, "rb") as file:
                return pickle.load(file)
            
    def save(self):
        with open(self.path, "w") as file:
            pickle.dump(self.items, file)
            
    def add_object(self, item):
        self.items.append(item)
        return self.items
            
    def __bool__(self):
        return self.exists
    