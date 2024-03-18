import pickle
import inspect
from logging import INFO, WARN

class Save:
    def __init__(self, logger):
        self.logger = logger
        self.classes = []
        
    def save_model(self, model, path, name = ''):
        model.save(path)
        self.logger.log(INFO, f'Saved model {name}')
        
    def save_other(self, object, path):
        with open(path, 'wb') as f:
            pickle.dump(object, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    def new_object(self, obj):
        self.classes.append(obj)
        self.logger.log(INFO, f'loaded object {type(obj).__name__}')
        
    def save_all(self): 
        for i in self.classes:
            try:
                kwargs = {}
                a = inspect.getfullargspec(i.save)
                if "saver" in a[0]:
                    kwargs["saver"] = self
                i.save(**kwargs)
            except AttributeError:
                self.logger.log(WARN, f'Object {type(i).__name__} has no method save()')
                continue