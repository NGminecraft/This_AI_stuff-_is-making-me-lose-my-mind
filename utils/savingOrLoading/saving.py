import pickle
from logging import INFO

class Save:
    def __init__(self, logger):
        self.logger = logger
        
    def save_model(self, model, path, name = ''):
        model.save(path)
        self.logger.log(INFO, f'Saved model {name}')
        
    def save_other(self, object, path):
        with open(path, 'w') as f:
            pickle.dump(object, f, protocol=pickle.HIGHEST_PROTOCOL)