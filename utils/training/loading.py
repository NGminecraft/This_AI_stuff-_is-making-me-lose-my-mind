import json
import os
import copy
import logging
import keras_tuner as kt

class Loader:
    def __init__(self, directory, objective='val_loss', logger = None):
        self.directory = directory
        self.objective = objective
        self.logger = logger

    def get_models(self, base_model=None, tuner_obj=None):
        all_losses = {}
        for subdir in os.listdir(self.directory):
            subdir_path = os.path.join(self.directory, subdir)
            if os.path.isdir(subdir_path):
                with open(os.path.join(subdir_path, 'trial.json'), 'r') as f:
                    data = json.load(f)
                    all_losses[data['score']] = os.path.join(subdir_path, 'trial.json')

        losses = copy.deepcopy(all_losses)
        for i in all_losses.keys():
            if i is None or i < 0:
                del losses[i]
        keys = sorted(losses.keys())

        if not self.logger is None:
            self.logger.log(logging.INFO, f'Best loss is {keys[0]} from {losses[keys[0]].split("/")[-2]}')
        else:
            print(f'Best loss is {keys[0]} from {losses[keys[0]].split("/")[-2]}')

        if not tuner_obj is None:
            b = tuner_obj.load_model("/".join([losses[keys[0]].split("/")[0:-1], "checkpoint"]))
            print(b)

if __name__ == '__main__':
    loader = Loader(directory='Reward/Data/Models/Training attempt 2')
    loader.get_models()