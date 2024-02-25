import json
import os
import copy
import logging

class Loader:
    def __init__(self, directory, objective='val_loss', logger = None):
        self.directory = directory
        self.objective = objective
        self.logger = logger

    def get_models(self):
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

if __name__ == '__main__':
    loader = Loader(directory='Reward/Data/Models/Reward Model')
    loader.get_models()