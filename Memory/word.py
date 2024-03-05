import logging

class Word:
    def __init__(self, values:dict, name_ascii:str, name_num:float, exceptions, logger):
        self.exceptions = exceptions
        self.logger = logger
        self.str_name = name_ascii
        self.number = name_num
        self.val_dict = values
        self.logger.log(logging.INFO, f'Created word {self.str_name}')
        
    def get_dict(self):
        return self.val_dict
    
    def __add__(self, object):
        return self.number + object