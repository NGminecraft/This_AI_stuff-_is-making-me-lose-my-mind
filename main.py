from Reward.reward2 import Reward
import logging
import sys
import os
from utils.savingOrLoading.loading import FileLoader as file_loader
from utils.inputs_preparation.formatter import Formatter
from utils.savingOrLoading.cls_loader import CLS_Loader as cls_loader
from Memory.memory import Memory
import exceptions
from utils.savingOrLoading.saving import Save as file_save

try:
    with open("logs/Info.log", 'r') as log:
        with open('logs/old_info.log', 'w') as file:
            file.write(log.read())
            os.remove("logs/Info.log")
            

    with open("logs/Warnings.log", 'r') as log:
        with open('logs/old_warnings.log', 'w') as file:
            file.write(log.read())
            os.remove("logs/Warnings.log")
    
    with open("logs/Debug.log", "r") as log:
        with open("logs/old_debug", "w") as file:
            file.write(log.read())
            os.remove("logs/Debug.log")
            
except FileNotFoundError:
    print("No previous logs found")

logger = logging.getLogger('MainLogger')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setLevel(logging.INFO)

debug_handler = logging.StreamHandler(stream=sys.stdout)
debug_handler.setLevel(logging.DEBUG)

debug_file_handler = logging.FileHandler("logs/Debug.log")
debug_file_handler.setLevel(logging.DEBUG)

info_file_handler = logging.FileHandler('logs/Info.log')
info_file_handler.setLevel(logging.INFO)

warning_file_handler = logging.FileHandler('logs/Warnings.log')
warning_file_handler.setLevel(logging.WARNING)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
info_file_handler.setFormatter(formatter)
warning_file_handler.setFormatter(formatter)
debug_handler.setFormatter(formatter)
debug_file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(info_file_handler)
logger.addHandler(warning_file_handler)
logger.addHandler(debug_handler)
logger.addHandler(debug_file_handler)

logger.log(logging.INFO, 'Initializing')
logger.log(logging.WARNING, 'Initializing Warnings')
logger.propagate = False

loader = cls_loader(logger=logger, exceptions=exceptions, formatter=Formatter(logger=logger), file_save=file_save, file_load=file_loader)

memory_class = loader.load(Memory)

loader.formatter.tokenizer.memory = memory_class

loader.formatter = memory_class

reward_class = loader.load(Reward, formatter=Formatter)

loader.begin_save()
