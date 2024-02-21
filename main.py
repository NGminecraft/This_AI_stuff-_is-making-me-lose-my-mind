from utils.Hasher.hasher import Hasher
from Reward.reward2 import Reward
import logging
import sys

try:
    with open("logs/Info.log", 'r') as log:
        with open('logs/old_info.log', 'w') as file:
            file.write(log.read())

    with open("logs/Warnings.log", 'r') as log:
        with open('logs/old_warnings.log', 'w') as file:
            file.write(log.read())
except FileNotFoundError:
    print("No previous logs found")

logger = logging.getLogger('MainLogger')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setLevel(logging.INFO)

info_file_handler = logging.FileHandler('logs/Info.log')
info_file_handler.setLevel(logging.INFO)

warning_file_handler = logging.FileHandler('logs/Warnings.log')
warning_file_handler.setLevel(logging.WARNING)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
info_file_handler.setFormatter(formatter)
warning_file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(info_file_handler)
logger.addHandler(warning_file_handler)

logger.log(logging.INFO, 'Initializing')
logger.log(logging.WARNING, 'Initializing Warnings')

text = Hasher()
a = Reward(text, logger=logger)