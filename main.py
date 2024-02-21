from utils.Hasher.hasher import Hasher
from Reward.reward2 import Reward
import logging

with open("example.log", 'r') as log:
    with open('old_log.log', 'w') as file:
        file.write(log.read())
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.INFO, filemode='w')


text = Hasher()
a = Reward(text)