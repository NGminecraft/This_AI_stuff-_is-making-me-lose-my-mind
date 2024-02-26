import keras_tuner as kt
from Reward.reward2 import Reward

reward = Reward(build_model = False)
# DO NOT RUN IF TRAINING IS IN PROGRESS
tuner = kt.BayesianOptimization(
    reward._model_build,
    objective='val_loss',
    max_trials=1000,
    directory='Reward/Data/Models',
    project_name='Training attempt 2',
    overwrite=False
)

try:
    b = tuner.oracle.get_trial('trial_0014')
    tuner.hypermodel.build(b)
    b.summary()
except KeyError:
    c = tuner.oracle.get_best_trials(5)
    print(c)