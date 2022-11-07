# -*- coding: utf-8 -*-
from peaks import assistant
# %% Synthetic time serie 
lengthOfSerie = 250
# lengthOfSerie = 2500
# lengthOfSerie = 25000
Ts = assistant.T(lengthOfSerie)
# T.add_noise(0.5)
Ts.add_sinSerie(-.045, 0.5)
Ts.add_sinSerie(.00045, 0.75)
Ts.add_sinSerie(.05, 0.25)
Ts.add_sinSerie(-.15, 0.25)
Ts.add_sinSerie(-1.15, 0.25)
Ts.add_noise(0.25)

df = Ts.df
# %%
# https://docs.numer.ai/tournament/learn
# convex optimization
# true contribution
# train a model to make predictions on tournament data
# train a model to make predictions on tournament data

import pandas as pd
from xgboost import XGBRegressor

# training data contains features and targets
training_data = df['y']

# tournament data contains features only
# tournament_data = pd.read_csv("numerai_tournament_data.csv").set_index("id")
# feature_names = [f for f in training_data.columns if "feature" in f]

# train a model to make predictions on tournament data
model = XGBRegressor(max_depth=5, learning_rate=0.01, \
                     n_estimators=2000, colsample_bytree=0.1)
model.fit(training_data[feature_names], training_data["target"])

# submit predictions to numer.ai
predictions = model.predict(tournament_data[feature_names])
predictions.to_csv("predictions.csv")
