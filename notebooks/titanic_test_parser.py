import pandas as pd
import os


ID_NAME = "PassengerId"
TARGET_NAME = "Survived"

train = pd.read_csv(os.path.join(os.environ["DATA"], "train.csv")).set_index(ID_NAME)
test = pd.read_csv("test.csv").set_index(ID_NAME)