import pandas as pd
import os

ID_NAME = "PassengerId"
TARGET_NAME = "Survived"

#data load
train = pd.read_csv(os.path.join(os.environ["DATA"], "train.csv")).set_index(ID_NAME)
test = pd.read_csv(os.path.join(os.environ["DATA"], "test.csv")).set_index(ID_NAME)
