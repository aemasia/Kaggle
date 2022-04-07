import pandas as pd
import os

ID_NAME = "PassengerId"
TARGET_NAME = "Survived"

#data load
train = pd.read_csv(os.path.join(os.environ["DATA"], "train.csv")).set_index(ID_NAME)
test = pd.read_csv(os.path.join(os.environ["DATA"], "test.csv")).set_index(ID_NAME)

#feature engineering and selection
train["SibSp_bin"] = (train["SibSp"] > 0).astype(int)
test["SibSp_bin"] = (test["SibSp"] > 0).astype(int)
train["Parch_bin"] = (train["Parch"] > 0).astype(int)
test["Parch_bin"] = (test["Parch"] > 0).astype(int)

train["Title"] = train["Name"].apply(lambda x: x.split(", ")[1].split(".")[0])
test["Title"] = test["Name"].apply(lambda x: x.split(", ")[1].split(".")[0])

train["Title"].value_counts()
test["Title"].value_counts()

num_features = [
    "Age",
    #"SibSp",
    #"Parch",
    "Fare",
]
cat_features = [
    "Pclass",
    "Sex",
    "SibSp_bin",
    "Parch_bin",
    #"Embarked",
]

#data preprocessing
#missing data

train[num_features + cat_features + [TARGET_NAME]].info()
test[num_features + cat_features].info()

title_age_dict = train.groupby("Title")["Age"].mean().to_dict()
train["Age"] = train["Age"].fillna(train["Title"].map(title_age_dict))
train[num_features + cat_features + [TARGET_NAME]].info()

test["Age"] = test["Age"].fillna(test["Title"].map(title_age_dict))
test[num_features + cat_features].info()