import pandas as pd
import numpy as np
import pickle


df = pd.read_csv("salary_predict_dataset.csv")

df["experience"].fillna(0, inplace = True)

df["test_score"].fillna(df["test_score"].mean(),inplace = True)

df["interview_score"].fillna(df["interview_score"].mean(),inplace = True)

def string_to_number(word):
    dict = {"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,
          "eleven":11,"twelve":12, 0:0, "thirteen" : 13,"fifteen" : 15}
    return dict[word]

df["experience"] = df["experience"].apply(lambda x: string_to_number(x))

X = df.iloc[:,:3]

y = df.iloc[:,-1]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X, y)

pickle.dump(regressor, open("model.pkl", "wb"))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))
