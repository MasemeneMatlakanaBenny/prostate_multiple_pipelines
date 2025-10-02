import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

dataset=pd.read_csv("prostate.csv")

train_df,test_df=train_test_split(dataset,test_size=0.2,random_state=42)

train_df.to_csv("train_prostate.csv")
test_df.to_csv("test_prostate.csv")  