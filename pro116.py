import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
import statistics as st
import random as rd
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split as tts 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score as AS

data_file = pd.read_csv("c116/Admission_Predict.csv")

data_file_GREScore = data_file["GREScore"].tolist()
data_file_TOEFLScore =  data_file["TOEFLScore"].tolist()
data_file_Chanceofadmit =  data_file["Chanceofadmit"].tolist()

data_file_plot_graph  = px.scatter(x = data_file_GREScore , y =  data_file_TOEFLScore )
# data_file_plot_graph.show()

colors = []

for i in data_file_Chanceofadmit:
    if i ==1 :
        colors.append("green")
    else:
        colors.append("red")
        
        
color_graph =  go.Figure(data = go.Scatter(x = data_file_TOEFLScore , y = data_file_GREScore ,mode =  "markers" , marker = dict(color = colors)))
color_graph.show()

factors = data_file["GREScore"]
outcome = data_file["TOEFLScore"]

salary_train , salary_test , TOEFLScore_train , TOEFLScore_test =  tts(factors,outcome,test_size = 0.25 ,random_state=0)

sc = StandardScaler()

salary_train = sc.fit_transform(salary_train)
salary_test = sc.transform(salary_test)

lr = LogisticRegression(random_state = 0 )

TOEFLScore_prediction = lr.fit(salary_test,TOEFLScore_train)

result  = AS(TOEFLScore_test , TOEFLScore_prediction)
print(result)

