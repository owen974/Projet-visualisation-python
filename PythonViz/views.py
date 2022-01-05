# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 14:30:27 2022

@author: prath
"""

from flask import Flask, render_template, request
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl

df1=pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
df1 = df1.rename(columns={'FAVC': 'frequent_cons_high_caloric_food', 'FCVC': 'frequency_cons_veg', 
                           'NCP': 'nb_main_meal', 'CAEC': 'cons_between_meal', 
                           'SCC': 'Suivi_calories', 'FAF': 'frequency_phys_act', 
                           'TUE': 'time_tech', 'CALC': 'cons_alcohol'})
df1['Gender'] = df1['Gender'].replace(['Male','Female'],[0,1])
df1['family_history_with_overweight'] = df1['family_history_with_overweight'].replace(['yes','no'],[1,0])
df1['frequent_cons_high_caloric_food'] = df1['frequent_cons_high_caloric_food'].replace(['yes','no'],[1,0])
df1['cons_between_meal'] = df1['cons_between_meal'].replace(['no','Sometimes','Frequently','Always'],[0,1,2,3])
df1['SMOKE'] = df1['SMOKE'].replace(['yes','no'],[1,0])
df1['Suivi_calories'] = df1['Suivi_calories'].replace(['yes','no'],[1,0])
df1['MTRANS'] = df1['MTRANS'].map({'Public_Transportation':0,
                             'Walking':1,'Automobile':2,
                             'Motorbike':3,'Bike':4},na_action=None)
df1['cons_alcohol'] = df1['cons_alcohol'].map({'no':0,
                             'Sometimes':1,'Frequently':2,
                             'Always':3},na_action=None)
df1['NObeyesdad'] = df1['NObeyesdad'].map({'Insufficient_Weight':0,
                             'Normal_Weight':1,'Overweight_Level_I':2,
                             'Overweight_Level_II':3,'Obesity_Type_I':4,
                            'Obesity_Type_II':5,'Obesity_Type_III':6},na_action=None)

X = df1.drop('NObeyesdad', 1)
X=X.drop('Height',1)
X=X.drop('Weight',1)
Y=df1['NObeyesdad']


from sklearn import svm
svr = svm.SVR(kernel='linear')
from sklearn.model_selection import GridSearchCV
parameters = {  'gamma' : [0.01, 0.1, 0.5]           }
grid1       = GridSearchCV(svm.SVR(), parameters, n_jobs=-1, cv=5)
grid1.fit(X,Y)

parameters = {  'C'      : [0.5, 1, 1.5]             ,
                'gamma'  : [0.5, 0.1, 0.15]      }
grid2 = GridSearchCV(svm.SVR(), parameters, n_jobs=-1)
grid2.fit(X,Y)

from sklearn.ensemble import BaggingRegressor
bagging = BaggingRegressor()
bagging.fit(X, Y)

from sklearn.ensemble import AdaBoostClassifier
adaboost = AdaBoostClassifier(n_estimators=50,learning_rate=1)
adaboost.fit(X, Y)

from sklearn.ensemble import ExtraTreesClassifier
extratree = ExtraTreesClassifier(n_estimators=100, random_state=0)
extratree.fit(X, Y)

from sklearn.ensemble import GradientBoostingRegressor
gradboosting = GradientBoostingRegressor(max_depth=1)
gradboosting.fit(X, Y)

from sklearn.ensemble import RandomForestClassifier
rdforest = RandomForestClassifier(max_depth=2, random_state=0)
rdforest.fit(X, Y)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X, Y)

df1['NObeyesdad'] = df1['NObeyesdad'].map({'Insufficient_Weight':0,
                             'Normal_Weight':1,'Overweight_Level_I':2,
                             'Overweight_Level_II':3,'Obesity_Type_I':4,
                            'Obesity_Type_II':5,'Obesity_Type_III':6},na_action=None)

def Predictions(X,algorithme):
    L=algorithme.predict(X)
    
    x=L[0]
    x = ('Insufficient_Weight' if x==0 else 'Normal_Weight' if x==1 else 'Overweight_Level_I' if x==2 else 'Overweight_Level_II' if x==3 else 'Obesity_Type_I' if x==4 else 'Obesity_Type_II' if x==5 else 'Obesity_Type_III')
    return x

app = Flask(__name__)


@app.route("/")
def main():
    print("test2")
    return render_template('index.html')

@app.route('/', methods=['POST'])
def homepredict():
    
    
    columns=['Gender', 'Age','family_history_with_overweight',
       'frequent_cons_high_caloric_food', 'frequency_cons_veg', 'nb_main_meal',
       'cons_between_meal', 'SMOKE', 'CH2O', 'Suivi_calories',
       'frequency_phys_act', 'time_tech', 'cons_alcohol', 'MTRANS',]
    X=pd.DataFrame(columns=columns)
    
    gender=request.form['gender-select']
    gender = (0 if gender == 'Male' else 1)
    fwho=request.form['fhwo-select']
    fwho = (1 if fwho=='yes' else 0)
    cal=request.form['caloric_track-select']
    cal = (1 if cal=='yes' else 0)
    btw_meals=request.form['betw_meal-select']
    btw_meals = (1 if btw_meals=='yes' else 0)
    smoke=request.form['smoke-select']
    smoke= (0 if smoke=='No' else 1 if smoke=='Sometimes' else 2 if smoke=='Frequently' else 3)
    cal_track=request.form['caloric_track-select']
    cal_track = (1 if cal_track=='yes' else 0)
    alcohol=request.form['alcohol-select']
    alcohol= (0 if alcohol=='No' else 1 if alcohol=='Sometimes' else 2 if alcohol=='Frequently' else 3)
    transp=request.form['transport-select']
    transp= (0 if transp=='Public transportation' else 1 if transp=='Walking' else 2 if transp=='Automobile' else 3 if transp=='Motorbike' else 4)
    
    
    X=X.append({'Gender':gender, 'Age':int(request.form['age']), 'family_history_with_overweight':fwho,
       'frequent_cons_high_caloric_food':cal, 'frequency_cons_veg':int(request.form['veg']), 'nb_main_meal':int(request.form['meal']),
       'cons_between_meal':btw_meals, 'SMOKE':smoke, 'CH2O':int(request.form['water']), 'Suivi_calories':cal_track,
       'frequency_phys_act':int(request.form['activity']), 'time_tech':int(request.form['techno']), 'cons_alcohol':alcohol, 'MTRANS':transp},ignore_index=True)
    
    model=request.form['activity']
    
    algo = (grid1 if model=='GridSearchCv-1 parameter' else grid2 if model=='GridSearchCv-2 Sparameters' else gradboosting if model=='Gradient Boosting' else bagging if model=='Bagging' else extratree if model=='Extra Tree'else rdforest if model=='Random Forest' else adaboost if model=='Adaboost' else knn)
    
    p='PREDICTION : '
    p+=Predictions(X,algo)
    
    return render_template('index.html',res=p)
                  
if __name__ == "__main__":
    app.run(debug=True)