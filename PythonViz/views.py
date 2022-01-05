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
X = df1.drop('NObeyesdad', 1)
Y=df1['NObeyesdad']

from sklearn.ensemble import ExtraTreesClassifier
algorithme = ExtraTreesClassifier(n_estimators=100, random_state=0)
algorithme.fit(X, Y)

def Predictions(X,algorithme):
    L=algorithme.predict(X)
    return [round(i,0) for i in L]

app = Flask(__name__)


@app.route("/")
def main():
    print("test")
    return render_template('index.html')

@app.route('/', methods=['POST'])
def homepredict():
    print(int(request.form['caloric-select']))
    
    columns=['Gender', 'Age','family_history_with_overweight',
       'frequent_cons_high_caloric_food', 'frequency_cons_veg', 'nb_main_meal',
       'cons_between_meal', 'SMOKE', 'CH2O', 'Suivi_calories',
       'frequency_phys_act', 'time_tech', 'cons_alcohol', 'MTRANS',]
    X=pd.dataframe(columns=columns)
    X=X.append({'Gender':int(request.form['gender-select']), 'Age':int(request.form['age']), 'family_history_with_overweight':int(request.form['fhwo-select']),
       'frequent_cons_high_caloric_food':int(request.form['caloric-select']), 'frequency_cons_veg':int(request.form['veg']), 'nb_main_meal':int(request.form['meal']),
       'cons_between_meal':int(request.form['betw_meal-select']), 'SMOKE':int(request.form['smoke-select']), 'CH2O':int(request.form['water']), 'Suivi_calories':int(request.form['caloric_track-select']),
       'frequency_phys_act':int(request.form['activity']), 'time_tech':int(request.form['techno']), 'cons_alcohol':int(request.form['alcohol-select']), 'MTRANS':int(request.form['transport-select'])})
    

    p=Predictions(X,algorithme)
    print(p)
    
    return render_template('index.html',res=p)
                  
if __name__ == "__main__":
    app.run(debug=True)