# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 00:46:07 2022

@author: prath
"""

from flask import Flask, render_template, request

app=Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def ARG():
    return render_template('index.html', prediction = request.form['Hour'])

app.run(host='127.0.0.1', port=8080, debug=False)