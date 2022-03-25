# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 23:18:39 2021

@author: Rushi
"""
#api libraries
from flask import Flask, request, render_template

#predicting libraries
from keras.models import model_from_json
import joblib
import os

DATA_PATH = "D:/RK/Marwadi University/Sem-7/Project/base"
os.chdir(DATA_PATH)

#loading trained model and weights
json_file = open("DNN.json","r")
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("DNN.h5")
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#loading predefined scaler
norm_x = joblib.load("scaler.nor")

def predict_sepsis(X):
    input_f = X
    transformed_input_f = norm_x.transform(input_f)
    output = model.predict(transformed_input_f)
    rounded = [round(x[0]) for x in output]
    output = rounded
    return output
    

app = Flask(__name__)


@app.route('/', methods=["GET","POST"])
def result():
    if request.method == "POST":
        hr = request.form.get('hr')
        o2sat = request.form.get('o2sat')
        temp = request.form.get('temp')
        sbp = request.form.get('sbp')
        mp = request.form.get('mp')
        dbp = request.form.get('dbp')
        resp = request.form.get('resp')
        etco2 = request.form.get('etco2')
        basex = request.form.get('basex')
        hco3 = request.form.get('hco3')
        fio2 = request.form.get('fio2')
        ph = request.form.get('ph')
        paco2 = request.form.get('paco2')
        sao2 = request.form.get('sao2')
        ast = request.form.get('ast')
        bun = request.form.get('bun')
        alkpho = request.form.get('alkpho')
        calc = request.form.get('calc')
        chol = request.form.get('chol')
        creati = request.form.get('creati')
        bilird = request.form.get('bilird')
        gluc = request.form.get('gluc')
        lact = request.form.get('lact')
        magne = request.form.get('magne')
        phos = request.form.get('phos')
        potas = request.form.get('potas')
        bilirt = request.form.get('bilirt')
        trap = request.form.get('trop')
        hct = request.form.get('hct')
        hgb = request.form.get('hgb')
        ptt = request.form.get('ptt')
        wbc = request.form.get('wbc')
        fibri = request.form.get('fibri')
        plate = request.form.get('plate')
        age = request.form.get('age')
        gender = request.form.get('gender')
        if (gender != 1 or gender != 0):
            gender = gender.lower()
            if gender == "female":
                gender = 0
            else:
                gender = 1
        hosat = request.form.get('hosat')
        iculos = request.form.get('iculos')
        
        features = [[hr,o2sat,temp,sbp,mp,dbp,resp,etco2,basex,
                hco3,fio2,ph,paco2,sao2,ast,bun,alkpho,calc,
                chol,creati,bilird,gluc,lact,magne,phos,potas,
                bilirt,trap,hct,hgb,ptt,wbc,fibri,plate,age,
                gender,hosat,iculos]]
        
        sepsis_prediction = predict_sepsis(features)
        
        if (sepsis_prediction[0] == 1):
            return("The patient will develop Sepsis.")
        else:
            return("The patient will not develop Sepsis.")
        
    return render_template("sepsis.html")

    
    


if __name__ == "__main__":
    app.run()