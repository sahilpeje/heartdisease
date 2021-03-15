from flask import Flask,render_template,request
import pickle
import jsonify
import sklearn
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open('G:\\HeartDisease\\knn_model.pkl', 'rb'))


@app.route('/',methods=['GET'])
def home():
	return render_template('final.html')

standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():    
    if request.method== 'POST':
        Age=int(request.form['age'])
        Bp=int(request.form['trestbps'])
        Chol=int(request.form['chol'])
        Thalach=int(request.form['thalach'])
        Oldpeak=float(request.form['oldpeak'])
        ca=int(request.form['ca'])
        sex=int(request.form['sex'])
        cp=int(request.form['cp'])
        fbs=int(request.form['fbs'])   
        restecg=int(request.form['restecg'])
        exang=int(request.form['exang'])
        slope=int(request.form['slope'])  
        thal=int(request.form['thal'])
        
        prediction = np.array([[Age,Bp,Chol,Thalach,Oldpeak,ca,fbs,restecg,sex,cp,exang,slope,thal]])
        #data=np.array([temp_pr])
        #prediction=model.transform(prediction)
        
        Y_pred=model.predict(prediction)
        #my_pred=int(knn.predict(data)[0])
        
        if Y_pred==1:
            return render_template('final.html',prediction_text="You have not heart disease")
        else:
            return render_template('final.html',prediction_text="Sorry You have heart disease")
    else:
        return render_template('final.html')

if __name__=="__main__":
    app.run(debug=True)
        
        
        
        

        
        
        

            


        