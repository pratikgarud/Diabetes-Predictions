from flask import Flask,render_template,request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('Diabetes_Model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        preg = request.form.get('preg')
        glucose = request.form.get('glucose')
        bp = request.form.get('bp')
        ins = request.form.get('ins')
        bmi = request.form.get('bmi')
        dpf = request.form.get('dpf')
        age = request.form.get('age')
        all_value = np.array([preg,glucose,bp,ins,bmi,dpf,age])
        output = model.predict([all_value])
        pred = np.asscalar(output)
    return render_template('index.html',text1=pred)


if __name__ == '__main__':
    app.run(debug=True)