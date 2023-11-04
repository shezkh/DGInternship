import numpy as np
from flask import Flask, request, render_template
import pickle

#creating an application by calling flask
app = Flask(__name__)

#loading the model already saved
model = pickle.load(open('model2.pkl', 'rb'))

#creating endpoints for homepage (default) and prediction
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI

    '''
    int_features = [int(x) for x in request.form.values()] 
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = round(prediction[0], 2)
    
    return render_template('index.html', prediction_text='Salary should be $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)