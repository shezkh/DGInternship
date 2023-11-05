import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle

#creating an application by calling flask
app = Flask(__name__)

#loading the model already saved
model = pickle.load(open('model2.pkl', 'rb'))

#creating endpoints for homepage (default) and prediction
@app.route('/')
def home():
    return render_template('index.html')

#this route is to be used with webapp
@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI

    '''
    int_features = [int(x) for x in request.form.values()] 
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    prediction = float(prediction[0])

    output = round(prediction, 2)
    
    return render_template('index.html', prediction_text='Salary should be $ {}'.format(output))

#this route is to be used with api
@app.route('/api_predict/')
def price_predict():
    years = request.args.get('YearsExperience')
        
    test_df = pd.DataFrame({'YearsExperience':[years]})
    
    pred_salary = model.predict(test_df)
    return jsonify({'Your salary should be $': str(np.round(pred_salary,2))})


if __name__ == "__main__":
    app.run(debug=True)
