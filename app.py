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
    
    prediction = float(prediction[0])

    output = round(prediction, 2)
    
    return render_template('index.html', prediction_text='Salary should be $ {}'.format(output))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        # Assuming you expect an array of input values as 'features'
        features = data['features']
        input_features = [int(x) for x in features]
        final_features = [np.array(input_features)
        prediction = model.predict(final_features)

        prediction = float(prediction[0])
        output = round(prediction, 2)

        return jsonify({"prediction": output})
    except Exception as e:
        return jsonify({"error": "Invalid input data"})

if __name__ == "__main__":
    app.run(debug=True)
