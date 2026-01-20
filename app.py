from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the saved model
model_path = os.path.join('model', 'titanic_survival_model.pkl')
model = joblib.load(model_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        try:
            # Collect inputs from form
            data = {
                'Pclass': [int(request.form['Pclass'])],
                'Sex': [request.form['Sex']],
                'Age': [float(request.form['Age'])],
                'SibSp': [int(request.form['SibSp'])],
                'Parch': [int(request.form['Parch'])]
            }
            
            df_input = pd.DataFrame(data)
            prediction = model.predict(df_input)[0]
            
            result = "Survived" if prediction == 1 else "Did Not Survive"
        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)