from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load("ev_model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        kwh = float(request.form['kwh'])
        range_val = float(request.form['range'])
        efficiency = float(request.form['efficiency'])
        fast_charge = float(request.form['fast_charge'])
        top_speed = float(request.form['top_speed'])
        acceleration = float(request.form['acceleration'])
        drive = int(request.form['drive'])
        seats = int(request.form['seats'])

        input_data = np.array([[ 
            kwh,
            range_val,
            efficiency,
            fast_charge,
            top_speed,
            acceleration,
            drive,
            seats
        ]])

        print("Input shape:", input_data.shape)

        prediction = model.predict(input_data)
        print("Prediction:", prediction)

        return render_template("index.html",
                               prediction_text=f"Estimated Price: £ {round(prediction[0],2)}")

    except Exception as e:
        print("ERROR:", e)
        return render_template("index.html",
                               prediction_text=f"Error: {e}")


if __name__ == "__main__":
    app.run(debug=True)
