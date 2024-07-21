from flask import Flask, render_template, request
import pickle
import numpy as np
import xgboost as xgb

# Load the saved model
model = pickle.load(open("rf.pkl", "rb"))

# Load the scaler
with open("apple_scaler.pkl", "rb") as f:
    sc = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

@app.route("/index")
def index():
    return render_template("index.html")

@app.route('/result', methods=["POST"])
def prediction():
    Size = request.form["Size"]
    Weight = request.form["Weight"]
    Sweetness = request.form["Sweetness"]
    Crunchiness = request.form["Crunchiness"]
    Juiciness = request.form["Juiciness"]
    Ripeness = request.form["Ripeness"]
    Acidity = request.form["Acidity"]

    x_test = [[float(Size), float(Weight), float(Sweetness), float(Crunchiness), float(Juiciness), float(Ripeness), float(Acidity)]]
    print(x_test)

    # Scale the input features
    x_test = sc.transform(np.array(x_test))
    
    # Predict the quality
    prediction = model.predict(x_test)

 # Determine if the quality is "Good" or "Bad"
    if prediction == 0:
        text = "👎 Bad Quality: The apples show signs of poor quality, with suboptimal size, sweetness, and firmness, indicating they are not ready for harvesting."
    else:
        text = "👍 Good Quality: The apples are of excellent quality, with ideal size, sweetness, and firmness, making them perfect for harvesting."

    return render_template("result.html", prediction_text=text)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=False)
