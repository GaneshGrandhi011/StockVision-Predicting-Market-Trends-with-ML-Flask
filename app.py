from flask import Flask, request, render_template
import os

# This line helps reduce some of the TensorFlow startup messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Import your predict_stock function
from model import predict_stock

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        ticker = request.form.get("ticker").upper()
        if not ticker:
            return render_template("index.html", error="Please enter a stock ticker.")
        
        # Call the function from model.py and get the results dictionary
        results = predict_stock(ticker)
        
        # Pass the entire results dictionary to the template
        return render_template("index.html", results=results)

    except Exception as e:
        # Handle errors gracefully (e.g., invalid ticker)
        error_message = f"An error occurred: {e}"
        return render_template("index.html", error=error_message)

if __name__ == "__main__":
    app.run(debug=True)