from flask import  Flask,request,jsonify
import pandas as pd
import pickle
import numpy as np
from flask_cors import CORS



app=Flask(__name__)
CORS(app)
# note
# When your frontend sends a request to the backend, the browser says:
# "Nope! You can't talk to 5000 from 5500."
# This is because modern browsers have a rule called CORS.
# CORS = Cross-Origin Resource Sharing
# The browser says:
# "I don’t trust other websites by default."
# " I’ll block requests made from one place (like port 5500) to another (like port 5000) unless the server (5000) says it’s okay."

# Solution: flask-cors
# This is a helper tool in Flask to tell the browser:
# "Hey browser, I allow other websites (like frontend on 5500) to talk to me."
# So if you use flask-cors, your Flask app will say:
# ✅ “It’s okay, let them send requests to me.”



with open("model.pkl","rb") as f:
    model=pickle.load(f)

with open("carname_encoder.pkl","rb") as f:
     carname_le = pickle.load(f)

with open("fueltype_encoder.pkl", "rb") as f:
    fueltype_le = pickle.load(f)

with open("carbody_encoder.pkl", "rb") as f:
    carbody_le = pickle.load(f)



@app.route("/")
def home():
    return "welcome to the car prediaction website"

@app.route("/predict",methods=["POST"])
def predict():
    try:
        data=request.get_json() 
        
        # data = request.get_json()
        # Extracts the JSON data sent from the frontend
        # Converts it to a Python dictionary

        df=pd.DataFrame([data])




        # When someone sends data (like a form or JSON) to your Flask server, you use request to receive that data.

        # request.get_json()
        # Meaning:
        # It is used when data is sent in JSON format (like JavaScript object), usually from a frontend like React or Postman.
        # jnson data in directory formet we have to convert into list for makeing datafreame
        # Data is JSON (like a dictionary).
        # You access values like: data['name']
        # # It is used in API development, React, or JavaScript-based frontend.

#         project/
# │
# ├── app.py                         # Flask backend with CORS and API route
# ├── model.pkl                      # Trained model
# ├── carname_encoder.pkl            # LabelEncoder for CarName
# ├── fueltype_encoder.pkl           # LabelEncoder for fueltype
# ├── carbody_encoder.pkl            # LabelEncoder for carbody
# │
# └── frontend/
#     └── index.html                 # React + Tailwind form, calls /predict via fetch()


        
        # ✅ request.form
        # # <form method="POST" action="/predict">
        #    <input type="text" name="name">
        #     <button type="submit">Submit</button>
        #      </form>
        # Data is form data (not JSON).
        # You access values like: request.form['name']
        # It is used in simple HTML forms (no JavaScript).

#         project/
# │
# ├── app.py                         # Main Flask backend
# ├── model.pkl                      # Trained ML model
# ├── carname_encoder.pkl            # LabelEncoder for CarName
# ├── fueltype_encoder.pkl           # LabelEncoder for fueltype
# ├── carbody_encoder.pkl            # LabelEncoder for carbody
# │
# └── templates/
#     └── form.html                  # HTML form for user input




# JSON (JavaScript Object Notation)
# {"name": "Alice", "age": 25}
# ✅ Standard text format ->it's not python directory data type
# Readable by JavaScript, APIs, etc.



        # checking data is given or not
        if not data:
            return jsonify({"error":"data is not found"})
        
        # jsonify()
        # It's a Flask function that converts a Python dictionary to a JSON response.
        # JSON is used to send data to the frontend (HTML, JS, React) or other APIs 
        # Flask can't return dictionaries directly to the browser.
        # The browser/React frontend understands only JSON.
        # So jsonify() prepares the response correctly with the right Content-Type.




        
        req_col=['CarName','fueltype','carbody','enginesize','stroke','horsepower']
        # checking all box data is present or not

        if not all(col in df.columns for col in req_col):
         return  jsonify({"error": "All required fields are not present"})
        
        # encoded the cat data

        df['CarName'] = carname_le.transform(df['CarName'])
        df['fueltype'] = fueltype_le.transform(df['fueltype'])
        df['carbody'] = carbody_le.transform(df['carbody'])

        features = df[['CarName','fueltype','carbody','enginesize','stroke','horsepower']]

        prediction = model.predict(features)

        return jsonify({"price": round(prediction[0], 2)})
    
    except Exception as e:
        return jsonify({"error": str(e)})
    





if __name__=="__main__":
    app.run(debug=True)



# how to run 
# html file->go live
# run->flask server->python app.py

# explantion
# // JS code sends a POST request to Flask
# fetch("http://127.0.0.1:5000/predict", {
#   method: "POST",
#   headers: {
#     "Content-Type": "application/json"  // JSON format
#   },
#   body: JSON.stringify({
#     CarName: "honda",
#     fueltype: "gas",
#     carbody: "sedan",
#     enginesize: 130,
#     stroke: 3.2,
#     horsepower: 90
#   })
# });


# Backend: What happens step-by-step

# The user fills form and clicks submit.
# JavaScript fetch() sends data as JSON to http://127.0.0.1:5000/predict.
# Flask receives the POST request at /predict.
# The function @app.route("/predict", methods=["POST"]) gets triggered.
# Flask sees it's a JSON request, so we use:
# data=request.get_json() converts JSON into a Python dictionary.
# data == {
#   "CarName": "honda",
#   "fueltype": "gas",
#   ...
# }
# You turn it into a DataFrame:
# You encode strings (like CarName) into numbers using your LabelEncoders.
# You make the prediction:
# price = model.predict(df)[0]
# return jsonify({"price": price})
# 

# CASE 2: Plain HTML Form → uses request.form
# <form action="/predict" method="POST">
#   <input name="CarName" />
#   <input name="fueltype" />
#   ...
#   <button type="submit">Submit</button>
# </form>


# Backend: What happens step-by-step

# User fills form and clicks submit.
# Browser sends a form-encoded POST request to Flask /predict.
# Flask receives it and again calls the same function:

# @app.route("/predict", methods=["POST"])
# def predict():

# Flask knows this is not JSON, it’s HTML form data.
# You fetch each field like:

# CarName = request.form["CarName"]
# fueltype = request.form["fueltype"]
# enginesize = float(request.form["enginesize"])
# ...
# You then collect into DataFrame like:

# df = pd.DataFrame([{
#   "CarName": CarName,
#   ...
# }])

# Continue with encoding, predicting, and returning:
# return jsonify({"price": 12345})

# ✅ In this case, the form values are fetched one by one from request.form.



