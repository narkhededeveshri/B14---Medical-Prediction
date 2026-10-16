import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('medical.pkl', 'rb'))

# HTML Template with Tailwind CSS for a modern, attractive UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Insurance Cost Predictor</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gradient-to-br from-blue-50 to-indigo-100 min-h-screen flex items-center justify-center p-4">
    <div class="max-w-xl w-full bg-white rounded-2xl shadow-xl overflow-hidden p-8">
        <div class="text-center mb-8">
            <span class="bg-indigo-100 text-indigo-800 text-xs font-semibold px-3 py-1 rounded-full uppercase tracking-wider">Machine Learning Web App</span>
            <h1 class="text-3xl font-bold text-gray-900 mt-2">Medical Cost Predictor</h1>
            <p class="text-gray-500 text-sm mt-1">Enter your details below to estimate medical insurance charges.</p>
        </div>

        <form method="POST" action="/" class="space-y-5">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <!-- Age -->
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">Age</label>
                    <input type="number" name="age" required min="1" max="120" placeholder="e.g., 30"
                        class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                </div>

                <!-- BMI -->
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">BMI</label>
                    <input type="number" step="0.1" name="bmi" required placeholder="e.g., 25.5"
                        class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                </div>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <!-- Children -->
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">Children</label>
                    <input type="number" name="children" required min="0" max="10" placeholder="e.g., 2"
                        class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                </div>

                <!-- Sex -->
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">Sex</label>
                    <select name="sex" class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none bg-white">
                        <option value="1">Male</option>
                        <option value="0">Female</option>
                    </select>
                </div>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <!-- Smoker -->
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">Smoker</label>
                    <select name="smoker" class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none bg-white">
                        <option value="1">Yes</option>
                        <option value="0">No</option>
                    </select>
                </div>

                <!-- Region -->
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">Region</label>
                    <select name="region" class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none bg-white">
                        <option value="0">Northeast</option>
                        <option value="1">Northwest</option>
                        <option value="2">Southeast</option>
                        <option value="3">Southwest</option>
                    </select>
                </div>
            </div>

            <button type="submit" 
                class="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3 px-4 rounded-lg transition duration-200 shadow-md">
                Predict Charges
            </button>
        </form>

        {% if prediction %}
        <div class="mt-6 p-4 bg-indigo-50 border border-indigo-200 rounded-xl text-center">
            <h2 class="text-sm font-medium text-indigo-600 uppercase tracking-wide">Estimated Insurance Cost</h2>
            <p class="text-3xl font-extrabold text-indigo-900 mt-1">${{ prediction }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Extract inputs from form
            age = float(request.form['age'])
            sex = float(request.form['sex'])
            bmi = float(request.form['bmi'])
            children = float(request.form['children'])
            smoker = float(request.form['smoker'])
            region = float(request.form['region'])

            # Create input array for the model
            features = np.array([[age, sex, bmi, children, smoker, region]])
            
            # Predict
            pred = model.predict(features)[0]
            prediction = round(pred, 2)
        except Exception as e:
            prediction = f"Error in calculation: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
