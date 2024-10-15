from django.shortcuts import render
import joblib
import numpy as np

# Load your trained logistic regression model
model = joblib.load('F:/Student-Performance/STUDENT_PERFORMANCE_APP/logistic_regression_model.pkl')

def performance(request):
    if request.method == 'POST':
        # Get form data
        gender = int(request.POST.get('gender'))
        test = int(request.POST.get('test'))
        math_score = int(request.POST.get('math-score'))
        reading_score = int(request.POST.get('reading-score'))
        writing_score = int(request.POST.get('writing-score'))

        # Prepare input data for prediction
        input_data = np.array([[gender, test, math_score, reading_score, writing_score]])

        # Make prediction
        prediction_numeric = model.predict(input_data)[0]

        # Map numeric predictions back to grades
        grade_mapping = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}
        prediction = grade_mapping.get(prediction_numeric, "Unknown")

        return render(request, 'performance.html', {'prediction': prediction})
    else:
        return render(request, 'performance.html')
