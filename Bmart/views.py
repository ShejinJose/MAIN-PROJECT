from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
# def home(request):
#     return HttpResponse("Hello, Django!")

import os
import pickle
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings 

# Construct the absolute path to model.pkl
model_path = os.path.join(settings.BASE_DIR, 'Bmart', 'model.pkl')

# Load the model
model = pickle.load(open(model_path, "rb"))

# # Load the pre-trained model
# model = pickle.load(open("model.pkl", "rb"))

print(model_path)  # Print the full path to model.pkl

def home(request):
    return render(request, 'index.html')

def predict(request):
    if request.method == 'POST':
        try:
            # Get form data
            weight = float(request.POST.get('weight'))
            price = float(request.POST.get('price'))
            visibility = float(request.POST.get('visibility'))
            outlet_size = int(request.POST.get('outlet_size'))
            outlet_age = int(request.POST.get('outlet_age'))

            # Prepare features for prediction
            features = [[weight, price, visibility, outlet_size, outlet_age]]

            # Predict using the model
            prediction = model.predict(features)[0]

            return render(request, 'result.html', {'prediction': prediction})

        except Exception as e:
            return JsonResponse({'error': str(e)})
