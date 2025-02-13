from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
# def home(request):
#     return HttpResponse("Hello, Django!")
from django.shortcuts import render

def home(request):
    return render(request, 'home.html')  # This will render a template called 'home.html'



import pandas as pd
import joblib
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.core.files.storage import FileSystemStorage

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current file

# Load the trained model
model_path = os.path.join(BASE_DIR, 'sales_prediction_model.pkl')  # Construct the file path
model = joblib.load(model_path)

# Load the label encoder
encoder_path = os.path.join(BASE_DIR, 'label_encoder.pkl')  # Construct the file path
encoder = joblib.load(encoder_path)

# Single Prediction View
def predict_sales(request):
    if request.method == 'POST':
        # Map dropdown values to match the encoder's expected input
        dropdown_mappings = {
            'item_fat_content': {
                'Low Fat': 'Low Fat',
                'Regular': 'Regular'
            },
            'outlet_size': {
                'Small': 'Small',
                'Medium': 'Medium',
                'Large': 'Large'
            },
            'outlet_location_type': {
                'Tier 1': 'Tier 1',
                'Tier 2': 'Tier 2',
                'Tier 3': 'Tier 3'
            },
            'outlet_type': {
                'Grocery Store': 'Grocery Store',
                'Supermarket Type1': 'Supermarket Type1',
                'Supermarket Type2': 'Supermarket Type2',
                'Supermarket Type3': 'Supermarket Type3'
            }
        }

        # Gather input data from the form
        try:
            data = {
                'Item_Identifier': request.POST['item_identifier'],
                'Item_Weight': float(request.POST['item_weight']),
                'Item_Fat_Content': dropdown_mappings['item_fat_content'][request.POST['item_fat_content']],
                'Item_Visibility': float(request.POST['item_visibility']),
                'Item_Type': request.POST['item_type'],
                'Item_MRP': float(request.POST['item_mrp']),
                'Outlet_Identifier': request.POST['outlet_identifier'],
                'Outlet_Size': dropdown_mappings['outlet_size'][request.POST['outlet_size']],
                'Outlet_Location_Type': dropdown_mappings['outlet_location_type'][request.POST['outlet_location_type']],
                'Outlet_Type': dropdown_mappings['outlet_type'][request.POST['outlet_type']],
                'Outlet_Establishment_Year': int(request.POST['outlet_establishment_year']),
            }
        except KeyError as e:
            return render(request, 'predict_sales.html', {'error': f"Invalid input: {e}"})

        # Prepare data for prediction
        input_data = pd.DataFrame([data])

        # Align the input data columns with the model's expected feature order
        expected_features = [
            'Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility',
            'Item_Type', 'Item_MRP', 'Outlet_Identifier', 'Outlet_Establishment_Year',
            'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'
        ]
        try:
            input_data = input_data[expected_features]
        except KeyError as e:
            return render(request, 'predict_sales.html', {'error': f"Feature mismatch: {e}"})

        # Encode categorical features with OrdinalEncoder
        categorical_cols = [
            'Item_Identifier', 'Item_Fat_Content', 'Item_Type', 
            'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'
        ]

        try:
            # Check if encoder is fitted
            if not hasattr(encoder, 'categories_'):
                return render(request, 'predict_sales.html', {'error': "Encoder is not properly fitted."})

            for idx, col in enumerate(categorical_cols):
                # Get the categories for the current column
                valid_categories = encoder.categories_[idx]

                # Map unseen categories to "Unknown"
                input_data[col] = input_data[col].apply(
                    lambda x: x if x in valid_categories else 'Unknown'
                )

                # Update the encoder categories if "Unknown" is not already present
                if 'Unknown' not in valid_categories:
                    encoder.categories_[idx] = list(valid_categories) + ['Unknown']

            # Transform the categorical columns
            input_data[categorical_cols] = encoder.transform(input_data[categorical_cols])

        except Exception as e:
            return render(request, 'predict_sales.html', {'error': f"Encoding error: {e}"})

        # Predict sales
        try:
            prediction = model.predict(input_data)
            result = round(prediction[0], 2)
        except Exception as e:
            return render(request, 'predict_sales.html', {'error': f"Prediction error: {e}"})

        # Render result on the page
        return render(request, 'predict_sales.html', {'predicted_sales': result})

    # Render the form initially
    return render(request, 'predict_sales.html')



def batch_sales_prediction(request):
    if request.method == 'POST' and request.FILES['file']:
        uploaded_file = request.FILES['file']
        fs = FileSystemStorage()
        file_path = fs.save(uploaded_file.name, uploaded_file)

        # Load the uploaded file
        data = pd.read_csv(fs.path(file_path))

        # Ensure categorical columns are encoded
        categorical_cols = [
            'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 
            'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'
        ]
        for col in categorical_cols:
            # Map unseen categories to 'Unknown' and transform
            data[col] = data[col].apply(lambda x: x if x in encoder.classes_ else 'Unknown')
            encoder.classes_ = list(encoder.classes_) + ['Unknown']  # Add 'Unknown' to classes
            data[col] = encoder.transform(data[col])

        # Predict sales
        predictions = model.predict(data)
        data['Predicted_Sales'] = predictions

        # Save results
        result_file = fs.path(file_path).replace('.csv', '_results.csv')
        data.to_csv(result_file, index=False)

        return JsonResponse({'result_file': result_file})

    return render(request, 'batch_upload.html')


# from django.http import JsonResponse
# from django.shortcuts import render
# from django.core.files.storage import FileSystemStorage
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# import xgboost as xgb

# # Load your pre-trained model (ensure you load it correctly)
# # For example:
# # model = joblib.load('path/to/your/model.pkl')

# # Prepare a global LabelEncoder dictionary to encode all categorical columns
# encoders = {col: LabelEncoder() for col in [
#     'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 
#     'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'
# ]}

# import numpy as np
# import os
# import numpy as np
# import pandas as pd
# import xgboost as xgb
# import pickle
# from django.core.files.storage import FileSystemStorage
# from django.http import JsonResponse
# from django.shortcuts import render

# import os
# import pickle
# import xgboost as xgb

# # Define the base directory
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# # Paths for the saved model and encoders
# MODEL_PATH = os.path.join(BASE_DIR, "xgboost_model.json")
# ENCODERS_PATH = os.path.join(BASE_DIR,  "encoders.pkl")

# # Check if files exist
# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
# if not os.path.exists(ENCODERS_PATH):
#     raise FileNotFoundError(f"Encoders file not found at {ENCODERS_PATH}")

# # Load the saved model and encoders
# model = xgb.Booster()
# model.load_model(MODEL_PATH)

# with open(ENCODERS_PATH, "rb") as f:
#     encoders = pickle.load(f)


# def batch_sales_prediction(request):
#     if request.method == 'POST' and request.FILES.get('file'):
#         # Retrieve uploaded file
#         uploaded_file = request.FILES['file']
#         fs = FileSystemStorage()
#         file_path = fs.save(uploaded_file.name, uploaded_file)

#         # Load uploaded CSV into DataFrame
#         data = pd.read_csv(fs.path(file_path))

#         # Define expected features
#         expected_features = [
#             'Item_Identifier', 'Item_Weight', 'Item_Fat_Content',
#             'Item_Visibility', 'Item_Type', 'Item_MRP',
#             'Outlet_Identifier', 'Outlet_Establishment_Year',
#             'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'
#         ]

#         # Ensure all required features are present in the uploaded file
#         missing_features = [feature for feature in expected_features if feature not in data.columns]
#         if missing_features:
#             return JsonResponse({'error': f'Missing required features: {", ".join(missing_features)}'})

#         # Encode categorical columns
#         categorical_cols = [
#             'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier',
#             'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'
#         ]

#         for col in categorical_cols:
#             if col in data.columns:
#                 # Check if the LabelEncoder for this column has been fitted
#                 if not hasattr(encoders[col], 'classes_'):
#                     # Fit the LabelEncoder with the unique values from the training data or current column
#                     encoders[col].fit(data[col].dropna().unique())

#                 # Handle unseen categories by mapping them to "Unknown"
#                 data[col] = data[col].apply(
#                     lambda x: x if x in encoders[col].classes_ else 'Unknown'
#                 )

#                 # Update LabelEncoder to include "Unknown" if necessary
#                 if 'Unknown' not in encoders[col].classes_:
#                     encoders[col].classes_ = np.append(encoders[col].classes_, 'Unknown')

#                 # Transform categorical column
#                 data[col] = encoders[col].transform(data[col])

#         # Align input data with model's expected feature order
#         data = data[expected_features]

#         # Convert data to DMatrix for prediction
#         dtest = xgb.DMatrix(data)

#         # Make predictions
#         try:
#             predictions = model.predict(dtest)
#         except ValueError as e:
#             return JsonResponse({'error': f'Prediction error: {e}'})

#         # Add predictions to DataFrame
#         data['Predicted_Sales'] = predictions

#         # Save DataFrame with predictions as a new CSV file
#         result_file_path = fs.path(file_path).replace('.csv', '_results.csv')
#         data.to_csv(result_file_path, index=False)

#         return JsonResponse({'result_file': result_file_path})

#     return render(request, 'batch_upload.html')
