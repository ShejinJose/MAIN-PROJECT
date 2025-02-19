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

# import os

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current file

# # Load the trained model
# model_path = os.path.join(BASE_DIR, 'sales_prediction_model.pkl')  # Construct the file path
# model = joblib.load(model_path)

# # Load the label encoder
# encoder_path = os.path.join(BASE_DIR, 'label_encoder.pkl')  # Construct the file path
# encoder = joblib.load(encoder_path)

# # Single Prediction View
# def predict_sales(request):
#     if request.method == 'POST':
#         # Map dropdown values to match the encoder's expected input
#         dropdown_mappings = {
#             'item_fat_content': {
#                 'Low Fat': 'Low Fat',
#                 'Regular': 'Regular'
#             },
#             'outlet_size': {
#                 'Small': 'Small',
#                 'Medium': 'Medium',
#                 'Large': 'Large'
#             },
#             'outlet_location_type': {
#                 'Tier 1': 'Tier 1',
#                 'Tier 2': 'Tier 2',
#                 'Tier 3': 'Tier 3'
#             },
#             'outlet_type': {
#                 'Grocery Store': 'Grocery Store',
#                 'Supermarket Type1': 'Supermarket Type1',
#                 'Supermarket Type2': 'Supermarket Type2',
#                 'Supermarket Type3': 'Supermarket Type3'
#             }
#         }

#         # Gather input data from the form
#         try:
#             data = {
#                 'Item_Identifier': request.POST['item_identifier'],
#                 'Item_Weight': float(request.POST['item_weight']),
#                 'Item_Fat_Content': dropdown_mappings['item_fat_content'][request.POST['item_fat_content']],
#                 'Item_Visibility': float(request.POST['item_visibility']),
#                 'Item_Type': request.POST['item_type'],
#                 'Item_MRP': float(request.POST['item_mrp']),
#                 'Outlet_Identifier': request.POST['outlet_identifier'],
#                 'Outlet_Size': dropdown_mappings['outlet_size'][request.POST['outlet_size']],
#                 'Outlet_Location_Type': dropdown_mappings['outlet_location_type'][request.POST['outlet_location_type']],
#                 'Outlet_Type': dropdown_mappings['outlet_type'][request.POST['outlet_type']],
#                 'Outlet_Establishment_Year': int(request.POST['outlet_establishment_year']),
#             }
#         except KeyError as e:
#             return render(request, 'predict_sales.html', {'error': f"Invalid input: {e}"})

#         # Prepare data for prediction
#         input_data = pd.DataFrame([data])

#         # Align the input data columns with the model's expected feature order
#         expected_features = [
#             'Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility',
#             'Item_Type', 'Item_MRP', 'Outlet_Identifier', 'Outlet_Establishment_Year',
#             'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'
#         ]
#         try:
#             input_data = input_data[expected_features]
#         except KeyError as e:
#             return render(request, 'predict_sales.html', {'error': f"Feature mismatch: {e}"})

#         # Encode categorical features with OrdinalEncoder
#         categorical_cols = [
#             'Item_Identifier', 'Item_Fat_Content', 'Item_Type', 
#             'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'
#         ]

#         try:
#             # Check if encoder is fitted
#             if not hasattr(encoder, 'categories_'):
#                 return render(request, 'predict_sales.html', {'error': "Encoder is not properly fitted."})

#             for idx, col in enumerate(categorical_cols):
#                 # Get the categories for the current column
#                 valid_categories = encoder.categories_[idx]

#                 # Map unseen categories to "Unknown"
#                 input_data[col] = input_data[col].apply(
#                     lambda x: x if x in valid_categories else 'Unknown'
#                 )

#                 # Update the encoder categories if "Unknown" is not already present
#                 if 'Unknown' not in valid_categories:
#                     encoder.categories_[idx] = list(valid_categories) + ['Unknown']

#             # Transform the categorical columns
#             input_data[categorical_cols] = encoder.transform(input_data[categorical_cols])

#         except Exception as e:
#             return render(request, 'predict_sales.html', {'error': f"Encoding error: {e}"})

#         # Predict sales
#         try:
#             prediction = model.predict(input_data)
#             result = round(prediction[0], 2)
#         except Exception as e:
#             return render(request, 'predict_sales.html', {'error': f"Prediction error: {e}"})

#         # Render result on the page
#         return render(request, 'predict_sales.html', {'predicted_sales': result})

#     # Render the form initially
#     return render(request, 'predict_sales.html')
from django.shortcuts import render
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained model and encoder
model_path = os.path.join(BASE_DIR, 'sales_prediction_model.pkl')
encoder_path = os.path.join(BASE_DIR, 'label_encoder.pkl')

model = joblib.load(model_path)
encoder = joblib.load(encoder_path)

# Define expected feature order (should match training features)
EXPECTED_FEATURES = [
    'Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility',
    'Item_Type', 'Item_MRP', 'Outlet_Identifier', 'Outlet_Establishment_Year',
    'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'
]

# Function to encode categorical features
def encode_categorical_features(input_data, encoder):
    """ Convert categorical features into numerical format using the stored encoder """
    categorical_cols = ['Item_Identifier', 'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
    
    for col in categorical_cols:
        if col in input_data:
            valid_categories = encoder.categories_[categorical_cols.index(col)]
            input_data[col] = input_data[col].apply(lambda x: x if x in valid_categories else 'Unknown')

    input_data[categorical_cols] = encoder.transform(input_data[categorical_cols])
    
    return input_data

# Helper function for insights
def generate_insights(predicted_sales, item_mrp, item_visibility, outlet_type, outlet_location):
    insights = {}

    # Classify sales level
    if predicted_sales < 1000:
        insights["Sales Level"] = "Low Sales (Consider Discounts or Promotions)"
    elif predicted_sales < 3000:
        insights["Sales Level"] = "Medium Sales (Optimize Product Placement)"
    else:
        insights["Sales Level"] = "High Sales (High Demand Item)"

    # Pricing Strategy
    if predicted_sales < 2000 and item_mrp > 200:
        insights["Pricing Strategy"] = "Consider reducing MRP slightly to increase sales."
    elif predicted_sales > 3000:
        insights["Pricing Strategy"] = "Demand is high! Consider premium pricing or promotions."

    # Outlet Strategy
    if outlet_type == "Supermarket Type1" and predicted_sales > 2500:
        insights["Outlet Strategy"] = "Supermarkets have high sales—stock more in these outlets."
    elif outlet_type == "Grocery Store" and predicted_sales < 1500:
        insights["Outlet Strategy"] = "Grocery store sales are lower—consider promoting this item more."

    # Visibility Analysis
    if item_visibility < 0.02:
        insights["Product Placement"] = "Low visibility—consider better in-store positioning."
    elif item_visibility > 0.1:
        insights["Product Placement"] = "High visibility but check if sales match expectations."

    return insights

# Generate a sales distribution plot
def generate_sales_graph(predicted_sales):
    plt.figure(figsize=(6, 4))
    sns.histplot([predicted_sales], bins=10, kde=True, color="blue")
    plt.axvline(predicted_sales, color="red", linestyle="dashed", linewidth=2)
    plt.xlabel("Predicted Sales")
    plt.ylabel("Frequency")
    plt.title("Sales Prediction Distribution")

    # Save plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    graph_url = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    
    return f"data:image/png;base64,{graph_url}"

# Sales Prediction View
def predict_sales(request):
    if request.method == 'POST':
        try:
            # Collect user input from form
            input_data = {
                'Item_Identifier': request.POST['item_identifier'],
                'Item_Weight': float(request.POST['item_weight']),
                'Item_Fat_Content': request.POST['item_fat_content'],
                'Item_Visibility': float(request.POST['item_visibility']),
                'Item_Type': request.POST['item_type'],
                'Item_MRP': float(request.POST['item_mrp']),
                'Outlet_Identifier': request.POST['outlet_identifier'],
                'Outlet_Establishment_Year': int(request.POST['outlet_establishment_year']),
                'Outlet_Size': request.POST['outlet_size'],
                'Outlet_Location_Type': request.POST['outlet_location_type'],
                'Outlet_Type': request.POST['outlet_type']
            }

            # Convert input data to DataFrame
            input_df = pd.DataFrame([input_data])

            # Encode categorical features before passing to model
            input_df = encode_categorical_features(input_df, encoder)

            # Ensure feature order matches training
            input_df = input_df[EXPECTED_FEATURES]

            # Convert all columns to float (required for XGBoost)
            input_df = input_df.astype(float)

            # Predict sales
            prediction = model.predict(input_df)
            predicted_sales = round(prediction[0], 2)

            # Generate insights
            insights = generate_insights(predicted_sales, input_data['Item_MRP'], input_data['Item_Visibility'], input_data['Outlet_Type'], input_data['Outlet_Location_Type'])

            # Generate a sales distribution graph
            sales_graph = generate_sales_graph(predicted_sales)

            # Render the results
            return render(request, 'predict_sales.html', {
                'predicted_sales': predicted_sales,
                'insights': insights,
                'sales_graph': sales_graph
            })

        except Exception as e:
            return render(request, 'predict_sales.html', {'error': f"Error: {e}"})

    return render(request, 'predict_sales.html')

# # Single Prediction View
# def predict_sales(request):
#     if request.method == 'POST':
#         try:
#             # Collect form data
#             item_mrp = float(request.POST['item_mrp'])
#             item_visibility = float(request.POST['item_visibility'])
#             outlet_type = request.POST['outlet_type']
#             outlet_location = request.POST['outlet_location_type']

#             # Prepare data for prediction
#             input_data = pd.DataFrame([{
#                 'Item_MRP': item_mrp,
#                 'Item_Visibility': item_visibility,
#                 'Outlet_Type': outlet_type,
#                 'Outlet_Location_Type': outlet_location
#             }])

#             # Predict sales
#             prediction = model.predict(input_data)
#             predicted_sales = round(prediction[0], 2)

#             # Generate insights
#             insights = generate_insights(predicted_sales, item_mrp, item_visibility, outlet_type, outlet_location)

#             # Render the results with insights
#             return render(request, 'predict_sales.html', {
#                 'predicted_sales': predicted_sales,
#                 'insights': insights
#             })

#         except Exception as e:
#             return render(request, 'predict_sales.html', {'error': f"Error: {e}"})

#     return render(request, 'predict_sales.html')







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

