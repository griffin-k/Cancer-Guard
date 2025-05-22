from django.shortcuts import render
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from django.http import HttpResponse
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from weasyprint import HTML
from datetime import datetime
from PyPDF2 import PdfReader, PdfWriter
from io import BytesIO



# Load dataset
def load_dataset():
    data = pd.read_csv("model/data.csv")
    data.columns = data.columns.str.strip()  
    X = data.drop(columns=["LUNG_CANCER"])
    return X

# Load trained model
def load_model():
    try:
        model = joblib.load("model/logistic_regression_model.pkl")
    except FileNotFoundError:
        raise FileNotFoundError("Model file not found. Please ensure 'logistic_regression_model.pkl' is in the same directory.")
    return model

# Preprocess user input
def preprocess_input(user_inputs, categorical_features, numerical_features, preprocessor):
    input_data = pd.DataFrame([user_inputs])
    # Apply preprocessing
    input_data_transformed = preprocessor.transform(input_data)
    return input_data_transformed

# Convert 'Yes'/'No' to 2/1 respectively
def map_yes_no_to_numeric(value):
    value = value.strip().upper()  
    return 2 if value == "YES" else 1 

# Django view function
def lung_cancer_prediction(request):
    X = load_dataset()
    categorical_features = X.select_dtypes(include=["object"]).columns
    numerical_features = X.select_dtypes(exclude=["object"]).columns
    model = load_model()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_features),
        ]
    )
    preprocessor.fit(X)

    if request.method == "POST":
        user_inputs = {
            "GENDER": request.POST.get("gender"),
            "AGE": int(request.POST.get("age")),
            "SMOKING": map_yes_no_to_numeric(request.POST.get("smoking")),
            "YELLOW_FINGERS": map_yes_no_to_numeric(request.POST.get("yellow_fingers")),
            "ANXIETY": map_yes_no_to_numeric(request.POST.get("anxiety")),
            "PEER_PRESSURE": map_yes_no_to_numeric(request.POST.get("peer_pressure")),
            "CHRONIC DISEASE": map_yes_no_to_numeric(request.POST.get("chronic_disease")),
            "FATIGUE": map_yes_no_to_numeric(request.POST.get("fatigue").strip()),
            "ALLERGY": map_yes_no_to_numeric(request.POST.get("allergy").strip()),
            "WHEEZING": map_yes_no_to_numeric(request.POST.get("wheezing")),
            "ALCOHOL CONSUMING": map_yes_no_to_numeric(request.POST.get("alcohol_consuming")),
            "COUGHING": map_yes_no_to_numeric(request.POST.get("coughing")),
            "SHORTNESS OF BREATH": map_yes_no_to_numeric(request.POST.get("shortness_of_breath")),
            "SWALLOWING DIFFICULTY": map_yes_no_to_numeric(request.POST.get("swallowing_difficulty")),
            "CHEST PAIN": map_yes_no_to_numeric(request.POST.get("chest_pain")),
        }


        print("User Inputs:")
        for key, value in user_inputs.items():
            print(f"{key}: {value}")

        
        input_data = preprocess_input(user_inputs, categorical_features, numerical_features, preprocessor)
        
        try:

            prediction = model.predict(input_data)[0]
            result = "Lung Cancer Detected" if prediction == "YES" else "No Lung Cancer Detected"
        except Exception as e:
            result = f"An error occurred during prediction: {e}"


        return render(request, "dashboard/result.html", {"user_inputs": user_inputs, "result": result})

    return render(request, "dashboard/detection.html")









def generate_pdf(request):
    user_inputs = request.POST.dict()
    result = request.POST.get("result", "No Result Provided")
    user_inputs.pop("csrfmiddlewaretoken", None)


    template_pdf_path = "model/sample.pdf"  
   


    current_year = datetime.now().year


    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <style>
            body {{
                font-family: 'Arial', sans-serif;
                margin: 0;
                padding: 0;
                color: #333;
                line-height: 1.6;
            }}
            .header {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 10px 20px;
                background-color: #f5f5f5;
                border-bottom: 2px solid #ddd;
            }}
            .note {{
                margin: 20px auto;
                font-size: 13px;
                font-style: italic;
                color: #555;
                text-align: center;
                border: 1px dashed #ccc;
                padding: 10px;
                width: 90%;
            }}
            .result-box {{
                color: white;
                padding: 15px;
                font-size: 18px;
                font-weight: bold;
                border-radius: 8px;
                margin: 20px auto;
                width: 60%;
                background-color: {'#e74c3c' if result == 'Lung Cancer Detected' else '#27ae60'};
                text-align: center;
            }}
            .title {{
                font-size: 18px;
                font-weight: bold;
                margin: 20px 0 10px;
                text-align: center;
                color: #2c3e50;
            }}
            .table-container {{
                margin: 20px auto;
                width: 90%;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 0 auto;
                font-size: 12px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: center;
            }}
            th {{
                background-color: #f0f0f0;
                font-weight: bold;
                color: #34495e;
            }}
            .status-yes {{
                color: #27ae60;
                font-weight: bold;
            }}
            .status-no {{
                color: #e74c3c;
                font-weight: bold;
            }}
            .footer {{
                text-align: center;
                margin-top: 30px;
                font-size: 12px;
                color: #7f8c8d;
            }}
            @page {{
                size: A4;
                margin: 20mm;
            }}
        </style>
    </head>
    <body>
        <div class="note">
            This report is generated using a machine learning-based prediction system. 
            The prediction may or may not be accurate.
        </div>
        <div class="result-box">{result}</div>
        <div class="title">Medical History & Symptoms</div>
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Sr. No</th>
                        <th>Disease</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
    """


    for idx, (key, value) in enumerate(user_inputs.items(), start=1):
        if value:  
            field_name = key.replace("_", " ").title()
            status_class = "status-yes" if value == "1" else "status-no"
            status_text = "YES" if value == "1" else "NO"
            html_content += f"""
                    <tr>
                        <td>{idx}</td>
                        <td>{field_name}</td>
                        <td class="{status_class}">{status_text}</td>
                    </tr>
            """


    html_content += f"""
                </tbody>
            </table>
        </div>
        <div class="footer">
            © {current_year} Cancer Guard. All Rights Reserved.
        </div>
    </body>
    </html>
    """

    dynamic_pdf = HTML(string=html_content).write_pdf()
    custom_template = PdfReader(template_pdf_path)
    output_pdf = PdfWriter()
    dynamic_pdf_reader = PdfReader(BytesIO(dynamic_pdf))
    dynamic_page = dynamic_pdf_reader.pages[0]
    template_page = custom_template.pages[0]
    template_page.merge_page(dynamic_page)
    output_pdf.add_page(template_page)
    response = HttpResponse(content_type="application/pdf")
    response["Content-Disposition"] = 'attachment; filename="lung_cancer_report.pdf"'


    output_pdf.write(response)

    return response





def view_question(request):
    return render(request, "dashboard/ask_dr.html")

def view_support(request):
    return render(request, "dashboard/contact.html")

def view_about(request):
    return render(request, "dashboard/about_us.html")

def view_home(request):
    return render(request, "dashboard/home.html")


def view_index(request):
    return render(request, "dashboard/index.html")


from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib import messages

# --- Signup View ---
def view_register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)  # Automatically log the user in after signup
            messages.success(request, "Account created successfully.")
            return redirect('home')  # Replace with your homepage URL name
        else:
            messages.error(request, "Please correct the errors below.")
    else:
        form = UserCreationForm()

    return render(request, 'dashboard/register.html', {'form': form})


# --- Login View ---
def view_login(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(request, "Login successful.")
            return redirect('home')  # Replace with your homepage URL name
        else:
            messages.error(request, "Invalid username or password.")
    else:
        form = AuthenticationForm()

    return render(request, 'dashboard/login.html', {'form': form})
