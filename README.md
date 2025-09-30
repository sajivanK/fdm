
# 🚗 Vehicle CO₂ Emissions Predictor  

A machine learning web application that predicts **vehicle CO₂ emissions (g/km)** based on specifications such as engine size, cylinders, fuel type, and fuel consumption.  

This project was developed as part of the **Fundamentals of Data Mining** module. It demonstrates the complete ML workflow: data cleaning, preprocessing, model training, evaluation, and deployment on **Streamlit Cloud**.  

---

## 📌 Project Overview  

- ✅ **Data Cleaning & Preprocessing**  
  - Added synthetic rows with missing values, duplicates, and outliers for cleaning tasks.  
  - Handled missing values (median imputation for numeric, mode for categorical).  
  - Encoded categorical variables (fuel type).  
  - Scaled numerical features for fair comparison.  

- ✅ **Exploratory Data Analysis (EDA)**  
  - Distribution plots of CO₂ emissions.  
  - Correlation heatmaps.  
  - Scatter plots showing relationships (e.g., emissions vs engine size).  
  - Boxplots across categorical features (fuel type, transmission, vehicle class).  

- ✅ **Modeling**  
  - Algorithm: **Multiple Linear Regression (MLR)**.  
  - Performance:  
    - R² Score: **0.9957**  
    - RMSE: **3.61 g/km**  
    - MAE: **1.97 g/km**  
  - Interpretation: Very high accuracy with small prediction errors.  

- ✅ **Deployment**  
  - Implemented as a **Streamlit web app**.  
  - Users can enter vehicle details and instantly get predicted emissions.  
  - Includes model performance metrics and interactive graphs.  

---

## 🖥️ Demo  

👉 [Live App on Streamlit Cloud](https://vehicle-co2-emission-predictor-afgsfzxjds6e2beiqtd75i.streamlit.app)  

---

## 📊 Features of the Web App  

1. **User Input Form**  
   - Engine size (L)  
   - Number of cylinders  
   - Fuel type (Petrol, Diesel, Ethanol, Natural Gas, Premium Petrol)  
   - Combined fuel consumption (L/100 km)  

2. **Prediction Results**  
   - Predicted CO₂ emissions (g/km).  
   - Comparison with dataset average.  

3. **Model Insights**  
   - Actual vs Predicted scatter plot.  
   - Residuals distribution.  
   - Vehicle vs Dataset comparison chart.  

---

## ⚙️ Tech Stack  

- **Language:** Python  
- **Libraries:**  
  - Streamlit  
  - Pandas, NumPy  
  - Scikit-learn  
  - Matplotlib, Seaborn  
  - Joblib  

---

## 📂 Repository Structure  

```

vehicle-co2-emission-predictor/
│── app.py                     # Streamlit app
│── final_model.pkl            # Trained ML model (pipeline + regression)
│── cleaned_vehicle_dataset.csv # Dataset used for predictions & averages
│── requirements.txt           # Dependencies
│── README.md                  # Project documentation

````

---

## 👨‍👩‍👦‍👦 Team Members  

- **Sajivan K.**   
- **Mathushan K.**  
- **Kethushan M.**  
- **Ajaniya K.**  

📚 Module: **Fundamentals of Data Mining**  

---

## 🚀 How to Run Locally  

1. Clone this repo:
```bash
   git clone https://github.com/sajivanK/vehicle-co2-emission-predictor.git
   cd vehicle-co2-emission-predictor
```

2. Install dependencies:

```bash
   pip install -r requirements.txt
```


3. Run the app:

```bash
   streamlit run app.py
```

---

## 📈 Future Improvements

* Add more ML models (Ridge, Lasso, Random Forest) for comparison.
* Support for additional features (vehicle weight, horsepower).
* Deploy with a database for saving user predictions.
* Enhanced UI (dark mode, dashboard-style layout).

---

## 🙌 Acknowledgements

* Dataset: [Vehicle Fuel Consumption Ratings](https://www.kaggle.com/datasets)
* Streamlit for easy deployment.
* Fundamentals of Data Mining lecturers for guidance.



