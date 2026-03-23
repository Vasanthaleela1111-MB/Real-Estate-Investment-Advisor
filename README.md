# Real-Estate-Investment-Advisor
# 🏠 Real Estate Investment Analysis using Machine Learning

## 📌 Project Overview

This project is a Machine Learning-based Real Estate Investment Analysis System that helps users:

* Predict whether a property is a **Good Investment** (Classification)
* Predict the **Future Property Price after 5 Years** (Regression)

The system uses feature engineering, data preprocessing, machine learning models, and a Streamlit web application for predictions.

---

## 🚀 Features

* Good Investment Prediction (Classification Model)
* Future Property Price Prediction after 5 Years (Regression Model)
* Feature Engineering (Amenities Score, Age of Property, etc.)
* Data Preprocessing and Encoding
* StandardScaler for Feature Scaling
* Machine Learning Models (Random Forest / XGBoost)
* Streamlit Web Application
* Interactive User Input Interface
* Real Estate Investment Insights

---

## 🧠 Machine Learning Models Used

### Classification Model

Predicts whether the property is a **Good Investment** based on:

* Property size
* Price
* Location
* Amenities
* Transport accessibility
* Property type
* Availability status
* Nearby schools and hospitals

### Regression Model

Predicts the **Future Property Price after 5 years** based on:

* Price per SqFt
* Amenities
* Location
* Property age
* Furnishing status
* Property type
* Owner type
* Transport accessibility

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost / Random Forest
* Matplotlib / Seaborn
* Streamlit
* Pickle
* Machine Learning

---

## 📂 Project Structure

```
RealEstate_Project/
│
├── cleaned_data.csv
├── Feature_data.csv
├── india_housing_prices.csv
├── classification.ipynb
├── regression.ipynb
├── Data_clening.ipynb
├── Real Estate.ipynb
├── classification.pkl
├── classification_scaler.pkl
├── model.pkl
├── real.py
├── README.md
```

## 📁 Dataset and Model Files

Some dataset and model files are large and cannot be uploaded directly to GitHub.
You can download the CSV files and model files from the Google Drive link below:

🔗 Google Drive Link:
https://drive.google.com/drive/folders/1i69cV9woGf8ERlBgUSvkfmHQbBNfN2E3?usp=sharing

After downloading, place the files in the project folder as shown in the project structure before running the Streamlit application.

## ▶️ How to Run the Project

1. Install required libraries:

```
pip install pandas numpy scikit-learn streamlit matplotlib seaborn xgboost
```

2. Run the Streamlit app:

```
streamlit run real.py
```

3. Open browser:

```
http://localhost:8501
```

---

## 📊 Output

The system provides:

* Good Investment Prediction with confidence
* Future Property Price after 5 years

---

## 📌 Conclusion

This project helps users make better real estate investment decisions by predicting investment quality and future property price using machine learning models and data analysis.

---

## 👩‍💻 Author

Vasantha Leela MB
This project was developed as part of my Machine Learning and Web Application development work, focusing on real estate investment prediction using classification and regression models with Streamlit deployment.
