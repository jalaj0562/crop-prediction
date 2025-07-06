🌾 Smart Agriculture Assistant

A Flask-based machine learning web app for **crop recommendation and yield prediction**, powered by historical weather and agriculture data.

 🚀 Features

- Predicts the most suitable crop based on user input (soil nutrients, weather, and pH).
- Estimates historical yield for the predicted crop.
- Visualizes:
  - 🌡️ Feature Correlation Heatmap
  - 🌱 Feature Importance (Random Forest)
  - 🌾 Crop Distribution
  - 📈 Year-wise Yield Trends by State

 🧠 Models Used

- **Crop Recommendation**: `RandomForestClassifier`
- **Yield Prediction**: `LinearRegression`
- Features: `N, P, K, temperature, humidity, ph, rainfall`
- Data Preprocessing: `StandardScaler`

 🗂️ Dataset Sources

- `Crop_recommendation.csv`
- `rain-agriculture.csv`

> Note: You must place these datasets in your local system and **update the paths** in the script if necessary.

 🖥️ Tech Stack

- **Frontend**: HTML + Bootstrap 5
- **Backend**: Python (Flask)
- **Visualization**: Matplotlib, Seaborn
- **ML Libraries**: scikit-learn, pandas, numpy

 ⚙️ How to Run

1. **Install dependencies**  
   ```bash
   pip install flask pandas numpy scikit-learn matplotlib seaborn
