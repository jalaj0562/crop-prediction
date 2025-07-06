
from flask import Flask, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from scipy import stats

app = Flask(__name__)

# Load both datasets
crop_df = pd.read_csv(r'C:\Users\sneha\Downloads\Crop_recommendation.csv')
rain_df = pd.read_csv(r'C:\Users\sneha\Downloads\rain-agriculture.csv')

# Process rain-agriculture data
rain_df['total_rainfall'] = rain_df[['JUN', 'JUL', 'AUG', 'SEP']].sum(axis=1)
id_vars = ['State Name', 'subdivision', 'YEAR', 'total_rainfall']
yield_vars = [col for col in rain_df.columns if 'YIELD' in col]
rain_long = rain_df.melt(
    id_vars=id_vars,
    value_vars=yield_vars,
    var_name='Crop_Yield_Column',
    value_name='historical_yield'
)
rain_long['crop'] = rain_long['Crop_Yield_Column'].str.replace(' YIELD.*', '', regex=True).str.lower()

# Prepare crop recommendation data
crop_df['crop'] = crop_df['label'].str.lower()
merged_df = pd.merge(crop_df, rain_long, on='crop', how='inner')

# Clean data and build models
merged_df = merged_df.dropna()
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X = merged_df[features]
y_crop = merged_df['crop']
y_yield = merged_df['historical_yield']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
crop_model = RandomForestClassifier(random_state=42)
crop_model.fit(X_scaled, y_crop)
yield_model = LinearRegression()
yield_model.fit(X_scaled, y_yield)

# Plotting functions
def plot_correlation_heatmap():
    corr = merged_df[features].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("🔗 Feature Correlation Heatmap")
    return fig_to_base64()

def plot_feature_importance():
    importance = pd.Series(crop_model.feature_importances_, index=features)
    plt.figure(figsize=(8, 4))
    sns.barplot(x=importance.values, y=importance.index, palette="viridis")
    plt.title("🌱 Feature Importance for Crop Prediction")
    plt.xlabel("Importance")
    return fig_to_base64()

def plot_yield_by_year(crop):
    df = merged_df[merged_df['crop'] == crop]
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=df, x='YEAR', y='historical_yield', hue='subdivision', marker='o')
    plt.title(f"📊 Yield Trend of {crop.title()} by State")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig_to_base64()

def plot_crop_distribution():
    plt.figure(figsize=(10, 5))
    sns.countplot(y=merged_df['crop'], order=merged_df['crop'].value_counts().index, palette='coolwarm')
    plt.title("🌾 Crop Distribution in Dataset")
    plt.xlabel("Count")
    plt.ylabel("Crop")
    plt.tight_layout()
    return fig_to_base64()

def fig_to_base64():
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    encoded = base64.b64encode(img.read()).decode('utf8')
    plt.close()
    return encoded

@app.route('/', methods=['GET', 'POST'])
def index():
    heatmap_img = plot_correlation_heatmap()
    feature_img = plot_feature_importance()
    dist_img = plot_crop_distribution()
    crop_prediction = None
    yield_prediction = None
    state_crop_data = None
    trend_img = None

    if request.method == 'POST':
        try:
            user_input = [
                float(request.form['N']),
                float(request.form['P']),
                float(request.form['K']),
                float(request.form['temperature']),
                float(request.form['humidity']),
                float(request.form['ph']),
                float(request.form['rainfall'])
            ]
            scaled_input = scaler.transform([user_input])
            crop_prediction = crop_model.predict(scaled_input)[0]
            yield_prediction = round(yield_model.predict(scaled_input)[0], 2)
            state_crop_data = merged_df[merged_df['crop'] == crop_prediction][['subdivision', 'YEAR', 'historical_yield']].dropna().sort_values(by='YEAR', ascending=False).head(5).to_html(classes="table table-bordered")
            trend_img = plot_yield_by_year(crop_prediction)
        except Exception as e:
            crop_prediction = 'Invalid input.'
            yield_prediction = 'N/A'

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>🌾 Statewise Crop Recommendation by Jalaj</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
        <style>
            body {{ background-color: #f4f6f9; }}
            .tab-content img {{ max-width: 100%; height: auto; }}
            footer {{ text-align: center; padding: 1em; color: gray; }}
        </style>
    </head>
    <body>
    <div class="container my-4">
        <h1 class="mb-4 text-center">🌿 Smart Agriculture Assistant by Jalaj </h1>

        <form method="post" class="bg-light p-4 rounded mb-4">
            <div class="row">
                <div class="col"><input type="number" name="N" class="form-control" placeholder="Nitrogen (N)" required></div>
                <div class="col"><input type="number" name="P" class="form-control" placeholder="Phosphorus (P)" required></div>
                <div class="col"><input type="number" name="K" class="form-control" placeholder="Potassium (K)" required></div>
            </div><br>
            <div class="row">
                <div class="col"><input type="number" step="0.1" name="temperature" class="form-control" placeholder="Temperature (°C)" required></div>
                <div class="col"><input type="number" step="0.1" name="humidity" class="form-control" placeholder="Humidity (%)" required></div>
                <div class="col"><input type="number" step="0.01" name="ph" class="form-control" placeholder="Soil pH" required></div>
            </div><br>
            <div class="row">
                <div class="col"><input type="number" step="0.1" name="rainfall" class="form-control" placeholder="Rainfall (mm)" required></div>
            </div><br>
            <button type="submit" class="btn btn-success">Get Crop Recommendation</button>
        </form>

        {f'<h4 class="alert alert-success">🌾 Recommended Crop: <strong>{crop_prediction}</strong></h4>' if crop_prediction else ''}
        {f'<h5 class="alert alert-info">📈 Estimated Yield: <strong>{yield_prediction} kg/ha</strong></h5>' if yield_prediction else ''}
        {state_crop_data if state_crop_data else ''}

        <ul class="nav nav-tabs mt-4" id="myTab" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="heat-tab" data-bs-toggle="tab" data-bs-target="#heat" type="button" role="tab">Correlation</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="impact-tab" data-bs-toggle="tab" data-bs-target="#impact" type="button" role="tab">Importance</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="dist-tab" data-bs-toggle="tab" data-bs-target="#dist" type="button" role="tab">Crop Distribution</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="trend-tab" data-bs-toggle="tab" data-bs-target="#trend" type="button" role="tab">Yield Trend</button>
            </li>
        </ul>

        <div class="tab-content p-4 bg-white border border-top-0 rounded-bottom">
            <div class="tab-pane fade show active" id="heat">
                <h4>🔗 Feature Correlation</h4>
                <img src="data:image/png;base64,{heatmap_img}" class="img-fluid mt-3"/>
            </div>
            <div class="tab-pane fade" id="impact">
                <h4>🌱 Feature Importance</h4>
                <img src="data:image/png;base64,{feature_img}" class="img-fluid mt-3"/>
            </div>
            <div class="tab-pane fade" id="dist">
                <h4>🌾 Crop Occurrence in Dataset</h4>
                <img src="data:image/png;base64,{dist_img}" class="img-fluid mt-3"/>
            </div>
            <div class="tab-pane fade" id="trend">
                <h4>📈 Yield Trend</h4>
                {f'<img src="data:image/png;base64,{trend_img}" class="img-fluid mt-3"/>' if trend_img else '<p>Select a crop to see trend.</p>'}
            </div>
        </div>
    </div>

    <footer>
        &copy; 2025 Smart Agriculture — Flask & ML
    </footer>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """
    return html

if __name__ == '__main__':
    app.run(debug=True)
