import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import hashlib
import sqlite3
import datetime
import base64
import os
from sklearn.linear_model import LinearRegression

# Set page configuration with a fresh background and modern icon
st.set_page_config(
    page_title="Rainfall Prediction App - Weather Wizard",
    page_icon="🌧",  # safer version without variation selector
    layout="wide",
    initial_sidebar_state="expanded"
)


def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def add_clouds():
    cloud_css = """
    <style>
    .cloud {
        width: 200px;
        height: 60px;
        background: #fff;
        border-radius: 100px;
        position: absolute;
        top: 100px;
        left: 50%;
        margin-left: -100px;
        animation: float 60s linear infinite;
        opacity: 0.6;
        z-index: 0;
    }

    .cloud:before,
    .cloud:after {
        content: '';
        position: absolute;
        background: #fff;
        width: 100px;
        height: 80px;
        position: absolute;
        top: -30px;
        left: 10px;
        border-radius: 100px;
    }

    .cloud:after {
        width: 120px;
        height: 120px;
        top: -60px;
        left: auto;
        right: 15px;
    }

    @keyframes float {
        0% { left: -200px; }
        100% { left: 100%; }
    }
    </style>

    <div class="cloud"></div>
    <div class="cloud" style="top: 200px; animation-delay: 10s;"></div>
    <div class="cloud" style="top: 300px; animation-delay: 20s;"></div>
    """
    st.markdown(cloud_css, unsafe_allow_html=True)



# Usage
set_background(r"D:\Users\Rakshitha M R\Downloads\Rainfall_Prediction\blue.jpeg")
add_clouds()


# Custom CSS for creative, modern look with gradients, shadows, and animations
st.markdown(
    """
    <style>
    /* Fullscreen background */
    html, body {
        height: 100%;
        margin: 0;
        background: linear-gradient(to bottom, #1a2a6c, #b21f1f, #fdbb2d); 
        overflow: hidden;
    }

    /* Sky layers */
    .sky {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(to top, #0f2027, #203a43, #2c5364);
        z-index: -3;
    }

    /* Floating Clouds */
    .cloud {
        position: absolute;
        background: rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        filter: blur(20px);
        animation: moveClouds 60s linear infinite;
    }

    .cloud1 {
        width: 300px;
        height: 100px;
        top: 100px;
        left: -400px;
        animation-delay: 0s;
    }

    .cloud2 {
        width: 250px;
        height: 80px;
        top: 200px;
        left: -500px;
        animation-delay: 20s;
    }

    .cloud3 {
        width: 200px;
        height: 70px;
        top: 300px;
        left: -300px;
        animation-delay: 40s;
    }

    @keyframes moveClouds {
        0% { left: -400px; }
        100% { left: 100%; }
    }

    /* Rain drops */
    .rain {
        position: fixed;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -2;
    }

    .raindrop {
        position: absolute;
        width: 2px;
        height: 15px;
        background: rgba(255, 255, 255, 0.4);
        animation: rainFall 0.7s linear infinite;
    }

    @keyframes rainFall {
        0% {
            top: -10%;
            opacity: 0;
        }
        30% {
            opacity: 1;
        }
        100% {
            top: 100%;
            opacity: 0;
        }
    }
    </style>

    <!-- Sky Background -->
    <div class="sky"></div>

    <!-- Floating Clouds -->
    <div class="cloud cloud1"></div>
    <div class="cloud cloud2"></div>
    <div class="cloud cloud3"></div>

    <!-- Rain Effect -->
    <div class="rain">
        """ +
        ''.join([
            f'<div class="raindrop" style="left:{i}%; animation-delay: {i * 0.1}s;"></div>'
            for i in range(0, 100, 3)
        ]) +
    """
    </div>
    """,
    unsafe_allow_html=True
)

# Initialize database
def init_db():
    conn = sqlite3.connect('rainfall_app.db')
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE,
            password TEXT,
            created_at TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            date TIMESTAMP,
            region TEXT,
            predicted_amount REAL,
            is_above_average INTEGER,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    conn.commit()
    conn.close()

# Initialize database
init_db()
DB_PATH = 'rainfall_app.db'

# Utility to connect DB
def get_db_connection():
    return sqlite3.connect(DB_PATH)

# Save prediction
def save_prediction(user_id, region, predicted_amount, is_above_average):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO predictions (user_id, date, region, predicted_amount, is_above_average)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, datetime.datetime.now(), region, predicted_amount, is_above_average))
    conn.commit()
    conn.close()

# Greet user based on time
def dynamic_greeting():
    hour = datetime.datetime.now().hour
    if 5 <= hour < 12:
        return "Good Morning"
    elif 12 <= hour < 18:
        return "Good Afternoon"
    elif 18 <= hour < 22:
        return "Good Evening"
    else:
        return "Good Night"

# Main app
def main():
    st.title("🌤️ Rainfall Prediction App - Weather Wizard")
    st.markdown(f"### {dynamic_greeting()}, welcome to your personalized rainfall predictor!")

    
    # Load data and models
    @st.cache_data
    def load_data():
        try:
            if not os.path.exists('models'):
                os.makedirs('models')
            if not os.path.exists('data'):
                os.makedirs('data')
            try:
                with open('models/rainfall_models.pkl', 'rb') as f:
                    models = pickle.load(f)
            except FileNotFoundError:
                st.warning("Model file not found. Creating sample models for demo.")
                models = create_sample_models()
                with open('models/rainfall_models.pkl', 'wb') as f:
                    pickle.dump(models, f)
            try:
                historical_rainfall = pd.read_csv('data/rainfall_data.csv')
            except FileNotFoundError:
                st.warning("Data file not found. Creating sample data for demo.")
                historical_rainfall = create_sample_data()
                historical_rainfall.to_csv('data/rainfall_data.csv', index=False)
            return models, historical_rainfall, True
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None, None, False

    def create_sample_models():
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        sample_data = np.random.rand(100, 16)
        scaler.fit(sample_data)
        classifier = RandomForestClassifier(n_estimators=10, random_state=42)
        classifier.fit(sample_data, np.random.randint(0, 2, 100))
        regressor = RandomForestRegressor(n_estimators=10, random_state=42)
        regressor.fit(sample_data, np.random.rand(100) * 1000)
        return {'scaler': scaler, 'classifier': classifier, 'regressor': regressor}

    def create_sample_data():
        subdivisions = [
            'ANDAMAN & NICOBAR ISLANDS', 'ARUNACHAL PRADESH', 'ASSAM & MEGHALAYA',
            'NAGA MANI MIZO TRIPURA', 'SUB HIMALAYAN WEST BENGAL & SIKKIM',
            'GANGETIC WEST BENGAL', 'ORISSA', 'JHARKHAND', 'BIHAR', 'EAST UTTAR PRADESH',
            'WEST UTTAR PRADESH', 'UTTARAKHAND', 'HARYANA DELHI & CHANDIGARH',
            'PUNJAB', 'HIMACHAL PRADESH', 'JAMMU & KASHMIR', 'WEST RAJASTHAN',
            'EAST RAJASTHAN', 'WEST MADHYA PRADESH', 'EAST MADHYA PRADESH',
            'GUJARAT REGION', 'SAURASHTRA & KUTCH', 'KONKAN & GOA',
            'MADHYA MAHARASHTRA', 'MARATHWADA', 'VIDARBHA', 'COASTAL ANDHRA PRADESH',
            'TELANGANA', 'RAYALSEEMA', 'TAMIL NADU', 'COASTAL KARNATAKA',
            'NORTH INTERIOR KARNATAKA', 'SOUTH INTERIOR KARNATAKA', 'KERALA',
            'LAKSHADWEEP'
        ]
        years = list(range(1901, 2021))
        data = []
        for subdivision in subdivisions:
            for year in years:
                jan = max(0, np.random.normal(20, 10))
                feb = max(0, np.random.normal(25, 15))
                mar = max(0, np.random.normal(30, 20))
                apr = max(0, np.random.normal(40, 25))
                may = max(0, np.random.normal(60, 30))
                jun = max(0, np.random.normal(150, 50))
                jul = max(0, np.random.normal(250, 70))
                aug = max(0, np.random.normal(220, 60))
                sep = max(0, np.random.normal(180, 50))
                oct = max(0, np.random.normal(70, 30))
                nov = max(0, np.random.normal(30, 15))
                dec = max(0, np.random.normal(20, 10))
                jan_feb = jan + feb
                mar_may = mar + apr + may
                jun_sep = jun + jul + aug + sep
                oct_dec = oct + nov + dec
                annual = jan_feb + mar_may + jun_sep + oct_dec
                trend_factor = (year - 1901) / 120
                if subdivision in ['KERALA', 'TAMIL NADU', 'COASTAL KARNATAKA']:
                    annual *= (1 + 0.1 * trend_factor)
                else:
                    annual *= (1 - 0.05 * trend_factor)
                data.append({
                    'SUBDIVISION': subdivision,
                    'YEAR': year,
                    'JAN': round(jan, 1),
                    'FEB': round(feb, 1),
                    'MAR': round(mar, 1),
                    'APR': round(apr, 1),
                    'MAY': round(may, 1),
                    'JUN': round(jun, 1),
                    'JUL': round(jul, 1),
                    'AUG': round(aug, 1),
                    'SEP': round(sep, 1),
                    'OCT': round(oct, 1),
                    'NOV': round(nov, 1),
                    'DEC': round(dec, 1),
                    'Jan-Feb': round(jan_feb, 1),
                    'Mar-May': round(mar_may, 1),
                    'Jun-Sep': round(jun_sep, 1),
                    'Oct-Dec': round(oct_dec, 1),
                    'ANNUAL': round(annual, 1)
                })
        df = pd.DataFrame(data)
        return df

    models, historical_rainfall, data_loaded = load_data()
    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    seasons = ['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']
    if data_loaded:
        tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
            "Home", "Make Prediction", "Explore Data", "Model Insights",
            "Regional Map", "Compare Regions", "Future Forecast",
            "Climate Impact", "Extreme Weather Events", "About", "Settings"
        ])

      
        # --- Tab 1: Home ---
        with tab0:
            st.header("Welcome to the Rainfall Prediction App 🌦️")
            st.write("""
                This app helps you explore historical rainfall data for various regions and make simple rainfall predictions for future years.
        
                **Use the tabs to navigate through the data, prediction tool, and your saved predictions.**
                """)
        
        # Tab 1 content
        with tab1:
            st.header("🌦️ Rainfall Prediction ")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Enter Monthly Rainfall (mm)")
                jan = st.number_input('January', min_value=0.0, max_value=1000.0, value=20.0, key="jan")
                feb = st.number_input('February', min_value=0.0, max_value=1000.0, value=30.0, key="feb")
                mar = st.number_input('March', min_value=0.0, max_value=1000.0, value=40.0, key="mar")
                apr = st.number_input('April', min_value=0.0, max_value=1000.0, value=50.0, key="apr")
                may = st.number_input('May', min_value=0.0, max_value=1000.0, value=100.0, key="may")
                jun = st.number_input('June', min_value=0.0, max_value=1000.0, value=200.0, key="jun")
            with col2:
                st.subheader("Enter Monthly Rainfall (mm)")
                jul = st.number_input('July', min_value=0.0, max_value=1000.0, value=300.0, key="jul")
                aug = st.number_input('August', min_value=0.0, max_value=1000.0, value=250.0, key="aug")
                sep = st.number_input('September', min_value=0.0, max_value=1000.0, value=200.0, key="sep")
                oct = st.number_input('October', min_value=0.0, max_value=1000.0, value=100.0, key="oct")
                nov = st.number_input('November', min_value=0.0, max_value=1000.0, value=50.0, key="nov")
                dec = st.number_input('December', min_value=0.0, max_value=1000.0, value=20.0, key="dec")

            st.markdown("### Select Region and Year for Prediction")
            selected_region = st.selectbox(
                "Choose Region",
                options=sorted(historical_rainfall['SUBDIVISION'].unique()),
                index=0,
                key="region_selector"
            )
            prediction_year = st.number_input(
                "Prediction Year",
                min_value=int(datetime.datetime.now().year),
                max_value=int(datetime.datetime.now().year + 10),
                value=int(datetime.datetime.now().year),
                key="pred_year"
            )

            jan_feb = jan + feb
            mar_may = mar + apr + may
            jun_sep = jun + jul + aug + sep
            oct_dec = oct + nov + dec
            annual_total = jan + feb + mar + apr + may + jun + jul + aug + sep + oct + nov + dec

            st.markdown("### Seasonal Rainfall Summary")
            t2col1, t2col2, t2col3, t2col4 = st.columns(4)
            t2col1.metric("Jan-Feb", f"{jan_feb:.1f} mm")
            t2col2.metric("Mar-May", f"{mar_may:.1f} mm")
            t2col3.metric("Jun-Sep", f"{jun_sep:.1f} mm")
            t2col4.metric("Oct-Dec", f"{oct_dec:.1f} mm")
            st.metric("Annual Total", f"{annual_total:.1f} mm")

            if st.button('Predict Rainfall'):
                with st.spinner('Crunching numbers... just a moment!'):
                    input_data = np.array([[jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec,
                                            jan_feb, mar_may, jun_sep, oct_dec]])
                    input_scaled = models['scaler'].transform(input_data)
                    try:
                        if hasattr(models['classifier'], 'predict_proba'):
                            rain_prob = models['classifier'].predict_proba(input_scaled)[0][1]
                            rain_class = models['classifier'].predict(input_scaled)[0]
                        else:
                            rain_class = models['classifier'].predict(input_scaled)[0]
                            rain_prob = rain_class
                        rain_amount = models['regressor'].predict(input_scaled)[0]

                        st.markdown("## 🌦️ Prediction Results")
                        if hasattr(models['classifier'], 'predict_proba'):
                            st.metric("Probability of Above Average Rainfall", f"{rain_prob:.2f}")
                        if rain_class == 1:
                            st.success("✅ Prediction: Above Average Rainfall Expected")
                        else:
                            st.warning("⚠️ Prediction: Below Average Rainfall Expected")
                        st.metric("Predicted Annual Rainfall Amount", f"{rain_amount:.2f} mm")
                        
                        avg_rainfall = historical_rainfall['ANNUAL'].mean()
                        comparison_text = f"This is {'above' if rain_amount > avg_rainfall else 'below'} the historical average of {avg_rainfall:.2f} mm"
                        st.info(comparison_text)

                        if 'user_id' in st.session_state and st.session_state.user_id is not None:
                            save_prediction(
                                st.session_state.user_id, selected_region, rain_amount, 1 if rain_class == 1 else 0
                            )
                            st.balloons()
                            st.success("Prediction saved to your account history! 🎉")

                        # Enhanced Visualization with vibrant colors and tooltips
                        fig, ax = plt.subplots(figsize=(10, 6))
                        months_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        monthly_values = [jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec]
                        colors = plt.cm.plasma(np.linspace(0, 1, len(months_labels)))
                        bars = ax.bar(months_labels, monthly_values, color=colors)

                        ax.set_title(f"Monthly Rainfall Prediction for {selected_region} in {prediction_year}",
                                     fontsize=16, fontweight='bold')
                        ax.set_xlabel('Month', fontsize=12)
                        ax.set_ylabel('Rainfall (mm)', fontsize=12)
                        ax.grid(axis='y', linestyle='--', alpha=0.7)

                        # Add tooltips
                        for bar, value in zip(bars, monthly_values):
                            tooltip = f"{bar.get_x()}: {value:.2f} mm"
                            # Use a library like altair or a custom solution for interactive tooltips in Streamlit
                            #  st.write(tooltip) # Simplest approach, but not ideal for rich tooltips

                        st.pyplot(fig)  # Display the chart

                        # Create a download section for prediction results
                        st.subheader("Download Prediction Results")
                
                        # Create a DataFrame with the prediction results
                        prediction_results = pd.DataFrame({
                            'Month': months_labels,
                            'Rainfall (mm)': monthly_values
                        })

                        # Add seasonal and annual data
                        prediction_summary = pd.DataFrame({
                            'Period': ['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec', 'Annual Total', 'Predicted Annual', 'Average Annual'],
                            'Rainfall (mm)': [jan_feb, mar_may, jun_sep, oct_dec, annual_total, rain_amount, avg_rainfall]
                        })

                        # Create CSV for download
                        csv = prediction_results.to_csv(index=False)
                        summary_csv = prediction_summary.to_csv(index=False)
                
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                            label="Download Monthly Data",
                            data=csv,
                            file_name="rainfall_prediction_monthly.csv",
                            mime="text/csv",
                        )
                        
                        with col2:
                            st.download_button(
                            label="Download Summary",
                            data=summary_csv,
                            file_name="rainfall_prediction_summary.csv",
                            mime="text/csv",
                        )
                

                    except Exception as e:
                        st.error(f"Error during prediction: {e}")

        with tab2:
            st.header("Explore Historical Rainfall Data")
            st.subheader("Select Region and Year Range")
            region_select = st.selectbox("Choose a region to explore",
                                         options=sorted(historical_rainfall['SUBDIVISION'].unique()),
                                         index=0)
            year_range = st.slider("Select Year Range",
                                   min_value=1901,
                                   max_value=2020,
                                   value=(1950, 2020),
                                   step=1)

            # Filter data
            filtered_data = historical_rainfall[
                (historical_rainfall['SUBDIVISION'] == region_select) &
                (historical_rainfall['YEAR'] >= year_range[0]) &
                (historical_rainfall['YEAR'] <= year_range[1])
                ]

            # Calculate annual average rainfall
            annual_avg = filtered_data.groupby('YEAR')['ANNUAL'].mean().reset_index()

            # Display some metrics
            st.subheader(f"Historical Rainfall Overview for {region_select}")
            mean_rainfall = annual_avg['ANNUAL'].mean()
            max_rainfall = annual_avg['ANNUAL'].max()
            min_rainfall = annual_avg['ANNUAL'].min()

            st.metric("Average Annual Rainfall", f"{mean_rainfall:.2f} mm")
            st.metric("Highest Annual Rainfall", f"{max_rainfall:.2f} mm")
            st.metric("Lowest Annual Rainfall", f"{min_rainfall:.2f} mm")

            # Plotting
            st.subheader("Annual Average Rainfall Trend")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(annual_avg['YEAR'], annual_avg['ANNUAL'], marker='o', linestyle='-', color='#1e90ff')
            ax.set_title(f"Annual Average Rainfall for {region_select} ({year_range[0]} - {year_range[1]})",
                         fontsize=16, fontweight='bold')
            ax.set_xlabel("Year", fontsize=12)
            ax.set_ylabel("Average Rainfall (mm)", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)

            # Display the filtered data
            st.subheader("Filtered Data Table")
            st.dataframe(filtered_data)
            

        with tab3:
            st.header("Model Insights")
            
            # Feature importance
            st.subheader("Feature Importance")
        
            # Get feature names
            feature_names = months + seasons
        
            # Classification model feature importance
            if hasattr(models['classifier'], 'feature_importances_'):
                st.write("Classification Model Feature Importance")
            
                # Get feature importances
                importances = models['classifier'].feature_importances_
                indices = np.argsort(importances)[::-1]
            
                # Plot feature importances with colorful gradient
                fig, ax = plt.subplots(figsize=(12, 6))
                colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
                bars = ax.bar(range(len(feature_names)), importances[indices], 
                         align='center', color=colors[indices], alpha=0.8)
                ax.set_xticks(range(len(feature_names)))
                ax.set_xticklabels([feature_names[i] for i in indices], rotation=90)
                ax.set_title('Feature Importance for Rainfall Classification', fontsize=16, fontweight='bold')
                ax.set_ylabel('Importance', fontsize=12)
                ax.set_xlabel('Feature', fontsize=12)
            
                # Add grid and background colors
                ax.grid(axis='y', linestyle='--', alpha=0.3)
                fig.set_facecolor('#f0f2f6')
                ax.set_facecolor('#f8f9fa')
            
                st.pyplot(fig)
            
                # Show top 5 important features
                st.write("Top 5 Important Features for Classification:")
                top_features = pd.DataFrame({
                    'Feature': [feature_names[i] for i in indices[:5]],
                    'Importance': importances[indices[:5]]
                })
                st.dataframe(top_features)
        
            # Regression model feature importance
            if hasattr(models['regressor'], 'feature_importances_'):
                st.write("Regression Model Feature Importance")
            
                # Get feature importances
                importances = models['regressor'].feature_importances_
                indices = np.argsort(importances)[::-1]
            
                # Plot feature importances with colorful gradient
                fig, ax = plt.subplots(figsize=(12, 6))
                colors = plt.cm.plasma(np.linspace(0, 1, len(feature_names)))
                bars = ax.bar(range(len(feature_names)), importances[indices], 
                         align='center', color=colors[indices], alpha=0.8)
                ax.set_xticks(range(len(feature_names)))
                ax.set_xticklabels([feature_names[i] for i in indices], rotation=90)
                ax.set_title('Feature Importance for Rainfall Amount Prediction', fontsize=16, fontweight='bold')
                ax.set_ylabel('Importance', fontsize=12)
                ax.set_xlabel('Feature', fontsize=12)
            
                # Add grid and background colors
                ax.grid(axis='y', linestyle='--', alpha=0.3)
                fig.set_facecolor('#f0f2f6')
                ax.set_facecolor('#f4f6f8')
            
                st.pyplot(fig)
            
                # Show top 5 important features
                st.write("Top 5 Important Features for Regression:")
                top_features = pd.DataFrame({
                    'Feature': [feature_names[i] for i in indices[:5]],
                    'Importance': importances[indices[:5]]
                })
                st.dataframe(top_features)

            st.subheader("Linear Regression Coefficients")
            if 'regressor' in models:
                importances = models['regressor'].feature_importances_
                coefficients = models['regressor'].feature_importances_
                month_coefs = dict(zip(months + seasons, coefficients))
                sorted_month_coefs = sorted(month_coefs.items(), key=lambda item: abs(item[1]), reverse=True)

                for month, coef in sorted_month_coefs:
                    st.markdown(f"**{month}:** {coef:.4f}")

                st.info(
                    "These coefficients indicate the influence of each month's rainfall on the annual prediction.  "
                    "Larger magnitude coefficients suggest a stronger influence."
                )
            else:
                st.warning("Regressor model not available.")

            st.subheader("Classifier Feature Importance")
            if 'classifier' in models and hasattr(models['classifier'], 'feature_importances_'):
                importances = models['classifier'].feature_importances_
                month_importances = dict(zip(months + seasons, importances))
                sorted_month_importances = sorted(month_importances.items(), key=lambda item: item[1], reverse=True)

                for month, importance in sorted_month_importances:
                    st.markdown(f"**{month}:** {importance:.4f}")

                st.info(
                    "Feature importances from the classifier indicate which month's rainfall is most crucial for predicting  "
                    "whether the rainfall will be above or below average."
                )
            elif 'classifier' in models:
                st.warning("Classifier does not support feature importances.")
            else:
                st.warning("Classifier model not available.")

        with tab4:
            st.header("Regional Rainfall Map")

            # Select year for visualization
            years = sorted(historical_rainfall['YEAR'].unique())
            selected_year = st.selectbox("Select Year", years, index=len(years)-1)
        
            # Filter data for selected year
            year_data = historical_rainfall[historical_rainfall['YEAR'] == selected_year]
        
            # Create a DataFrame with subdivision and annual rainfall
            map_data = year_data[['SUBDIVISION', 'ANNUAL']].copy()
        
            # Display the map
            st.write(f"Annual Rainfall Distribution Across India in {selected_year}")
        
            # Use Plotly for interactive map (if you have coordinates for subdivisions)
            # For now, we'll use a bar chart as a placeholder
            fig, ax = plt.subplots(figsize=(20, 16))
        
            # Sort by rainfall amount for better visualization
            map_data = map_data.sort_values('ANNUAL', ascending=False)
        
            # Create colorful bars with a rainfall-themed colormap
            colors = plt.cm.Blues(np.linspace(0.3, 1, len(map_data)))
            bars = ax.barh(map_data['SUBDIVISION'], map_data['ANNUAL'], color=colors)
        
            ax.set_title(f'Annual Rainfall by Region in {selected_year}', fontsize=16, fontweight='bold')
            ax.set_xlabel('Rainfall (mm)', fontsize=12)
            ax.set_ylabel('Region', fontsize=12)
            ax.grid(axis='x', linestyle='--', alpha=0.3)
        
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.annotate(f'{width:.1f}',
                        xy=(width, bar.get_y() + bar.get_height()/2),
                        xytext=(5, 0),  # 5 points horizontal offset
                        textcoords="offset points",
                        ha='left', va='center', fontsize=9)
        
            fig.set_facecolor('#f0f2f6')
            ax.set_facecolor('#f8f9fa')
        
            st.pyplot(fig)

 
            st.warning("Interactive map visualization is complex and may require additional libraries or a different approach.  A static representation is shown.")
            # Create a simplified representation of a map -  replace with actual map if possible
            regions = historical_rainfall['SUBDIVISION'].unique()
            region_rainfall = {}
            for region in regions:
                region_rainfall[region] = historical_rainfall[historical_rainfall['SUBDIVISION'] == region]['ANNUAL'].mean()

            # Example: Create a DataFrame for the map (replace with actual coordinates if available)
            map_data = pd.DataFrame({
                'Region': list(region_rainfall.keys()),
                'Average Rainfall': list(region_rainfall.values()),
                'Latitude': [20, 27, 25, 24, 26, 23, 20, 24, 25, 26, 27, 30, 29, 31, 32, 34, 26, 25, 23, 22, 21, 20, 15, 18, 19, 21, 17, 18, 16, 13, 14, 15, 16, 10, 11],  # Example
                'Longitude': [73, 93, 92, 94, 88, 87, 85, 86, 87, 82, 80, 79, 76, 75, 77, 76, 72, 74, 75, 76, 71, 70, 74, 73, 77, 79, 80, 81, 79, 80, 75, 76, 77, 76, 72] # Example
            })

            # Create a scatter plot to represent rainfall (replace with a real map)
            fig, ax = plt.subplots(figsize=(12, 8))
            scatter = ax.scatter(map_data['Longitude'], map_data['Latitude'], s=map_data['Average Rainfall'] / 10,  # Size represents rainfall
                                 c=map_data['Average Rainfall'], cmap='viridis', alpha=0.7)  # Color represents rainfall
            ax.set_title('Average Annual Rainfall by Region', fontsize=18, fontweight='bold')
            ax.set_xlabel('Longitude', fontsize=12)
            ax.set_ylabel('Latitude', fontsize=12)

            # Add region labels (optional, for clarity)
            for i, row in map_data.iterrows():
                ax.text(row['Longitude'], row['Latitude'], row['Region'][:3], fontsize=8, ha='center', va='bottom')

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, label='Average Rainfall (mm)')
            st.pyplot(fig)

        with tab5:
            st.header("Compare Regional Rainfall")
            regions_to_compare = st.multiselect("Select regions to compare",
                                               options=sorted(historical_rainfall['SUBDIVISION'].unique()),
                                               default=[historical_rainfall['SUBDIVISION'].unique()[0],
                                                        historical_rainfall['SUBDIVISION'].unique()[1]])

            if len(regions_to_compare) > 0:
                comparison_data = historical_rainfall[historical_rainfall['SUBDIVISION'].isin(regions_to_compare)]
                annual_avg_comparison = comparison_data.groupby(['SUBDIVISION', 'YEAR'])['ANNUAL'].mean().reset_index()

                #  Plotting
                fig, ax = plt.subplots(figsize=(12, 6))
                for region in regions_to_compare:
                    region_data = annual_avg_comparison[annual_avg_comparison['SUBDIVISION'] == region]
                    ax.plot(region_data['YEAR'], region_data['ANNUAL'], marker='o', linestyle='-', label=region)

                ax.set_title('Annual Average Rainfall for Selected Regions', fontsize=16, fontweight='bold')
                ax.set_xlabel('Year', fontsize=12)
                ax.set_ylabel('Average Rainfall (mm)', fontsize=12)
                ax.legend(fontsize=10)
                ax.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig)
            else:
                st.warning("Please select at least one region to compare.")
                    
            # Select subdivision
            subdivisions = sorted(historical_rainfall['SUBDIVISION'].unique())
       
            col1, col2 = st.columns(2)
            with col1:
                region1 = st.selectbox("Select First Region", subdivisions, index=0, key="region1")
            with col2:
                region2 = st.selectbox("Select Second Region", subdivisions, index=min(1, len(subdivisions)-1), key="region2")
        
            # Filter data for selected regions
            region1_data = historical_rainfall[historical_rainfall['SUBDIVISION'] == region1]
            region2_data = historical_rainfall[historical_rainfall['SUBDIVISION'] == region2]
        
        # Monthly comparison
            st.subheader("Monthly Rainfall Comparison")
        
            # Calculate monthly averages
            region1_monthly = region1_data[months].mean()
            region2_monthly = region2_data[months].mean()
        
            # Create DataFrame for plotting
            monthly_comparison = pd.DataFrame({
                'Month': months,
                region1: region1_monthly.values,
                region2: region2_monthly.values
            })
        
            # Plot
            fig, ax = plt.subplots(figsize=(12, 6))
        
            x = np.arange(len(months))
            width = 0.35
        
            # Create grouped bar chart
            bars1 = ax.bar(x - width/2, monthly_comparison[region1], width, label=region1, color='#bcbd22', alpha=0.8)
            bars2 = ax.bar(x + width/2, monthly_comparison[region2], width, label=region2, color='#9467bd', alpha=0.8)
        
            ax.set_title(f'Monthly Rainfall Comparison: {region1} vs {region2}', fontsize=16, fontweight='bold')
            ax.set_xlabel('Month', fontsize=12)
            ax.set_ylabel('Average Rainfall (mm)', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(months)
            ax.legend(fontsize=10)
            ax.grid(axis='y', linestyle='--', alpha=0.3)
        
            fig.set_facecolor('#f0f2f6')
            ax.set_facecolor('#f8f9fa')
        
            st.pyplot(fig)

        with tab6:
            st.header("Future Rainfall Forecast")
            st.subheader("Select Region to Forecast")
            forecast_region = st.selectbox("Choose a region for future forecast",
                                               options=sorted(historical_rainfall['SUBDIVISION'].unique()),
                                               index=0)
            years_to_forecast = st.slider("Select the number of years to forecast",
                                                min_value=1,
                                                max_value=20,
                                                value=5,
                                                step=1)

            # Prepare data for forecasting (simplified - replace with actual model)
            region_data = historical_rainfall[historical_rainfall['SUBDIVISION'] == forecast_region].copy()
            region_data['YEAR_NUM'] = (region_data['YEAR'] - 1900).astype(int)  # Create a numerical year feature

            # Use a simple linear regression for demonstration
            model = LinearRegression()
            model.fit(region_data[['YEAR_NUM']], region_data['ANNUAL'])

            # Generate forecast years
            last_year = region_data['YEAR'].max()
            forecast_years = np.array(range(last_year + 1, last_year + years_to_forecast + 1)).reshape(-1, 1)
            forecast_years_num = np.array(range((last_year - 1900) + 1, (last_year - 1900) + years_to_forecast + 1)).reshape(-1, 1)

            # Make predictions
            forecast_predictions = model.predict(forecast_years_num)

            # Plotting
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(region_data['YEAR'], region_data['ANNUAL'], marker='o', linestyle='-', label='Historical Data')
            ax.plot(forecast_years, forecast_predictions, marker='o', linestyle='--', color='#ff7f0e', label='Forecast')
            ax.set_title(f"Rainfall Forecast for {forecast_region}", fontsize=16, fontweight='bold')
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel('Annual Rainfall (mm)', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)

            # Display forecast values
            st.subheader("Forecasted Rainfall Values:")
            for year, prediction in zip(forecast_years, forecast_predictions):
                st.markdown(f"**{int(year)}:** {prediction:.2f} mm")

        with tab7:
            st.header("Climate Change Impact")
            st.subheader("Select Region to Analyze")
            climate_region = st.selectbox("Choose a region to analyze climate change impact",
                                               options=sorted(historical_rainfall['SUBDIVISION'].unique()),
                                               index=0)

            # Filter data for the selected region
            climate_data = historical_rainfall[historical_rainfall['SUBDIVISION'] == climate_region].copy()
            climate_data['DECADE'] = (climate_data['YEAR'] // 10) * 10  # Group by decade

            # Calculate average rainfall per decade
            decade_avg_rainfall = climate_data.groupby('DECADE')['ANNUAL'].mean().reset_index()

            # Plotting
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(decade_avg_rainfall['DECADE'], decade_avg_rainfall['ANNUAL'], marker='o', linestyle='-', color='#2ca02c')
            ax.set_title(f"Decadal Average Rainfall in {climate_region}", fontsize=16, fontweight='bold')
            ax.set_xlabel('Decade', fontsize=12)
            ax.set_ylabel('Average Rainfall (mm)', fontsize=12)
            ax.set_xticks(decade_avg_rainfall['DECADE'])
            ax.set_xticklabels([f"{int(decade)}s" for decade in decade_avg_rainfall['DECADE']])
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)

            # Display analysis
            st.subheader("Analysis of Rainfall Trends")
            if len(decade_avg_rainfall) > 1:
                rainfall_change = decade_avg_rainfall['ANNUAL'].iloc[-1] - decade_avg_rainfall['ANNUAL'].iloc[0]
                percentage_change = (rainfall_change / decade_avg_rainfall['ANNUAL'].iloc[0]) * 100
                st.markdown(
                    f"The average rainfall in {climate_region} has changed by {rainfall_change:.2f} mm"
                    f" ( {percentage_change:.2f}%) between the first and last decades."
                )
                if rainfall_change > 0:
                    st.success("Rainfall has increased, which could indicate a trend towards a wetter climate.")
                else:
                    st.warning("Rainfall has decreased, which could indicate a trend towards a drier climate.")
            else:
                st.info("Insufficient data to analyze long-term trends.")

        # Fix: Properly indent the tab8 section to be inside the if data_loaded block
        with tab8:
            st.header("Extreme Weather Events")

            # Select time periods to compare
            st.subheader("Compare Rainfall Patterns Across Different Time Periods")
        
            col1, col2 = st.columns(2)
            with col1:
                start_year1 = st.number_input("Early Period Start Year", min_value=int(historical_rainfall['YEAR'].min()), 
                                         max_value=int(historical_rainfall['YEAR'].max() - 30), value=1901)
                end_year1 = st.number_input("Early Period End Year", min_value=int(start_year1), 
                                       max_value=int(start_year1 + 29), value=int(start_year1 + 29))
        
            with col2:
                start_year2 = st.number_input("Recent Period Start Year", min_value=int(end_year1 + 1), 
                                         max_value=int(historical_rainfall['YEAR'].max() - 10), 
                                         value=int(historical_rainfall['YEAR'].max() - 30))
                end_year2 = st.number_input("Recent Period End Year", min_value=int(start_year2), 
                                       max_value=int(historical_rainfall['YEAR'].max()), 
                                       value=int(historical_rainfall['YEAR'].max()))
        
            # Filter data for the two periods
            period1_data = historical_rainfall[(historical_rainfall['YEAR'] >= start_year1) & 
                                         (historical_rainfall['YEAR'] <= end_year1)]
            period2_data = historical_rainfall[(historical_rainfall['YEAR'] >= start_year2) & 
                                         (historical_rainfall['YEAR'] <= end_year2)]
        
            # Calculate average rainfall by subdivision for each period
            period1_avg = period1_data.groupby('SUBDIVISION')['ANNUAL'].mean().reset_index()
            period2_avg = period2_data.groupby('SUBDIVISION')['ANNUAL'].mean().reset_index()
        
            # Merge the data
            comparison_df = pd.merge(period1_avg, period2_avg, on='SUBDIVISION', suffixes=('_early', '_recent'))
        
            # Calculate change
            comparison_df['absolute_change'] = comparison_df['ANNUAL_recent'] - comparison_df['ANNUAL_early']
            comparison_df['percent_change'] = (comparison_df['absolute_change'] / comparison_df['ANNUAL_early']) * 100
        
            # Sort by percent change
            comparison_df = comparison_df.sort_values('percent_change')
        
            # Display the comparison
            st.subheader(f"Rainfall Change: {start_year1}-{end_year1} vs {start_year2}-{end_year2}")
        
            # Visualization
            fig, ax = plt.subplots(figsize=(12, 8))
        
            # Create colorful bars with diverging colormap
            colors = plt.cm.RdBu_r(np.linspace(0, 1, len(comparison_df)))
            bars = ax.barh(comparison_df['SUBDIVISION'], comparison_df['percent_change'], color=colors)
        
            ax.set_title('Percent Change in Annual Rainfall by Region', fontsize=16, fontweight='bold')
            ax.set_xlabel('Percent Change (%)', fontsize=12)
            ax.set_ylabel('Region', fontsize=12)
            ax.grid(axis='x', linestyle='--', alpha=0.3)
        
            # Add a vertical line at 0
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.7)
        
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                label_color = 'black' if abs(width) < 30 else 'white'
                ax.annotate(f'{width:.1f}%',
                        xy=(width, bar.get_y() + bar.get_height()/2),
                        xytext=(5 if width >= 0 else -5, 0),
                        textcoords="offset points",
                        ha='left' if width >= 0 else 'right', 
                        va='center', 
                        fontsize=9,
                        color=label_color)
        
            fig.set_facecolor('#f0f2f6')
            ax.set_facecolor('#f8f9fa')
        
            st.pyplot(fig)
        
            # Show the data table
            st.subheader("Detailed Comparison Data")
            st.dataframe(comparison_df[['SUBDIVISION', 'ANNUAL_early', 'ANNUAL_recent', 'absolute_change', 'percent_change']])

            # Select region for analysis
            extreme_region = st.selectbox("Select Region for Analysis", subdivisions, key="extreme_region")
        
            # Filter data for selected region
            extreme_data = historical_rainfall[historical_rainfall['SUBDIVISION'] == extreme_region]
        
            # Calculate statistics
            mean_rainfall = extreme_data['ANNUAL'].mean()
            std_rainfall = extreme_data['ANNUAL'].std()
        
            # Define extreme events (e.g., +/- 1.5 standard deviations from mean)
            threshold_high = mean_rainfall + 1.5 * std_rainfall
            threshold_low = mean_rainfall - 1.5 * std_rainfall
        
            # Identify extreme years
            extreme_high_years = extreme_data[extreme_data['ANNUAL'] > threshold_high]
            extreme_low_years = extreme_data[extreme_data['ANNUAL'] < threshold_low]

        
            col1, col2, col3 = st.columns(3)
            col1.metric("Average Annual Rainfall", f"{mean_rainfall:.1f} mm")
            col2.metric("High Rainfall Threshold", f"{threshold_high:.1f} mm")
            col3.metric("Low Rainfall Threshold", f"{threshold_low:.1f} mm")
        
            # Visualize all years with extreme events highlighted
            fig, ax = plt.subplots(figsize=(12, 6))
        
            # Plot all years
            ax.scatter(extreme_data['YEAR'], extreme_data['ANNUAL'], 
                  color='#1f77b4', alpha=0.6, s=50, label='Normal Years')
        
            # Highlight extreme high years
            ax.scatter(extreme_high_years['YEAR'], extreme_high_years['ANNUAL'], 
                  color='#e377c2', s=80, label='Extremely Wet Years')
        
            # Highlight extreme low years
            ax.scatter(extreme_low_years['YEAR'], extreme_low_years['ANNUAL'], 
                  color='#bcbd22', s=80, label='Extremely Dry Years')
        
            # Add threshold lines
            ax.axhline(y=threshold_high, color='#d62728', linestyle='--', alpha=0.7, 
                  label=f'High Threshold ({threshold_high:.1f} mm)')
            ax.axhline(y=threshold_low, color='#9467bd', linestyle='--', alpha=0.7, 
                  label=f'Low Threshold ({threshold_low:.1f} mm)')
            ax.axhline(y=mean_rainfall, color='black', linestyle='-', alpha=0.5, 
                  label=f'Mean ({mean_rainfall:.1f} mm)')
        
            ax.set_title(f'Extreme Rainfall Events in {extreme_region}', fontsize=16, fontweight='bold')
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel('Annual Rainfall (mm)', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, loc='upper left')
        
            fig.set_facecolor('#f0f2f6')
            ax.set_facecolor('#f8f9fa')
        
            st.pyplot(fig)

            # Analyze frequency of extreme events over time
            st.subheader("Frequency of Extreme Events Over Time")
        
            # Create time periods (e.g., decades)
            extreme_data['Decade'] = (extreme_data['YEAR'] // 10) * 10
        
            # Re-filter with the Decade column included
            extreme_high_years = extreme_data[extreme_data['ANNUAL'] > threshold_high].copy()
            extreme_low_years = extreme_data[extreme_data['ANNUAL'] < threshold_low].copy()
        
            # Count extreme events by decade
            high_events_by_decade = extreme_high_years.groupby('Decade').size().reset_index(name='High Events')
            low_events_by_decade = extreme_low_years.groupby('Decade').size().reset_index(name='Low Events')
        
            # Merge the data
            events_by_decade = pd.merge(high_events_by_decade, low_events_by_decade, on='Decade', how='outer').fillna(0)
        
            # Visualize
            fig, ax = plt.subplots(figsize=(12, 6))
        
            x = events_by_decade['Decade']
            width = 3
        
            # Create grouped bar chart
            bars1 = ax.bar(x - width/2, events_by_decade['High Events'], width, label='Extremely Wet Years', color='#d62728', alpha=0.8)
            bars2 = ax.bar(x + width/2, events_by_decade['Low Events'], width, label='Extremely Dry Years', color='#17becf', alpha=0.8)
        
            ax.set_title(f'Frequency of Extreme Rainfall Events by Decade in {extreme_region}', fontsize=16, fontweight='bold')
            ax.set_xlabel('Decade', fontsize=12)
            ax.set_ylabel('Number of Events', fontsize=12)
            ax.set_xticks(events_by_decade['Decade'])
            ax.set_xticklabels([f"{int(decade)}s" for decade in events_by_decade['Decade']])
            ax.legend(fontsize=10)
            ax.grid(axis='y', linestyle='--', alpha=0.3)
        
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.annotate(f'{int(height)}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=9)
        
            fig.set_facecolor('#f0f2f6')
            ax.set_facecolor('#f8f9fa')
        
            st.pyplot(fig)
        
        # --- Tab 9: About ---
        with tab9:
            st.header("About This App")
            st.write("""
            **Rainfall Prediction App** is a demo project built with Streamlit to showcase how to analyze historical rainfall data and perform simple predictions.
        
            - Developer: Rakshitha M R ||@rakshithamr319   
            - Data: Synthetic sample data generated for demo purposes  
            - Model: Simple average-based prediction with random noise  
        
            This app is for learning and demonstration only.
            """)
            st.markdown("[GitHub Repository](https://github.com/RakshithaMR2002/Rainfall-Prediction-in-India)")

        # --- Tab 10: Settings ---
        with tab10:
            st.header("User Settings")
            st.write("This is a placeholder for settings such as password change, profile update, etc.")
            st.info("Settings features are not implemented in this demo.")


        # Add a footer
        st.markdown("<footer>© 2025 Rainfall Prediction App | Weather Wizard</footer>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
