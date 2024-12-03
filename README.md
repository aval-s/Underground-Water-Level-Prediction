### Underground Water Level Prediction


 --- 

## 🌍 Introduction
Urban groundwater plays a critical role in addressing challenges related to weather resilience and sustainable water management. Groundwater levels are dynamic, influenced by weather, land use, and withdrawal rates. Accurate predictions are crucial to mitigate overexploitation and manage this vital resource sustainably.

This project focuses on forecasting groundwater levels in India using machine learning and time-series techniques. The goal is to help stakeholders, such as engineers and water managers, make informed decisions for sustainable water use.

## 🔍 Problem Statement
Groundwater overexploitation leads to:

- Significant water-level declines.
- Increased pumping costs.
- Aquifer compaction and reduced water quality.

To secure future water availability, there is an urgent need for reliable groundwater-level predictions, especially in arid and semi-arid regions.

## 🎯 Objectives
- Develop a highly accurate groundwater level prediction model.
- Use historical data to forecast future groundwater levels.
- Leverage time-series techniques like SARIMA and ARIMA to analyze seasonal and trend variations.

## 🛠️ Technical Design

##### Approach:
1. _**Exploratory Data Analysis (EDA)**_: Understand data characteristics and detect anomalies.
2. _**Preprocessing**_: Handle missing values, outliers, and make data stationary.
3. _**Modeling**_: Build predictive models using ARIMA and SARIMA.
4. _**Visualization**_: Provide insights through intuitive charts and GUI for user interaction.

##### Architecture:
- Input: Raw groundwater data (1990–2015) sourced from Central Ground Water Board (CGWB).
- Processing: Cleaned, seasonal decomposition, and transformed into stationarity.
- Modeling:
	* **ARIMA**: Predict non-seasonal variations.
	* **SARIMA**: Capture seasonal components.
- Output: Forecast groundwater levels for the next five years (2015–2020).

## 📊 Key Features
- _**Time-Series Modeling**_: Analysis using ACF/PACF plots and optimal parameter tuning.
- _**Data Cleaning**_: Robust handling of missing values and outliers.
- _**GUI Integration**_: Interactive platform for district-specific predictions.
- _**Visualization**_: Trend and seasonal decomposition plots for actionable insights.

## 📈 Results
- Achieved reliable predictions with an R² score > 0.8.
- Visualized 5-year forecasts to assist water management policies.

## 🖥️ Technologies Used
- _**Languages**_: Python (3.8+)
- _**Libraries**_: Pandas, NumPy, Seaborn, Matplotlib, StatsModels, Scikit-Learn.
- _**Visualization**_: Tkinter GUI for user interaction.
- _**Platform**_: Cross-platform compatibility with cloud options (AWS, Google Cloud, Azure).

## 🚀 How to Run
Clone the repository:

1. Clone the repository:
   ```bash
   git clone https://github.com/aval-s/Underground-Water-Level-Prediction.git
   ```

2. Install dependencies:
	```bash
	pip install -r requirements.txt
	```
3. Run the project:
	```bash
	python main.py
 	```

## 🛡️ Future Work
- Enhance model precision using deep learning techniques like LSTMs.
- Incorporate real-time IoT sensor data for adaptive forecasting.
- Scale the application for other regions with diverse aquifer properties.
