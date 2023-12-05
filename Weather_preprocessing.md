# Preprocessing - Weather Data

## Data Source

The weather data is sourced from the KMA Weather Data Open Portal, specifically the Disaster Prevention Weather Observation (AWS) dataset. The data is available at [KMA Data Portal](https://data.kma.go.kr/data/grnd/selectAwsRltmList.do?pgmNo=56).

## Data Preprocessing Steps

1. **Data Loading and Merging:**

   - Two CSV files (`OBS_AWS_TIM_20231112111131.csv` and `OBS_AWS_TIM_20231112110248.csv`) are loaded and concatenated into a single DataFrame (`merged_df`).

2. **Data Cleaning:**

   - Check for missing values using `train.info()` and handle them appropriately.
   - Create 'date' and 'time' columns based on the '일시' column.
   - Convert the '일시' column to datetime format and store it in the 'date' column.
   - Generate the 'rain' column based on the '강수량(mm)' column.

3. **Weather-related Features:**

   - Calculate the 'wind_chill' using the `calculate_wind_chill` function.
   - Create 'heat_warning' and 'heat_alert' features using the `check_heat_warning` and `check_heat_alert` functions.
   - Generate 'cold_warning' and 'cold_alert' features using the `check_cold_warning` and `check_cold_alert` functions.

4. **Save Processed Data:**
   - Save the final preprocessed DataFrame to a CSV file named 'preprocessed_weather.csv'.

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/jeonghyeonee/23_BA_preprocessing_modeling.git
   cd YOUR_REPOSITORY
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the preprocessing script:

   ```bash
   python weather_preprocessing.py
   ```

4. Find the preprocessed data in the 'preprocessed_weather.csv' file.

Feel free to customize the code and adapt it to your specific requirements.

---
