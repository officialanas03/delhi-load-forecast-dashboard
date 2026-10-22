# Delhi Load Forecasting & Demand-Response Dashboard

## How this project started

I am an M.Tech student in Electrical Power Systems and Management at Jamia Millia Islamia, Delhi. Earlier, I completed an economic dispatch project using real grid data and transmission losses. That project worked out the cheapest way to supply a load, but it started with the load already known.

This raised a simple question for me: where does that load figure come from? In a real grid, operators do not know tomorrow's demand in advance. They have to forecast it. I decided to build my next project around this question, and I chose Delhi because i am doing my masters from here and its demand changes a lot between seasons.

The aim is to forecast Delhi's electricity demand from past load and weather data, and to show the result in a simple dashboard that marks expected peak hours.

## How I carried it out

### Step 1: Collecting the data

I used two sources:

- **Load data: Delhi's electricity demand recorded every 5 minutes, starting from April 2023. I downloaded it from a Kaggle dataset of Delhi SLDC data. It has about 293,000 rows.
- **Weather data:daily temperature and humidity for Delhi, taken from the Open-Meteo API with a small Python script.

I chose weather data because temperature is expected to affect demand strongly. People use more air conditioners in summer and heaters in winter.

### Step 2: Cleaning and merging

Data every 5 minutes is very detailed and noisy. I converted it to hourly averages, which gave about 24,400 rows. I then joined it with the weather data by date and added a few columns: hour of the day, day of the week, weekend or not, and month. A forecasting model needs these to learn daily and weekly patterns.

### Step 3: Looking at the data

Before building any model, I plotted the data to see what patterns exist. This is what I found.

//Daily pattern

![Average load by hour of day](reports/figures/load_by_hour.png)

Load is lowest around 4 AM (about 3,450 MW on average). It rises quickly from 5 to 10 AM, stays high through the afternoon at about 4,700 to 4,850 MW, and peaks near 3 PM.

//Seasonal pattern**

![Average load by month](reports/figures/load_by_month.png)

Summer is the heaviest season. June has the highest average load (about 5,750 MW), and May to September all stay high. February, March and November are the lowest, at about 3,050 to 3,100 MW.

//Effect of temperature

![Daily average load vs temperature](reports/figures/load_vs_temperature.png)

Load is lowest when the daily average temperature is between 17 and 23 degrees C (about 3,000 MW). Above 25 degrees it rises steeply, reaching about 7,000 MW near 38 to 39 degrees. It also rises a little below 12 degrees, probably because of heating.

The points are quite spread out. At the same temperature, daily load can differ by a few thousand MW. So temperature alone is not enough, and the model will also use humidity, hour, day of the week and month.

## What I plan to do next

1. Build a simple baseline forecast (SARIMA or Prophet).
2. Build a stronger model (XGBoost or LSTM) and compare it with the baseline.
3. Add a rule that flags expected peak-demand hours.
4. Build a Streamlit dashboard to show forecasts and alerts.
5. Deploy the dashboard and write up the results.

## Progress

- [x] Collected load and weather data
- [x] Cleaned and merged the data
- [x] Studied the patterns in the data
- [ ] Baseline forecasting model
- [ ] Stronger forecasting model
- [ ] Peak-demand alerts
- [ ] Dashboard
- [ ] Deployment and write-up

## Project files

- `src/fetch_data.py` downloads the weather data and checks the load file
- `src/preprocess.py` cleans and merges the data
- `src/eda.py` creates the charts in `reports/figures/`
- `app/` will contain the dashboard
- `notebooks/` and `models/` are for the modelling work

## How to run it

The raw data files are not stored in this repository. Download the load CSV from Kaggle and save it as `data/raw/delhi_load_raw.csv`. Then run:

    pip install -r requirements.txt
    python src/fetch_data.py
    python src/preprocess.py
    python src/eda.py
