import glob
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

kaggle.api.authenticate()
kaggle.api.dataset_download_files('olistbr/brazilian-ecommerce', path = '.', unzip=True)

#| Weather data from API

import requests
import pandas as pd

#| Visual Crossing API
API_KEY = 'DPLDYY46MS6UJADMQGFKMGVRP'
location = 'Brazil'
start_date = '2017-01-01'
end_date = '2018-12-31'
unit_group = 'metric'

#| Construct the API endpoint
url = (
    f'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/weatherdata/history'
    f'?&aggregateHours=24'
    f'&startDateTime={start_date}T00:00:00'
    f'&endDateTime={end_date}T23:59:59'
    f'&unitGroup={unit_group}'
    f'&location={location}'
    f'&key={API_KEY}'
    f'&contentType=json'
)

#| Make the request
response = requests.get(url)

#| Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()
    
    # Extract the relevant data from the JSON
    weather_data = data['locations'][location]['values']
    
    # Convert the extracted data to a pandas DataFrame
    df = pd.DataFrame(weather_data)
    
    # Display the DataFrame
    print(df)

#| Wrangle weather data    
weather_data = df[['temp', 'cloudcover', 'datetimeStr', 'precip', 'weathertype']]
weather_data['date_string'] = weather_data['datetimeStr'].str[:10]
weather_data['date'] = pd.to_datetime(weather_data['date_string'])
weather_data = weather_data[['date', 'temp', 'cloudcover', 'precip']]

#| Holidays via API - free api account
#| limits you to only the last years worth of data 

#| Set up the API key and base endpoint
API_KEY = 'd53abde5-f4ae-4072-8387-dcc98fa3f986' 
country = 'BR'  # Brazil country code

#| List to hold holiday data
all_holidays = []

#| Loop over the years to get holidays from 2017 to 2019
for year in range(2017, 2020):
    # Construct the API endpoint for Holiday API
    url = f'https://holidayapi.com/v1/holidays?key={API_KEY}&country={country}&year={year}'

    # Make the request
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        holidays = response.json().get('holidays', [])
        
        # Extract relevant data and append it to the list
        for holiday in holidays:
            holiday_data = {
                'date': holiday['date'],
                'name': holiday['name'],
                'type': ', '.join(holiday['type']),  # Join the list of holiday types into a string
                'year': year
            }
            all_holidays.append(holiday_data)
    else:
        print(f"Failed to retrieve holidays for {year}. Status code: {response.status_code}")
        print(f"Response: {response.text}")

# Convert the list of holidays into a DataFrame
holidays = pd.DataFrame(all_holidays)

#| Set the path for the folder containing the CSV files
path = '/Users/seth/Library/Mobile Documents/com~apple~CloudDocs/Documents/Freelance/Projects/eCommerce Forecasting/'  # Change this to the folder where your CSVs are stored
csv_files = glob.glob(path + "*.csv")  # This finds all CSVs in the folder

#| Read all CSVs into a dictionary and rename them
dataframes = {}

for file in csv_files:
    #| Create a name for the DataFrame (remove the path and .csv extension)
    df_name = file.split("/")[-1].replace(".csv", "").replace("olist_", "")
    
    #| Read the CSV file
    dataframes[df_name] = pd.read_csv(file)

#| Extract dataframes from dictionary of datasets
customers_df = dataframes['customers_dataset']
geolocation_df = dataframes['geolocation_dataset']
order_items_df = dataframes['order_items_dataset']
order_payments_df = dataframes['order_payments_dataset']
order_reviews_df = dataframes['order_reviews_dataset']
orders_df = dataframes['orders_dataset']
products_df = dataframes['products_dataset']
sellers_df = dataframes['sellers_dataset']
translation_df = dataframes['product_category_name_translation']
holidays = dataframes['holidays']

#| Combine categories in translation_df
def map_category(value):
    if  value in ['art', 'cds_dvds_musicals', 'dvds_blu_ray',
                  'flowers', 'arts_and_craftmanship']:
        return 'arts'
    elif value == 'auto':
        return 'auto'
    elif value in ['books_technical', 'books_general_interest',
                   'books_imported']:
        return 'books'
    elif value in ['garden_tools', 'construction_tools_construction',
                   'construction_tools_tools', 'home_construction',
                   'construction_tools_lights', 'construction_tools_safety',
                   'construction_tools_garden']:
        return 'construction'
    elif value in ['computers_accessories', 'tablets_printing_image',
                   'small_appliances', 'consoles_games', 'audio',
                   'air_conditioning', 'electronics', 'home_appliances',
                   'computers', 'cine_photo', 'small_appliances_home_oven_and_coffee']:
        return 'electronics'
    elif value in ['fashion_shoes', 'fashion_male_clothing', 
                   'fashion_underwear_beach', 'fashion_sport',
                   'fashion_childrens_clothes', 'fashion_bags_accessories',
                   'fashio_female_clothing']: # intentional typo
        return 'fashion'
    elif value in ['food_drink', 'food', 'drinks', 'la_cuisine']:
        return 'food'
    elif value in ['bed_bath_table', 'furniture_decor',
                   'kitchen_dining_laundry_garden_furniture',
                   'office_furniture', 'furniture_mattress_and_upholstery',
                   'furniture_bedroom']:
        return 'furniture'
    elif value in ['health_beauty', 'perfumery']:
        return 'health'
    elif value in ['housewares', 'home_comfort', 'home_appliances_2',
                   'home_comfort_2']:
        return 'homeware'
    elif value in ['music', 'musical_instruments']:
        return 'music'
    else:
        return 'other'

#| Apply the function to recategorise data
translation_df['category_combined'] = translation_df['product_category_name_english'].apply(map_category)

#| Join datasets together to create complete set for forecasting.
#| Join orders and order payments
order_payments_df = order_payments_df.drop_duplicates(subset = 'order_id')
orders = pd.merge(orders_df, order_payments_df, how='left', on='order_id')

#| Join product names and their translation
products = pd.merge(products_df, translation_df, how = 'inner', on = 'product_category_name')

#| Join products to order items
order_items_df = order_items_df.drop_duplicates(subset = 'order_id')
product_items = pd.merge(order_items_df, products, how = 'left', on = 'product_id')
print(len(order_items_df)) 
product_items = product_items.drop_duplicates(subset = 'order_id')
print(len(product_items)) 

#| Join orders and product_items
joined_df = pd.merge(orders, product_items, how = 'left', on = 'order_id')

#| Drop NAs from joining process
cleaned = joined_df.dropna()

#| Filter to only necessary columns
print(cleaned.info()) 
working_df = cleaned[['order_id', 'customer_id', 'order_purchase_timestamp','price', 'payment_value', 'category_combined']]
pd.set_option('display.max_columns', None)
print(working_df.head(10))

#| Working out if there is a promotion
#| Flag entries where payment_value is less than price. 
working_df['promotion_flag'] = (working_df['payment_value'] < working_df['price']).astype(int)
print(working_df.head(10))

#| Format order_purchase_timestamp
working_df['date'] =  pd.to_datetime(working_df['order_purchase_timestamp']).dt.date
working_df['date'] = pd.to_datetime(working_df['date'])

#| Add weather data for additional variables
working_df = pd.merge(working_df, weather_data, how='left', on='date')

#| Add holiday values
working_df['is_holiday'] = working_df.date.isin(holidays['Date'])  # Flag holidays

