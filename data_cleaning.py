import numpy as np
import pandas as pd
from data_import import working_df
from data_import import translation_df
from data_import import order_payments_df
from data_import import orders_df
from data_import import products_df
from data_import import order_items_df
from data_import import weather_data
from data_import import holidays

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

#| Remove duplicates
cleaned = cleaned.drop_duplicates(subset = 'order_id')
print(len(cleaned)) 

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

