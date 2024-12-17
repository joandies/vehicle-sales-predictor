import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import datetime as dt
import yaml

# Function to handle rare categories in categorical columns
def group_rare_categories(df, column_name, min_count):
    value_counts = df[column_name].value_counts()
    rare_categories = value_counts[value_counts < min_count].index
    df[column_name] = df[column_name].apply(lambda x: 'other' if x in rare_categories else x)
    return df

def get_raw_data_path(config_path='../../configs/cleaning.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config.get('raw_data_path')

def clean_car_data():
    # Get the raw data path from the YAML file
    raw_data_path = get_raw_data_path()
    df = pd.read_csv(raw_data_path)

    # Remove duplication caused by case differences in string columns
    Object_col = df.select_dtypes(include="object").columns.to_list()
    for column in Object_col:
        df[column] = df[column].str.title()

    # Standardize 'make' column
    df['make'] = df['make'].replace({
        'Bmw': 'BMW', 'Gmc': 'GMC', 'Ram': 'RAM', 'Gmc Truck': 'GMC', 'Dodge Tk': 'Dodge',
        'Mazda Tk': 'Mazda', 'Hyundai Tk': 'Hyundai', 'Mercedes-B': 'Mercedes', 'Mercedes-Benz': 'Mercedes',
        'Vw': 'VW', 'Chev Truck': 'Chevrolet', 'Ford Tk': 'Ford', 'Ford Truck': 'Ford'
    })

    # Handle rare categories in 'make' column
    df = group_rare_categories(df, 'make', 28)

    # Handle rare categories in 'trim' column
    df = group_rare_categories(df, 'trim', 101)

    # Calculate car age
    saledate = df['saledate'].str.split(expand=True)
    df = pd.merge(left=df, right=saledate.iloc[:, 3], left_index=True, right_index=True)
    df.rename(columns={3: 'salesyear'}, inplace=True)
    df['salesyear'] = df['salesyear'].fillna(2015).astype(np.int64)
    df['car_age'] = df['salesyear'] - df['year']
    df['car_age'] = df['car_age'].apply(lambda x: 0 if x == -1 else x)

    # Handle missing values in 'state' column
    df['state'] = df['state'].apply(lambda x: np.nan if x[0] == '3' else x)
    df['state'].replace({
        'Ca': 'California', 'Tx': 'Texas', 'Pa': 'Pennsylvania', 'Mn': 'Minnesota',
        'Az': 'Arizona', 'Wi': 'Wisconsin', 'Tn': 'Tennessee', 'Md': 'Maryland', 'Fl': 'Florida',
        'Ne': 'Nebraska', 'Nj': 'New Jersey', 'Nv': 'Nevada', 'Oh': 'Ohio', 'Mi': 'Michigan',
        'Ga': 'Georgia', 'Va': 'Virginia', 'Sc': 'South Carolina', 'Nc': 'North Carolina',
        'In': 'Indiana', 'Il': 'Illinois', 'Co': 'Colorado', 'Ut': 'Utah', 'Mo': 'Missouri',
        'Ny': 'New York', 'Ma': 'Massachusetts', 'Pr': 'Puerto Rico', 'Or': 'Oregon', 'La': 'Louisiana',
        'Wa': 'Washington', 'Hi': 'Hawaii', 'Qc': 'Quebec', 'Ab': 'Alberta', 'On': 'Ontario',
        'Ok': 'Oklahoma', 'Ms': 'Mississippi', 'Nm': 'New Mexico', 'Al': 'Alabama', 'Ns': 'Nova Scotia'
    }, inplace=True)

    # Handle rare categories in 'state' column
    df = group_rare_categories(df, 'state', 201)
    
    # Fix erroneous values in 'transmission' and 'body' columns
    for i, row in df.iterrows():
        if row['transmission'] == 'Sedan':
            df.loc[i, 'body'] = 'Sedan'
            df.loc[i, 'transmission'] = np.nan
    
    # Handle 'body' column categories
    df['body'].replace({
        'G37 Coupe': 'Coupe', 'Cts Wagon': 'Wagon', 'Cts-V Wagon': 'Wagon', 'G37 Convertible': 'Convertible',
        'G Sedan': 'Sedan', 'G Convertible': 'Convertible', 'G Coupe': 'Coupe', 'Granturismo Convertible': 'Convertible',
        'Ram Van': 'Van', 'Transit Van': 'Van', 'Q60 Convertible': 'Convertible', 'Q60 Coupe': 'Coupe',
        'Tsx Sport Wagon': 'Wagon', 'Beetle Convertible': 'Convertible', 'E-Series Van': 'Van',
        'Elantra Coupe': 'Coupe', 'Genesis Coupe': 'Coupe', 'Koup': 'Coupe', 'Cts Coupe': 'Coupe',
        'Cts-V Coupe': 'Coupe', 'Promaster Cargo Van': 'Van', 'Supercrew': 'Cab', 'Double Cab': 'Cab',
        'Access Cab': 'Cab', 'King Cab': 'Cab', 'Extended Cab': 'Cab', 'Supercab': 'Cab', 'Regular Cab': 'Cab',
        'Quad Cab': 'Cab', 'Club Cab': 'Cab', 'Xtracab': 'Cab', 'Mega Cab': 'Cab', 'Cab Plus 4': 'Cab',
        'Cab Plus': 'Cab', 'Crewmax Cab': 'Cab', 'Crew Cab': 'Cab', 'Regular-Cab': 'Cab'
    }, inplace=True)


    # Handle 'color' column
    def clean_color_values(x):
        del_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '—']
        for i in str(x):
            if str(i) in del_list:
                return np.nan
        return x

    df['color'] = df['color'].apply(clean_color_values)
    df = group_rare_categories(df, 'color', 251)

    # Handle 'interior' column
    df['interior'].replace({'—': np.nan}, inplace=True)
    df = group_rare_categories(df, 'interior', 201)

    # Handle 'condition' column
    df['condition'] = df['condition'].apply(lambda x: x * 10 if x < 10 else x)
    
    # Handle 'odometer' column
    df['odometer'] = pd.to_numeric(df['odometer'])
    df['odometer'] = df.apply(lambda x: np.nan if (x['odometer'] < 2 and x['car_age'] > 0) or (x['odometer'] > 990000) else x['odometer'], axis=1)

    # Clean 'sellingprice' column and filter outliers
    df['price_res'] = df['mmr'] - df['sellingprice']
    df = df[(df['sellingprice']>150) & (df['price_res']<25000) & (df['price_res']>-25000)]
    
    # Drop irrelevant columns
    df.drop(columns=['year', 'vin', 'saledate', 'salesyear', 'price_res', 'seller'], inplace=True)
    # Drop rows with missing values in essential columns like 'make'
    df.dropna(axis=0, subset=['make', 'model'], inplace=True)
    df.to_csv('../../data/processed/test.csv', index=False)

    return df

if __name__ == '__main__':
    cleaned_df = clean_car_data()
    # You can export the cleaned dataframe to a file if necessary:
    cleaned_df.to_csv('../../data/processed/car_prices_after_FE_TESTPY.csv', index=False)
