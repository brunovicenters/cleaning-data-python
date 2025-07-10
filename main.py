import pandas as pd
import numpy as np
import datetime as dt
import tabulate as tb

# import data
df = pd.read_json('dirty-data.json')

# print data
print(tb.tabulate(df.head(), headers='keys', tablefmt='grid'))

# get info about df
df.info()
print(df.describe())

# deal with duplicates
duplicates = df.duplicated()
df.drop_duplicates(inplace=True)

# duplicates with different info
dup_person = df.duplicated(subset=['name', 'blood_type', 'siblings'], keep='last')
df = df[~dup_person]

# duplicated ids
dup_id = df.duplicated(subset=['id'], keep='first')

for i in df[dup_id].index:
    df.at[i, 'id'] = df['id'].max() + 1

# resort values
df = df.sort_values(by = 'id')

# fix birthdate - multiple formats
df['birthdate_parsed'] = pd.to_datetime(df['birthdate'], errors='coerce', format="%Y-%m-%d")

mask = df['birthdate_parsed'].isna()
df.loc[mask, 'birthdate_parsed'] = pd.to_datetime(df.loc[mask, "birthdate"], errors='coerce', format="%Y/%m/%d")

df['birthdate'] = df['birthdate_parsed']

df.drop(columns=['birthdate_parsed'], inplace=True)

# fix and validate age
df['age'] = pd.to_numeric(df['age'], errors='coerce')

today = pd.Timestamp.today().normalize()

# calculate age
df["calculated_age"] = (
    (today - df["birthdate"]).dt.days // 365
)

mask = df["age"].notna()
df["age_valid"] = False
df.loc[mask, "age_valid"] = (
    df.loc[mask, "age"].astype(int) == df.loc[mask, "calculated_age"]
)

# assign valid age
df.loc[df['birthdate'].notna() & (df['age_valid'] == False), 'age'] =  df.loc[df['birthdate'].notna() & (df["age_valid"] == False), "calculated_age"]

df.drop(columns=['age_valid', 'calculated_age'], inplace=True)

# now, turn birthdate in date only
df["birthdate"] = df["birthdate"].dt.strftime("%Y-%m-%d")

# clean age
df.loc[df['age'] > 120, 'age'] = None
df['age'] = df['age'].astype('Int64')

assert df['age'].dtype == 'Int64', 'Age with wrong type'

# clean name's column
df['name'] = df['name'].str.lower()
df['name'] = df['name'].str.strip()

# clean blood_type
accepted_bt = ['A', 'B', 'O', 'AB']

inc_bt = set(df['blood_type']).difference(accepted_bt)
inc_rows = df['blood_type'].isin(inc_bt)

df.loc[inc_rows, 'blood_type'] = None
df['blood_type'] = df['blood_type'].astype('category')

assert df['blood_type'].dtype == 'category', 'Blood type with wrong type'

# clean gender
df['gender'] = df['gender'].str.strip()
df['gender'] = df['gender'].str.lower()
gender_mapping = {'woman': 'w', 'w': 'w', 'female': 'w',
                  'man': 'm',  'm': 'm', 'male': 'm'}
df['gender'] = df['gender'].replace(gender_mapping)

df['gender'] = df['gender'].astype('category')

# clean weight
df['weight'] = df['weight'].str.lower()

# separate pounds and convert it to kg
mask = df['weight'].str.contains('lb', na=False)
pounds_df = df.loc[mask, 'weight'].str.replace('lb', '', regex=False).str.strip()
pounds_df = round(pd.to_numeric(pounds_df, errors='coerce') / 2.205, 2)
df.loc[mask, 'weight'] = pounds_df

# strip char and convert do float
df.loc[df['weight'].str.contains('kg', na=False), 'weight'] = df.loc[df['weight'].str.contains('kg', na=False), 'weight'].str.replace('kg', '', regex=False)
df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
df.loc[df['weight'] < 0, 'weight'] = -df['weight']
assert df['weight'].dtype == 'float64', "Weight with wrong type"

df.rename(columns={'weight': 'weight_kg'}, inplace=True)

# clean wait_time_minutes
df.loc[df['wait_time_minutes'] < 0, 'wait_time_minutes'] = - df['wait_time_minutes']

# create ranges of wait_time
label_ranges = [0, 10, 30, np.inf]
label_names = ['short', 'average', 'long']

df['wait_time_range'] = pd.cut(df['wait_time_minutes'], bins= label_ranges, labels=label_names)

# clean and assert number of siblings
df.loc[df['brothers'] < 0, 'brothers'] = 0
df.loc[df['sisters'] < 0, 'sisters'] = 0
df.loc[df['siblings'] < 0, 'siblings'] = 0

df.loc[df['brothers'] + df['sisters'] != df['siblings'], 'siblings'] = df['brothers'] + df['sisters']

mask = (df['brothers'] + df['sisters']) != df['siblings']
assert mask.sum() == 0, "Some rows have inconsistent sibling counts!"

# final columns
df = df[['id', 'name', 'age', 'birthdate', 'gender', 'weight_kg', 'blood_type', 'wait_time_minutes', 'wait_time_range', 'brothers', 'sisters', 'siblings']]

print(tb.tabulate(df, headers='keys'))

df.to_json('clean-data.json', orient='records', indent=2, date_format='iso')