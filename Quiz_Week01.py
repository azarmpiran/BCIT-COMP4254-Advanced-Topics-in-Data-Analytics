# Azarm Piran
# A01195657
# Advanced Topics in Data Analytics - Quiz 1

import pandas as pd


# Looping Through DataFrames
# Individual cell values can be accessed by specifying the row number and column name
# Example 11: Accessing Cell Values
import pandas as pd
df = pd.DataFrame()
df = df.append({"Full Name":"Kyle Lowry", "Street Address":"1 Main St."},ignore_index=True)
df = df.append({"Full Name":"Jamal Murray", "Street Address":"12 Meyer St."},ignore_index=True)
df = df.append({"Full Name":"Alex Caruso", "Street Address":"11 Tree St."},ignore_index=True)

# Loop through data frame and show rows.
for i in range(0, len(df)):
    print(df.iloc[i]['Full Name'])




# Assigning One Cell at a Time
# Example 12: Assigning One Cell at a Time
import pandas as pd
df = pd.DataFrame()
df = df.append({"Full Name":"Kyle Lowry"},   ignore_index=True)
df = df.append({"Full Name":"Jamal Murray"}, ignore_index=True)
df = df.append({"Full Name":"Alex Caruso"},  ignore_index=True)

# Create and initialize column with whole number if the column stores integers.
df['Count'] = 0

print(df)
# iat needs the column position so this function will find it.
def getColumnPosition(df, columnNameInput):
    columnNames = df.keys()
    for i in range(0, len(columnNames)):
        if(columnNames[i]==columnNameInput):
            return i

colPosition = getColumnPosition(df, 'Count')

# Store integers in each cell o the 'Count' column.
# The values are generated with i*100 for the sake of demonstration.
for i in range(0, len(df)):
    df.iat[i, colPosition] = i*100 # Store integers.
print(df)




# Numeric DataFrame Summaries
# Grouping on Columns
# Example 16: Simple Grouping Summary

import pandas as pd

# Create and show DataFrame with four columns.
dataSet = {'First': ['Bobbie', 'Jennie', 'Rita', 'Jacky', 'Jane', "May"],
           'Last':  ["H.", "K.", "J.", "Z.", "A.", "H."],
           'Age':   [4, 5, 4, 6, 5, 5],
           'Prov':  ['BC', 'AB', 'BC', 'BC','AB', "BC"]}

df     = pd.DataFrame(dataSet, columns= ['First', 'Last', 'Age', 'Prov'])
print(df)

# Get count children per Province. Rename summary column to ‘Children Per Province’
dfStats = df.groupby('Prov')['Last'].count().reset_index().rename(columns={'Last': 'Children Per Province'})
print(dfStats)
















# Part A:
import pandas as pd
dataSet = {
    'Cities': ['Kelowna', 'Edmonton', 'Vancouver', 'Portland','Calgary', 'Los Angeles', 'Tacoma', 'Dallas'],
    'Region': ['BC', 'AB', 'BC', 'WA', 'AB', 'CA', 'WA', 'TX'],
    'Temperature': [-3, -16, 5, 8, -3, 20, 9, 24]
}

df = pd.DataFrame(data=dataSet)
df['Description'] = ""
#print(df)


def getColumnPosition(df, columnNameInput):
    columnNames = df.keys()
    for i in range(0, len(columnNames)):
        if(columnNames[i]==columnNameInput):
            return i

colPosition = getColumnPosition(df, 'Description')

print(colPosition)

for i in range(0, len(df)):
    if(df.iloc[i]['Temperature']) <= 0:
        df.iat[i, colPosition] = "It is freezing"

    elif(df.iloc[i]['Temperature']) <= 12:
        df.iat[i, colPosition] = "It is cool"
    else:
        df.iat[i, colPosition] = "It is warm"

print(df)








# Part B:
import pandas as pd
dataSet = {
    'Cities': ['Kelowna', 'Edmonton', 'Vancouver', 'Portland','Calgary', 'Los Angeles', 'Tacoma', 'Dallas'],
    'Region': ['BC', 'AB', 'BC', 'WA', 'AB', 'CA', 'WA', 'TX'],
    'Temperature': [-3, -16, 5, 8, -3, 20, 9, 24]
}

df = pd.DataFrame(data=dataSet)
df['Description'] = ""
#print(df)


def getColumnPosition(df, columnNameInput):
    columnNames = df.keys()
    for i in range(0, len(columnNames)):
        if(columnNames[i]==columnNameInput):
            return i

colPosition = getColumnPosition(df, 'Description')

print(colPosition)

for i in range(0, len(df)):
    if(df.iloc[i]['Temperature']) <= 0:
        df.iat[i, colPosition] = "It is freezing"

    elif(df.iloc[i]['Temperature']) <= 12:
        df.iat[i, colPosition] = "It is cool"
    else:
        df.iat[i, colPosition] = "It is warm"

#print(df)
dfStats = df.groupby('Region')['Temperature'].mean().reset_index().rename(columns={'Temperature': 'Average Temp'})
#print(dfStats)
df1 = df.groupby('Region')['Temperature'].min().reset_index().rename(columns={'Temperature': 'Min Temp'})
dfStats['Min Temp'] = df1['Min Temp']
#print(dfStats)
df1 = df.groupby('Region')['Temperature'].max().reset_index().rename(columns={'Temperature': 'Max Temp'})
dfStats['Max Temp'] = df1['Max Temp']
print(dfStats)



# Test

# Part B:
import pandas as pd
dataSet = {
    'Cities': ['Kelowna', 'Edmonton', 'Vancouver', 'Portland','Calgary', 'Los Angeles', 'Tacoma', 'Dallas'],
    'Region': ['BC', 'AB', 'BC', 'WA', 'AB', 'CA', 'WA', 'TX'],
    'Temperature': [-3, -16, 5, 8, -3, 20, 9, 24]
}

df = pd.DataFrame(data=dataSet)
df['Description'] = ""
#print(df)


def getColumnPosition(df, columnNameInput):
    columnNames = df.keys()
    for i in range(0, len(columnNames)):
        if(columnNames[i]==columnNameInput):
            return i

colPosition = getColumnPosition(df, 'Description')

print(colPosition)

for i in range(0, len(df)):
    if(df.iloc[i]['Temperature']) <= 0:
        df.iat[i, colPosition] = "It is freezing"

    elif(df.iloc[i]['Temperature']) <= 12:
        df.iat[i, colPosition] = "It is cool"
    else:
        df.iat[i, colPosition] = "It is warm"

#print(df)
dfStats = df.groupby('Region')['Temperature'].mean().reset_index().rename(columns={'Temperature': 'Average Temp'})
#print(dfStats)
df1 = df.groupby('Region')['Temperature'].min().reset_index().rename(columns={'Temperature': 'Min Temp'})
dfStats['Min Temp'] = df1['Min Temp']
#print(dfStats)
df1 = df.groupby('Region')['Temperature'].max().reset_index().rename(columns={'Temperature': 'Max Temp'})
dfStats['Max Temp'] = df1['Max Temp']
print(dfStats)