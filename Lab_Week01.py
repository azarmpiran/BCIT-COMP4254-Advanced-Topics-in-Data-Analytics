# Lab Week 1
# Advanced topics for Data Analytics
# Azarm Piran | A01195657


import pandas as pd



# Exercise 1
def showFullName(firstName, lastName):
    output = "* Full Name: " + firstName + " " + lastName
    print(output)

showFullName("Azarm","Piran")



# Example 1
# Using return at the end of the function to return the result of the function
def addTwoNumbers( operandA, operandB ):
   result = operandA + operandB;
   return result;

sum = addTwoNumbers(3,4) # This is the calling instruction.
print("The result from adding 3 and 4 is " + str(sum))




# Exercise 2
def tipCalculator(amount):
    result = amount + (0.15*amount)
    return result

amountAfterTip = tipCalculator(15.25)
print(f"Your total after tip is: {amountAfterTip}")



# Example 2: Functions with Parameters and a Conditional Block
# This function receives a package weight and country as parameters.
def showBillingStatement(packageWeight, country):
    print("\n==========================================================")
    CHARGE_PER_KG = 13.60
    flatFee       = 0.0

    variableCharge = CHARGE_PER_KG * packageWeight
    if(country == "Canada"): # Double equals sign is used for equals comparison.
        flatFee = 25.00
    elif(country == "US"):
        flatFee = 32.00

    subTotal = variableCharge + flatFee

    weightDesc = "Weight:   " + str(packageWeight) + "kg @ $" + str(CHARGE_PER_KG)
    feeDesc    = "Flat fee: $" + str(flatFee)

    print(weightDesc)
    print(feeDesc)

    print("----------------------------------------------------------")
    totalDesc = "Total:    $" + str(subTotal)
    print(totalDesc)

# These instructions call the function.
showBillingStatement(2.7, "Canada")
showBillingStatement(3.8, "US")




# Exercise 4
def showBillingStatement(packageWeight, country):
    print("\n==========================================================")
    CHARGE_PER_KG = 13.60
    flatFee = 0.0

    variableCharge = CHARGE_PER_KG * packageWeight
    if (country == "Canada"):
        flatFee = 25.00
    elif (country == "US"):
        flatFee = 32.00
    else: # I used "else" because it asks for any country outside Canada and US
        flatFee = 42.00


    subTotal = variableCharge + flatFee

    weightDesc = "Weight:   " + str(packageWeight) + "kg @ $" + str(CHARGE_PER_KG)
    feeDesc = "Flat fee: $" + str(flatFee)

    print(weightDesc)
    print(feeDesc)

    print("----------------------------------------------------------")
    totalDesc = "Total:    $" + str(subTotal)
    print(totalDesc)

showBillingStatement(2.7, "Canada")
showBillingStatement(3.8, "US")
showBillingStatement(4.8, "Mexico")



# Exercise 5
def showBillingStatement(packageWeight, country):
    print("\n==========================================================")
    CHARGE_PER_KG = 13.60
    flatFee = 0.0

    variableCharge = CHARGE_PER_KG * packageWeight
    if (country == "Canada"):
        flatFee = 25.00
    elif (country == "US"):
        flatFee = 32.00
    else: # I used "else" because it asks for any country outside Canada and US
        flatFee = 42.00


    subTotal = variableCharge + flatFee

    weightDesc = "Weight:   " + str(packageWeight) + "kg @ $" + str(CHARGE_PER_KG) + " = " + str(variableCharge)
    feeDesc = "Flat fee: $" + str(flatFee)

    print(f"Country:    {country}")
    print(weightDesc)
    print(feeDesc)

    print("----------------------------------------------------------")
    totalDesc = "Total:    $" + str(subTotal)
    print(totalDesc)

showBillingStatement(2.7, "Canada")
showBillingStatement(3.8, "US")
showBillingStatement(4.8, "Mexico")



# Example 3
# Three different ways to iterate through an array and print it
islands = ["Hawaii", "Maui", "Oahu"]

print("The array length is " + str(len(islands))  + ".")

# Use for loop and index.
for index in range(0, len(islands)):
    print("index " + str(index) + ": " + islands[index])
print("\nHere is the list again.\n")

# Use for loop and object.
for island in islands:
    print(island)

# Use while loop and counter.
print("\nHere is the list again.\n")
counter = 0
while(counter < len(islands)):
    print("counter " + str(counter) + ": " + islands[counter])
    counter +=1


# Exercise 6
dictionaries = []
dictionaries.append({"First":"Bob", "Last":"Jones"})
dictionaries.append({"First":"Harpreet", "Last":"Kaur"})
dictionaries.append({"First":"Mohamad",  "Last":"Argani"})

for i in range(0, len(dictionaries)):
    print(dictionaries[i]['First'] + " " + dictionaries[i]['Last'])




# Exercise 7
dictionaries = []
dictionaries.append({"First":"Bob", "Last":"Jones", "favouriteColour":"red"})
dictionaries.append({"First":"Harpreet", "Last":"Kaur", "favouriteColour":"green"})
dictionaries.append({"First":"Mohamad",  "Last":"Argani", "favouriteColour":"blue"})

for i in range(0, len(dictionaries)):
    print(dictionaries[i]['First'] + " " + dictionaries[i]['Last'] + " likes " + dictionaries[i]['favouriteColour'] + " colour")





# Exercise 8
dictionaries = []
dictionaries.append({"First":"Bob", "Last":"Jones", "favouriteColour":"red"})
dictionaries.append({"First":"Harpreet", "Last":"Kaur", "favouriteColour":"green"})
dictionaries.append({"First":"Mohamad",  "Last":"Argani", "favouriteColour":"blue"})

for i in range(0, len(dictionaries)):
    if (dictionaries[i]['First'] == "Bob") or (dictionaries[i]['Last'] == "Kaur"):
    print(dictionaries[i]['First'] + " " + dictionaries[i]['Last'])




# Exercise 9
dictionaries = []
dictionaries.append({"First":"Bob", "Last":"Jones", "favouriteColour":"red"})
dictionaries.append({"First":"Harpreet", "Last":"Kaur", "favouriteColour":"green"})
dictionaries.append({"First":"Mohamad",  "Last":"Argani", "favouriteColour":"blue"})

i = 0
while (i < len(dictionaries)):
    if (dictionaries[i]['First'] == "Bob") or (dictionaries[i]['Last'] == "Kaur"):
        print(dictionaries[i]['First'] + " " + dictionaries[i]['Last'])
    i = i + 1



# Example 5
fullName      = "Bob Jones"
spacePosition = fullName.find(" ")
startPosition = 0
endPosition   = spacePosition
firstName     = fullName[startPosition:endPosition]
print(firstName)



# Exercise 10
fullName      = "Bob Jones"
spacePosition = fullName.find(" ")
startPosition = spacePosition
endPosition   = len(fullName)
lastName     = fullName[startPosition+1:endPosition]
print(lastName)

fullName      = "Azarm Piran"
spacePosition = fullName.find(" ")
startPosition = spacePosition
endPosition   = len(fullName)
lastName     = fullName[startPosition+1:endPosition]
print(lastName)



# Example 6: Finding Multiple Occurrences of a Substring
text = "A lazy dog jumped over a log."
positions = []
for i in range(0,len(text)-1):
    if(text[i:i+2] == "og"):
        positions.append(i)
print(positions)



# saving firstname and lastname and city seprately in 3 different lists
def getEventHostData():
    hostData = []
    hostData.append("Jill Richler, Vancouver June 16")
    hostData.append("May Chen, Toronto June 17")
    hostData.append("Belinda Sigardson, Calgary July 20")
    return hostData

eventHosts = getEventHostData()

firstName = []
lastName = []
city = []

for eventHost in eventHosts:
    firstNameEndPosition = eventHost.find(" ")
    firstNameStartPosition = 0
    firstName.append(eventHost[firstNameStartPosition:firstNameEndPosition])

    lastNameFirstPosition = eventHost.find(" ")
    lastNameEndPosition = eventHost.find(",")
    lastName.append(eventHost[lastNameFirstPosition+1:lastNameEndPosition])

    positions = []
    for i in range(0, len(eventHost) - 1):
        if (eventHost[i:i + 1] == " "):
            positions.append(i)
    cityStartPosition = positions[1]
    cityEndPosition = positions[2]
    city.append(eventHost[cityStartPosition+1:cityEndPosition])

    print(f"{eventHost[firstNameStartPosition:firstNameEndPosition]} {eventHost[lastNameFirstPosition+1:lastNameEndPosition]}, {eventHost[cityStartPosition+1:cityEndPosition]}")


print(firstName)
print(lastName)
print(city)




# Extracting only full name and city from each element of the hostData and print it
# Exercise 11
def getEventHostData():
    hostData = []
    hostData.append("Jill Richler, Vancouver June 16")
    hostData.append("May Chen, Toronto June 17")
    hostData.append("Belinda Sigardson, Calgary July 20")
    return hostData

eventHosts = getEventHostData()
for eventHost in eventHosts:

    positions = []

    for i in range(0, len(eventHost) - 1):
        if (eventHost[i:i + 1] == " "):
            positions.append(i)

    print(eventHost[0:positions[2]])



# Example 8
sentenceArray = ['A', 'lazy', 'dog', 'jumped', 'over', 'a', 'log.']
delimiter     = ' '
newSentence   = delimiter.join(sentenceArray)
print(newSentence)


# Exercise 12
sentenceArray = ['A', 'lazy', 'dog', 'jumped', 'over', 'a', 'log.']
delimiter     = '*'
newSentence   = delimiter.join(sentenceArray)
print(newSentence)




# Example 9: Creating a DataFrame
import pandas as pd
df = pd.DataFrame()
df = df.append({"Full Name":"Eric Choi", "Street Address":"1 Main St."},ignore_index=True)
df = df.append({"Full Name":"Will Lum", "Street Address":"12 Meyer St."},ignore_index=True)
print(df)


# Exercise 13
import pandas as pd
df = pd.DataFrame()
df = df.append({"Full Name":"Eric Choi", "Street Address":"1 Main St."},ignore_index=True)
df = df.append({"Full Name":"Will Lum", "Street Address":"12 Meyer St."},ignore_index=True)
df = df.append({"Full Name":"Azarm Piran", "Street Address":"3 Commercial St."},ignore_index=True)
print(df)





# Example 10: Adding Columns to DataFrames
import pandas as pd
df = pd.DataFrame()
df = df.append({"Full Name":"Eric Choi", "Street Address":"1 Main St."},ignore_index=True)
df = df.append({"Full Name":"Will Lum", "Street Address":"12 Meyer St."},ignore_index=True)
df['Colour'] = ['red','green']
print(df)



# Exercise 14
import pandas as pd
df = pd.DataFrame()
df = df.append({"Full Name":"Eric Choi", "Street Address":"1 Main St."},ignore_index=True)
df = df.append({"Full Name":"Will Lum", "Street Address":"12 Meyer St."},ignore_index=True)
df['Colour'] = ['red','green']
df['City'] = ['Vancouver','Burnaby']
print(df)



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



# Exercise 15
import pandas as pd
df = pd.DataFrame()
df = df.append({"Full Name":"Kyle Lowry", "Street Address":"1 Main St."},ignore_index=True)
df = df.append({"Full Name":"Jamal Murray", "Street Address":"12 Meyer St."},ignore_index=True)
df = df.append({"Full Name":"Alex Caruso", "Street Address":"11 Tree St."},ignore_index=True)

for i in range(0, len(df)):
    print(df.iloc[i]['Street Address'])





# Exercise 16
import pandas as pd
df = pd.DataFrame()
df = df.append({"Full Name":"Kyle Lowry", "Street Address":"1 Main St."},ignore_index=True)
df = df.append({"Full Name":"Jamal Murray", "Street Address":"12 Meyer St."},ignore_index=True)
df = df.append({"Full Name":"Alex Caruso", "Street Address":"11 Tree St."},ignore_index=True)


print(df.iloc[[0]])
print("\n")
print(df.iloc[0])
print("\n")
print(df.loc[0,:])



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




# Exercise 17
import pandas as pd
df = pd.DataFrame()
df = df.append({"Full Name":"Kyle Lowry"},   ignore_index=True)
df = df.append({"Full Name":"Jamal Murray"}, ignore_index=True)
df = df.append({"Full Name":"Alex Caruso"},  ignore_index=True)

df['Count'] = 0.0000

def getColumnPosition(df, columnNameInput):
    columnNames = df.keys()
    for i in range(0, len(columnNames)):
        if(columnNames[i]==columnNameInput):
            return i

colPosition = getColumnPosition(df, 'Count')

for i in range(0, len(df)):
    df.iat[i, colPosition] = i*100.5783
print(df)



# Exercise 18
import pandas as pd
df = pd.DataFrame()
df = df.append({"Full Name":"Kyle Lowry"},   ignore_index=True)
df = df.append({"Full Name":"Jamal Murray"}, ignore_index=True)
df = df.append({"Full Name":"Alex Caruso"},  ignore_index=True)

df['Count'] = 0.0

def getColumnPosition(df, columnNameInput):
    columnNames = df.keys()
    for i in range(0, len(columnNames)):
        if(columnNames[i]==columnNameInput):
            return i

colPosition = getColumnPosition(df, 'Count')

for i in range(0, len(df)):
    df.iat[i, colPosition] = i*100.5783
print(df)



# Example 13: Extracting Column Subsets
import pandas as pd

# Create and show DataFrame with four columns.
dataSet = {'First': ['Bobbie', 'Jennie', 'Rita', 'Jacky', 'Jane'],
           'Last':  ["H.", "K.", "J.", "Z.", "A."],
           'Age':   [4, 5, 4, 6, 5],
           'Prov':  ['BC', 'AB', 'BC', 'BC','AB']}

df     = pd.DataFrame(dataSet, columns= ['First', 'Last', 'Age', 'Prov'])
print(df)

# Create and show subset of DataFrame.
print("\n*******************")
dfNames = df[["First", "Last"]]
print (dfNames)



# Exercise 19
import pandas as pd

# Create and show DataFrame with four columns.
dataSet = {'First': ['Bobbie', 'Jennie', 'Rita', 'Jacky', 'Jane'],
           'Last':  ["H.", "K.", "J.", "Z.", "A."],
           'Age':   [4, 5, 4, 6, 5],
           'Prov':  ['BC', 'AB', 'BC', 'BC','AB']}

df     = pd.DataFrame(dataSet, columns= ['First', 'Last', 'Age', 'Prov'])
print(df)


print("\n*******************")
dfNames = df[["First", "Prov"]]
print (dfNames)





# Analyzing Frequencies
# Example 14: Frequencies and Ranges
print("\n*******************\nAge Frequency:")
print(df['Age'].value_counts())





# Exercise 20
import pandas as pd

dataSet = {'First': ['Bobbie', 'Jennie', 'Rita', 'Jacky', 'Jane'],
           'Last':  ["H.", "K.", "J.", "Z.", "A."],
           'Age':   [4, 5, 4, 6, 5],
           'Prov':  ['BC', 'AB', 'BC', 'BC','AB']}
df     = pd.DataFrame(dataSet, columns= ['First', 'Last', 'Age', 'Prov'])


print("\n*******************\nCount of children per each province:")
print(df['Prov'].value_counts())




# Example 15: Sorting DataFrame Content
import pandas as pd
dataSet = {'First': ['Bobbie', 'Jennie', 'Rita', 'Jacky', 'Jane'],
           'Last':  ["H.", "K.", "J.", "Z.", "A."],
           'Age':   [4, 5, 4, 6, 5],
           'Prov':  ['BC', 'AB', 'BC', 'BC','AB']}
df     = pd.DataFrame(dataSet, columns= ['First', 'Last', 'Age', 'Prov'])

print("\n*******************")
dfNames = df[["First", "Last"]]
print (dfNames)

dfSorted = df.sort_values(['Prov', 'Age'], ascending=[True, True])
print(dfSorted)




# Exercise 21
import pandas as pd
dataSet = {'First': ['Bobbie', 'Jennie', 'Rita', 'Jacky', 'Jane'],
           'Last':  ["H.", "K.", "J.", "Z.", "A."],
           'Age':   [4, 5, 4, 6, 5],
           'Prov':  ['BC', 'AB', 'BC', 'BC','AB']}
df     = pd.DataFrame(dataSet, columns= ['First', 'Last', 'Age', 'Prov'])

print("\n*******************")
dfNames = df[["First", "Last"]]
print (dfNames)
print("\n")

dfSorted = df.sort_values(['First'], ascending=[True])
print(dfSorted)




# Filtering a DataFrame
# Exercise 22
import pandas as pd
dataSet = {'First': ['Bobbie', 'Jennie', 'Rita', 'Jacky', 'Jane'],
           'Last':  ["H.", "K.", "J.", "Z.", "A."],
           'Age':   [4, 5, 4, 6, 5],
           'Prov':  ['BC', 'AB', 'BC', 'BC','AB']}
df     = pd.DataFrame(dataSet, columns= ['First', 'Last', 'Age', 'Prov'])

print(df)
print("\n*******************")
print(df[(df['Age']>5) & (df['Prov']=='BC')])
print("\n*******************")




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






# Exercise 23
import pandas as pd

dataSet = {'First': ['Bobbie', 'Jennie', 'Rita', 'Jacky', 'Jane', "May"],
           'Last':  ["H.", "K.", "J.", "Z.", "A.", "H."],
           'Age':   [4, 5, 4, 6, 5, 5],
           'Prov':  ['BC', 'AB', 'BC', 'BC','AB', "BC"]}

df     = pd.DataFrame(dataSet, columns= ['First', 'Last', 'Age', 'Prov'])
print(df)

# Avg child age per province
dfStats = df.groupby('Prov')['Age'].mean().reset_index().rename(columns={'Avg': 'Average Child Age'})
print(dfStats)



# Example 17: Grouping Summaries
import pandas as pd

# Create and show DataFrame with four columns.
dataSet = {'First': ['Bobbie', 'Jennie', 'Rita', 'Jacky', 'Jane', "May"],
           'Last':  ["H.", "K.", "J.", "Z.", "A.", "H."],
           'Age':   [4, 5, 4, 6, 5, 5],
           'Prov':  ['BC', 'AB', 'BC', 'BC','AB', "BC"]}

df     = pd.DataFrame(dataSet, columns= ['First', 'Last', 'Age', 'Prov'])

# Get count of children per Province.
dfStats = df.groupby('Prov')['Last'].count().reset_index().rename(columns={'Last': 'Child Count'})

# Get maximum age per Province.
df3 = df.groupby('Prov')['Age'].max().reset_index().rename(columns={'Age': 'Maximum Age'})
dfStats['Maximum Age'] = df3['Maximum Age']
print(dfStats)



# Exercise 24
import pandas as pd

# Create and show DataFrame with four columns.
dataSet = {'First': ['Bobbie', 'Jennie', 'Rita', 'Jacky', 'Jane', "May"],
           'Last':  ["H.", "K.", "J.", "Z.", "A.", "H."],
           'Age':   [4, 5, 4, 6, 5, 5],
           'Prov':  ['BC', 'AB', 'BC', 'BC','AB', "BC"]}

df     = pd.DataFrame(dataSet, columns= ['First', 'Last', 'Age', 'Prov'])

# Get count of children per Province.
dfStats = df.groupby('Prov')['Last'].count().reset_index().rename(columns={'Last': 'Child Count'})

# Get maximum age per Province.
df3 = df.groupby('Prov')['Age'].max().reset_index().rename(columns={'Age': 'Maximum Age'})
dfStats['Maximum Age'] = df3['Maximum Age']

# Get min age per Province.
df4 = df.groupby('Prov')['Age'].min().reset_index().rename(columns={'Age': 'Minimum Age'})
dfStats['Minimum Age'] = df4['Minimum Age']


# Get avg age per Province.
df5 = df.groupby('Prov')['Age'].mean().reset_index().rename(columns={'Age': 'Average Age'})
dfStats['Average Age'] = df5['Average Age']

print(dfStats)






# Exercise 25

FOLDER = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
FILE   = "fruit.csv"
df = pd.read_csv(FOLDER + FILE)
print(df.head())

# Generate a calculated column.
df['Revenue'] = round(df['Price'] * df['Quantity'], 2)
print(df)


# Generate revenue summary per region.
dfStats = df.groupby(['Product','Region'])['Revenue'].sum().reset_index().rename(columns={'Revenue': 'Total Revenue'})
print(dfStats)








# Example 18: Querying DataFrames with SQL

import pandas as pd
from sqlalchemy import create_engine

# Read file.
FOLDER = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
FILE   = "fruit.csv"
df = pd.read_csv(FOLDER + FILE)

# Placed query in this function to enable code re-usuability.
def showQueryResult(sql, df, tableName):
    # This code creates an in-memory table called 'FruitTable'.
    engine     = create_engine('sqlite://', echo=False)
    connection = engine.connect()
    df.to_sql(name=tableName, con=connection, if_exists='replace', index=False)

    # This code performs the query.
    queryResult = pd.read_sql(sql, connection)
    return queryResult

# Read all rows from the table.
SQL       = "SELECT * FROM FruitTable"
queryDf   = showQueryResult(SQL, df, 'FruitTable')
print(queryDf)




# Exercise 26
import pandas as pd
from sqlalchemy import create_engine

# Read file.
FOLDER = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
FILE   = "fruit.csv"
df = pd.read_csv(FOLDER + FILE)

# Placed query in this function to enable code re-usuability.
def showQueryResult(sql, df, tableName):
    # This code creates an in-memory table called 'FruitTable'.
    engine     = create_engine('sqlite://', echo=False)
    connection = engine.connect()
    df.to_sql(name=tableName, con=connection, if_exists='replace', index=False)

    # This code performs the query.
    queryResult = pd.read_sql(sql, connection)
    return queryResult

# Read all rows from the table.
SQL       = "SELECT Product,Quantity,Region FROM FruitTable WHERE Product = 'apples'"
queryDf   = showQueryResult(SQL, df, 'FruitTable')
print(queryDf)












# Example 19: SQL for Grouping and Summary Queries with DataFrames
import pandas as pd
from sqlalchemy import create_engine

# Read file.
FOLDER = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
FILE   = "fruit.csv"
df = pd.read_csv(FOLDER + FILE)

# Placed query in this function to enable code re-usuability.
def showQueryResult(sql, df, tableName):
    # This code creates an in-memory table called 'FruitTable'.
    engine     = create_engine('sqlite://', echo=False)
    connection = engine.connect()
    df.to_sql(name=tableName, con=connection, if_exists='replace', index=False)

    # This code performs the query.
    queryResult = pd.read_sql(sql, connection)
    return queryResult

# Read all rows from the table.
SQL = "SELECT Product,SUM(Price*Quantity) AS Revenue FROM FruitTable GROUP BY Product"
queryDf   = showQueryResult(SQL, df, 'FruitTable')
print(queryDf)



# Exercise 27
import pandas as pd
from sqlalchemy import create_engine

# Read file.
FOLDER = "C:\\Users\\Azarm\\Desktop\\BCIT\\AdvancedTopics\\DataSets\\"
FILE   = "fruit.csv"
df = pd.read_csv(FOLDER + FILE)

# Placed query in this function to enable code re-usuability.
def showQueryResult(sql, df, tableName):
    # This code creates an in-memory table called 'FruitTable'.
    engine     = create_engine('sqlite://', echo=False)
    connection = engine.connect()
    df.to_sql(name=tableName, con=connection, if_exists='replace', index=False)

    # This code performs the query.
    queryResult = pd.read_sql(sql, connection)
    return queryResult

# Read all rows from the table.
SQL = "SELECT Region,SUM(Price*Quantity) AS Revenue FROM FruitTable GROUP BY Region"
queryDf   = showQueryResult(SQL, df, 'FruitTable')
print(queryDf)




