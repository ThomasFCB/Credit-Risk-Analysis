import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Load public data
df = pd.read_csv('german_credit_data.csv')
sns.countplot(x=df["Risk"])
plt.title("Distribuição da variável alvo")

# Identify categorical columns
categorical_columns = ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose", "Risk"]

# Perform one-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Save the encoded dataframe to a new file
#encoded_file_path = "german_credit_data_encoded.csv"
#df_encoded.to_csv(encoded_file_path, index=False)

# Show first rows
print(df.columns)
print(df.head())
print(df_encoded.head())

#Histogram for some important variables
df[['Age','Job','Credit amount','Duration']].hist(figsize=(12, 10), bins=20)

# Define colors for risk categories
colors = {"good": "blue", "bad": "red"}

# Create scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="Age", y="Credit amount", hue="Risk", palette=colors, alpha=0.7)

plt.xlabel("Age")
plt.ylabel("Credit amount")
plt.title("Age vs Credit amount with Risk Categories")
plt.legend(title="Risk")
plt.grid(True)

plt.show()
