import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Carregar os dados públicos
df = pd.read_csv('german_credit_data.csv')

# Identify categorical columns
categorical_columns = ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose", "Risk"]

# Perform one-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Save the encoded dataframe to a new file
#encoded_file_path = "german_credit_data_encoded.csv"
#df_encoded.to_csv(encoded_file_path, index=False)

# Exibir as primeiras linhas
print(df.columns)
print(df.head())
print(df_encoded.head())

# Define colors for risk categories
colors = {"good": "blue", "bad": "red"}

# Create scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="Age", y="Credit amount", hue="Risk", palette=colors, alpha=0.7)

# Customize plot
plt.xlabel("Age")
plt.ylabel("Credit amount")
plt.title("Age vs Credit amount with Risk Categories")
plt.legend(title="Risk")
plt.grid(True)

# Show plot
plt.show()