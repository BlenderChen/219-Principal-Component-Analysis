import pandas as pd
import matplotlib.pyplot as plt
import re
import random
import numpy as np

# Load the data
df = pd.read_csv("Project1-ClassificationDataset.csv")

# Function to count alphanumeric characters
def extract_alphanumeric(text):
    return "".join(re.findall(r"[A-Za-z0-9]", text))

def count_alphanumeric(text):
    return len(re.findall(r"[A-Za-z0-9]", text))

def clean(text):
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    texter = re.sub(r"<br />", " ", text)
    texter = re.sub(r"&quot;", "\"",texter)
    texter = re.sub('&#39;', "\"", texter)
    texter = re.sub('\n', " ", texter)
    texter = re.sub(' u '," you ", texter)
    texter = re.sub('`',"", texter)
    texter = re.sub(' +', ' ', texter)
    texter = re.sub(r"(!)\1+", r"!", texter)
    texter = re.sub(r"(\?)\1+", r"?", texter)
    texter = re.sub('&amp;', 'and', texter)
    texter = re.sub('\r', ' ',texter)
    clean = re.compile('<.*?>')
    texter = texter.encode('ascii', 'ignore').decode('ascii')
    texter = re.sub(clean, '', texter)
    if texter == "":
        texter = ""
    return texter


# Hierarchy for the leaf_label
leaf_label_order = ['basketball', 'baseball', 'tennis', 'football', 'soccer', 'forest fire', 'flood', 'earthquake', 'drought', 'heatwave']

# Reading the CSV file of each column
full_text = df.iloc[:, 0]
leaf_label = df.iloc[:, 6]
root_label = df.iloc[:, 7]

# Add alphanumeric count column
df["alphanumeric_count"] = full_text.apply(count_alphanumeric)

# Convert leaf_label column to categorical with the specified order
df['leaf_label'] = pd.Categorical(df['leaf_label'], categories=leaf_label_order, ordered=True)

plt.hist(df["alphanumeric_count"], bins=100, color="skyblue", edgecolor="black", alpha=0.7)
plt.title("Distribution of Alphanumeric Characters per Row")
plt.xlabel("Number of Alphanumeric Characters")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Plot leaf label distribution with specified order
df['leaf_label'].value_counts().reindex(leaf_label_order).dropna().plot(kind='bar', color='lightgreen', edgecolor='black')
plt.title('Leaf Label Distribution')
plt.xlabel('Leaf Label')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Plot root label distribution
df['root_label'].value_counts().plot(kind='bar', color='lightcoral', edgecolor='black')
plt.title('Root Label Distribution')
plt.xlabel('Root Label')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

#Binary Classification
np.random.seed(42)
random.seed(42)
from sklearn.model_selection import train_test_split
#spliting the data set into 80% training and 20% testing randomly
train, test = train_test_split(df[["full_text", "root_label"]], test_size=0.2) 

