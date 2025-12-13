import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import os
from src.logger import info

def tag_data():
  df = pd.read_csv("artifacts/train_category.csv")

  df['full_text'] = df['Ticket Subject'].fillna('') + " " + df['Ticket Description'].fillna('')

  vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
  X = vectorizer.fit_transform(df['full_text'])

  kmeans = KMeans(n_clusters=5, random_state=42)
  clusters = kmeans.fit_predict(X)

  df['Ticket Type'] = clusters

  clusters_map = {
    0:'Account Access',
    1:'General Inquiry',
    2:'Firmware/Update Issue',
    3: 'Technical Support',
    4: 'Product Issue'
  }

  df['Ticket Type Name'] = df['Ticket Type'].map(clusters_map)

  info("Saving tagged data...")

  df.to_csv("artifacts/train_category.csv", index=False)

  df_test = pd.read_csv("artifacts/test_category.csv")

  df_test['full_text'] = df_test['Ticket Subject'].fillna('') + " " + df_test['Ticket Description'].fillna('')
  X_test = vectorizer.transform(df_test['full_text'])

  df_test['Ticket Type'] = kmeans.predict(X_test)
  df_test['Ticket Type Name'] = df_test['Ticket Type'].map(clusters_map)
  df_test.to_csv("artifacts/test_category.csv", index=False)

  info("Successfully relabed data ")
  info(df[['Ticket Type Name', 'full_text']].head(5))


if __name__ == "__main__":
    tag_data()  