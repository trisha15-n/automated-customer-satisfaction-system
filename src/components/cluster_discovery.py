import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def discover_topics():

    print("Loading data...")
    df = pd.read_csv("artifacts/train_category.csv")
    

    df['full_text'] = df['Ticket Subject'].fillna('') + " " + df['Ticket Description'].fillna('')
    

    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(df['full_text'])

    print("Running K-Means...")
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X)
    

    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    
    for i in range(5):
        print(f"\nCluster {i}:")

        top_terms = [terms[ind] for ind in order_centroids[i, :10]]
        print(f"Keywords: {', '.join(top_terms)}")
        print(f"Sample Subject: {df.iloc[kmeans.labels_ == i]['Ticket Subject'].values[0]}")

if __name__ == "__main__":
    discover_topics()