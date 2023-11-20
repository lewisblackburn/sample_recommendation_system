import pandas as pd
import scipy.sparse as sparse
from implicit.nearest_neighbours import bm25_weight
from implicit.als import AlternatingLeastSquares
import numpy as np

# Load your data from a CSV file (replace 'your_data.csv' with your actual file)
df = pd.read_csv('data.csv')

# Convert 'rating' and 'userid' columns to numeric
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['userid'] = pd.to_numeric(df['userid'], errors='coerce')

# Create mappings for user and film IDs
user_mapping = {user: i for i, user in enumerate(df['userid'].unique())}
film_mapping = {film: i for i, film in enumerate(df['film'].unique())}

# Use the mappings to create integer-based user and film IDs
df['userid'] = df['userid'].map(user_mapping)
df['film'] = df['film'].map(film_mapping)

# Create a sparse matrix
sparse_data = sparse.coo_matrix((df['rating'], (df['userid'], df['film'])))

# Use the implicit library's BM25 weighting
sparse_data = bm25_weight(sparse_data, K1=100, B=0.8)

# Transpose the matrix
user_film_matrix = sparse_data.T.tocsr()

# Create the model and fit it
model = AlternatingLeastSquares(factors=64, regularization=0.05, alpha=2.0, random_state=42)
model.fit(user_film_matrix)

for x in range(1, 5):
    # Choose a sample user ID for recommendations
    sample_userid = x

    # Get recommendations for the sample user
    ids, scores = model.recommend(sample_userid, user_film_matrix[sample_userid], N=10, filter_already_liked_items=False)

    # Inverse mappings to get original film names
    inverse_film_mapping = {i: film for film, i in film_mapping.items()}
    result = pd.DataFrame({"film": [inverse_film_mapping.get(i, 'Unknown') for i in ids], "score": scores})

    print(result)
