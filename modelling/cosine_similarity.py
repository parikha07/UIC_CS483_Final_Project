import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "/Users/gauthamys/Desktop/UIC_CS483_Final_Project/data"

tracks = pd.read_csv(f'{DATA_PATH}/final_similiar_tracks_list.csv')
album_sim = pd.read_csv(f'{DATA_PATH}/outputs.csv')
user_meta = pd.read_excel(f'{DATA_PATH}/all_friends_details.xlsx')

tag_sets = tracks['tags'].apply(lambda x: eval(x)) 
unique_tags = sorted(set(tag for tags in tag_sets for tag in tags))

tag_index = {tag: i for i, tag in enumerate(unique_tags)}

# Function to convert tags to feature vector
def tags_to_vector(tags, tag_index):
    vector = [0] * len(tag_index)
    for tag in tags:
        if tag in tag_index:
            vector[tag_index[tag]] = 1
    return vector

tracks['feature_vector'] = tag_sets.apply(lambda tags: tags_to_vector(tags, tag_index))

feature_matrix = tracks['feature_vector'].tolist()

# Compute cosine similarity between all feature vectors
similarity_matrix = cosine_similarity(feature_matrix)

# Convert to DataFrame for easier reading
similarity_df = pd.DataFrame(similarity_matrix, index=tracks['tracks'], columns=tracks['tracks'])

# Display the similarity matrix
print(similarity_df)