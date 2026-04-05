import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from kaggle_secrets import UserSecretsClient

# 1. Setup Pinecone Client
user_secrets = UserSecretsClient()
pc = Pinecone(api_key=user_secrets.get_secret("PINECONE_API_KEY"))

# 2. Create the Index if it doesn't exist
index_name = "fridge-ai-recipes"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

# Connect to the index
index = pc.Index(index_name)

# 3. Load your vectors from Step 2
df = pd.read_pickle('/kaggle/working/recipes_with_vectors.pkl')

# 4. Prepare data for Pinecone (ID, Vector, and Metadata)
# We batch this to avoid hitting network limits
batch_size = 100
for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]
    
    upsert_data = []
    for _, row in batch.iterrows():
        upsert_data.append({
            "id": str(row['id']), 
            "values": row['vector'], 
            "metadata": {
                "name": row['name'],
                "ingredients": row['ingredients'],
                "steps": row['steps'][:500] # Truncate steps to keep metadata small
            }
        })
    
    index.upsert(vectors=upsert_data)
    print(f"Upserted batch {i//batch_size + 1}")

print("All recipes are now in the cloud!")