import pandas as pd
from openai import OpenAI
from kaggle_secrets import UserSecretsClient
import time

# 1. Setup OpenAI Client using Kaggle Secrets
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def get_embeddings(text_list):
    """Sends a batch of strings to OpenAI and returns their vectors."""
    try:
        response = client.embeddings.create(
            input=text_list,
            model="text-embedding-3-small"
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"Error calling OpenAI: {e}")
        return None

# 2. Load your cleaned data from Step 1
df = pd.read_csv('/kaggle/working/cleaned_recipes.csv')

# For testing, let's just do the first 50 recipes
# You can remove '.head(50)' once you verify it works!
df = df.head(1000) 

# 3. Create the text string for embedding
# We combine name and ingredients so the AI understands both
df['combined_text'] = "Recipe: " + df['name'] + ". Ingredients: " + df['ingredients']

# 4. Process in batches (OpenAI allows multiple strings per request)
batch_size = 20
all_embeddings = []

print(f"Starting vectorization for {len(df)} recipes...")

for i in range(0, len(df), batch_size):
    batch_text = df['combined_text'].iloc[i:i+batch_size].tolist()
    embeddings = get_embeddings(batch_text)
    
    if embeddings:
        all_embeddings.extend(embeddings)
        print(f"Processed {i + len(batch_text)} / {len(df)}")
    
    # Small sleep to avoid hitting rate limits too fast
    time.sleep(0.5)

# 5. Save the vectors into the dataframe
df['vector'] = all_embeddings

# 6. Save as a Pickle file
# We use .pkl because CSVs don't handle lists of numbers very well
df.to_pickle('/kaggle/working/recipes_with_vectors.pkl')
print("\nSuccess! Your vectors are saved in 'recipes_with_vectors.pkl'")