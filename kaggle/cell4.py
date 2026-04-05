from openai import OpenAI
from pinecone import Pinecone
from kaggle_secrets import UserSecretsClient

# 1. Setup Clients
user_secrets = UserSecretsClient()
client = OpenAI(api_key=user_secrets.get_secret("OPENAI_API_KEY"))
pc = Pinecone(api_key=user_secrets.get_secret("PINECONE_API_KEY"))

# 2. Connect to your existing index
index_name = "fridge-ai-recipes"
index = pc.Index(index_name)

def search_fridge(human_sentence, top_k=3):
    # Step A: Human Sentence -> 1536 Vector
    response = client.embeddings.create(
        input=[human_sentence],
        model="text-embedding-3-small"
    )
    query_vector = response.data[0].embedding

    # Step B: Vector -> Pinecone Search
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True # This returns the name, ingredients, and steps!
    )

    # Step C: Display Results
    print(f"--- Results for: '{human_sentence}' ---")
    for match in results['matches']:
        print(f"\n[Score: {match['score']:.2f}] Recipe: {match['metadata']['name']}")
        print(f"Ingredients: {match['metadata']['ingredients']}")
        print(f"Steps: {match['metadata']['steps'][:150]}...") # Just a preview

# 3. TEST IT!
search_fridge("'broccoli', 'cabbage', 'carrots', 'green peas', 'cauliflower', 'romaine lettuce', 'spinach', 'radishes', 'tuscan kale'")