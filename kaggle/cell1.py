import pandas as pd
import os

def clean_recipe_data_on_kaggle():
    # 1. Define Kaggle-specific paths based on your screenshot
    input_file = '/kaggle/input/datasets/shuyangli94/food-com-recipes-and-user-interactions/RAW_recipes.csv'
    output_file = '/kaggle/working/cleaned_recipes.csv'
    
    print(f"--- Loading data from: {input_file} ---")
    
    # Check if file exists to avoid errors
    if not os.path.exists(input_file):
        print("Error: File not found. Double check the path!")
        return

    # 2. Load the dataset
    df = pd.read_csv(input_file)

    # 3. Apply Cleaning Logic
    initial_count = len(df)
    
    # Remove rows with missing essential info
    df = df.dropna(subset=['name', 'ingredients', 'steps'])
    
    # Text Normalization
    df['name'] = df['name'].str.lower()
    
    # Filter for Quality (3+ ingredients and name > 3 chars)
    df = df[df['ingredients'].str.count(',') >= 2] 
    df = df[df['name'].str.len() > 3]

    # Deduplication
    df = df.drop_duplicates(subset=['name'])

    # 4. Save to Kaggle's 'working' directory
    df.to_csv(output_file, index=False)
    
    print(f"Done! Cleaned {initial_count} recipes down to {len(df)} entries.")
    print(f"--- Saved cleaned data to: {output_file} ---")
    
    # Preview the first few rows
    return df.head()

# Execute the function in your Kaggle cell
clean_recipe_data_on_kaggle()
