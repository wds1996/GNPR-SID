# generate_poi_embeddings.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import argparse


def category2vec(csv_path, output_dir, model_name="all-MiniLM-L6-v2", n_components=None, category_column="category"):

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read CSV file
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path, on_bad_lines='skip', encoding='utf-8')
    
    # Extract unique categories
    categories = df[category_column].unique().tolist()
    print(f"Found {len(categories)} unique categories")
    
    # Load sentence transformer model
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = model.encode(categories, show_progress_bar=True)
    print(f"Original embedding shape: {embeddings.shape}")
    
    # Apply PCA if specified
    if n_components is not None and n_components < embeddings.shape[1]:
        print(f"Applying PCA to reduce dimensions to {n_components}")
        pca = PCA(n_components=n_components)
        embeddings_reduced = pca.fit_transform(embeddings)
        print(f"Reduced embedding shape: {embeddings_reduced.shape}")
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
        final_embeddings = embeddings_reduced
    else:
        final_embeddings = embeddings
    
    # Create mapping dictionary
    category_to_embedding = dict(zip(categories, final_embeddings))
    
    # Save results
    np.save(os.path.join(output_dir, "category_embeddings.npy"), final_embeddings)
    
    # Save mapping as dictionary (optional)
    import pickle
    with open(os.path.join(output_dir, "category_to_embedding.pkl"), 'wb') as f:
        pickle.dump(category_to_embedding, f)
    
    print(f"Results saved to: {output_dir}")
    print(f"- category_embeddings.npy: Embedding matrix")
    print(f"- categories.npy: Category names")
    print(f"- category_embeddings.csv: Human-readable embeddings")
    print(f"- category_to_embedding.pkl: Dictionary mapping")

if __name__ == "__main__":
    datafold = ''  
    path = f""

    if not os.path.exists(path):
        os.makedirs(path)
    parser = argparse.ArgumentParser(description="Generate POI category embeddings")
    parser.add_argument("--csv_path", default=f"", help="Path to input CSV file")
    parser.add_argument("--output_dir", default=f"", help="Output directory for results")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence transformer model name")
    parser.add_argument("--dim", type=int, default=64, help="Target dimensionality (PCA)")
    parser.add_argument("--column", default="category", help="Category column name")
    
    args = parser.parse_args()

    category2vec(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        model_name=args.model,
        n_components=args.dim,
        category_column=args.column
    )