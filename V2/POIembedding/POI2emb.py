# build_poi_semantic_vectors.py
import pandas as pd
import numpy as np
import pickle
import ast
import argparse
import os

def normalize_vector(x):
    norm = np.linalg.norm(x)
    return x / (norm + 1e-8)

def parse_time_dict(x):
    if pd.isna(x) or x == '' or x == '{}':
        return {}
    try:
        return ast.literal_eval(x)
    except:
        return {}

def extract_time_features(time_dict):
    vec = np.zeros(24)
    for h in range(24):
        vec[h] = time_dict.get(h, 0)
    
    if vec.sum() == 0:
        return np.zeros(27)  # 24 + 3
    
    hist = vec / vec.sum()
    hours = np.arange(24)
    mean_hour = (hours * hist).sum()
    variance = ((hours - mean_hour) ** 2 * hist).sum()
    peak_hour = hours[np.argmax(hist)]
    
    return np.concatenate([
        hist,
        [mean_hour / 24.0,          
         variance / (24.0**2),      
         peak_hour / 24.0]          
    ])

def extract_time_features2(time_dict):

    if not time_dict:
        return np.zeros(12)
    
    hours = np.array(list(time_dict.keys()))
    counts = np.array(list(time_dict.values()))
    total = counts.sum()
    
    if total == 0:
        return np.zeros(12)
    

    weights = counts / total  # shape: (n_hours,)


    fourier_features = []
    max_freq = 6 
    for k in range(1, max_freq + 1):
        sin_val = np.sum(weights * np.sin(2 * np.pi * k * hours / 24.0))
        cos_val = np.sum(weights * np.cos(2 * np.pi * k * hours / 24.0))
        fourier_features.extend([sin_val, cos_val])
    fourier_features = np.array(fourier_features)  # shape: (12,)

    # sorted_items = sorted(time_dict.items(), key=lambda item: item[1], reverse=True)
    # if len(sorted_items) >= 2:
    #     top1_count = sorted_items[0][1]
    #     top2_count = sorted_items[1][1]
    #     avg_count = total / 24.0
    #     multi_peak = 1.0 if (top2_count > 1.2 * avg_count) else 0.0
    # else:
    #     multi_peak = 0.0
    # fourier_features = np.concatenate([fourier_features, [multi_peak]])    

    return fourier_features


def latlon_to_3d(lat, lon):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    return np.array([
        np.cos(lat_rad) * np.cos(lon_rad),
        np.cos(lat_rad) * np.sin(lon_rad),
        np.sin(lat_rad)
    ])

def main(csv_path, category_pkl, cf_pkl, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading POI data...")
    df = pd.read_csv(csv_path, on_bad_lines='skip', encoding='utf-8')
    
    print("Loading pre-computed category embeddings...")
    with open(category_pkl, 'rb') as f:
        category_to_embedding = pickle.load(f)
    
    sample_cat_vec = next(iter(category_to_embedding.values()))
    cat_dim = len(sample_cat_vec)
    print(f"Category vector dimension: {cat_dim}")
    
    # print("Loading CF embeddings...")
    # with open(cf_pkl, 'rb') as f:
    #     cf_embeddings = pickle.load(f)
    
    # sample_cf_vec = next(iter(cf_embeddings.values()))
    # cf_dim = len(sample_cf_vec)
    # print(f"CF vector dimension: {cf_dim}")
    
    full_vectors = []
    valid_pids = []

    for _, row in df.iterrows():
        try:
            # CF 
            # cf_vec = cf_embeddings.get(row['pid'], np.zeros(cf_dim))
            # cf_vec = normalize_vector(cf_vec)
           
            # Category 
            cat_vec = category_to_embedding.get(row['category'], np.zeros(cat_dim))
            # cat_vec = normalize_vector(cat_vec)
            
            spatial_vec = latlon_to_3d(row['latitude'], row['longitude'])
            # spatial_vec = normalize_vector(spatial_vec)
            
            # Time
            time_dict = parse_time_dict(row['visit_time_and_count'])
            time_vec = extract_time_features2(time_dict)
            # time_vec = normalize_vector(time_vec)
            
            # concatenated = np.concatenate([cf_vec, cat_vec, spatial_vec, time_vec])
            concatenated = np.concatenate([cat_vec, spatial_vec, time_vec])
            
            # full_vec = normalize_vector(concatenated)
            full_vec = concatenated  
            
            full_vectors.append(full_vec)
            valid_pids.append(row['pid'])
            
        except Exception as e:
            print(f"Error processing pid {row['pid']}: {e}")
            continue
    
    full_vectors = np.array(full_vectors)
    print(f"Generated vectors for {len(full_vectors)} POIs")
    print(f"Total vector dimension: {full_vectors.shape[1]}")
    
    poi_vector_dict = dict(zip(valid_pids, full_vectors))
    output_path = os.path.join(output_dir, "poi_Emb_dict.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(poi_vector_dict, f)

    # csv_output_path = os.path.join(output_dir, "poi_embeddings.csv")
    # df_out = pd.DataFrame({
    #     'pid': valid_pids,
    #     'embedding': [vec.tolist() for vec in full_vectors]
    # })
    # df_out.to_csv(csv_output_path, index=False)
    
    print(f"Done! Results saved to {output_path}")

if __name__ == "__main__":
    datafold = ''  
    path = f""
    if not os.path.exists(path):
        os.makedirs(path)
    parser = argparse.ArgumentParser(description="Build POI semantic vectors for similarity-based RQ-VAE")
    parser.add_argument("--csv_path", default=f"", help="Path to POI CSV file")
    parser.add_argument("--category_pkl", default=f"", help="Path to category_to_embedding.pkl")
    parser.add_argument("--cf_pkl", default=f"", help="Path to CF_embedding.pkl")
    parser.add_argument("--output_dir", default=f"", help="Output directory")

    args = parser.parse_args()
    main(args.csv_path, args.category_pkl, args.cf_pkl, args.output_dir)