from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

def generalized_distance(vec1, vec2, degree):
    """
    Computes the mean distance of a given degree.
    degree=1: Mean Absolute Error (MAE)
    degree=2: Root Mean Squared Error (RMSE)
    """
    # Calculate the mean of the differences raised to the power of degree
    mean_diff = torch.mean(torch.abs(vec1 - vec2) ** degree)
    
    # Take the degree-th root to return to the original scale
    return mean_diff ** (1 / degree)

def main(model_id):
    # 2. Load the model
    # We use torch.float16 and device_map="auto" to efficiently manage 
    # the ~26GB of memory required for a 13B parameter model.
    print("Loading model weights (this may take a moment)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    embedding_layer = model.get_input_embeddings()

    # 3. Define text and tokenize
    token_ids_lists = [
        [
            21891, # learning
        ],
        [
            19668, # learn
            292, # ing
        ],
        [
            1945, # lear
            1076, # ning
        ],
        [
            21094, # running
        ],
        [
            3389, # run
            1076, # ning
        ],
        [
            21891, # learning
        ],
        [
            21094, # running
        ],
        [
            19510, # _interfaces
        ],
        [
            19455, # вые
        ],
    ]

    def print_embedding_token_average(tokens):
        print("="*40)
        input_ids_tensor = torch.tensor([tokens], dtype=torch.long).to(model.device)

        # 4. Access the embedding layer and extract the vectors
        with torch.no_grad():
            token_embeddings = embedding_layer(input_ids_tensor)

        # 5. Display the data
        print(f"Input IDs Tensor shape: {input_ids_tensor.shape}")
        print(f"Resulting Embedding Tensor shape: {token_embeddings.shape}")

        # 6. Display the vector for a specific token (e.g., the first token)
        for i, t in enumerate(tokens):
            first_token_vector = token_embeddings[0, i, :]
            print(f"--- Embedding Vector for '{t}' --- {first_token_vector}")

        average_vector = token_embeddings.mean(dim=1).squeeze(0)
        print(f"\nAverage vector: {average_vector}\n")
        return average_vector
    
    prev_avg = None
    for tl in token_ids_lists:
        avg = print_embedding_token_average(tl)
        if prev_avg is not None:
            print(f"Similarity score (deg 1): {generalized_distance(prev_avg, avg, 1)}")
        prev_avg = avg


def rank_token_pairs(model_id, output_file='top_100_closest_tokens.csv', top_n=100):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    embedding_matrix = model.get_input_embeddings().weight

    # Ensure matrix is on GPU if available for speed
    device = embedding_matrix.device
    num_tokens = embedding_matrix.shape[0]
    
    # Calculate pairwise squared Euclidean distance: |x-y|^2 = |x|^2 + |y|^2 - 2<x,y>
    norms = torch.sum(embedding_matrix**2, dim=1)
    dot_product = torch.mm(embedding_matrix, embedding_matrix.t())
    dist_sq = norms.view(-1, 1) + norms.view(1, -1) - 2 * dot_product
    
    # Set diagonal (self-similarity) to infinity so they are not included
    dist_sq.fill_diagonal_(float('inf'))
    
    # We want the smallest distances, so use 'largest=False'
    # We retrieve 2 * top_n because the matrix is symmetric (i,j and j,i)
    flat_dist = dist_sq.view(-1)
    top_values, top_indices = torch.topk(flat_dist, k=top_n * 2, largest=False)
    
    results = []
    seen_pairs = set()
    
    for val, idx in zip(top_values, top_indices):
        i = (idx // num_tokens).item()
        j = (idx % num_tokens).item()
        
        # Ensure we only add unique pairs (i, j)
        pair = tuple(sorted((i, j)))
        if pair not in seen_pairs:
            results.append({
                'Token_ID_1': pair[0], 
                'Token_ID_2': pair[1], 
                'Distance': torch.sqrt(val).item()
            })
            seen_pairs.add(pair)
            
        if len(results) >= top_n:
            break
            
    # Export to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Top {top_n} token pairs saved to {output_file}")


if __name__ == "__main__":
    model_id = "huggyllama/llama-13b"
    main(model_id)
    rank_token_pairs(model_id)
