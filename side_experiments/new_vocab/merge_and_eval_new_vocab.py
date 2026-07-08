import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "huggyllama/llama-7b"
NUM_OUTPUT_TOKENS = 10

# TOKENS = [" spaces", "hip"]
# TOKENS = [" space", "ship"]
# TEST_SENTENCE = "The spaceship accelerated toward the distant" # planet, star

# TOKENS = [" water", " meter"]
# TEST_SENTENCE = "I used the water meter to measure the amount of"

# TOKENS = [" picture", " frame"]
# TEST_SENTENCE = "I hung the picture frame on the" # wall

# TOKENS = [" tooth", "brush"]
# TEST_SENTENCE = "The toothbrush brushes my"

# ' chairs' is a bad example
# TOKENS = [" chair", "s"]
# TEST_SENTENCE = "I have multiple chairs on the" # ground
# TEST_SENTENCE = "I use chairs to" # sit

# ' cups' seems like a bad example too
# TOKENS = [" cup", "s"]
# TEST_SENTENCE = "I use cups to" # drink

# ' transgender'
# TOKENS = [" trans", "gender"]
# TEST_SENTENCE = "A transgender person is someone who"

# Reverse ` transgender`. Probably doesn't work since the LLM is trained to generate parts of the word sequentially.
# TOKENS = [" trans", "gender"]
# TEST_SENTENCE = "Someone who does not identify with the gender they were assigned at is called a"

# ' blocked'
# TOKENS = [" block", "ed"]
# TEST_SENTENCE = "This object blocked my"

# Setup: Load a base model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
FULL_WORD = "".join(TOKENS)

tokenizer_a = AutoTokenizer.from_pretrained(MODEL_NAME)
model_a = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

tokenizer_b = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer_b.add_tokens(FULL_WORD)
model_b = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
model_b.resize_token_embeddings(len(tokenizer_b))

print(f"IDs for '{FULL_WORD}': a-{tokenizer_a.encode(FULL_WORD)}, b-{tokenizer_b.encode(FULL_WORD)}")

# Ensure the model is on GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Initialize weights for the new token
new_id = tokenizer_b.convert_tokens_to_ids("".join(TOKENS))
a_id = tokenizer_a.convert_tokens_to_ids(TOKENS[0])
b_id = tokenizer_a.convert_tokens_to_ids(TOKENS[1])

with torch.no_grad():
    emb_b = model_b.get_input_embeddings().weight
    emb_b[new_id] = (emb_b[a_id] + emb_b[b_id]) / 2
    if hasattr(model_b, 'lm_head'):
        model_b.lm_head.weight[new_id] = (model_b.lm_head.weight[a_id] + model_b.lm_head.weight[b_id]) / 2

# Define the Validation Function
vocab_size_a = len(tokenizer_a)

def get_next_token_probs(model, tokenizer, sequence, num_output_tokens = 1, top_k = 3): 
    for i in range(num_output_tokens):
        inputs = tokenizer(sequence, return_tensors="pt").input_ids.to(device)
        print(f"  Inputs: {inputs.tolist()}")
        with torch.no_grad():
            outputs = model(inputs)
            # Apply softmax to the last position logits
            logits = outputs.logits[0, -1, :]
            probs = F.softmax(logits, dim=-1)
            # Get output greedily
            top_probs, top_indices = torch.topk(probs, top_k)
            print(f"    Output: ", end="")
            for i in range(top_k):
                token_id = top_indices[i].item()
                token_prob = top_probs[i].item()
                token_str = tokenizer.convert_ids_to_tokens([token_id])[0]
                print(f"'{token_str}' ({token_id}, {token_prob*100:.2f}%), ", end="")
            print("")
            sequence += tokenizer.convert_ids_to_tokens([top_indices[0]])[0]
    return probs[:vocab_size_a], sequence

# Run the Test
print("="*40)
probs_a, output_a = get_next_token_probs(model_a, tokenizer_a, TEST_SENTENCE, NUM_OUTPUT_TOKENS)
print(f"Test Sentence A tokens: {tokenizer_a.tokenize(TEST_SENTENCE)}\nText A: {output_a}")
print("="*40)
probs_b, output_b = get_next_token_probs(model_b, tokenizer_b, TEST_SENTENCE, NUM_OUTPUT_TOKENS)
print(f"Test Sentence B tokens: {tokenizer_b.tokenize(TEST_SENTENCE)}\nText B: {output_b}")
print("="*40)

# Compute KL Divergence
# KL Div requires log-probs for the first input
kl_div = F.kl_div(probs_b.log(), probs_a, reduction='batchmean')
print(f"KL Divergence: {kl_div.item():.6f}")

# Threshold check
if kl_div.item() < 0.1:
    print("Success: Distributions are well-aligned.")
else:
    print("Warning: High divergence. Model may need fine-tuning to map the new token.")
