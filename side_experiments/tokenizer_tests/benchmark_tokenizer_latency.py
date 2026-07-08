import time
import numpy as np
import os
from pathlib import Path
import csv
from vllm.transformers_utils.tokenizer import get_tokenizer

CURRENT_DIR = Path(__file__).resolve().parent

def benchmark_vllm_tokenizer(model_name: str, dataset: list[str], warmup_runs: int = 2):
    print(f"Loading tokenizer for model: {model_name}...")
    # Fetch the vLLM tokenizer wrapper
    tokenizer = get_tokenizer(model_name)

    # Save vocabulary
    vocab_file = Path(f"{CURRENT_DIR}/llm_vocabularies/{model_name}_vocab.csv")
    os.makedirs(vocab_file.parent, exist_ok=True)
    with open(vocab_file, mode="w", newline="",) as f:
        writer = csv.writer(f)
        for string, token in sorted(tokenizer.get_vocab().items(), key=lambda x: x[1]):
            writer.writerow([token, string])
    
    # Warmup to ensure caching or lazy loading doesn't skew results
    print("Warming up...")
    for _ in range(warmup_runs):
        for text in dataset[:10]:  # Use a small subset for warmup
            _ = tokenizer.encode(text)
            
    print(f"Benchmarking tokenization on {len(dataset)} strings...")
    
    # Track individual latencies
    latencies = []
    total_tokens = 0
    
    # Benchmark loop
    start_total = time.perf_counter()
    for text in dataset:
        start_single = time.perf_counter()
        tokens = tokenizer.encode(text)
        end_single = time.perf_counter()
        
        latencies.append(end_single - start_single)
        total_tokens += len(tokens)
        print(f"{tokens=}")
    end_total = time.perf_counter()
    
    # Calculate metrics
    total_time = end_total - start_total
    avg_latency_ms = np.mean(latencies) * 1000
    p95_latency_ms = np.percentile(latencies, 95) * 1000
    throughput_tokens_per_sec = total_tokens / total_time
    
    # Display Results
    print("\n" + "="*40)
    print("BENCHMARK RESULTS")
    print("="*40)
    print(f"Total Strings Processed : {len(dataset)}")
    print(f"Total Tokens Generated  : {total_tokens}")
    print(f"Total Execution Time    : {total_time:.4f} seconds")
    print(f"Average Latency/String  : {avg_latency_ms:.4f} ms")
    print(f"P95 Latency/String      : {p95_latency_ms:.4f} ms")
    print(f"Tokenizer Throughput    : {throughput_tokens_per_sec:.2f} tokens/sec")
    print("="*40)

if __name__ == "__main__":
    MODEL = "huggyllama/llama-7b"
    
    SAMPLE_DATASET = [
        "Hello, how are you today?",
        "What is the capital of France?",
        "Write a Python function to reverse a string and explain how it works line by line.",
        "Artificial intelligence is transforming industries ranging from healthcare to finance by automating complex tasks and analyzing massive datasets at scale.",
        "Short phrase."
    ] * 200
    
    benchmark_vllm_tokenizer(model_name=MODEL, dataset=SAMPLE_DATASET)
