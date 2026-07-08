# Tokenizer Tests

A set of scripts to experiment with vLLM tokenizers.

## Scripts and Notes
* `benchmark_tokenizer_latency.py`: Benchmark how long it takes to process and convert string input to tokens.
* `find_same_str_diff_token_output.py`: Script that attempts to achieve the same string output using different sequences of tokens.
    * Since models are trained on matching the longest possible string, then the model will most likely output the correct and longer version of the string, rather than breaking up the string into several tokens.
* `view_token_embedding_vectors.py`: Given groups of tokens (often representing the same string), we check whether the average value of the embeddings of multiple tokens match the embedding of the token that represents the entire string.
    * We find no 
