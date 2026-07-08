import time
import asyncio
from vllm import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs

async def get_token_ids(engine, text: str):
    """Helper to get token IDs from the vLLM engine's tokenizer."""
    tokenizer = engine.get_tokenizer()
    return tokenizer.encode(text, add_special_tokens=False)

async def generate_text(engine, prompt: str, logit_bias: dict = None):
    """Submits a request to the vLLM engine with specific sampling params."""
    sampling_params = SamplingParams(
        max_tokens=5,
        # temperature=0.0,  # Greedy to ensure deterministic choice after bias
        logit_bias=logit_bias or {}
    )
    
    request_id = f"req_{hash(prompt + str(logit_bias))}"
    results_generator = engine.generate(prompt, sampling_params, request_id)
    
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
        
    # Extract text and token IDs chosen by the model
    output_text = final_output.outputs[0].text
    output_token_ids = final_output.outputs[0].token_ids
    return output_text, output_token_ids

async def main(engine):
    # 2. Get the Token IDs for our target variations
    # Variation A: The full, standard single token
    EXPECTED_PREFIX = "learned"
    standard_tokens = await get_token_ids(engine, EXPECTED_PREFIX)    
    print(f"Target word '{EXPECTED_PREFIX}' tokenized naturally: {standard_tokens}")
    
    prompt = "Past tense of learn is "
    
    # --- Runs ---
    logit_bias = {}
    for i in range(200):
        text_split, tokens_split = await generate_text(engine, prompt, logit_bias=logit_bias)
        
        print(f"\n--- Run {i}: Forced Split ---")
        print(f"Output text: {text_split}")
        print(f"Output token IDs: {tokens_split}")

        logit_bias[tokens_split[0]] = -100.0 # Ban the token

if __name__ == "__main__":
    # 1. Initialize the AsyncLLMEngine
    engine_args = AsyncEngineArgs(model="huggyllama/llama-13b")
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    asyncio.run(main(engine))

    del engine
    time.sleep(3)
