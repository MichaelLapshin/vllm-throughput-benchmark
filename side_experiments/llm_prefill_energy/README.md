# Purpose of this side-experiment

See how the energy/runtime performs with different prefill batch sizes.

## Current Results & Next Steps

Currently, the retuls show the best performance to occur at around a prefill-only batch size of about 32-requests.

The curve seems unusual, so perhaps try testing the following paramters: (***This unusual behaviour has been identified as a bug and was fixed.***)
* Set the batch size via vLLM, rather than for-looping the batches of requests to generate.
* Perhaps randomize token IDs for the requests, rather than making themm share them.
