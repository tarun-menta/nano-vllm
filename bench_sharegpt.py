import argparse
import os
import time
from random import Random
import json
from tqdm import tqdm

from nanovllm import LLM, SamplingParams


# Download ShareGPT dataset from https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
# hf download anon8231489123/ShareGPT_Vicuna_unfiltered ShareGPT_V3_unfiltered_cleaned_split.json --local-dir ~/huggingface/sharegpt --repo-type dataset
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark with ShareGPT conversations"
    )
    parser.add_argument("model", type=str, help="Path to model weights")
    parser.add_argument("dataset_path", type=str, help="Path to dataset")
    parser.add_argument(
        "--num-prompts", type=int, default=256, help="Number of conversations to sample"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048, help="Max output tokens per request"
    )
    parser.add_argument(
        "--max-model-len", type=int, default=4096, help="Max model context length"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6, help="Sampling temperature"
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1, help="Tensor parallel size"
    )
    parser.add_argument(
        "--num-turns", type=int, default=1, help="Number of turns to sample"
    )
    args = parser.parse_args()

    assert args.num_turns % 2 == 1, "Number of turns must be odd"

    # Load ShareGPT dataset
    print("Loading ShareGPT dataset...")
    with open(args.dataset_path, "r") as f:
        sharegpt_conversations = json.load(f)
    # Some ShareGPT conversations start with a gpt response, for some reason
    sharegpt_conversations = [
        conv
        for conv in sharegpt_conversations
        if len(conv["conversations"]) > 0
        and conv["conversations"][0]["from"] == "human"
    ]

    # Extract first human turn from each conversation
    sampled_conversations = []
    Random(42).shuffle(sharegpt_conversations)
    for conversation in tqdm(
        sharegpt_conversations[: args.num_prompts], desc="Extracting prompts..."
    ):
        messages = conversation["conversations"]
        for message in messages:
            message["content"] = message.pop("value")
            role = message.pop("from")
            message["role"] = "user" if role == "human" else "assistant"
        sampled_conversations.append(messages[: args.num_turns])
    print(
        f"Extracted {len(sampled_conversations)} chat conversations from {len(sharegpt_conversations)} conversations"
    )

    # Initialize LLM
    model_path = os.path.expanduser(args.model)
    llm = LLM(
        model_path,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=True,
    )

    # Apply chat template
    prompts = [
        llm.tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True
        )
        for conv in sampled_conversations
    ]
    

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        ignore_eos=False,
    )

    # Warmup
    print("Warming up...")
    llm.generate(["Benchmark warmup."], SamplingParams())

    # Benchmark
    print(f"Running benchmark with {args.num_prompts} prompts...")
    t = time.time()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    elapsed = time.time() - t

    # Compute metrics
    total_input_tokens = sum(len(llm.tokenizer.encode(p)) for p in prompts)
    total_output_tokens = sum(len(o["token_ids"]) for o in outputs)
    mean_output_len = total_output_tokens / len(outputs)

    print(f"\n{'=' * 50}")
    print(f"ShareGPT Benchmark Results")
    print(f"{'=' * 50}")
    print(f"Num prompts:             {args.num_prompts}")
    print(f"Total time:              {elapsed:.2f}s")
    print(f"Total input tokens:      {total_input_tokens}")
    print(f"Total output tokens:     {total_output_tokens}")
    print(f"Output throughput:       {total_output_tokens / elapsed:.2f} tok/s")
    print(
        f"Total throughput (I+O):  {(total_input_tokens + total_output_tokens) / elapsed:.2f} tok/s"
    )
    print(f"Mean output length:      {mean_output_len:.1f} tokens")


if __name__ == "__main__":
    main()
