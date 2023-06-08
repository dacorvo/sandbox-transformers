import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from timeit import default_timer as timer


def text_generation(model_id, instruction, load_in_8bit):

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        load_in_8bit=load_in_8bit
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    start = timer()
    sequences = pipeline(
        instruction,
        max_length=200,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    print(f"Generation took: {timer() - start} s.")
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transformers test script')
    parser.add_argument('--bitwidth', type=int, default=32, help='weights and activations bitwidth')
    parser.add_argument('--test-tgen', action='store_true', help='test text generation')
    args = parser.parse_args()

    if args.test_tgen:
        model = "tiiuae/falcon-7b-instruct"
        instruction = "Write a poem about Valencia"
        text_generation(model, instruction, args.bitwidth == 8)