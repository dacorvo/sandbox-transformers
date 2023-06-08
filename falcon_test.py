import argparse
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import transformers
import torch
from datasets import load_dataset
from evaluate import evaluator
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


def bench_zero_shot_classification(model_id, load_in_8bit, n_samples=10):

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        load_in_8bit=load_in_8bit
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    eval_dataset = load_dataset("hellaswag", split="validation").select(range(n_samples))
    task_evaluator = evaluator("text-classification")
    qa_pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer
    )
    metric = task_evaluator.compute(model_or_pipeline=qa_pipe, data=eval_dataset, metric="hellaswag")
    print(metric)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transformers test script')
    parser.add_argument('--bitwidth', type=int, default=32, help='weights and activations bitwidth')
    parser.add_argument('--test-tgen', action='store_true', help='test text generation')
    parser.add_argument('--bench-zsc', action='store_true', help='bench zero-shot classification')
    args = parser.parse_args()

    if args.test_tgen:
        model = "tiiuae/falcon-7b-instruct"
        instruction = "Write a poem about Valencia"
        text_generation(model, instruction, args.bitwidth == 8)
    if args.bench_zsc:
        model = "tiiuae/falcon-7b-instruct"
        n_samples = 10
        bench_zero_shot_classification(model, args.bitwidth == 8, n_samples)