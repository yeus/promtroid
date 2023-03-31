#!/usr/bin/env python

# use the program like this:  ./pd.py gpt2 "Hello, my name is"

import argparse
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelWithHeads


def generate_text(model, tokenizer, prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids=input_ids, max_length=50, do_sample=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generates text using a GPT model')
    parser.add_argument('model', type=str, help='Hugging Face model name or path to local model directory')
    args, prompt = parser.parse_known_args(sys.argv[1:])
    prompt = " ".join(prompt) # join multiple words..

    if args.model == "gpt4all":
        from nomic.gpt4all import GPT4All

        m = GPT4All()
        m.open()
        print(m.prompt(prompt))
    else:
        # try: #check if it is a "normal" model, first
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model)
        # except:
        # model = AutoModelWithHeads.from_pretrained(args.model)
        # adapter_name = model.load_adapter(args.model, source="hf")
        # model.active_adapters = adapter_name

        print(generate_text(model, tokenizer, prompt))
