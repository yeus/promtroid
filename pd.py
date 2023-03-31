# following code was produced using the following prompt to
# chatgpt-4:
#
# write a small python linux cli script which give generic access to huggingface transformers gpt models

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_text(model, tokenizer, prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids=input_ids, max_length=50, do_sample=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generates text using a GPT model')
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Hugging Face model name or path to local model directory')
    parser.add_argument('--prompt', '-p', type=str, required=True,
                        help='Prompt to generate text from')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    print(generate_text(model, tokenizer, args.prompt))
