from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def generate_fake_news(prompt, max_length=50, num_return_sequences=5):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9
    )
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

if __name__ == "__main__":
    prompt = input("Enter a prompt for fake news: ")
    headlines = generate_fake_news(prompt)
    for i, headline in enumerate(headlines, 1):
        print(f"{i}: {headline}") 