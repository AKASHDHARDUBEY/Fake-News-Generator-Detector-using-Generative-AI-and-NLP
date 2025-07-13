from transformers import pipeline, set_seed

def main():
    generator = pipeline('text-generation', model='gpt2')
    set_seed(42)
    prompt = "Breaking News:"
    results = generator(prompt, max_length=30, num_return_sequences=5)
    print("Generated Fake News Headlines:")
    for i, result in enumerate(results):
        print(f"{i+1}: {result['generated_text']}")

if __name__ == "__main__":
    main() 