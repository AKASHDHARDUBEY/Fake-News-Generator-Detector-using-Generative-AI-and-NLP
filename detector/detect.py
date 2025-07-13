from transformers import pipeline

classifier = pipeline('text-classification', model='./bert_model')
headline = input("Enter a news headline: ")
result = classifier(headline)
print(result)

if __name__ == "__main__":
    main() 