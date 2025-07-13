import streamlit as st
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import os

st.title("Fake News Generator & Detector")

# Generator
st.header("Generate Fake News")
prompt = st.text_input("Prompt:", "Breaking news:")
if st.button("Generate"):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(input_ids, max_length=50, num_return_sequences=3)
    for i, output in enumerate(outputs):
        st.write(f"{i+1}: {tokenizer.decode(output, skip_special_tokens=True)}")

# Detector
st.header("Detect Fake News")
headline = st.text_input("Headline to check:")
if st.button("Detect"):
    if os.path.exists("detector/bert_model"):
        classifier = pipeline('text-classification', model="detector/bert_model")
        result = classifier(headline)
        st.write(result)
    else:
        st.warning("BERT model not found. Please train the detector first.")

# Dataset Visualization
st.header("Dataset Visualization")
data_path = "data/liar_train.csv"
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    st.write(df['label'].value_counts())
    st.bar_chart(df['label'].value_counts())
else:
    st.info("Dataset not found. Please add liar_train.csv to the data folder.") 