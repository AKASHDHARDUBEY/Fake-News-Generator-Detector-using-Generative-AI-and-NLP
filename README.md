# 🚨 Fake News Generator & Detector using Generative AI and NLP

## 🔍 Description
This project leverages the capabilities of Generative AI and Natural Language Processing (NLP) to both create and detect fake news, showcasing the dual-edged nature of modern AI technologies.

## 🧠 Key Components
- **Fake News Generator**: Built using GPT-2, capable of producing realistic yet fabricated news headlines or short articles.
- **Fake News Detector**: Utilizes a fine-tuned BERT model, trained on labeled datasets (e.g., LIAR dataset or FakeNewsNet) to classify news as real or fake.

## 🎯 Objective
To explore the ethical implications and technological solutions surrounding the rise of AI-generated misinformation by:
- Simulating the generation of deceptive content.
- Providing tools to detect and counteract it.

## 🧰 Tech Stack
- Python
- TensorFlow / PyTorch
- Hugging Face Transformers (gpt2, bert-base-uncased)
- Streamlit / Flask (for web UI)
- Pandas / Numpy (for data handling)

## 💡 Features
- Real-time fake news headline generation.
- Text classification with confidence score (Real or Fake).
- Interactive interface to test custom inputs.
- Dataset visualization and model performance metrics.

## 📁 Folder Structure
```
fake-news-ai/
├── generator/
│   ├── generate.py
│   └── generate_gpt2.py
├── detector/
│   ├── detect.py
│   ├── train_bert.py
│   └── bert_model/
├── app.py  # Streamlit UI
├── data/
│   ├── liar_train.csv
│   └── liar_test.csv
├── README.md
├── requirements.txt
```

## 🚀 Getting Started
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Download and place datasets** in the `data/` folder (e.g., `liar_train.csv`, `liar_test.csv`).
3. **Train the BERT detector:**
   ```bash
   python detector/train_bert.py
   ```
4. **Generate fake news headlines:**
   ```bash
   python generator/generate_gpt2.py
   ```
5. **Detect fake news:**
   ```bash
   python detector/detect.py
   ```
6. **Run the web app:**
   ```bash
   streamlit run app.py
   ```

## ⚖️ Impact
This project educates users about:
- The dangers of AI misuse in spreading misinformation.
- The importance of developing robust detectors.
- The need for responsible AI development and deployment.

## 📊 (Optional) Dataset Visualization
Add code in `app.py` or a notebook to visualize dataset stats and model performance.

## 📝 Ethical Disclaimer
This project is for educational and research purposes only. Generative AI can be misused to create and spread misinformation. Always use such technologies responsibly and ethically. 