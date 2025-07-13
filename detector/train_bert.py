from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load and preprocess dataset
# Adjust file paths and column names as needed for your dataset
train_file = '../data/liar_train.csv'
test_file = '../data/liar_test.csv'
dataset = load_dataset('csv', data_files={'train': train_file, 'test': test_file})
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Training
training_args = TrainingArguments(
    output_dir='./bert_model',
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
)

trainer.train()
trainer.save_model('./bert_model') 