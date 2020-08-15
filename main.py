from pathlib import Path
from torch.nn import functional as F
import torch
from transformers import BertForSequenceClassification, BertTokenizer

from transformers import AdamW

from text_dataset import SentimentAnalysisDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
model = BertForSequenceClassification.from_pretrained("DeepPavlov/rubert-base-cased", num_labels=3)
model.train().to(device)
epochs = 10
optimizer = AdamW(model.parameters(), lr=1e-3)
train_data = torch.utils.data.DataLoader(SentimentAnalysisDataset(Path('train.csv')), batch_size=16, shuffle=True)
test_data = torch.utils.data.DataLoader(SentimentAnalysisDataset(Path('test.csv')), batch_size=4)

tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')

for epoch in range(epochs):
    train_loss = 0.0
    for current_step, batch in enumerate(train_data):
        texts, labels = batch['text'], batch['label'].T.to(device)
        tokens = tokenizer(texts, return_tensors='pt', padding=True)
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = F.cross_entropy(labels, outputs[0])
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if current_step % 100 == 0:
            print(f'Epoch {epochs}, step {current_step} loss: {train_loss / current_step}')

    with torch.no_grad():
        model.eval()
        for batch in test_data:
            texts, labels = batch['text'], batch['label'].T.to('device')
            tokens = tokenizer(texts, return_tensors='pt', padding=True)
            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = F.cross_entropy(labels, outputs[0])
