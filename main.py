from pathlib import Path
import torch
from transformers import BertForSequenceClassification, BertTokenizer

from transformers import AdamW

from text_dataset import SentimentAnalysisDataset
from utils import compute_metrics

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# device = 'cpu'
model = BertForSequenceClassification.from_pretrained("DeepPavlov/rubert-base-cased", num_labels=3)
model.train().to(device)
epochs = 100
optimizer = AdamW(model.parameters(), lr=1e-3)
train_data = torch.utils.data.DataLoader(SentimentAnalysisDataset(Path('train.csv')), batch_size=32, shuffle=True)
test_data = torch.utils.data.DataLoader(SentimentAnalysisDataset(Path('test.csv')), batch_size=16)

tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')

model.resize_token_embeddings(len(tokenizer))
best_f1 = 0.0
for epoch in range(epochs):
    train_loss = 0.0
    for current_step, batch in enumerate(train_data):
        texts, labels = batch['text'], batch['label'].T.to(device).unsqueeze(0).long()
        inputs = tokenizer(texts, return_tensors='pt', padding=True).to(device)
        outputs = model(**inputs, labels=labels)
        loss = outputs[0]
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if current_step % 100 == 1:
            print(f'Epoch {epoch}, step {current_step} loss: {train_loss / current_step}')

    with torch.no_grad():
        model.eval()
        f1 = accuracy = precision = recall = 0.0
        for current_step, batch in enumerate(test_data):
            texts, labels = batch['text'], batch['label'].T.to(device).unsqueeze(0).long()
            inputs = tokenizer(texts, return_tensors='pt', padding=True).to(device)
            outputs = model(**inputs, labels=labels)
            loss, logits = outputs[:2]
            metrics = compute_metrics(logits.softmax(1), labels)
            f1 += metrics['f1']
            accuracy += metrics['accuracy']
            precision += metrics['precision']
            recall += metrics['recall']
        print(
            f'Test for epoch {epoch}/{epochs}. Avg accuracy: {accuracy / current_step}, f1: {f1 / current_step}, '
            f'precision: {precision / current_step}, recall: {recall / current_step}')
        if f1 / current_step > best_f1:
            best_f1 = f1 / current_step
            torch.save(model.state_dict(), 'res.pt')
