from pathlib import Path
import torch
from transformers import BertForSequenceClassification, BertTokenizer
import numpy as np
from transformers import AdamW

from text_dataset import SentimentAnalysisDataset
from utils import compute_metrics

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# device = 'cpu'
model = BertForSequenceClassification.from_pretrained('DeepPavlov/rubert-base-cased-conversational', num_labels=3, )
model.to(device)
epochs = 20
optimizer = AdamW(model.parameters(), lr=1e-3)
train_data = torch.utils.data.DataLoader(SentimentAnalysisDataset(Path('train.csv')), batch_size=16, shuffle=True)
test_data = torch.utils.data.DataLoader(SentimentAnalysisDataset(Path('test.csv')), batch_size=8)

tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased-conversational', model_max_length=512)

model.resize_token_embeddings(len(tokenizer))
best_f1 = 0.0
for epoch in range(epochs):
    train_loss = 0.0
    model.train()
    for current_step, batch in enumerate(train_data):
        texts, labels = batch['text'], batch['label'].T.to(device).unsqueeze(0).long()
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(device)
        outputs = model(**inputs, labels=labels)
        loss = outputs[0]
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if current_step % 100 == 1:
            print(f'Epoch {epoch}, step {current_step} loss: {train_loss / current_step}')

    with torch.no_grad():
        model.eval()
        accuracy = 0.0
        neg_f1 = neg_precision = neg_recall = 0.0
        neut_f1 = neut_precision = neut_recall = 0.0
        pos_f1 = pos_precision = pos_recall = 0.0
        total_steps = 0
        for current_step, batch in enumerate(test_data):
            texts, labels = batch['text'], batch['label'].T.to(device).unsqueeze(0).long()
            inputs = tokenizer(texts, return_tensors='pt', padding=True).to(device)
            outputs = model(**inputs, labels=labels)
            loss, logits = outputs[:2]
            metrics = compute_metrics(logits.softmax(1), labels)
            neg_f1 += metrics['neg_f1']
            neg_precision += metrics['neg_precision']
            neg_recall += metrics['neg_recall']
            neut_f1 += metrics['neut_f1']
            neut_precision += metrics['neut_precision']
            neut_recall += metrics['neut_recall']
            pos_f1 += metrics['pos_f1']
            pos_precision += metrics['pos_precision']
            pos_recall += metrics['pos_recall']
            accuracy += metrics['accuracy']
            total_steps += 1
        print(f'Test for epoch {epoch}/{epochs}. Accuracy: {accuracy / total_steps}')

        print(
            f'neg_f1: {neg_f1 / total_steps}, neg_precision: {neg_precision / total_steps}, neg_recall: {neg_recall / total_steps}')
        print(
            f'neut_f1: {neut_f1 / total_steps}, neut_precision: {neut_precision / total_steps}, neut_recall: {neut_recall / total_steps}')
        print(
            f'pos_f1: {pos_f1 / total_steps}, pos_precision: {pos_precision / total_steps}, pos_recall: {pos_recall / total_steps}')

        mean_f1 = np.mean([neg_f1, neut_f1, pos_f1])
        if mean_f1 / total_steps > best_f1:
            best_f1 = mean_f1 / current_step
            torch.save(model.state_dict(), 'res.pt')
