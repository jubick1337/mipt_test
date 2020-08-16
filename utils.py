import re
import torch
import torch.nn.functional as F
import numpy as np
from emot.emo_unicode import UNICODE_EMO

PRETRAINED_MODEL_NAME = 'DeepPavlov/rubert-base-cased-conversational'


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()

    losses = []
    correct_predictions = 0

    for data in data_loader:
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        targets = data["targets"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, predictions = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(predictions == targets)
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def get_predictions(model, data_loader):
    model = model.eval()
    device = next(model.parameters()).device
    texts = []
    predictions = []
    prediction_probabilities = []
    real_values = []

    with torch.no_grad():
        for data in data_loader:
            texts = data["review_text"]
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            targets = data["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, predictions = torch.max(outputs, dim=1)

            probabilities = F.softmax(outputs, dim=1)

            texts.extend(texts)
            predictions.extend(predictions)
            prediction_probabilities.extend(probabilities)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probabilities = torch.stack(prediction_probabilities).cpu()
    real_values = torch.stack(real_values).cpu()
    return texts, predictions, prediction_probabilities, real_values


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for data in data_loader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            targets = data["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


def normalize_whitespaces(text: str) -> str:
    return ' '.join(text.split()).strip()


def remove_urls(text: str) -> str:
    return re.sub(r'https?.\/\/\S*', '', text, flags=re.MULTILINE)


def convert_emojis(text: str) -> str:
    for emoji in UNICODE_EMO:
        text = text.replace(emoji, '_'.join(UNICODE_EMO[emoji].replace(',', '').replace(':', '').split()) + ' ')
    return text
