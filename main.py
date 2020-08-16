from collections import defaultdict
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from transformers import AdamW

from model import SentimentClassifier
from dataset import create_data_loader
from utils import train_epoch, eval_model, get_predictions, PRETRAINED_MODEL_NAME
import pandas as pd

tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_train, df_val = train_test_split(df_train, test_size=0.25)

BATCH_SIZE = 8
MAX_LEN = 512

train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SentimentClassifier(n_classes=3)
model = model.to(device)

EPOCHS = 10

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)

history = defaultdict(list)
best_accuracy = 0

for epoch in range(EPOCHS):

    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(df_train)
    )

    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(
        model,
        val_data_loader,
        loss_fn,
        device,
        len(df_val)
    )

    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)

    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'best_model_state.bin')
        best_accuracy = val_acc

test_acc, _ = eval_model(
    model,
    test_data_loader,
    loss_fn,
    device,
    len(df_test)
)

print(test_acc.item())

y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
    model,
    test_data_loader
)

print(classification_report(y_test, y_pred, target_names=[0, 1, 2]))