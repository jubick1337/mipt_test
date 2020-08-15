import re
from emot.emo_unicode import UNICODE_EMO
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def compute_metrics(predicted, ground_truth):
    predicted = predicted.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(ground_truth, predicted, average='binary')
    acc = accuracy_score(ground_truth, predicted)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def normalize_whitespaces(text):
    return ' '.join(text.split()).strip()


def remove_urls(text):
    return re.sub(r'https?.\/\/\S*', '', text, flags=re.MULTILINE)


def convert_emojis(text):
    for emot in UNICODE_EMO:
        text = text.replace(emot, "_".join(UNICODE_EMO[emot].replace(",", "").replace(":", "").split()) + ' ')
    return text
