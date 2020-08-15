import re
import torch
from emot.emo_unicode import UNICODE_EMO
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def compute_metrics(predicted, ground_truth):
    predicted = predicted.argmax(-1)
    ground_truth = torch.flatten(ground_truth)
    precision, recall, f1, _ = precision_recall_fscore_support(ground_truth, predicted,
                                                               labels=[0, 1, 2])
    acc = accuracy_score(ground_truth, predicted)
    return {
        'accuracy': acc,
        'neg_f1': f1[0],
        'neg_precision': precision[0],
        'neg_recall': recall[0],
        'neut_f1': f1[1],
        'neut_precision': precision[1],
        'neut_recall': recall[1],
        'pos_f1': f1[2],
        'pos_precision': precision[2],
        'pos_recall': recall[2],
    }


def normalize_whitespaces(text):
    return ' '.join(text.split()).strip()


def remove_urls(text):
    return re.sub(r'https?.\/\/\S*', '', text, flags=re.MULTILINE)


def convert_emojis(text):
    for emoji in UNICODE_EMO:
        text = text.replace(emoji, '_'.join(UNICODE_EMO[emoji].replace(',', '').replace(':', '').split()) + ' ')
    return text
