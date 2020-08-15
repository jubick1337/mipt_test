from pathlib import Path
import pandas as pd
import torch

from utils import convert_emojis, remove_urls, normalize_whitespaces


class SentimentAnalysisDataset(torch.utils.data.Dataset):

    def __init__(self, data_path: Path):
        self._df = pd.read_csv(data_path)

    def __getitem__(self, idx):
        sample = self._df.iloc[idx]
        text = sample.text
        label = sample.label + 1
        text = remove_urls(text)
        text = convert_emojis(text)
        text = normalize_whitespaces(text)
        return {'text': text, 'label': torch.Tensor([label])}

    def __len__(self):
        return len(self._df)
