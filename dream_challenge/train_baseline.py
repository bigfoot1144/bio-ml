import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

df = pl.read_csv('dream_train.csv')

# Define valid nucleotide mappings
nucleotide_mapping = {'T': 0, 'G': 1, 'C': 2, 'A': 3}

class SequenceDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.max_length = 142
        self.sequences = []
        self.expressions = []
        
        for row in tqdm(dataframe.iter_rows(named=True)):
            seq = row['sequence']
            expr = row['expression']
            
            one_hot = []
            # Process each character up to max_length
            for c in seq[:self.max_length]:
                if c in nucleotide_mapping:
                    vec = [0.0] * 4
                    vec[nucleotide_mapping[c]] = 1.0
                else:  # Handle 'N' or any invalid characters
                    vec = [0.0] * 4
                one_hot.append(vec)
            
            # Pad remaining slots with zero vectors
            while len(one_hot) < self.max_length:
                one_hot.append([0.0] * 4)
            
            # Convert to tensor and transpose to [4, 142]
            tensor_seq = torch.tensor(one_hot, dtype=torch.float32).permute(1, 0)
            self.sequences.append(tensor_seq)
            self.expressions.append(torch.tensor(expr, dtype=torch.float32))
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.expressions[idx]

dataset = SequenceDataset(df)