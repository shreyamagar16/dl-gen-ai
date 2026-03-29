import torch
from torch.utils.data import Dataset
from features import extract_mel

class AudioDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        mel = extract_mel(self.file_paths[idx])
        
        mel = torch.tensor(mel).unsqueeze(0).float()
        label = torch.tensor(self.labels[idx])
        
        return mel, label
