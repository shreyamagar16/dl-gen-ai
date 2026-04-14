import torch
from torch.utils.data import Dataset

from features import extract_mel


class AudioDataset(Dataset):
    """PyTorch ``Dataset`` of log-mel spectrograms built from audio file paths.

    Each item is a single-channel mel tensor and an integer genre label.
    An optional ``transforms`` callable can augment or modify the mel array
    after extraction and before it is converted to a tensor.
    """

    def __init__(self, file_paths, labels, transforms=None):
        """Build the dataset from parallel lists of paths and labels.

        Args:
            file_paths: Iterable of paths to audio files readable by ``extract_mel``.
            labels: Integer labels aligned with ``file_paths`` (same length).
            transforms: Optional callable applied to the mel ``numpy`` array from ``extract_mel`` (typically same shape), before tensor conversion.
                If ``None``, no extra processing is applied.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        mel = extract_mel(self.file_paths[idx])
        if self.transforms is not None:
            mel = self.transforms(mel)

        mel = torch.tensor(mel).unsqueeze(0).float()
        label = torch.tensor(self.labels[idx])

        return mel, label
