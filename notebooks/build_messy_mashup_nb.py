"""One-off script to generate messy_mashup_pipeline.ipynb"""
import json
from pathlib import Path

cells = []

def add_md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.strip().split("\n")]})

def add_code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in text.strip().split("\n")]})

add_md("""
# Messy Mashup — Genre classification (train + inference)

**Sections:** Config → Audio features (mix stems, mel, noise) → Dataset → CNN + CRNN → Train/val + Macro F1 → Submission.

Run **Save & Run All** on Kaggle. If paths differ, adjust `CONFIG` after listing `/kaggle/input` once.
""")

add_code("""
# =============================================================================
# 1) IMPORTS
# =============================================================================
import os
import random
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# Optional logging (disable if not installed)
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
""")

add_code("""
# =============================================================================
# 2) CONFIG — edit if your competition bundle uses different folder names
# =============================================================================
def discover_data_root():
    \"\"\"Pick first existing Kaggle/local layout.\"\"\"
    env = os.environ.get("KAGGLE_INPUT", "")
    roots = []
    if env:
        roots.append(Path(env) / "messy-mashup")
        roots.append(Path(env) / "messy_mashup")
        roots.append(Path(env) / "jan-2026-dl-gen-ai-project" / "messy_mashup")
        roots.append(Path(env) / "competitions" / "jan-2026-dl-gen-ai-project" / "messy_mashup")
    roots.append(Path("/kaggle/input/messy-mashup"))
    roots.append(Path("/kaggle/input/messy_mashup"))
    roots.append(Path("/kaggle/input/competitions/jan-2026-dl-gen-ai-project/messy_mashup"))
    roots.append(Path.cwd() / "data" / "messy_mashup")  # local placeholder
    for p in roots:
        if p.exists() and (p / "genres_stems").is_dir():
            return p
    raise FileNotFoundError(
        "Could not find genres_stems/. List /kaggle/input once, then set DATA_ROOT manually."
    )


try:
    DATA_ROOT = discover_data_root()
except FileNotFoundError:
    # Last-resort path from course bundle layout — change if your slug differs
    DATA_ROOT = Path("/kaggle/input/competitions/jan-2026-dl-gen-ai-project/messy_mashup")
    if not (DATA_ROOT / "genres_stems").is_dir():
        raise

STEMS_SUBDIR = "genres_stems"
# ESC-50 is often inside the same bundle; search common locations
NOISE_SEARCH_DIRS = [
    DATA_ROOT / "ESC-50-master" / "audio",
    DATA_ROOT / "esc-50" / "audio",
    DATA_ROOT / "ESC-50" / "audio",
]

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]
LABEL_MAP = {g: i for i, g in enumerate(GENRES)}
SR = 22050
N_MELS = 128
MEL_TIME = 128
N_CLASSES = len(GENRES)

# Training
SEED = 42
BATCH_SIZE = 16
EPOCHS = 25  # increase for leaderboard; start reasonable for one run
LR = 1e-3
VAL_FRAC = 0.15
NUM_WORKERS = 0  # librosa in workers can break on Kaggle; keep 0

# Model: "cnn" or "crnn"
MODEL_NAME = "crnn"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DATA_ROOT:", DATA_ROOT)
print("device:", device)
""")

add_code("""
# =============================================================================
# 3) NOISE FILES (ESC-50) — cache paths once
# =============================================================================
def collect_noise_files(search_dirs):
    files = []
    for d in search_dirs:
        if not d.exists():
            continue
        for root, _, names in os.walk(d):
            for n in names:
                if n.lower().endswith(".wav"):
                    files.append(os.path.join(root, n))
    return files


# Discover any ESC-style tree under the bundle (dataset layout varies)
for root, _, _ in os.walk(DATA_ROOT):
    r = root.replace("\\\\", "/")
    if r.endswith("/audio") and "ESC" in r.upper():
        NOISE_SEARCH_DIRS.append(Path(root))

NOISE_FILES = collect_noise_files(NOISE_SEARCH_DIRS)
print("ESC-50 wav files found:", len(NOISE_FILES))
if len(NOISE_FILES) == 0:
    print("Warning: no noise files — training will skip ESC-50 noise (still uses other augmentations).")
""")

add_code("""
# =============================================================================
# 4) FEATURES — mix stems, augment, mel (128 x 128), optional noise
# =============================================================================
def load_and_mix_stems(folder_path: str):
    \"\"\"Load drums/vocals/bass/others, align lengths, mix to one waveform.\"\"\"
    stems = ["drums.wav", "vocals.wav", "bass.wav", "others.wav"]
    signals = []
    for s in stems:
        p = os.path.join(folder_path, s)
        if os.path.isfile(p):
            y, _ = librosa.load(p, sr=SR, mono=True)
            signals.append(y.astype(np.float32))
    if not signals:
        return None
    min_len = min(len(x) for x in signals)
    trimmed = [x[:min_len] for x in signals]
    # Sum stems (closer to a mashup than average) then peak-normalize
    mixed = np.sum(trimmed, axis=0)
    peak = np.max(np.abs(mixed)) + 1e-9
    mixed = (mixed / peak).astype(np.float32)
    return mixed


def _random_noise_segment(target_len: int):
    \"\"\"Return a noise vector of length target_len (or padded), or zeros.\"\"\"
    if not NOISE_FILES:
        return np.zeros(target_len, dtype=np.float32)
    path = random.choice(NOISE_FILES)
    n, _ = librosa.load(path, sr=SR, mono=True)
    n = n.astype(np.float32)
    if len(n) >= target_len:
        start = random.randint(0, len(n) - target_len)
        return n[start : start + target_len]
    pad = target_len - len(n)
    return np.pad(n, (0, pad), mode="constant")


def augment_waveform(y: np.ndarray, training: bool) -> np.ndarray:
    \"\"\"Time shift, gain; optional light pitch shift (slow — low probability).\"\"\"
    if not training or y is None or len(y) == 0:
        return y
    y = y.copy()
    # Time shift (circular / reflect pad)
    if random.random() < 0.5:
        shift = random.randint(-SR // 4, SR // 4)
        y = np.roll(y, shift)
    # Volume
    y = y * float(random.uniform(0.5, 1.2))
    # Light pitch shift (optional, ~10% of batches — can comment out if too slow)
    if random.random() < 0.1:
        try:
            y = librosa.effects.pitch_shift(y, sr=SR, n_steps=float(random.uniform(-1.0, 1.0)))
        except Exception:
            pass
    return y.astype(np.float32)


def extract_mel_from_signal(y: np.ndarray, training: bool = True) -> np.ndarray:
    \"\"\"
    Build log-mel spectrogram of shape (N_MELS, MEL_TIME) = (128, 128).
    Training: ESC-50 noise + waveform augmentations. Inference: no random noise.
    \"\"\"
    if y is None or len(y) < 512:
        y = np.zeros(SR, dtype=np.float32)  # ~1s silence fallback

    y = augment_waveform(y, training=training)

    if training and NOISE_FILES and random.random() < 0.7:
        noise = _random_noise_segment(len(y))
        snr = random.uniform(0.05, 0.35)  # linear scale mix; robust to unknown ESC scaling
        y = y + snr * noise
        peak = np.max(np.abs(y)) + 1e-9
        y = (y / peak).astype(np.float32)

    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, n_fft=1024, hop_length=512, fmin=20, fmax=SR // 2
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Fixed 128 time frames
    if mel_db.shape[1] < MEL_TIME:
        mel_db = np.pad(mel_db, ((0, 0), (0, MEL_TIME - mel_db.shape[1])), mode="constant")
    else:
        mel_db = mel_db[:, :MEL_TIME]

    # Per-sample standardization — helps optimization a lot vs raw dB
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-5)
    return mel_db.astype(np.float32)


def sanity_check_mel(mel: np.ndarray, tag=""):
    \"\"\"Debug: catch constant / empty mels.\"\"\"
    std = float(mel.std())
    if std < 1e-6:
        print(f"[WARN] mel nearly constant {tag} mean={mel.mean():.4f} std={std}")
    return std
""")

add_code("""
# =============================================================================
# 5) DATASET — one sample = one song folder (mixed stems); tensor (1, 128, 128)
# =============================================================================
class MashupTrainDataset(Dataset):
    def __init__(self, folder_paths, labels, training=True, debug_once=True):
        self.folder_paths = folder_paths
        self.labels = labels
        self.training = training
        self.debug_once = debug_once

    def __len__(self):
        return len(self.folder_paths)

    def __getitem__(self, idx):
        folder = self.folder_paths[idx]
        y = load_and_mix_stems(folder)
        mel = extract_mel_from_signal(y if y is not None else np.zeros(SR, np.float32), training=self.training)
        if self.debug_once and idx == 0:
            sanity_check_mel(mel, tag="first_sample")
            print(f"[debug] mel shape {mel.shape} mean={mel.mean():.4f} std={mel.std():.4f}")
            self.debug_once = False
        x = torch.from_numpy(mel).unsqueeze(0)  # (1, 128, 128)
        lab = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, lab


def build_file_list():
    folders, labels = [], []
    for g in GENRES:
        gdir = DATA_ROOT / STEMS_SUBDIR / g
        if not gdir.is_dir():
            continue
        for name in sorted(os.listdir(gdir)):
            fp = gdir / name
            if fp.is_dir():
                folders.append(str(fp))
                labels.append(LABEL_MAP[g])
    return folders, labels


all_folders, all_labels = build_file_list()
print("Total training folders:", len(all_folders))
assert len(all_folders) > 0, "No training folders found"

# Stratified train / val split
train_f, val_f, train_y, val_y = train_test_split(
    all_folders, all_labels, test_size=VAL_FRAC, random_state=SEED, stratify=all_labels
)
print("train:", len(train_f), "val:", len(val_f))
""")

add_code("""
# =============================================================================
# 6) MODELS — CNN baseline + CRNN (CNN + GRU temporal)
# =============================================================================
class CNNModel(nn.Module):
    def __init__(self, n_classes=N_CLASSES, p_drop=0.35):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # 128 -> 64 -> 32 -> 16; 128x128 -> 16x16 after 3 pools
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p_drop),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)


class CRNNModel(nn.Module):
    \"\"\"2D CNN (pool mostly along mel axis) + GRU over time; input (B,1,128,128).\"\"\"
    def __init__(self, n_classes=N_CLASSES, hidden=128, p_drop=0.35):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # 64 x 128
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # 32 x 128
            nn.Conv2d(64, 96, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # 16 x 128
            nn.Conv2d(96, 96, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # 8 x 128
        )
        # (B, 96, 8 mel bins, 128 time) -> sequence length 128, feat 96*8=768
        feat_dim = 96 * 8
        self.gru = nn.GRU(
            feat_dim, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3
        )
        self.head = nn.Sequential(nn.Dropout(p_drop), nn.Linear(hidden * 2, n_classes))

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)
        out, _ = self.gru(x)
        x = out[:, -1, :]
        return self.head(x)


def build_model(name: str):
    name = name.lower()
    if name == "cnn":
        return CNNModel()
    if name == "crnn":
        return CRNNModel()
    raise ValueError(name)
""")

add_code("""
# =============================================================================
# 7) TRAINING + VALIDATION (Macro F1 each epoch)
# =============================================================================
train_ds = MashupTrainDataset(train_f, train_y, training=True, debug_once=True)
val_ds = MashupTrainDataset(val_f, val_y, training=False, debug_once=True)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available()
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available()
)

model = build_model(MODEL_NAME).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
criterion = nn.CrossEntropyLoss()

if HAS_WANDB:
    try:
        wandb.init(project="messy-mashup", config=dict(model=MODEL_NAME, lr=LR, epochs=EPOCHS))
    except Exception as _e:
        print("wandb.init failed, continuing without wandb:", _e)
        HAS_WANDB = False

best_f1 = 0.0
for epoch in range(EPOCHS):
    model.train()
    running = 0.0
    n_seen = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        running += loss.item() * xb.size(0)
        n_seen += xb.size(0)
    train_loss = running / max(n_seen, 1)

    model.eval()
    preds_all, y_all = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            pr = logits.argmax(dim=1).cpu().numpy()
            preds_all.append(pr)
            y_all.append(yb.numpy())
    preds_all = np.concatenate(preds_all)
    y_all = np.concatenate(y_all)
    macro_f1 = f1_score(y_all, preds_all, average="macro")
    scheduler.step(macro_f1)

    print(f"Epoch {epoch+1}/{EPOCHS}  train_loss={train_loss:.4f}  val_macro_f1={macro_f1:.4f}")
    if HAS_WANDB:
        try:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_macro_f1": macro_f1, "lr": optimizer.param_groups[0]["lr"]})
        except Exception:
            pass

    if macro_f1 > best_f1:
        best_f1 = macro_f1
        torch.save(model.state_dict(), "best_model.pt")
        print("  saved best_model.pt")

print("Best val macro F1:", best_f1)
""")

add_code("""
# =============================================================================
# 8) INFERENCE — test.csv + mashups; same mel path as validation (no train noise)
# =============================================================================
def find_test_csv_and_audio_dir():
    \"\"\"Locate test.csv and folder of wav mashups.\"\"\"
    candidates = [
        DATA_ROOT / "test.csv",
        DATA_ROOT / "test" / "test.csv",
    ]
    test_csv = None
    for c in candidates:
        if c.is_file():
            test_csv = c
            break
    if test_csv is None:
        for p in DATA_ROOT.rglob("test.csv"):
            test_csv = p
            break
    if test_csv is None:
        raise FileNotFoundError("test.csv not found under DATA_ROOT")

    parent = test_csv.parent
    # Common layouts: same dir, or test/audio/
    audio_dirs = [parent, parent / "audio", parent / "test_audio", DATA_ROOT / "test", DATA_ROOT / "test_audio"]
    return test_csv, audio_dirs


test_csv_path, audio_dir_candidates = find_test_csv_and_audio_dir()
test_df = pd.read_csv(test_csv_path)
print("test.csv columns:", test_df.columns.tolist())
print(test_df.head())

id_col = "id" if "id" in test_df.columns else test_df.columns[0]


def resolve_wav_path(sample_id, audio_dirs):
    sid = str(sample_id).strip()
    alt = None
    if sid.isdigit():
        try:
            alt = f"{int(sid):04d}.wav"
        except ValueError:
            alt = None
    for d in audio_dirs:
        if not d.is_dir():
            continue
        for name in (f"{sid}.wav", alt):
            if not name:
                continue
            p = d / name
            if p.is_file():
                return str(p)
        try:
            for f in os.listdir(d):
                if f.endswith(".wav") and (f.startswith(sid) or sid in f):
                    return str(d / f)
        except FileNotFoundError:
            pass
    return None


# Reload best weights
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval()


def wav_to_tensor(wav_path: str):
    y, _ = librosa.load(wav_path, sr=SR, mono=True)
    mel = extract_mel_from_signal(y.astype(np.float32), training=False)
    sanity_check_mel(mel, tag=wav_path)
    return torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)


rows = []
missing = []
for _, row in test_df.iterrows():
    sid = row[id_col]
    path = resolve_wav_path(sid, audio_dir_candidates)
    if path is None:
        missing.append(sid)
        # uniform fallback — should not happen if layout is correct
        mel = extract_mel_from_signal(np.zeros(SR, np.float32), training=False)
        x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(device)
    else:
        x = wav_to_tensor(path).to(device)
    with torch.no_grad():
        pred = model(x).argmax(dim=1).item()
    rows.append({"id": sid, "genre": GENRES[pred]})

if missing:
    print("WARNING: missing audio for ids:", missing[:20], "..." if len(missing) > 20 else "")

sub = pd.DataFrame(rows)
sub_path = Path("/kaggle/working/submission.csv")
sub.to_csv(sub_path, index=False)
print("Saved", sub_path)
print(sub.head())
""")

add_md("""
### Notes for viva / tuning
- **Train vs test gap:** Training uses **mixed stems** from one song + noise/augmentation; test uses **cross-song mashups** + noise. Stronger overlap → more epochs, tune noise probability/SNR, try mixing random subsets of stems or two folders from same genre.
- **0.8 Macro F1** is an ambitious leaderboard target; if validation F1 stalls, consider pretrained audio backbones (e.g. PaSST, AST), ensembling, or longer clips — this notebook is a clean, correct baseline, not a guaranteed 0.8 solution.
""")

nb = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "name": "python",
            "version": "3.10.0",
            "mimetype": "text/x-python",
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
    },
    "cells": cells,
}

out = Path(__file__).resolve().parent / "messy_mashup_pipeline.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("Wrote", out)
