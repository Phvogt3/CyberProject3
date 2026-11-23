import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF


def load_16k_mono(path):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = torch.from_numpy(wav)
    if sr != 16000:
        wav = AF.resample(wav, sr, 16000)
    return wav
