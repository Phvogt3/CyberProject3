import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "aasist"))
from models.AASIST import Model
from audio_utils import load_16k_mono

conf = json.load(open(ROOT / "aasist/config/AASIST.conf"))
model = Model(conf["model_config"])
model.load_state_dict(torch.load(ROOT / "aasist/models/weights/AASIST.pth", map_location="cpu"))
model.eval()

N_SAMP = conf["model_config"]["nb_samp"]


def prep(path):
    wav = load_16k_mono(path)
    if wav.shape[0] < N_SAMP:
        wav = wav.repeat(N_SAMP // wav.shape[0] + 1)
    return wav[:N_SAMP].unsqueeze(0)


files = sys.argv[1:] or sorted(str(p) for p in (ROOT / "samples").rglob("*.wav"))

print("AASIST spoof detection (ASVspoof 2019 LA weights)")
with torch.no_grad():
    for f in files:
        _, out = model(prep(f))
        prob = torch.softmax(out, dim=1)[0, 1].item()
        label = "BONA FIDE" if prob >= 0.5 else "SPOOF"
        print(f"{Path(f).name:22s} score={out[0, 1].item():8.4f} p(bona fide)={prob:.4f} -> {label}")
