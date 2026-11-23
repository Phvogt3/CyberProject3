import sys
from speechbrain.inference.speaker import SpeakerRecognition
from audio_utils import load_16k_mono

model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
)

if len(sys.argv) == 3:
    pairs = [(sys.argv[1], sys.argv[2], "custom pair")]
else:
    pairs = [
        ("samples/real/spk1_snt1.wav", "samples/real/spk1_snt2.wav", "same speaker"),
        ("samples/real/spk1_snt1.wav", "samples/real/spk2_snt1.wav", "different speaker"),
    ]

print("ECAPA-TDNN speaker verification (cosine similarity, threshold 0.25)")
for enroll, test, label in pairs:
    a = load_16k_mono(enroll).unsqueeze(0)
    b = load_16k_mono(test).unsqueeze(0)
    score, decision = model.verify_batch(a, b)
    print(f"{label:18s} {enroll} vs {test}: score={score.item():.4f} match={bool(decision.item())}")
