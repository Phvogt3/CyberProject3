# Voice Verification and Deepfake Detection (DSC 4354 Project 3)

Modality: voice

Verification model: ECAPA-TDNN (SpeechBrain, speechbrain/spkrec-ecapa-voxceleb)
Detection model: AASIST (clovaai/aasist, ASVspoof 2019 LA pretrained weights)

## Setup (macOS, Python 3.10 to 3.13)

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Run

./make_fake_sample.sh      # creates a TTS fake sample with macOS say
python verify_ecapa.py     # speaker verification test
python detect_aasist.py    # spoof detection on every wav in samples/

Real samples in samples/real come from the SpeechBrain test set (Apache 2.0).
AASIST code and weights are from clovaai/aasist (MIT).
