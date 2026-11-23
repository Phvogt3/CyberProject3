# Voice Verification and Deepfake Detection (DSC 4354 Project 3)

Modality: voice

Verification model: ECAPA-TDNN (SpeechBrain, speechbrain/spkrec-ecapa-voxceleb)
Detection model: AASIST (clovaai/aasist, ASVspoof 2019 LA pretrained weights)

## Setup (macOS, Python 3.12)

conda create -n p3voice python=3.12 -y
conda activate p3voice
pip install -r requirements.txt

## Run

./make_fake_sample.sh      # creates a TTS fake sample with macOS say
python verify_ecapa.py     # speaker verification test
python detect_aasist.py    # spoof detection on every wav in samples/

Real samples in samples/real come from the SpeechBrain test set (Apache 2.0).
AASIST code and weights are from clovaai/aasist (MIT).

## Team and Labor Division

| Member | Responsibilities |
|---|---|
| Phillip Vogt | Environment setup and dependency resolution, model selection, inference testing of both models, GitHub repository |
| Bryan Stuhr | Audio preprocessing (resampling to 16 kHz, silence trimming, length normalization), consent forms, and enrollment data collection |
| Mason Pluard | Benchmark fake sample download (ASVspoof), enrollment flow, and web interface |
