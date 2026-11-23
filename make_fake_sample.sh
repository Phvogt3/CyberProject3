#!/bin/bash
say -o samples/fake/tts_fake.aiff "Please verify my identity so I can access my account."
afconvert -f WAVE -d LEI16@16000 samples/fake/tts_fake.aiff samples/fake/tts_fake.wav
echo "Created samples/fake/tts_fake.wav"
