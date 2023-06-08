# Some tests with the transformers library

Primarily focussing on inference on user setup with a 16 GB graphic card.

## Text generation with Falcon/7B/Instruct

```
python falcon_test.py --test-gen --bitwidth (8|32)
```

The text generation appears to be 4x facter in 8-bit on an NVIDIA Tesla T4