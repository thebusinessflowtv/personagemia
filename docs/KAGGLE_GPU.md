# Kaggle GPU runner for Mr. Uncut

This is the zero-per-generation-cost proof-of-concept path for PersonagemIA.

## Architecture

```text
GitHub Actions (CPU/orchestration)
        -> Kaggle API
        -> private Kaggle Script Kernel
        -> NvidiaTeslaT4
        -> Kokoro TTS
        -> EchoMimicV3-Flash
        -> mr_uncut_kaggle_test.mp4
        -> GitHub Actions artifact
```

## Required GitHub repository secrets

Open:

`personagemia -> Settings -> Secrets and variables -> Actions`

Create these repository secrets:

- `KAGGLE_USERNAME`: your Kaggle username/slug.
- `KAGGLE_API_TOKEN`: a Kaggle API/personal token created in your Kaggle account settings.

Do not commit the token or `kaggle.json` to the repository.

## Run the test

Open:

`Actions -> Mr Uncut Kaggle GPU Test -> Run workflow`

Choose one of the free Kokoro audition voices:

- `puck`
- `adam`
- `liam`

The first proof-of-concept deliberately uses EchoMimic at 512-class resolution and float16 for T4 compatibility. Once it passes, increase to 640 and 768 and compare quality/runtime.

## Current limitations

Kaggle GPU access has a weekly quota and availability can vary. This path is intended to avoid metered video-generation APIs; it is not guaranteed unlimited cloud compute.

The first run is slower because it downloads the EchoMimic and Wan model weights. If the proof-of-concept works, the next optimization is to cache the model bundle as a Kaggle Dataset/Model source so future jobs spend their GPU allocation on rendering instead of setup/downloads.
