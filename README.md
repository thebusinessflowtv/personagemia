# PersonagemIA

PersonagemIA is an AI virtual-presenter pipeline designed for a fixed studio and a consistent fictional character.

The system separates **identity**, **personality**, **acting direction**, **voice/audio**, **avatar rendering**, and **final composition** so the visual engine can be upgraded without rewriting the whole product.

## Goals

- One persistent fictional presenter with a stable visual identity.
- English-first delivery with a strong, expressive, opinionated on-camera style.
- Fixed studio/background to concentrate rendering quality on the person.
- Acting direction for eyes, face, head, shoulders, hands, arms and posture.
- Pluggable avatar engines (local/open-source GPU engines can be added behind one interface).
- GitHub Actions orchestration with optional self-hosted GPU runners.
- API + CLI so MediaForge or another system can submit render jobs.

## Pipeline

```text
Topic / script
      |
      v
Personality + Acting Director
      |
      v
Scene / performance plan
      |
      +--> existing or external TTS --> narration.wav
      |
      v
Avatar Engine
      |
      v
Fixed-studio presenter footage
      |
      v
FFmpeg composition / export
      |
      v
Final video
```

## Status

Foundation/bootstrap in progress. The repository is intentionally engine-agnostic at this stage; the next phase is to define the master character, studio image and production avatar backend.
