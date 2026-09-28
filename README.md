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

## Current foundation

The bootstrap layer is now in place:

- FastAPI service and CLI entrypoints.
- Deterministic Acting Director producing structured performance beats.
- Replaceable Avatar Engine interface.
- Command adapter for a local/open-source GPU renderer.
- Fixed-character and fixed-studio configuration templates.
- GitHub Actions CI and self-hosted GPU render workflow.
- Tests for core acting behavior.

The production avatar model itself is intentionally not locked yet. The next phase is to create and approve the master fictional character and studio image, then benchmark and connect the highest-quality compatible avatar renderer behind the existing engine contract.
