# PersonagemIA — Mr. Uncut

PersonagemIA is the production pipeline for **Mr. Uncut**, a persistent fictional English-speaking virtual creator recorded in one fixed studio.

The project is **local-first**: avatar animation, TTS and final rendering run on our own GPU machine with open-source models and no per-video API charge. Anthropic remains optional for script writing/research because that is an explicit editorial choice, not a rendering dependency.

## Production pipeline

```text
Portuguese voice note / opinion
          |
          v
Transcription
          |
          v
Anthropic Script Writer (optional paid API)
Mr. Uncut: first-person, strong, direct English
          |
          v
LOCAL TTS
Chatterbox (primary) / Kokoro (fallback + auditions)
          |
          v
LOCAL AVATAR
EchoMimicV3-Flash
face + eyes + head + torso + arms/hands
          |
          v
MuseTalk 1.5 (optional)
extra lip-sync refinement
          |
          v
FFmpeg / MediaForge
          |
          v
Final video
```

## Zero-cost generation stack

| Stage | Engine | Per-generation API cost |
| --- | --- | ---: |
| Voice | Chatterbox | $0 |
| Voice fallback/auditions | Kokoro | $0 |
| Human animation | EchoMimicV3-Flash | $0 |
| Lip refinement | MuseTalk 1.5 | $0 |
| Composition | FFmpeg | $0 |
| Script writer | Anthropic (optional) | external API pricing applies |

"Zero-cost" here means no metered generation/API fee. The self-hosted computer still uses its own electricity and hardware.

## Mr. Uncut identity

The canonical character/studio frame lives at:

```text
assets/mr_uncut_master.jpg
```

Character behavior and editorial rules live at:

```text
config/mr_uncut.yaml
```

The fixed studio is intentional. We spend model capacity on the presenter: mouth, eyes, brows, head, shoulders, hands, arms and posture instead of regenerating a moving background.

## Local runtime layout

By default heavy models are kept outside Git:

```text
~/.personagemia-ai/
├── engines/
│   ├── echomimic_v3/
│   └── MuseTalk/              # optional
├── models/
│   └── echomimic_v3/flash/
│       ├── Wan2.1-Fun-V1.1-1.3B-InP/
│       ├── chinese-wav2vec2-base/
│       └── transformer/
│           └── diffusion_pytorch_model.safetensors
└── venvs/
    ├── chatterbox/
    ├── kokoro/
    ├── echomimic/
    └── musetalk/              # optional
```

Each AI engine gets its own Python environment to avoid Torch/CUDA dependency conflicts.

## One-time installation on the GPU runner

Linux + NVIDIA is the production target.

```bash
git clone https://github.com/thebusinessflowtv/personagemia.git
cd personagemia
python3.11 -m venv .venv
. .venv/bin/activate
pip install -e .
bash scripts/bootstrap_local_ai.sh
personagemia local-doctor
```

The bootstrap installs Chatterbox and Kokoro, clones EchoMimicV3 and downloads the Flash models. MuseTalk is optional:

```bash
INSTALL_MUSETALK=1 bash scripts/bootstrap_local_ai.sh
```

## First local video

```bash
personagemia local-test \
  --image assets/mr_uncut_master.jpg \
  --script "Everybody keeps pretending this is normal. I don't buy it. I'm Mr. Uncut, and I'm going to say exactly what I think." \
  --voice electric \
  --tts-engine chatterbox \
  --no-refine-lips
```

Outputs are written under:

```text
outputs/local-tests/<job-id>/
```

## Generate three free voice candidates

Kokoro provides quick built-in American voices for auditions:

```bash
personagemia voice-samples --presets puck,adam,liam
```

For the final Mr. Uncut voice, Chatterbox can use an approved reference WAV so the same vocal identity remains stable while we change performance intensity.

## Performance presets

Chatterbox presets:

- `electric` — default Mr. Uncut performance.
- `intense` — maximum emphasis/aggression.
- `controlled` — still expressive but more deliberate.

Kokoro audition voices:

- `puck`
- `adam`
- `liam`

## GitHub Actions

`.github/workflows/local-avatar-test.yml` is the production test workflow. It runs on:

```text
[self-hosted, linux, personagemia]
```

GitHub orchestrates the job; the user's NVIDIA machine performs the expensive inference. The workflow runs `local-doctor` before rendering and uploads the resulting test artifact.

## Commands

```text
personagemia write-script     Rewrite an opinion in Mr. Uncut voice using Anthropic
personagemia plan             Build an acting/performance plan
personagemia local-doctor     Validate GPU/models/local runtimes
personagemia voice-samples    Generate local voice auditions
personagemia local-test       TTS -> EchoMimic -> optional MuseTalk
personagemia render           Generic pluggable avatar render entrypoint
```

## Current phase

The software path is ready for a local GPU test. The next physical requirement is a Linux NVIDIA runner with the local runtime bootstrapped. After the first 10-second render we will tune:

1. final voice identity,
2. Chatterbox expressiveness,
3. EchoMimic hand/body intensity,
4. eye/head motion,
5. whether MuseTalk materially improves the final mouth quality.
