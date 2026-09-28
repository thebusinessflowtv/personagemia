# PersonagemIA architecture

## Design principle

The presenter is the product. The studio is deliberately fixed so compute and quality can be concentrated on facial performance, eyes, head motion, shoulders, hands, arms and posture.

## Core layers

1. **Character identity** — master image, wardrobe, proportions and voice identity.
2. **Personality** — stable editorial/behavioral profile for the fictional creator.
3. **Acting Director** — turns script meaning into structured performance beats.
4. **Avatar Engine** — replaceable renderer that consumes image + audio + performance plan.
5. **Studio preservation** — fixed camera, lighting and background.
6. **Orchestration** — API, CLI and GitHub Actions.

## Engine contract

A production engine receives four inputs:

- character image
- narration audio
- performance plan JSON
- output path

The current `command` adapter means the heavy GPU renderer may live outside the Python application. A wrapper can integrate LongCat, EchoMimic, Wan-based animation or another future model without changing API clients.

## Performance plan

Every sentence/beat can carry:

- emotion
- semantic gesture
- energy
- eye-contact strength
- head-motion intensity
- brow-motion intensity

Later versions can add precise hand poses, body keypoints, timing from forced alignment, blink cadence and phoneme-level facial controls.

## Fixed studio rules

The background should have no animated screens, moving people or unnecessary motion. When supported by the selected renderer, background pixels should be preserved and only the performer region regenerated/composited.
