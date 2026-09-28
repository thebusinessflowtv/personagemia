import argparse
from pathlib import Path

import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--exaggeration", type=float, default=0.85)
    parser.add_argument("--cfg-weight", type=float, default=0.30)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxTTS.from_pretrained(device=device)

    kwargs = {
        "exaggeration": args.exaggeration,
        "cfg_weight": args.cfg_weight,
    }
    if args.reference:
        kwargs["audio_prompt_path"] = args.reference

    wav = model.generate(args.text, **kwargs)
    ta.save(str(output), wav, model.sr)
    print(output)


if __name__ == "__main__":
    main()
