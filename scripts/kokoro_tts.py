import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro import KPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--voice", default="am_puck")
    parser.add_argument("--speed", type=float, default=1.12)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pipeline = KPipeline(lang_code="a")
    chunks = []
    for _, _, audio in pipeline(
        args.text,
        voice=args.voice,
        speed=args.speed,
        split_pattern=r"\n+",
    ):
        chunks.append(np.asarray(audio, dtype=np.float32))

    if not chunks:
        raise RuntimeError("Kokoro returned no audio")
    sf.write(str(output), np.concatenate(chunks), 24000)
    print(output)


if __name__ == "__main__":
    main()
