import math
import shutil
import subprocess
import uuid
import wave
from pathlib import Path

from app.config import Settings


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
    if rate <= 0:
        raise ValueError(f"Invalid WAV sample rate: {rate}")
    return frames / float(rate)


def compatible_video_length(seconds: float, fps: int) -> int:
    """Return a Wan-compatible 4n+1 frame count covering the audio."""
    wanted = max(1, math.ceil(seconds * fps))
    remainder = (wanted - 1) % 4
    if remainder:
        wanted += 4 - remainder
    return max(81, wanted)


class EchoMimicFlashEngine:
    name = "echomimic_v3_flash"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()

    def render(self, image_path: Path, audio_path: Path, output_path: Path) -> Path:
        self._validate(image_path, audio_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings.echo_output_dir.mkdir(parents=True, exist_ok=True)

        # EchoMimic names its output after the image. Give every job a unique image stem
        # so reruns never get skipped because an old output already exists.
        job_stem = f"mr_uncut_{uuid.uuid4().hex[:10]}"
        job_image = output_path.parent / f"{job_stem}{image_path.suffix.lower()}"
        shutil.copy2(image_path, job_image)

        duration = wav_duration_seconds(audio_path)
        video_length = compatible_video_length(duration, self.settings.echo_fps)
        model_root = self.settings.echo_model_root
        expected = self.settings.echo_output_dir / f"{job_stem}_output.mp4"

        if self.settings.echo_weight_dtype not in {"float16", "bfloat16"}:
            raise ValueError("echo_weight_dtype must be float16 or bfloat16")

        command = [
            str(self.settings.echo_python),
            "infer_flash.py",
            "--image_path",
            str(job_image.resolve()),
            "--audio_path",
            str(audio_path.resolve()),
            "--prompt",
            (
                "A confident young American male creator speaks directly to camera in a fixed "
                "studio. Natural restrained upper-body motion, stable face identity, realistic "
                "eye contact, subtle blinking and expressive eyebrows. Mouth movement stays "
                "natural and controlled without exaggerated jaw opening."
            ),
            "--num_inference_steps",
            str(self.settings.echo_steps),
            "--config_path",
            "config/config.yaml",
            "--model_name",
            str((model_root / "Wan2.1-Fun-V1.1-1.3B-InP").resolve()),
            "--ckpt_idx",
            "50000",
            "--transformer_path",
            str((model_root / "transformer/diffusion_pytorch_model.safetensors").resolve()),
            "--save_path",
            str(self.settings.echo_output_dir.resolve()),
            "--wav2vec_model_dir",
            str((model_root / "chinese-wav2vec2-base").resolve()),
            "--sampler_name",
            "Flow_Unipc",
            "--video_length",
            str(video_length),
            "--guidance_scale",
            str(self.settings.echo_guidance_scale),
            "--audio_guidance_scale",
            str(self.settings.echo_audio_guidance_scale),
            "--audio_scale",
            "1.0",
            "--neg_scale",
            "1.0",
            "--neg_steps",
            "0",
            "--seed",
            str(self.settings.quality_seed),
            "--enable_teacache",
            "--teacache_threshold",
            "0.08",
            "--num_skip_start_steps",
            "5",
            "--enable_riflex",
            "--riflex_k",
            "6",
            "--ulysses_degree",
            "1",
            "--ring_degree",
            "1",
            "--weight_dtype",
            self.settings.echo_weight_dtype,
            "--sample_size",
            str(self.settings.echo_size),
            str(self.settings.echo_size),
            "--fps",
            str(self.settings.echo_fps),
            "--shift",
            "5.0",
        ]
        if video_length > 138:
            command.extend(["--use_dynamic_cfg", "--use_dynamic_acfg"])

        subprocess.run(command, cwd=self.settings.echo_root, check=True)
        if not expected.exists():
            raise RuntimeError(f"EchoMimic finished but did not create {expected}")

        shutil.move(str(expected), str(output_path))
        return output_path

    def _validate(self, image_path: Path, audio_path: Path) -> None:
        required = {
            "character image": image_path,
            "audio": audio_path,
            "EchoMimic Python": self.settings.echo_python,
            "EchoMimic repository": self.settings.echo_root / "infer_flash.py",
            "Wan base model": self.settings.echo_model_root / "Wan2.1-Fun-V1.1-1.3B-InP",
            "Wav2Vec model": self.settings.echo_model_root / "chinese-wav2vec2-base",
            "Flash transformer": (
                self.settings.echo_model_root / "transformer/diffusion_pytorch_model.safetensors"
            ),
        }
        for label, path in required.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing {label}: {path}")


class LatentSyncRefiner:
    """Temporally coherent mouth/teeth refinement for Avatar Quality V2."""

    name = "latentsync_1_5"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()

    def refine(self, video_path: Path, audio_path: Path, output_path: Path) -> Path:
        root = self.settings.latentsync_root
        python = self.settings.latentsync_python
        checkpoint = self.settings.latentsync_checkpoint
        required = {
            "LatentSync Python": python,
            "LatentSync repository": root / "scripts/inference.py",
            "LatentSync stage2 config": root / "configs/unet/stage2.yaml",
            "LatentSync checkpoint": checkpoint,
            "LatentSync Whisper": root / "checkpoints/whisper/tiny.pt",
        }
        for label, path in required.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing {label}: {path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = output_path.parent / f"latentsync_{uuid.uuid4().hex[:10]}"
        subprocess.run(
            [
                str(python),
                "-m",
                "scripts.inference",
                "--unet_config_path",
                "configs/unet/stage2.yaml",
                "--inference_ckpt_path",
                str(checkpoint.resolve()),
                "--inference_steps",
                str(self.settings.latentsync_steps),
                "--guidance_scale",
                str(self.settings.latentsync_guidance_scale),
                "--seed",
                str(self.settings.quality_seed),
                "--enable_deepcache",
                "--video_path",
                str(video_path.resolve()),
                "--audio_path",
                str(audio_path.resolve()),
                "--video_out_path",
                str(output_path.resolve()),
                "--temp_dir",
                str(temp_dir.resolve()),
            ],
            cwd=root,
            check=True,
        )
        if not output_path.exists():
            raise RuntimeError("LatentSync completed without an MP4 output")
        return output_path


class AvatarQualityV2Engine:
    """Generate short deterministic chunks and reset identity before drift accumulates."""

    name = "mr_uncut_avatar_quality_v2"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.motion = EchoMimicFlashEngine(self.settings)
        self.lips = LatentSyncRefiner(self.settings)

    def render(self, image_path: Path, audio_path: Path, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        duration = wav_duration_seconds(audio_path)
        chunk_seconds = max(1.0, self.settings.avatar_chunk_seconds)
        if duration <= chunk_seconds + 0.05:
            raw = output_path.with_name(f"{output_path.stem}_motion.mp4")
            self.motion.render(image_path, audio_path, raw)
            return self.lips.refine(raw, audio_path, output_path)

        work = output_path.parent / f"quality_v2_{uuid.uuid4().hex[:10]}"
        work.mkdir(parents=True, exist_ok=True)
        refined_parts: list[Path] = []
        count = math.ceil(duration / chunk_seconds)
        for index in range(count):
            start = index * chunk_seconds
            length = min(chunk_seconds, duration - start)
            chunk_audio = work / f"audio_{index:03d}.wav"
            raw_video = work / f"motion_{index:03d}.mp4"
            refined_video = work / f"refined_{index:03d}.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    f"{start:.3f}",
                    "-t",
                    f"{length:.3f}",
                    "-i",
                    str(audio_path),
                    "-acodec",
                    "pcm_s16le",
                    str(chunk_audio),
                ],
                check=True,
            )
            self.motion.render(image_path, chunk_audio, raw_video)
            self.lips.refine(raw_video, chunk_audio, refined_video)
            refined_parts.append(refined_video)

        concat = work / "concat.txt"
        concat.write_text(
            "".join(f"file '{part.resolve()}'\n" for part in refined_parts),
            encoding="utf-8",
        )
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat),
                "-c:v",
                "libx264",
                "-crf",
                "16",
                "-preset",
                "medium",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(output_path),
            ],
            check=True,
        )
        if not output_path.exists():
            raise RuntimeError("Avatar Quality V2 completed without an MP4 output")
        return output_path


class MuseTalkRefiner:
    name = "musetalk_1_5"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()

    def refine(self, video_path: Path, audio_path: Path, output_path: Path) -> Path:
        root = self.settings.musetalk_root
        python = self.settings.musetalk_python
        if not python.exists():
            raise FileNotFoundError(f"Missing MuseTalk Python: {python}")
        if not (root / "scripts/inference.py").exists():
            raise FileNotFoundError(f"Missing MuseTalk repository: {root}")

        job = uuid.uuid4().hex[:10]
        config_path = output_path.parent / f"musetalk_{job}.yaml"
        result_dir = output_path.parent / f"musetalk_{job}"
        config_path.write_text(
            "task_0:\n"
            f"  video_path: \"{video_path.resolve()}\"\n"
            f"  audio_path: \"{audio_path.resolve()}\"\n",
            encoding="utf-8",
        )
        subprocess.run(
            [
                str(python),
                "-m",
                "scripts.inference",
                "--inference_config",
                str(config_path.resolve()),
                "--result_dir",
                str(result_dir.resolve()),
                "--unet_model_path",
                "models/musetalkV15/unet.pth",
                "--unet_config",
                "models/musetalkV15/musetalk.json",
                "--version",
                "v15",
            ],
            cwd=root,
            check=True,
        )
        candidates = sorted(result_dir.rglob("*.mp4"), key=lambda item: item.stat().st_mtime)
        if not candidates:
            raise RuntimeError("MuseTalk completed without an MP4 output")
        shutil.copy2(candidates[-1], output_path)
        return output_path
