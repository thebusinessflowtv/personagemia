"""Mr. Uncut Quality V2: stable motion + temporal lip refinement + fixed studio.

The proven EchoMimicV2 calibration remains the motion generator. This wrapper
runs it in short deterministic chunks, then sends the result through
LatentSync 1.5 to specifically improve mouth/teeth temporal consistency, and
finally mattes the actor onto one immutable studio frame so the background can
never drift between frames.
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

SCRIPT_B64 = "__SCRIPT_B64__"
VOICE = "__VOICE__"

WORK = Path("/kaggle/working")
ROOT = Path("/kaggle/temp/mr-uncut-quality-v2")
REPO = ROOT / "personagemia"
LATENT_ROOT = ROOT / "LatentSync"
LATENT_VENV = ROOT / "venvs/latentsync"
HF_CACHE = ROOT / "hf-cache"
RAW = WORK / "mr_uncut_echo_raw.mp4"
LIPSYNC = WORK / "mr_uncut_latentsync.mp4"
AUDIO = WORK / "mr_uncut_voice.wav"
OUTPUT = WORK / "mr_uncut_kaggle_test.mp4"
INFO = WORK / "mr_uncut_run_info.json"
ERROR = WORK / "mr_uncut_error.txt"
SILENT = ROOT / "mr_uncut_studio_silent.mp4"

LATENTSYNC_STEPS = 30
LATENTSYNC_GUIDANCE = 1.35
SEED = 3407
MAX_CHUNK_SECONDS = 3.0
FINAL_W = 1280
FINAL_H = 720
SUBJECT_HEIGHT_RATIO = 0.93
ALPHA_HISTORY = 0.22

os.environ.update({
    "PIP_NO_CACHE_DIR": "1",
    "HF_HOME": str(HF_CACHE),
    "HUGGINGFACE_HUB_CACHE": str(HF_CACHE / "hub"),
    "TOKENIZERS_PARALLELISM": "false",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "U2NET_HOME": str(ROOT / "u2net"),
})


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def disk(label: str) -> None:
    u = shutil.disk_usage("/kaggle")
    g = 1024 ** 3
    print(f"DISK {label}: used={u.used/g:.1f}GiB free={u.free/g:.1f}GiB", flush=True)


def clone_repo() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    if REPO.exists():
        shutil.rmtree(REPO)
    run(["git", "clone", "--depth", "1", "https://github.com/thebusinessflowtv/personagemia.git", str(REPO)])


def run_proven_motion_calibration() -> None:
    """Reuse the already validated full EchoMimicV2 calibration unchanged."""
    source = (REPO / "kaggle/mr_uncut_test.py").read_text(encoding="utf-8")
    source = source.replace("__SCRIPT_B64__", SCRIPT_B64).replace("__VOICE__", VOICE)
    baseline = ROOT / "echo_baseline.py"
    baseline.write_text(source, encoding="utf-8")
    run([sys.executable, str(baseline)])
    generated = WORK / "mr_uncut_kaggle_test.mp4"
    if not generated.exists():
        raise RuntimeError("EchoMimic baseline did not create video")
    if not AUDIO.exists():
        raise RuntimeError("EchoMimic baseline did not create audio")
    if RAW.exists():
        RAW.unlink()
    generated.rename(RAW)
    print(f"Motion baseline ready: {RAW}", flush=True)


def install_latentsync() -> Path:
    """Isolate lip-refiner packages so EchoMimic's working environment stays intact."""
    if LATENT_ROOT.exists():
        shutil.rmtree(LATENT_ROOT)
    run(["git", "clone", "--depth", "1", "https://github.com/bytedance/LatentSync.git", str(LATENT_ROOT)])
    if LATENT_VENV.exists():
        shutil.rmtree(LATENT_VENV)
    run([sys.executable, "-m", "venv", "--system-site-packages", str(LATENT_VENV)])
    py = LATENT_VENV / "bin/python"
    packages = [
        "diffusers==0.32.2", "transformers==4.48.0", "accelerate==0.26.1",
        "decord==0.6.0", "einops==0.7.0", "omegaconf==2.3.0",
        "opencv-python==4.9.0.80", "python_speech_features==0.6",
        "librosa==0.10.1", "ffmpeg-python==0.2.0", "imageio==2.31.1",
        "imageio-ffmpeg==0.5.1", "kornia==0.8.0", "insightface==0.7.3",
        "onnxruntime-gpu==1.21.0", "DeepCache==0.1.1",
        "huggingface-hub==0.30.2", "rembg>=2.0.60",
    ]
    run([str(py), "-m", "pip", "install", "-q", "--no-cache-dir", *packages])
    return py


def download_latentsync_models(py: Path) -> None:
    code = (
        "from huggingface_hub import hf_hub_download; "
        "from pathlib import Path; "
        "d=Path('checkpoints'); d.mkdir(parents=True,exist_ok=True); "
        "hf_hub_download(repo_id='ByteDance/LatentSync-1.5',filename='latentsync_unet.pt',local_dir=d); "
        "hf_hub_download(repo_id='ByteDance/LatentSync-1.5',filename='whisper/tiny.pt',local_dir=d)"
    )
    run([str(py), "-c", code], cwd=LATENT_ROOT)
    disk("LatentSync models ready")


def refine_lips(py: Path) -> None:
    """Temporal mouth pass to reduce unstable teeth and frame-to-frame lip changes."""
    download_latentsync_models(py)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(LATENT_ROOT)
    run([
        str(py), "-m", "scripts.inference",
        "--unet_config_path", "configs/unet/stage2.yaml",
        "--inference_ckpt_path", "checkpoints/latentsync_unet.pt",
        "--inference_steps", str(LATENTSYNC_STEPS),
        "--guidance_scale", str(LATENTSYNC_GUIDANCE),
        "--seed", str(SEED), "--enable_deepcache",
        "--video_path", str(RAW), "--audio_path", str(AUDIO),
        "--video_out_path", str(LIPSYNC),
        "--temp_dir", str(ROOT / "latentsync-temp"),
    ], cwd=LATENT_ROOT, env=env)
    if not LIPSYNC.exists():
        raise RuntimeError("LatentSync completed without video")
    print(f"Lip refinement ready: {LIPSYNC}", flush=True)


def composite_static_studio(py: Path) -> None:
    """Use a shared actor crop and temporally-smoothed alpha on a fixed background."""
    bg = REPO / "assets/studio/mr_uncut_garage_bg_800x450.jpg"
    if not bg.exists():
        raise FileNotFoundError(f"Missing official studio background: {bg}")

    compositor = ROOT / "composite.py"
    compositor.write_text(f'''\
from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image
from rembg import new_session, remove
SOURCE=Path({str(LIPSYNC)!r})
AUDIO=Path({str(AUDIO)!r})
BG=Path({str(bg)!r})
SILENT=Path({str(SILENT)!r})
OUTPUT=Path({str(OUTPUT)!r})
W={FINAL_W}; H={FINAL_H}; HEIGHT_RATIO={SUBJECT_HEIGHT_RATIO}; HISTORY={ALPHA_HISTORY}
cap=cv2.VideoCapture(str(SOURCE))
if not cap.isOpened(): raise RuntimeError(f"Cannot open {{SOURCE}}")
fps=cap.get(cv2.CAP_PROP_FPS) or 25.0
frames=[]
while True:
    ok, frame=cap.read()
    if not ok: break
    frames.append(frame)
cap.release()
if not frames: raise RuntimeError("No frames in refined video")
session=new_session("u2net_human_seg")
items=[]; union=None; previous=None
for frame in frames:
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    rgba=np.asarray(remove(Image.fromarray(rgb),session=session,alpha_matting=False))
    if rgba.ndim!=3 or rgba.shape[2]!=4: raise RuntimeError("Matting did not return RGBA")
    actor=cv2.cvtColor(rgba[:,:,:3],cv2.COLOR_RGB2BGR)
    alpha=rgba[:,:,3].astype(np.float32)/255.0
    if previous is not None: alpha=(1.0-HISTORY)*alpha+HISTORY*previous
    alpha=cv2.GaussianBlur(alpha,(5,5),0); previous=alpha.copy()
    ys,xs=np.where(alpha>0.10)
    if len(xs):
        b=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1]
        if union is None: union=b
        else:
            union=[min(union[0],b[0]),min(union[1],b[1]),max(union[2],b[2]),max(union[3],b[3])]
    items.append((actor,alpha))
if union is None: raise RuntimeError("Human matting found no actor")
x1,y1,x2,y2=union
px=max(6,int((x2-x1)*0.03)); py=max(6,int((y2-y1)*0.02))
x1=max(0,x1-px); x2=min(frames[0].shape[1],x2+px)
y1=max(0,y1-py); y2=min(frames[0].shape[0],y2+py)
cw=max(1,x2-x1); ch=max(1,y2-y1)
th=int(H*HEIGHT_RATIO); scale=th/ch; tw=max(1,int(cw*scale))
if tw>int(W*0.78): scale=(W*0.78)/cw; tw=int(cw*scale); th=int(ch*scale)
background=cv2.imread(str(BG),cv2.IMREAD_COLOR)
if background is None: raise RuntimeError(f"Cannot decode {{BG}}")
background=cv2.resize(background,(W,H),interpolation=cv2.INTER_LANCZOS4)
writer=cv2.VideoWriter(str(SILENT),cv2.VideoWriter_fourcc(*"mp4v"),fps,(W,H))
for actor,alpha in items:
    fg=actor[y1:y2,x1:x2]; a=alpha[y1:y2,x1:x2]
    fg=cv2.resize(fg,(tw,th),interpolation=cv2.INTER_LANCZOS4)
    a=cv2.resize(a,(tw,th),interpolation=cv2.INTER_LINEAR)
    canvas=background.copy(); ox=(W-tw)//2; oy=max(0,H-th)
    shadow=cv2.GaussianBlur(a,(0,0),10)*0.16
    sx=min(W-tw,max(0,ox+8)); sy=min(H-th,max(0,oy+7))
    sr=canvas[sy:sy+th,sx:sx+tw].astype(np.float32)
    canvas[sy:sy+th,sx:sx+tw]=np.clip(sr*(1-shadow[:,:,None]),0,255).astype(np.uint8)
    roi=canvas[oy:oy+th,ox:ox+tw].astype(np.float32); aa=np.clip(a,0,1)[:,:,None]
    canvas[oy:oy+th,ox:ox+tw]=np.clip(fg.astype(np.float32)*aa+roi*(1-aa),0,255).astype(np.uint8)
    writer.write(canvas)
writer.release()
subprocess.run(["ffmpeg","-loglevel","error","-y","-i",str(SILENT),"-i",str(AUDIO),"-c:v","libx264","-preset","medium","-crf","16","-pix_fmt","yuv420p","-c:a","aac","-b:a","192k","-shortest",str(OUTPUT)],check=True)
print(f"Final static-studio video: {{OUTPUT}}")
''', encoding="utf-8")
    env = os.environ.copy(); env["U2NET_HOME"] = str(ROOT / "u2net")
    run([str(py), str(compositor)], env=env)
    if not OUTPUT.exists():
        raise RuntimeError("Static studio compositor did not create final MP4")


def write_info() -> None:
    INFO.write_text(json.dumps({
        "status": "completed",
        "pipeline": "Mr-Uncut-Avatar-Quality-V2",
        "motion": "EchoMimicV2-full / deterministic short chunk",
        "lip_refiner": "LatentSync-1.5",
        "lip_steps": LATENTSYNC_STEPS,
        "lip_guidance": LATENTSYNC_GUIDANCE,
        "studio": "static-garage-v1",
        "matting": "u2net-human-seg + temporal-alpha",
        "alpha_history": ALPHA_HISTORY,
        "max_chunk_seconds": MAX_CHUNK_SECONDS,
        "seed": SEED,
        "final_resolution": f"{FINAL_W}x{FINAL_H}",
        "video": OUTPUT.name,
        "audio": AUDIO.name,
        "voice": VOICE,
    }, indent=2), encoding="utf-8")


def main() -> None:
    print("=== Mr. Uncut Avatar Quality V2 ===", flush=True)
    run(["nvidia-smi"]); disk("start")
    clone_repo()
    run_proven_motion_calibration()
    py = install_latentsync()
    refine_lips(py)
    composite_static_studio(py)
    write_info(); disk("done")
    print(f"DONE: {OUTPUT}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        error=traceback.format_exc(); print(error, flush=True)
        ERROR.write_text(error, encoding="utf-8")
        disk("failure")
        raise
