import json
import time
from pathlib import Path
from typing import Any

import httpx

from app.config import Settings


class SyncLabsClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        if not self.settings.sync_api_key:
            raise RuntimeError("PERSONAGEMIA_SYNC_API_KEY is not configured")
        self.headers = {"x-api-key": self.settings.sync_api_key}

    def list_voices(self) -> list[dict[str, Any]]:
        with httpx.Client(timeout=60) as client:
            response = client.get(
                f"{self.settings.sync_base_url}/v2/voices",
                headers=self.headers,
            )
            response.raise_for_status()
            data = response.json()
        if not isinstance(data, list):
            raise RuntimeError("Unexpected Sync Labs voices response")
        return data

    def synthesize(
        self,
        script: str,
        *,
        voice_id: str | None = None,
        stability: float = 0.28,
        similarity_boost: float = 0.80,
    ) -> dict[str, Any]:
        selected_voice = voice_id or self.settings.sync_voice_id
        if not selected_voice:
            raise RuntimeError("No Sync Labs / ElevenLabs voice id configured")
        payload = {
            "script": script,
            "voiceId": selected_voice,
            "provider": "elevenlabs",
            "stability": stability,
            "similarityBoost": similarity_boost,
        }
        with httpx.Client(timeout=120) as client:
            response = client.post(
                f"{self.settings.sync_base_url}/v2/tts",
                headers={**self.headers, "Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    def generate_from_image(
        self,
        image_path: Path,
        *,
        audio_url: str,
        output_name: str = "mr_uncut_test",
    ) -> dict[str, Any]:
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        input_payload = json.dumps([{"type": "audio", "url": audio_url}])
        with image_path.open("rb") as handle, httpx.Client(timeout=180) as client:
            files = {
                "image": (image_path.name, handle, "image/png"),
            }
            data = {
                "model": "sync-3",
                "input": input_payload,
                "outputFileName": output_name,
            }
            response = client.post(
                f"{self.settings.sync_base_url}/v2/generate",
                headers=self.headers,
                files=files,
                data=data,
            )
            response.raise_for_status()
            return response.json()

    def wait_for_generation(
        self,
        generation_id: str,
        *,
        timeout_seconds: int = 900,
        poll_seconds: float = 5.0,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        with httpx.Client(timeout=90) as client:
            while time.monotonic() < deadline:
                response = client.get(
                    f"{self.settings.sync_base_url}/v2/generate/{generation_id}",
                    headers=self.headers,
                    params={"wait": "true"},
                )
                response.raise_for_status()
                data = response.json()
                status = data.get("status")
                if status == "COMPLETED":
                    return data
                if status in {"FAILED", "REJECTED"}:
                    raise RuntimeError(
                        f"Sync Labs generation {status}: "
                        f"{data.get('errorCode') or data.get('error') or 'unknown error'}"
                    )
                time.sleep(poll_seconds)
        raise TimeoutError(f"Sync Labs generation {generation_id} timed out")
