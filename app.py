import os
import threading
import uuid
import json
import time
from urllib.parse import urlencode
from urllib.request import urlopen
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import requests
from yt_dlp import YoutubeDL


BASE_DIR = Path(__file__).resolve().parent
DOWNLOAD_DIR = BASE_DIR / "tmp_audio"
DOWNLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder=".", static_url_path="")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
cors_origins = (
    [origin.strip() for origin in CORS_ORIGINS.split(",") if origin.strip()]
    if CORS_ORIGINS != "*"
    else "*"
)
CORS(
    app,
    resources={r"/api/*": {"origins": cors_origins}},
)

DOWNLOAD_FORMAT = os.getenv("YTDLP_FORMAT", "bestaudio[abr<=160]/bestaudio/best")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "whisper-1")
OPENAI_TIMEOUT_SEC = int(os.getenv("OPENAI_TIMEOUT_SEC", "600"))

jobs = {}
jobs_lock = threading.Lock()
translation_cache = {}
translation_lock = threading.Lock()


def extract_video_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def transcribe_with_openai(audio_path: Path, language: str = "en") -> list[dict]:
    if not OPENAI_API_KEY:
        raise RuntimeError("缺少 OPENAI_API_KEY，無法呼叫外部語音 API")

    with audio_path.open("rb") as audio_fp:
        files = {"file": (audio_path.name, audio_fp, "audio/mpeg")}
        data = {
            "model": OPENAI_TRANSCRIBE_MODEL,
            "response_format": "verbose_json",
            "temperature": "0",
        }
        if language:
            data["language"] = language

        resp = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            files=files,
            data=data,
            timeout=OPENAI_TIMEOUT_SEC,
        )

    if resp.status_code >= 400:
        raise RuntimeError(f"外部語音 API 失敗：HTTP {resp.status_code} {resp.text[:240]}")

    payload = resp.json()
    raw_segments = payload.get("segments") or []
    if not raw_segments:
        text = (payload.get("text") or "").strip()
        if text:
            return [{"start": 0.0, "end": 3.0, "text": text}]
        return []

    parsed = []
    for seg in raw_segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        parsed.append(
            {
                "start": round(float(seg.get("start", 0.0)), 3),
                "end": round(float(seg.get("end", seg.get("start", 0.0))), 3),
                "text": text,
            }
        )
    return parsed


def translate_text_to_zh_tw(text: str) -> str:
    normalized = (text or "").strip()
    if not normalized:
        return ""

    with translation_lock:
        if normalized in translation_cache:
            return translation_cache[normalized]

    query = urlencode(
        {
            "client": "gtx",
            "sl": "auto",
            "tl": "zh-TW",
            "dt": "t",
            "q": normalized,
        }
    )
    url = f"https://translate.googleapis.com/translate_a/single?{query}"
    with urlopen(url, timeout=12) as resp:
        payload = resp.read().decode("utf-8")
    data = json.loads(payload)
    translated = "".join(part[0] for part in data[0] if part and part[0]).strip()
    if not translated:
        translated = normalized

    with translation_lock:
        translation_cache[normalized] = translated
    return translated


def download_audio(job_id: str, video_id: str, out_path: Path) -> tuple[Path, float]:
    def on_progress(d):
        if d.get("status") != "downloading":
            return
        total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
        downloaded = d.get("downloaded_bytes") or 0
        percent = int((downloaded / total) * 100) if total else 0
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id]["message"] = f"下載音訊中 {percent}%"
                jobs[job_id]["progressPercent"] = min(30, int(percent * 0.3))

    ydl_opts = {
        "format": DOWNLOAD_FORMAT,
        "outtmpl": str(out_path.with_suffix(".%(ext)s")),
        "quiet": True,
        "noplaylist": True,
        "progress_hooks": [on_progress],
    }
    url = extract_video_url(video_id)
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded = Path(ydl.prepare_filename(info))
    duration = float(info.get("duration") or 0.0)
    return downloaded, duration


def transcribe_worker(job_id: str, video_id: str):
    def progress_pulse_worker(pulse_job_id: str, estimated_seconds: float):
        start_ts = time.time()
        # 辨識階段保底心跳進度：避免卡在 30% 直到第一個 seg 才跳動。
        while True:
            with jobs_lock:
                job = jobs.get(pulse_job_id)
                if not job:
                    return
                if job.get("status") in {"done", "error"}:
                    return
                if not str(job.get("message", "")).startswith("語音辨識中"):
                    return

                elapsed = max(0.0, time.time() - start_ts)
                ratio = min(1.0, elapsed / max(20.0, estimated_seconds))
                pulsed_progress = 30 + int(ratio * 62)  # 30 -> 92
                job["progressPercent"] = max(job.get("progressPercent", 30), min(92, pulsed_progress))

            time.sleep(0.8)

    audio_file = None
    try:
        with jobs_lock:
            jobs[job_id]["status"] = "running"
            jobs[job_id]["message"] = "下載音訊中"

        audio_prefix = DOWNLOAD_DIR / f"{job_id}"
        audio_file, audio_duration = download_audio(job_id, video_id, audio_prefix)

        with jobs_lock:
            jobs[job_id]["message"] = f"語音辨識中（外部 API: {OPENAI_TRANSCRIBE_MODEL}）"
            jobs[job_id]["progressPercent"] = max(jobs[job_id].get("progressPercent", 0), 30)

        estimated_transcribe_seconds = max(25.0, audio_duration * 1.2)
        threading.Thread(
            target=progress_pulse_worker,
            args=(job_id, estimated_transcribe_seconds),
            daemon=True,
        ).start()

        built_segments = transcribe_with_openai(audio_file, language="en")
        with jobs_lock:
            jobs[job_id]["segments"] = built_segments
            jobs[job_id]["processedSegments"] = len(built_segments)
            jobs[job_id]["progressPercent"] = max(jobs[job_id]["progressPercent"], 99)

        with jobs_lock:
            jobs[job_id]["status"] = "done"
            jobs[job_id]["message"] = "完成"
            jobs[job_id]["progressPercent"] = 100
    except Exception as exc:
        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(exc)
    finally:
        if audio_file and audio_file.exists():
            try:
                audio_file.unlink()
            except OSError:
                pass


@app.route("/")
def root():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify(
        {
            "ok": True,
            "service": "openai-transcribe-backend",
            "openai_key_configured": bool(OPENAI_API_KEY),
            "transcribe_model": OPENAI_TRANSCRIBE_MODEL,
        }
    )


@app.route("/api/transcribe/start", methods=["POST"])
def start_transcribe():
    payload = request.get_json(silent=True) or {}
    video_id = (payload.get("videoId") or "").strip()
    if not video_id:
        return jsonify({"error": "缺少 videoId"}), 400

    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {
            "status": "queued",
            "message": "排隊中",
            "progressPercent": 0,
            "segments": [],
            "processedSegments": 0,
            "error": "",
            "videoId": video_id,
        }

    thread = threading.Thread(target=transcribe_worker, args=(job_id, video_id), daemon=True)
    thread.start()

    return jsonify({"jobId": job_id})


@app.route("/api/transcribe/status/<job_id>", methods=["GET"])
def transcribe_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "找不到工作"}), 404
        return jsonify(job)


@app.route("/api/translate", methods=["POST"])
def translate_api():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    if not text:
        return jsonify({"error": "缺少 text"}), 400
    try:
        translated = translate_text_to_zh_tw(text)
        return jsonify({"translatedText": translated})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5500, debug=True)
