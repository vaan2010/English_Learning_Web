import os
import threading
import uuid
import json
import time
from urllib.parse import urlencode
from urllib.request import urlopen
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from faster_whisper import WhisperModel
from yt_dlp import YoutubeDL


BASE_DIR = Path(__file__).resolve().parent
DOWNLOAD_DIR = BASE_DIR / "tmp_audio"
DOWNLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder=".", static_url_path="")

MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "3"))
VAD_FILTER = os.getenv("WHISPER_VAD_FILTER", "true").lower() == "true"
DOWNLOAD_FORMAT = os.getenv("YTDLP_FORMAT", "bestaudio[abr<=160]/bestaudio/best")

model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

jobs = {}
jobs_lock = threading.Lock()
translation_cache = {}
translation_lock = threading.Lock()


def extract_video_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


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
            jobs[job_id]["message"] = f"語音辨識中（模型 {MODEL_SIZE}）"
            jobs[job_id]["progressPercent"] = max(jobs[job_id].get("progressPercent", 0), 30)

        estimated_transcribe_seconds = max(25.0, audio_duration * 1.2)
        threading.Thread(
            target=progress_pulse_worker,
            args=(job_id, estimated_transcribe_seconds),
            daemon=True,
        ).start()

        segments, _ = model.transcribe(
            str(audio_file),
            vad_filter=VAD_FILTER,
            word_timestamps=False,
            beam_size=BEAM_SIZE,
        )

        built_segments = []
        estimated_total_segments = max(20, int(audio_duration / 2.8)) if audio_duration > 0 else 80
        for seg in segments:
            text = (seg.text or "").strip()
            if text:
                built_segments.append(
                    {
                        "start": round(float(seg.start), 3),
                        "end": round(float(seg.end), 3),
                        "text": text,
                    }
                )

            with jobs_lock:
                jobs[job_id]["segments"] = built_segments
                jobs[job_id]["processedSegments"] = len(built_segments)
                # 辨識進度：不依賴文字是否非空，只要辨識時間前進就更新，避免長時間卡在 30%。
                time_ratio = (
                    min(1.0, float(seg.end) / audio_duration) if audio_duration and audio_duration > 0 else 0.0
                )
                segment_ratio = min(1.0, len(built_segments) / estimated_total_segments)
                ratio = max(time_ratio, segment_ratio * 0.9)
                progress = 30 + int(ratio * 69)
                jobs[job_id]["progressPercent"] = max(jobs[job_id]["progressPercent"], min(99, progress))

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
    return jsonify({"ok": True, "service": "whisper-transcribe-backend"})


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
