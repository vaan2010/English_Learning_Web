import os
import threading
import uuid
import json
import time
import base64
from urllib.parse import urlencode
from urllib.request import urlopen
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import google.generativeai as genai
from yt_dlp import YoutubeDL


BASE_DIR = Path(__file__).resolve().parent
DOWNLOAD_DIR = BASE_DIR / "tmp_audio"
DOWNLOAD_DIR.mkdir(exist_ok=True)
COOKIES_PATH = BASE_DIR / "yt_cookies.txt"

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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_TIMEOUT_SEC = int(os.getenv("GEMINI_TIMEOUT_SEC", "600"))
YTDLP_COOKIES_B64 = os.getenv("YTDLP_COOKIES_B64", "").strip()
YTDLP_COOKIES_TEXT = os.getenv("YTDLP_COOKIES_TEXT", "").strip()

jobs = {}
jobs_lock = threading.Lock()
translation_cache = {}
translation_lock = threading.Lock()


def setup_cookiefile_from_env() -> Path | None:
    """
    Allow cookie injection on cloud hosts (e.g. Render) where
    --cookies-from-browser is unavailable.
    """
    cookie_text = ""
    if YTDLP_COOKIES_B64:
        try:
            cookie_text = base64.b64decode(YTDLP_COOKIES_B64).decode("utf-8")
        except Exception:
            cookie_text = ""
    elif YTDLP_COOKIES_TEXT:
        cookie_text = YTDLP_COOKIES_TEXT

    if not cookie_text.strip():
        return None

    # Must be Netscape cookie format for yt-dlp cookiefile.
    COOKIES_PATH.write_text(cookie_text, encoding="utf-8")
    return COOKIES_PATH


COOKIEFILE = setup_cookiefile_from_env()
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def extract_video_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _extract_json_array(text: str) -> list:
    raw = (text or "").strip()
    if not raw:
        return []
    if raw.startswith("```"):
        raw = raw.strip("`")
        if "\n" in raw:
            raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
    start = raw.find("[")
    end = raw.rfind("]")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def transcribe_with_gemini(audio_path: Path, audio_duration: float, language: str = "en") -> list[dict]:
    if not GEMINI_API_KEY:
        raise RuntimeError("缺少 GEMINI_API_KEY，無法呼叫 Gemini API")

    prompt = (
        "You are a speech-to-text engine. "
        "Transcribe this audio and return ONLY a JSON array. "
        "Each item must be an object: {\"start\": number, \"end\": number, \"text\": string}. "
        "Use seconds for start/end. Keep segments in chronological order. "
        "Do not include markdown. "
        f"Target language: {language}. "
        f"Audio duration seconds: {max(0.0, float(audio_duration)):.3f}."
    )
    uploaded = genai.upload_file(path=str(audio_path))
    try:
        wait_started = time.time()
        while getattr(uploaded, "state", None) and uploaded.state.name == "PROCESSING":
            if time.time() - wait_started > GEMINI_TIMEOUT_SEC:
                raise RuntimeError("Gemini 檔案處理逾時")
            time.sleep(1.2)
            uploaded = genai.get_file(uploaded.name)

        if getattr(uploaded, "state", None) and uploaded.state.name == "FAILED":
            raise RuntimeError("Gemini 檔案處理失敗")

        model = genai.GenerativeModel(model_name=GEMINI_MODEL)
        response = model.generate_content(
            [prompt, uploaded],
            generation_config={
                "temperature": 0.0,
                "response_mime_type": "application/json",
            },
            request_options={"timeout": GEMINI_TIMEOUT_SEC},
        )
        text_parts = [response.text or ""]
    finally:
        try:
            if uploaded and getattr(uploaded, "name", None):
                genai.delete_file(uploaded.name)
        except Exception:
            pass

    parsed = _extract_json_array("\n".join(text_parts))

    segments = []
    for seg in parsed:
        if not isinstance(seg, dict):
            continue
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start + 1.0))
        if end < start:
            end = start + 0.5
        segments.append({"start": round(start, 3), "end": round(end, 3), "text": text})

    if segments:
        return segments

    # Fallback: no structured segment returned.
    plain_text = "\n".join(text_parts).strip()
    if plain_text:
        return [{"start": 0.0, "end": max(1.0, float(audio_duration or 3.0)), "text": plain_text}]
    return []


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

    url = extract_video_url(video_id)
    base_opts = {
        "outtmpl": str(out_path.with_suffix(".%(ext)s")),
        "quiet": True,
        "noplaylist": True,
        "progress_hooks": [on_progress],
        "extractor_args": {
            "youtube": {
                # Helps some cloud environments reduce bot checks.
                "player_client": ["android", "web"],
            }
        },
    }
    if COOKIEFILE and COOKIEFILE.exists():
        base_opts["cookiefile"] = str(COOKIEFILE)

    last_error = None
    for fmt in [DOWNLOAD_FORMAT, "bestaudio/best"]:
        ydl_opts = {**base_opts, "format": fmt}
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                downloaded = Path(ydl.prepare_filename(info))
            duration = float(info.get("duration") or 0.0)
            return downloaded, duration
        except Exception as exc:
            last_error = exc
            with jobs_lock:
                if job_id in jobs:
                    jobs[job_id]["message"] = f"音訊格式回退重試中（{fmt} 失敗）"

    raise RuntimeError(f"下載音訊失敗：{last_error}")


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
            jobs[job_id]["message"] = f"語音辨識中（外部 API: {GEMINI_MODEL}）"
            jobs[job_id]["progressPercent"] = max(jobs[job_id].get("progressPercent", 0), 30)

        estimated_transcribe_seconds = max(25.0, audio_duration * 1.2)
        threading.Thread(
            target=progress_pulse_worker,
            args=(job_id, estimated_transcribe_seconds),
            daemon=True,
        ).start()

        built_segments = transcribe_with_gemini(audio_file, audio_duration, language="en")
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
            err_text = str(exc)
            if "Sign in to confirm you’re not a bot" in err_text:
                err_text += (
                    " | 你目前在雲端環境，請在 Render 設定 YTDLP_COOKIES_B64（YouTube cookies 的 base64）"
                )
            jobs[job_id]["error"] = err_text
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
            "service": "gemini-transcribe-backend",
            "gemini_key_configured": bool(GEMINI_API_KEY),
            "transcribe_model": GEMINI_MODEL,
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
