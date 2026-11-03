import os
import re
import json
import time
import mimetypes
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, Response
from werkzeug.utils import secure_filename
import torch
import torchaudio
import numpy as np
import av
from einops import rearrange
from torch.nn import functional as F
from model.batfd_plus import BatfdPlus

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024  # 250MB max upload
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global model instance
MODEL = None
DEVICE = "cpu"
CACHE_FILE = "sample_cache.json"

def get_model():
    global MODEL
    if MODEL is None:
        ckpt_path = "ckpt/chronosync_ai.ckpt"
        if not os.path.exists(ckpt_path):
            ckpt_path = "ckpt/batfd_plus_default.ckpt"
        if os.path.exists(ckpt_path):
            print(f"[Model] Loading ChronoSync AI checkpoint from {ckpt_path}...")
            MODEL = BatfdPlus.load_from_checkpoint(ckpt_path)
            MODEL.eval()
            torch.set_num_threads(min(8, os.cpu_count() or 4))
            print("[Model] ChronoSync AI model loaded successfully into memory.")
        else:
            print(f"[Model] Warning: checkpoint {ckpt_path} not found.")
    return MODEL

def send_file_partial(path):
    """Serve media files supporting HTTP 206 Partial Content for video seeking."""
    range_header = request.headers.get("Range", None)
    if not range_header:
        return send_file(path)
        
    size = os.path.getsize(path)
    byte1, byte2 = 0, None
    
    m = re.search(r"(\d+)-(\d*)", range_header)
    if m:
        g = m.groups()
        byte1 = int(g[0])
        if g[1]:
            byte2 = int(g[1])
            
    length = size - byte1
    if byte2 is not None:
        length = byte2 + 1 - byte1
        
    with open(path, "rb") as f:
        f.seek(byte1)
        data = f.read(length)
        
    content_type, _ = mimetypes.guess_type(path)
    if not content_type:
        content_type = "video/mp4"
        
    rv = Response(data, 206, mimetype=content_type, direct_passthrough=True)
    rv.headers.add("Content-Range", f"bytes {byte1}-{byte1 + length - 1}/{size}")
    rv.headers.add("Accept-Ranges", "bytes")
    rv.headers.add("Content-Length", str(length))
    return rv

def run_forensic_analysis(video_path, target_frames=512, fps=25, sample_rate=16000):
    t_start = time.time()
    model = get_model()
    if model is None:
        raise RuntimeError("Model checkpoint not available on server.")
        
    container = av.open(video_path)
    video_stream = container.streams.video[0]
    actual_fps = float(video_stream.average_rate or fps)
    
    # 1. Decode video frames
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_rgb().to_ndarray())
        if len(frames) >= target_frames:
            break
            
    actual_frames = len(frames)
    duration_sec = actual_frames / actual_fps
    if actual_frames == 0:
        raise ValueError("Video contains no valid video frames.")
        
    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
    if actual_frames < target_frames:
        video = F.pad(video, [0, 0, 0, 0, 0, 0, 0, target_frames - actual_frames])
    else:
        video = video[:target_frames]
        
    video = F.interpolate(video, size=(96, 96), mode="bicubic")
    video = rearrange(video, "t c h w -> c t h w").unsqueeze(0)
    
    # 2. Extract and resample audio
    container.seek(0)
    audio_frames = []
    if len(container.streams.audio) > 0:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
        for frame in container.decode(audio=0):
            resampled = resampler.resample(frame)
            if resampled is not None:
                for r in resampled:
                    audio_frames.append(r.to_ndarray())
                    
    target_audio_samples = int(target_frames / fps * sample_rate)
    if audio_frames:
        audio = torch.from_numpy(np.concatenate(audio_frames, axis=1)[0]).float()
    else:
        audio = torch.zeros(target_audio_samples)
        
    if len(audio) < target_audio_samples:
        audio = F.pad(audio, (0, target_audio_samples - len(audio)))
    else:
        audio = audio[:target_audio_samples]
        
    ms = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=321, n_mels=64, hop_length=160)
    mel = ms(audio)
    spec = torch.log(mel + 0.01)
    if spec.shape[1] > 2048:
        spec = spec[:, :2048]
    elif spec.shape[1] < 2048:
        spec = F.pad(spec, (0, 2048 - spec.shape[1]))
    spec = spec.unsqueeze(0)
    
    # 3. Model Inference (BatfdPlus)
    with torch.no_grad():
        outputs = model(video, spec)
        fusion_bm = outputs[0]
        v_cla = outputs[15]
        a_cla = outputs[16]
        v_probs = torch.sigmoid(v_cla.squeeze()).cpu().numpy().tolist()
        a_probs = torch.sigmoid(a_cla.squeeze()).cpu().numpy().tolist()
        bm = fusion_bm[0].cpu().numpy()
        
    valid = min(actual_frames, target_frames)
    v_valid = v_probs[:valid]
    a_valid = a_probs[:valid]
    
    v_mean = float(np.mean(v_valid))
    v_max = float(np.max(v_valid))
    a_mean = float(np.mean(a_valid))
    a_max = float(np.max(a_valid))
    bm_max = float(bm[:, :valid].max())
    
    # Temporal boundary proposals
    proposals = []
    for d in range(1, bm.shape[0]):
        for s in range(valid):
            e = s + d
            if e <= valid:
                score = float(bm[d, s])
                if score > 0.35:
                    proposals.append({
                        "start_sec": round(s / actual_fps, 2),
                        "end_sec": round(e / actual_fps, 2),
                        "score": round(score, 4),
                        "duration_sec": round(d / actual_fps, 2)
                    })
    proposals.sort(key=lambda x: x["score"], reverse=True)
    
    # Downsampled timeline for smooth charting
    step = max(1, valid // 60)
    timeline = []
    for i in range(0, valid, step):
        t_sec = round(i / actual_fps, 2)
        v_p = round(float(v_probs[i]), 4)
        a_p = round(float(a_probs[i]), 4)
        frame_bm = float(bm[:, i].max())
        timeline.append({
            "frame": i,
            "time_sec": t_sec,
            "visual_score": v_p,
            "audio_score": a_p,
            "boundary_score": round(frame_bm, 4),
            "anomaly_risk": round(max(v_p, a_p, frame_bm), 4)
        })
        
    is_fake = (v_max > 0.5) or (bm_max > 0.5)
    overall_confidence = max(v_max, bm_max, a_max)
    
    return {
        "duration": round(duration_sec, 2),
        "frames": actual_frames,
        "fps": round(actual_fps, 1),
        "v_mean": round(v_mean, 4),
        "v_max": round(v_max, 4),
        "a_mean": round(a_mean, 4),
        "a_max": round(a_max, 4),
        "bm_max": round(bm_max, 4),
        "overall_confidence": round(overall_confidence * 100, 1),
        "verdict": "DEEPFAKE" if is_fake else "AUTHENTIC",
        "top_proposals": proposals[:6],
        "timeline": timeline,
        "inference_time_sec": round(time.time() - t_start, 2)
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/samples")
def get_samples():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                data = json.load(f)
            samples_list = []
            for sid, val in data.items():
                samples_list.append({
                    "id": sid,
                    "title": val.get("title", sid),
                    "type": val.get("type", "real"),
                    "expected": val.get("expected", "AUTHENTIC"),
                    "verdict": val.get("verdict", "AUTHENTIC"),
                    "confidence": val.get("overall_confidence", 0),
                    "duration": val.get("duration", 0),
                    "thumb_url": val.get("thumb_url", ""),
                    "video_url": val.get("video_url", "")
                })
            return jsonify({"status": "success", "samples": samples_list})
        except Exception as e:
            print("Error reading cache:", e)
            
    # Fallback default sample metadata
    fallback = [
        {"id": "real_1", "title": "Normal 01 (Authentic)", "type": "real", "expected": "AUTHENTIC", "thumb_url": "/api/sample-image/1.jpg", "video_url": "/api/sample-video/real/normal 01.mp4"},
        {"id": "fake_1", "title": "Deepfake 01 (Manipulated)", "type": "fake", "expected": "DEEPFAKE", "thumb_url": "/api/sample-image/1.jpg", "video_url": "/api/sample-video/fake/deepfake 01.mp4"},
        {"id": "real_2", "title": "Normal 02 (Authentic)", "type": "real", "expected": "AUTHENTIC", "thumb_url": "/api/sample-image/2.jpeg", "video_url": "/api/sample-video/real/normal 02.mp4"},
        {"id": "fake_2", "title": "Deepfake 02 (Manipulated)", "type": "fake", "expected": "DEEPFAKE", "thumb_url": "/api/sample-image/2.jpeg", "video_url": "/api/sample-video/fake/deepfake 02.mp4"},
        {"id": "real_3", "title": "Normal 03 (Authentic)", "type": "real", "expected": "AUTHENTIC", "thumb_url": "/api/sample-image/3.jpg", "video_url": "/api/sample-video/real/normal 03.mp4"},
        {"id": "fake_3", "title": "Deepfake 03 (Manipulated)", "type": "fake", "expected": "DEEPFAKE", "thumb_url": "/api/sample-image/3.jpg", "video_url": "/api/sample-video/fake/deepfake 03.mp4"},
        {"id": "real_4", "title": "Normal 04 (Authentic)", "type": "real", "expected": "AUTHENTIC", "thumb_url": "/api/sample-image/4.jpg", "video_url": "/api/sample-video/real/normal 04.mp4"},
        {"id": "fake_4", "title": "Deepfake 04 (Manipulated)", "type": "fake", "expected": "DEEPFAKE", "thumb_url": "/api/sample-image/4.jpg", "video_url": "/api/sample-video/fake/deepfake 04.mp4"},
        {"id": "real_5", "title": "Normal 05 (Authentic)", "type": "real", "expected": "AUTHENTIC", "thumb_url": "/api/sample-image/5.jpg", "video_url": "/api/sample-video/real/normal 05.MOV"},
        {"id": "fake_5", "title": "Deepfake 05 (Manipulated)", "type": "fake", "expected": "DEEPFAKE", "thumb_url": "/api/sample-image/5.jpg", "video_url": "/api/sample-video/fake/deepfake 05.mov"},
    ]
    return jsonify({"status": "success", "samples": fallback})

@app.route("/api/sample/<sample_id>")
def get_sample_detail(sample_id):
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                data = json.load(f)
            if sample_id in data:
                return jsonify({"status": "success", "data": data[sample_id]})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    return jsonify({"status": "error", "message": "Sample not found in cache"}), 404

@app.route("/api/sample-video/<category>/<path:filename>")
def serve_sample_video(category, filename):
    folder = "test_data/video" if category in ["real", "normal"] else "test_data/deepfake"
    path = os.path.join(folder, filename)
    if os.path.exists(path):
        return send_file_partial(path)
    # Check alternate variations (with space or underscore)
    for alt in [filename.replace(" ", "_"), filename.replace("_", " ")]:
        alt_path = os.path.join(folder, alt)
        if os.path.exists(alt_path):
            return send_file_partial(alt_path)
    return jsonify({"error": "Video not found"}), 404

@app.route("/api/sample-image/<filename>")
def serve_sample_image(filename):
    path = os.path.join("test_data/image", filename)
    if os.path.exists(path):
        return send_file(path)
    return jsonify({"error": "Image not found"}), 404

@app.route("/api/video/<filename>")
def serve_uploaded_video(filename):
    safe_name = secure_filename(filename)
    path = os.path.join(UPLOAD_FOLDER, safe_name)
    if os.path.exists(path):
        return send_file_partial(path)
    return jsonify({"error": "Video not found"}), 404

@app.route("/api/analyze-upload", methods=["POST"])
def analyze_upload():
    if "video" not in request.files:
        return jsonify({"status": "error", "message": "No video file provided"}), 400
        
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "Empty file name"}), 400
        
    filename = secure_filename(file.filename)
    timestamp = int(time.time())
    saved_filename = f"{timestamp}_{filename}"
    save_path = os.path.join(UPLOAD_FOLDER, saved_filename)
    file.save(save_path)
    
    try:
        results = run_forensic_analysis(save_path)
        results["file_name"] = filename
        results["video_url"] = f"/api/video/{saved_filename}"
        return jsonify({"status": "success", "data": results})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/analyze-sample-live", methods=["POST"])
def analyze_sample_live():
    data = request.get_json() or {}
    sample_id = data.get("sample_id")
    if not sample_id:
        return jsonify({"status": "error", "message": "Missing sample_id"}), 400
        
    category, num = sample_id.split("_")
    num_padded = num.zfill(2)
    ext = ".MOV" if (num == "5" and category == "real") else (".mov" if num == "5" else ".mp4")
    if category == "real":
        path = f"test_data/video/normal {num_padded}{ext}"
    else:
        path = f"test_data/deepfake/deepfake {num_padded}{ext}"
    
    if not os.path.exists(path):
        return jsonify({"status": "error", "message": "Sample file not found"}), 404
        
    try:
        results = run_forensic_analysis(path)
        results["file_name"] = os.path.basename(path)
        results["video_url"] = f"/api/sample-video/{category}/{os.path.basename(path)}"
        return jsonify({"status": "success", "data": results})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    print("[Server] Initializing ChronoSync AI Multimodal Forensic Server on port 5000...")
    get_model()
    app.run(host="0.0.0.0", port=5000, debug=False)
