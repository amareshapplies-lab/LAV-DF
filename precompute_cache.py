import os
import json
import time
import torch
import torchaudio
import numpy as np
import av
from einops import rearrange
from torch.nn import functional as F
from model.batfd_plus import BatfdPlus

def load_and_analyze(model, video_path, target_frames=512, fps=25, sample_rate=16000):
    t_start = time.time()
    container = av.open(video_path)
    
    # Video extraction
    frames = []
    video_stream = container.streams.video[0]
    actual_fps = float(video_stream.average_rate or fps)
    
    for frame in container.decode(video=0):
        frames.append(frame.to_rgb().to_ndarray())
        if len(frames) >= target_frames:
            break
            
    actual_frames = len(frames)
    duration_sec = actual_frames / actual_fps
    
    if actual_frames == 0:
        raise ValueError(f"No frames found in {video_path}")
        
    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
    if actual_frames < target_frames:
        video = F.pad(video, [0, 0, 0, 0, 0, 0, 0, target_frames - actual_frames])
    else:
        video = video[:target_frames]
        
    video = F.interpolate(video, size=(96, 96), mode="bicubic")
    video = rearrange(video, "t c h w -> c t h w").unsqueeze(0)
    
    # Audio extraction
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
    
    # Model forward pass
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
    
    # Top proposals
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
    
    # Sample timeline down to ~60 points max for smooth web charting
    step = max(1, valid // 60)
    timeline = []
    for i in range(0, valid, step):
        t_sec = round(i / actual_fps, 2)
        v_p = round(float(v_probs[i]), 4)
        a_p = round(float(a_probs[i]), 4)
        # Find maximum boundary score covering this frame
        frame_bm = float(bm[:, i].max())
        timeline.append({
            "frame": i,
            "time_sec": t_sec,
            "visual_score": v_p,
            "audio_score": a_p,
            "boundary_score": round(frame_bm, 4),
            "anomaly_risk": round(max(v_p, a_p, frame_bm), 4)
        })
        
    # Verdict logic (from original accuracy analysis)
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

if __name__ == "__main__":
    print("Precomputing sample cache with ChronoSync AI model...")
    ckpt_path = "ckpt/chronosync_ai.ckpt" if os.path.exists("ckpt/chronosync_ai.ckpt") else "ckpt/batfd_plus_default.ckpt"
    model = BatfdPlus.load_from_checkpoint(ckpt_path)
    model.eval()
    
    samples = [
        {"id": "real_1", "type": "real", "title": "Normal 01 (Authentic)", "video": "test_data/video/normal 01.mp4", "thumb": "test_data/image/1.jpg", "expected": "AUTHENTIC"},
        {"id": "fake_1", "type": "fake", "title": "Deepfake 01 (Manipulated)", "video": "test_data/deepfake/deepfake 01.mp4", "thumb": "test_data/image/1.jpg", "expected": "DEEPFAKE"},
        {"id": "real_2", "type": "real", "title": "Normal 02 (Authentic)", "video": "test_data/video/normal 02.mp4", "thumb": "test_data/image/2.jpeg", "expected": "AUTHENTIC"},
        {"id": "fake_2", "type": "fake", "title": "Deepfake 02 (Manipulated)", "video": "test_data/deepfake/deepfake 02.mp4", "thumb": "test_data/image/2.jpeg", "expected": "DEEPFAKE"},
        {"id": "real_3", "type": "real", "title": "Normal 03 (Authentic)", "video": "test_data/video/normal 03.mp4", "thumb": "test_data/image/3.jpg", "expected": "AUTHENTIC"},
        {"id": "fake_3", "type": "fake", "title": "Deepfake 03 (Manipulated)", "video": "test_data/deepfake/deepfake 03.mp4", "thumb": "test_data/image/3.jpg", "expected": "DEEPFAKE"},
        {"id": "real_4", "type": "real", "title": "Normal 04 (Authentic)", "video": "test_data/video/normal 04.mp4", "thumb": "test_data/image/4.jpg", "expected": "AUTHENTIC"},
        {"id": "fake_4", "type": "fake", "title": "Deepfake 04 (Manipulated)", "video": "test_data/deepfake/deepfake 04.mp4", "thumb": "test_data/image/4.jpg", "expected": "DEEPFAKE"},
        {"id": "real_5", "type": "real", "title": "Normal 05 (Authentic)", "video": "test_data/video/normal 05.MOV", "thumb": "test_data/image/5.jpg", "expected": "AUTHENTIC"},
        {"id": "fake_5", "type": "fake", "title": "Deepfake 05 (Manipulated)", "video": "test_data/deepfake/deepfake 05.mov", "thumb": "test_data/image/5.jpg", "expected": "DEEPFAKE"},
    ]
    
    cache = {}
    for s in samples:
        if os.path.exists(s["video"]):
            print(f"Analyzing {s['id']}: {s['video']}...")
            try:
                res = load_and_analyze(model, s["video"])
                res["title"] = s["title"]
                res["type"] = s["type"]
                res["expected"] = s["expected"]
                res["thumb_url"] = f"/api/sample-image/{os.path.basename(s['thumb'])}"
                res["video_url"] = f"/api/sample-video/{s['type']}/{os.path.basename(s['video'])}"
                cache[s["id"]] = res
                print(f"  -> {res['verdict']} ({res['overall_confidence']}%) in {res['inference_time_sec']}s")
            except Exception as e:
                print(f"  -> Error: {e}")
        else:
            print(f"File not found: {s['video']}")
            
    with open("sample_cache.json", "w") as f:
        json.dump(cache, f, indent=2)
    print("Saved sample_cache.json successfully!")
