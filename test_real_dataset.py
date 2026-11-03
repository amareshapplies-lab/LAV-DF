import os
import av
import torch
import torchaudio
import numpy as np
from einops import rearrange
from torch.nn import functional as F
from model.batfd_plus import BatfdPlus

def load_video_and_audio(video_path, target_frames=512, fps=25, sample_rate=16000):
    container = av.open(video_path)
    
    # Extract video frames
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_rgb().to_ndarray())
    if len(frames) == 0:
        raise ValueError(f"No video frames found in {video_path}")
    
    actual_frames = len(frames)
    duration_sec = actual_frames / fps
    
    # Video tensor: [T, C, H, W] in [0, 1]
    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
    
    # Pad or trim video to target_frames
    if actual_frames < target_frames:
        pad_size = target_frames - actual_frames
        video = F.pad(video, pad=[0, 0, 0, 0, 0, 0, 0, pad_size])
    else:
        video = video[:target_frames]
        
    # Resize to 96x96 and rearrange to [C, T, H, W]
    video = F.interpolate(video, size=(96, 96), mode="bicubic")
    video = rearrange(video, "t c h w -> c t h w")
    
    # Extract audio
    container.seek(0)
    audio_frames = []
    if len(container.streams.audio) > 0:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
        for frame in container.decode(audio=0):
            resampled = resampler.resample(frame)
            if resampled is not None:
                for r in resampled:
                    audio_frames.append(r.to_ndarray())
    
    target_audio_samples = int(target_frames / fps * sample_rate)  # 327680
    if len(audio_frames) > 0:
        audio_data = np.concatenate(audio_frames, axis=1)[0]
        audio = torch.from_numpy(audio_data).float()
    else:
        audio = torch.zeros(target_audio_samples)
        
    if len(audio) < target_audio_samples:
        audio = F.pad(audio, (0, target_audio_samples - len(audio)))
    else:
        audio = audio[:target_audio_samples]
        
    # Compute Log-Mel Spectrogram [64, 2048]
    ms = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=321, n_mels=64, hop_length=160
    )
    mel = ms(audio)
    spec = torch.log(mel + 0.01)
    if spec.shape[1] > 2048:
        spec = spec[:, :2048]
    elif spec.shape[1] < 2048:
        spec = F.pad(spec, (0, 2048 - spec.shape[1]))
        
    return video, spec, actual_frames, duration_sec

def evaluate_video(model, video_path):
    video, spec, actual_frames, duration_sec = load_video_and_audio(video_path)
    
    v_in = video.unsqueeze(0)  # [1, 3, 512, 96, 96]
    a_in = spec.unsqueeze(0)   # [1, 64, 2048]
    
    with torch.no_grad():
        outputs = model(v_in, a_in)
        fusion_bm_map = outputs[0]
        v_frame_cla = outputs[15]
        a_frame_cla = outputs[16]
        
        # Probabilities for visual and audio frames
        v_probs = torch.sigmoid(v_frame_cla.squeeze()).cpu().numpy()
        a_probs = torch.sigmoid(a_frame_cla.squeeze()).cpu().numpy()
        
        # Boundary proposals from fusion boundary map [1, 40, 512]
        bm = fusion_bm_map[0].cpu().numpy()
        
    valid_frames = min(actual_frames, 512)
    # Mask to valid duration
    v_score_valid = v_probs[:valid_frames].mean()
    v_score_max = v_probs[:valid_frames].max()
    a_score_valid = a_probs[:valid_frames].mean()
    a_score_max = a_probs[:valid_frames].max()
    
    # Boundary map score
    bm_max = float(bm[:, :valid_frames].max())
    
    # Top predicted temporal boundaries:
    # rows = duration (in time steps), cols = start frame
    proposals = []
    for d in range(1, bm.shape[0]):
        for start in range(valid_frames):
            end = start + d
            if end <= valid_frames:
                score = bm[d, start]
                proposals.append((score, start / 25.0, end / 25.0))
                
    proposals.sort(key=lambda x: x[0], reverse=True)
    top_proposal = proposals[0] if proposals else (0.0, 0.0, 0.0)
    
    # Combined forgery confidence
    overall_confidence = max(v_score_max, a_score_max, bm_max)
    verdict = "DEEPFAKE" if overall_confidence > 0.5 else "REAL"
    
    return {
        "file": os.path.basename(video_path),
        "duration": f"{duration_sec:.2f}s",
        "frames": actual_frames,
        "v_score_mean": f"{v_score_valid:.4f}",
        "v_score_max": f"{v_score_max:.4f}",
        "a_score_max": f"{a_score_max:.4f}",
        "bm_score_max": f"{bm_max:.4f}",
        "overall_confidence": f"{overall_confidence:.4f}",
        "top_segment": f"[{top_proposal[1]:.2f}s - {top_proposal[2]:.2f}s] (score: {top_proposal[0]:.4f})",
        "verdict": verdict
    }

def main():
    print("=" * 80)
    print("CHRONOSYNC AI PRETRAINED MODEL BENCHMARK ON TEST DATASET")
    print("=" * 80)
    
    ckpt_path = "ckpt/chronosync_ai.ckpt" if os.path.exists("ckpt/chronosync_ai.ckpt") else "ckpt/batfd_plus_default.ckpt"
    print(f"Loading pretrained ChronoSync AI model from: {ckpt_path} ...")
    model = BatfdPlus.load_from_checkpoint(ckpt_path)
    model.eval()
    print("Model loaded successfully!\n")
    
    test_files = [
        ("test_data/video/normal 01.mp4", "Real Ground Truth"),
        ("test_data/deepfake/deepfake 01.mp4", "Deepfake Ground Truth"),
        ("test_data/video/normal 02.mp4", "Real Ground Truth"),
        ("test_data/deepfake/deepfake 02.mp4", "Deepfake Ground Truth"),
        ("test_data/video/normal 03.mp4", "Real Ground Truth"),
        ("test_data/deepfake/deepfake 03.mp4", "Deepfake Ground Truth"),
    ]
    
    results = []
    for path, gt_label in test_files:
        if os.path.exists(path):
            print(f"Testing {path} ({gt_label})...")
            res = evaluate_video(model, path)
            res["ground_truth"] = gt_label
            results.append(res)
        else:
            print(f"File not found: {path}")
            
    print("\n" + "=" * 105)
    print(f"{'File':<18} | {'Ground Truth':<22} | {'Verdict':<10} | {'Confidence':<10} | {'Visual Max':<10} | {'Top Forged Segment':<25}")
    print("-" * 105)
    for r in results:
        print(f"{r['file']:<18} | {r['ground_truth']:<22} | {r['verdict']:<10} | {r['overall_confidence']:<10} | {r['v_score_max']:<10} | {r['top_segment']:<25}")
    print("=" * 105)

if __name__ == "__main__":
    main()
