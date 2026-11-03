import os
import av
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg
import subprocess
from einops import rearrange
from model.batfd_plus import BatfdPlus

def get_font(size):
    try:
        # Standard Windows fonts
        return ImageFont.truetype("arial.ttf", size)
    except:
        return ImageFont.load_default()

def get_bold_font(size):
    try:
        return ImageFont.truetype("arialbd.ttf", size)
    except:
        return get_font(size)

def load_video_and_predict(model, video_path, target_frames=512, fps=25, sample_rate=16000):
    container = av.open(video_path)
    
    raw_frames = []
    for frame in container.decode(video=0):
        raw_frames.append(frame.to_rgb().to_ndarray())
        
    actual_frames = len(raw_frames)
    if actual_frames == 0:
        raise ValueError(f"No frames in {video_path}")
        
    actual_fps = float(container.streams.video[0].average_rate or fps)
    duration_sec = actual_frames / actual_fps
    
    # Process for model
    video = torch.from_numpy(np.stack(raw_frames)).permute(0, 3, 1, 2).float() / 255.0
    if actual_frames < target_frames:
        pad_size = target_frames - actual_frames
        v_model = F.pad(video, pad=[0, 0, 0, 0, 0, 0, 0, pad_size])
    else:
        v_model = video[:target_frames]
        
    v_model = F.interpolate(v_model, size=(96, 96), mode="bicubic")
    v_model = rearrange(v_model, "t c h w -> c t h w").unsqueeze(0)
    
    # Audio
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
    if len(audio_frames) > 0:
        audio_data = np.concatenate(audio_frames, axis=1)[0]
        audio = torch.from_numpy(audio_data).float()
    else:
        audio = torch.zeros(target_audio_samples)
        
    if len(audio) < target_audio_samples:
        audio = F.pad(audio, (0, target_audio_samples - len(audio)))
    else:
        audio = audio[:target_audio_samples]
        
    ms = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=321, n_mels=64, hop_length=160)
    spec = torch.log(ms(audio) + 0.01)
    if spec.shape[1] > 2048:
        spec = spec[:, :2048]
    elif spec.shape[1] < 2048:
        spec = F.pad(spec, (0, 2048 - spec.shape[1]))
    spec_model = spec.unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(v_model, spec_model)
        fusion_bm = outputs[0]
        v_cla = outputs[15]
        a_cla = outputs[16]
        v_probs = torch.sigmoid(v_cla.squeeze()).cpu().numpy()
        a_probs = torch.sigmoid(a_cla.squeeze()).cpu().numpy()
        bm = fusion_bm[0].cpu().numpy()
        
    # Extract top forged intervals
    valid_frames = min(actual_frames, target_frames)
    proposals = []
    for d in range(1, bm.shape[0]):
        for start in range(valid_frames):
            end = start + d
            if end <= valid_frames:
                score = bm[d, start]
                if score > 0.35:
                    proposals.append({
                        "start_sec": start / actual_fps,
                        "end_sec": end / actual_fps,
                        "score": float(score)
                    })
    proposals.sort(key=lambda x: x["score"], reverse=True)
    
    return raw_frames, actual_fps, duration_sec, v_probs, a_probs, proposals

def draw_hud(frame_np, frame_idx, total_frames, fps, v_probs, a_probs, proposals, is_ground_truth_fake):
    img = Image.fromarray(frame_np).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    w, h = img.size
    curr_time = frame_idx / fps
    total_time = total_frames / fps
    
    # Map frame_idx to model's 512 timeline
    model_idx = min(int(frame_idx * (512 / total_frames)), 511)
    v_score = float(v_probs[model_idx])
    a_score = float(a_probs[model_idx])
    
    # Check if inside any top forged interval
    in_forged_interval = False
    active_proposal = None
    for p in proposals[:3]:
        if p["start_sec"] <= curr_time <= p["end_sec"]:
            in_forged_interval = True
            active_proposal = p
            break
            
    is_fake = (v_score > 0.5) or in_forged_interval or (a_score > 0.7)
    current_confidence = max(v_score, a_score, active_proposal["score"] if active_proposal else 0.0)
    if not is_fake:
        display_conf = (1.0 - current_confidence) * 100
        status_text = "AUTHENTIC / REAL"
        badge_bg = (46, 125, 50, 230)      # Rich Green
        accent_color = (0, 230, 118, 255)
    else:
        display_conf = current_confidence * 100
        status_text = "DEEPFAKE DETECTED"
        badge_bg = (211, 47, 47, 230)      # Crimson Red
        accent_color = (255, 45, 85, 255)
        
    # Fonts
    font_title = get_bold_font(18)
    font_badge = get_bold_font(22)
    font_stat = get_bold_font(16)
    font_small = get_font(13)
    
    # 1. Top Glassmorphic Navigation Bar
    draw.rectangle([0, 0, w, 55], fill=(15, 18, 26, 210))
    draw.line([(0, 55), (w, 55)], fill=(60, 70, 90, 255), width=1)
    
    draw.text((25, 16), "CHRONOSYNC AI • AUDIO-VISUAL FORENSIC MONITOR", font=font_title, fill=(255, 255, 255, 255))
    gt_tag = "[ GROUND TRUTH: DEEPFAKE ]" if is_ground_truth_fake else "[ GROUND TRUTH: REAL ]"
    gt_color = (255, 82, 82, 255) if is_ground_truth_fake else (105, 240, 174, 255)
    draw.text((430, 18), gt_tag, font=font_small, fill=gt_color)
    
    time_str = f"TIME: {curr_time:05.2f}s / {total_time:05.2f}s  |  FRAME: {frame_idx+1:03d}/{total_frames:03d}  |  FPS: {fps:.1f}"
    draw.text((w - 420, 18), time_str, font=font_stat, fill=(200, 215, 230, 255))
    
    # 2. Main Live Verdict Card (Top-Left)
    card_x, card_y, card_w, card_h = 25, 75, 400, 165
    draw.rounded_rectangle([card_x, card_y, card_x + card_w, card_y + card_h], radius=10, fill=(18, 22, 34, 225), outline=(50, 65, 85, 255), width=2)
    
    # Status Badge
    draw.rounded_rectangle([card_x + 15, card_y + 15, card_x + 270, card_y + 55], radius=6, fill=badge_bg)
    draw.text((card_x + 25, card_y + 22), status_text, font=font_badge, fill=(255, 255, 255, 255))
    
    # Confidence Score Percentage
    conf_text = f"Confidence: {display_conf:.1f}%"
    draw.text((card_x + 15, card_y + 68), conf_text, font=font_stat, fill=accent_color)
    
    # Visual Bar
    draw.text((card_x + 15, card_y + 98), f"Visual Score:  {v_score*100:.1f}%", font=font_small, fill=(220, 220, 220, 255))
    bar_bg = [card_x + 140, card_y + 101, card_x + 380, card_y + 111]
    draw.rounded_rectangle(bar_bg, radius=3, fill=(40, 48, 65, 255))
    fill_w = int((card_x + 380 - (card_x + 140)) * min(v_score, 1.0))
    if fill_w > 0:
        bar_fill = (255, 82, 82, 255) if v_score > 0.5 else (0, 230, 118, 255)
        draw.rounded_rectangle([card_x + 140, card_y + 101, card_x + 140 + fill_w, card_y + 111], radius=3, fill=bar_fill)
        
    # Audio Bar
    draw.text((card_x + 15, card_y + 128), f"Audio Score:   {a_score*100:.1f}%", font=font_small, fill=(220, 220, 220, 255))
    bar_bg_a = [card_x + 140, card_y + 131, card_x + 380, card_y + 141]
    draw.rounded_rectangle(bar_bg_a, radius=3, fill=(40, 48, 65, 255))
    fill_w_a = int((card_x + 380 - (card_x + 140)) * min(a_score, 1.0))
    if fill_w_a > 0:
        bar_fill_a = (255, 82, 82, 255) if a_score > 0.5 else (0, 230, 118, 255)
        draw.rounded_rectangle([card_x + 140, card_y + 131, card_x + 140 + fill_w_a, card_y + 141], radius=3, fill=bar_fill_a)

    # 3. Active Segment Warning Banner
    if in_forged_interval and active_proposal is not None:
        banner_w, banner_h = 500, 45
        bx = (w - banner_w) // 2
        by = h - 130
        draw.rounded_rectangle([bx, by, bx + banner_w, by + banner_h], radius=8, fill=(180, 20, 20, 230), outline=(255, 50, 50, 255), width=2)
        alert_msg = f"⚠ FORGED INTERVAL: [{active_proposal['start_sec']:.2f}s - {active_proposal['end_sec']:.2f}s] (Conf: {active_proposal['score']*100:.1f}%)"
        draw.text((bx + 20, by + 12), alert_msg, font=font_stat, fill=(255, 255, 255, 255))
        
        # Flashing Border
        draw.rectangle([0, 0, w-1, h-1], outline=(255, 30, 30, 180), width=6)

    # 4. Bottom Interactive Scrubber Bar
    tl_y = h - 60
    draw.rectangle([0, tl_y - 10, w, h], fill=(15, 18, 26, 220))
    draw.text((25, tl_y + 8), "TIMELINE", font=font_stat, fill=(180, 190, 210, 255))
    
    tl_start_x = 120
    tl_end_x = w - 40
    tl_w = tl_end_x - tl_start_x
    
    # Base track
    draw.rounded_rectangle([tl_start_x, tl_y + 12, tl_end_x, tl_y + 24], radius=6, fill=(35, 42, 55, 255))
    
    # Mark forged periods on the track
    for p in proposals[:3]:
        p_x1 = tl_start_x + int((p["start_sec"] / total_time) * tl_w)
        p_x2 = tl_start_x + int((p["end_sec"] / total_time) * tl_w)
        p_x2 = max(p_x1 + 4, p_x2)
        draw.rounded_rectangle([p_x1, tl_y + 12, min(p_x2, tl_end_x), tl_y + 24], radius=4, fill=(255, 50, 50, 230))
        
    # Moving Head
    head_x = tl_start_x + int((curr_time / total_time) * tl_w)
    draw.ellipse([head_x - 7, tl_y + 6, head_x + 7, tl_y + 30], fill=(0, 230, 255, 255), outline=(255, 255, 255, 255), width=2)
    
    # Composite
    out = Image.alpha_composite(img, overlay)
    return np.array(out.convert("RGB"))

def generate_video(model, input_path, output_path, is_ground_truth_fake=True):
    print(f"\nProcessing {input_path} -> {output_path}...")
    raw_frames, fps, duration_sec, v_probs, a_probs, proposals = load_video_and_predict(model, input_path)
    
    temp_video = "temp_render.mp4"
    if os.path.exists(temp_video):
        os.remove(temp_video)
        
    h, w, _ = raw_frames[0].shape
    print(f"Total frames: {len(raw_frames)}, Resolution: {w}x{h}, FPS: {fps:.1f}")
    
    # Write annotated video
    out_container = av.open(temp_video, mode='w')
    stream = out_container.add_stream('libx264', rate=int(round(fps)))
    stream.width = w
    stream.height = h
    stream.pix_fmt = 'yuv420p'
    
    for idx, f in enumerate(raw_frames):
        annotated_frame = draw_hud(f, idx, len(raw_frames), fps, v_probs, a_probs, proposals, is_ground_truth_fake)
        frame_av = av.VideoFrame.from_ndarray(annotated_frame, format='rgb24')
        for packet in stream.encode(frame_av):
            out_container.mux(packet)
            
    for packet in stream.encode():
        out_container.mux(packet)
    out_container.close()
    
    # Remux with original audio using ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    if os.path.exists(output_path):
        os.remove(output_path)
        
    cmd = [
        ffmpeg_exe, "-y",
        "-i", temp_video,
        "-i", input_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-shortest",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    if os.path.exists(temp_video):
        os.remove(temp_video)
        
    print(f"Successfully created: {output_path}")

def main():
    os.makedirs("output_videos", exist_ok=True)
    
    ckpt_path = "ckpt/chronosync_ai.ckpt" if os.path.exists("ckpt/chronosync_ai.ckpt") else "ckpt/batfd_plus_default.ckpt"
    print(f"Loading ChronoSync AI model from {ckpt_path}...")
    model = BatfdPlus.load_from_checkpoint(ckpt_path)
    model.eval()
    
    # 1. Annotate Deepfake Video
    generate_video(model, "test_data/deepfake/deepfake 01.mp4", "output_videos/annotated_deepfake_01.mp4", is_ground_truth_fake=True)
    
    # 2. Annotate Real Video
    generate_video(model, "test_data/video/normal 01.mp4", "output_videos/annotated_normal_01.mp4", is_ground_truth_fake=False)
    
    print("\nAll videos rendered and saved in output_videos/ directory!")

if __name__ == "__main__":
    main()
