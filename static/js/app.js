/**
 * ChronoSync AI - Multimodal Deepfake Detector - Frontend Controller
 * Minimal, Aesthetic, Pinterest-inspired Interactive Logic
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("video-file-input");
    const filePreviewCard = document.getElementById("file-preview-card");
    const previewFilename = document.getElementById("preview-filename");
    const previewFilesize = document.getElementById("preview-filesize");
    const cancelUploadBtn = document.getElementById("cancel-upload-btn");
    const startAnalysisBtn = document.getElementById("start-analysis-btn");
    
    // Progress UI
    const progressCard = document.getElementById("analysis-progress-card");
    const progressTitle = document.getElementById("progress-status-title");
    const progressDesc = document.getElementById("progress-status-desc");
    const progressBarFill = document.getElementById("progress-bar-fill");
    
    // Results UI
    const resultsSection = document.getElementById("results-section");
    const verdictBannerCard = document.getElementById("verdict-banner-card");
    const verdictBadge = document.getElementById("verdict-badge");
    const verdictText = document.getElementById("verdict-text");
    const verdictHeadline = document.getElementById("verdict-headline");
    const verdictSummary = document.getElementById("verdict-summary");
    const gaugeBar = document.getElementById("gauge-bar");
    const gaugePercentage = document.getElementById("gauge-percentage");
    
    // Meta stats
    const metaDuration = document.getElementById("meta-duration");
    const metaFrames = document.getElementById("meta-frames");
    const metaLatency = document.getElementById("meta-latency");
    
    // Video Player
    const inspectionVideo = document.getElementById("inspection-video");
    const playPauseBtn = document.getElementById("play-pause-btn");
    const playIcon = document.getElementById("play-icon");
    const seekSlider = document.getElementById("video-seek-slider");
    const muteBtn = document.getElementById("mute-btn");
    const volumeIcon = document.getElementById("volume-icon");
    const playerTimeDisplay = document.getElementById("player-time-display");
    
    // Timeline Canvas
    const timelineCanvas = document.getElementById("timeline-canvas");
    const timelineScrubber = document.getElementById("timeline-scrubber");
    const proposalsContainer = document.getElementById("proposals-container");
    
    // Modality Metrics
    const modVMax = document.getElementById("mod-v-max");
    const modVMean = document.getElementById("mod-v-mean");
    const modAMax = document.getElementById("mod-a-max");
    const modAMean = document.getElementById("mod-a-mean");
    const modBmMax = document.getElementById("mod-bm-max");
    const modProposalsCount = document.getElementById("mod-proposals-count");
    
    // Gallery & Reset Buttons
    const sampleCardsGrid = document.getElementById("sample-cards-grid");
    const resetBtn = document.getElementById("reset-btn");
    const reanalyzeBtn = document.getElementById("reanalyze-btn");
    const exportReportBtn = document.getElementById("export-report-btn");
    
    // App State
    let currentFile = null;
    let currentAnalysisData = null;
    let currentSampleId = null;
    let progressTimer = null;
    
    // -------------------------------------------------------------------------
    // 1. File Upload & Drag-and-Drop Handlers
    // -------------------------------------------------------------------------
    dropzone.addEventListener("click", () => fileInput.click());
    
    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("drag-over");
    });
    
    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("drag-over");
    });
    
    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("drag-over");
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileSelection(e.dataTransfer.files[0]);
        }
    });
    
    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });
    
    function handleFileSelection(file) {
        if (!file.type.startsWith("video/") && !file.name.match(/\.(mp4|mov|avi|mkv|webm)$/i)) {
            alert("Please upload a valid video file (.mp4, .mov, .webm, .avi)");
            return;
        }
        currentFile = file;
        currentSampleId = null;
        
        previewFilename.textContent = file.name;
        previewFilesize.textContent = (file.size / (1024 * 1024)).toFixed(2) + " MB";
        
        dropzone.classList.add("hidden");
        filePreviewCard.classList.remove("hidden");
        progressCard.classList.add("hidden");
        resultsSection.classList.add("hidden");
    }
    
    cancelUploadBtn.addEventListener("click", () => {
        currentFile = null;
        fileInput.value = "";
        filePreviewCard.classList.add("hidden");
        dropzone.classList.remove("hidden");
    });
    
    startAnalysisBtn.addEventListener("click", () => {
        if (!currentFile) return;
        runUploadAnalysis(currentFile);
    });
    
    // -------------------------------------------------------------------------
    // 2. Analysis Execution & Progress Pipeline
    // -------------------------------------------------------------------------
    function startProgressAnimation() {
        filePreviewCard.classList.add("hidden");
        progressCard.classList.remove("hidden");
        resultsSection.classList.add("hidden");
        
        const steps = [
            { title: "Decompressing Video Stream...", desc: "Sampling 25fps frames & extracting spatiotemporal bounding volumes", step: 1, pct: 25 },
            { title: "Synthesizing Log-Mel Spectrogram...", desc: "Computing 64-band acoustic harmonics at 16,000 Hz", step: 2, pct: 50 },
            { title: "Executing BA-TFD Neural Attention...", desc: "Synchronizing visual articulatory motion against audio phonemes", step: 3, pct: 75 },
            { title: "Synthesizing Boundary Proposals...", desc: "Filtering temporal anomaly proposals & computing confidence", step: 4, pct: 90 },
        ];
        
        let stepIdx = 0;
        progressBarFill.style.width = "10%";
        
        progressTimer = setInterval(() => {
            if (stepIdx < steps.length) {
                const s = steps[stepIdx];
                progressTitle.textContent = s.title;
                progressDesc.textContent = s.desc;
                progressBarFill.style.width = s.pct + "%";
                
                document.querySelectorAll(".step-badge").forEach((b, idx) => {
                    b.classList.toggle("active", idx <= s.step - 1);
                });
                stepIdx++;
            }
        }, 1200);
    }
    
    function stopProgressAnimation() {
        if (progressTimer) clearInterval(progressTimer);
        progressBarFill.style.width = "100%";
        progressCard.classList.add("hidden");
    }
    
    async function runUploadAnalysis(file) {
        startProgressAnimation();
        const formData = new FormData();
        formData.append("video", file);
        
        try {
            const resp = await fetch("/api/analyze-upload", {
                method: "POST",
                body: formData
            });
            const result = await resp.json();
            stopProgressAnimation();
            
            if (result.status === "success") {
                renderResults(result.data);
            } else {
                alert("Analysis failed: " + (result.message || "Unknown error"));
                filePreviewCard.classList.remove("hidden");
            }
        } catch (err) {
            stopProgressAnimation();
            alert("Network or inference error: " + err.message);
            filePreviewCard.classList.remove("hidden");
        }
    }
    
    async function runSampleAnalysis(sampleId, isLive = false) {
        startProgressAnimation();
        currentSampleId = sampleId;
        currentFile = null;
        
        try {
            let resp;
            if (isLive) {
                resp = await fetch("/api/analyze-sample-live", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ sample_id: sampleId })
                });
            } else {
                resp = await fetch(`/api/sample/${sampleId}`);
            }
            
            const result = await resp.json();
            stopProgressAnimation();
            
            if (result.status === "success") {
                renderResults(result.data);
            } else {
                alert("Could not load sample: " + (result.message || "Unknown error"));
                dropzone.classList.remove("hidden");
            }
        } catch (err) {
            stopProgressAnimation();
            alert("Sample loading error: " + err.message);
            dropzone.classList.remove("hidden");
        }
    }
    
    // -------------------------------------------------------------------------
    // 3. Render Forensic Results Showcase
    // -------------------------------------------------------------------------
    function renderResults(data) {
        currentAnalysisData = data;
        dropzone.classList.add("hidden");
        filePreviewCard.classList.add("hidden");
        progressCard.classList.add("hidden");
        resultsSection.classList.remove("hidden");
        
        // 1. Verdict & Gauge
        const isDeepfake = (data.verdict === "DEEPFAKE");
        if (isDeepfake) {
            verdictBadge.className = "verdict-pill-badge deepfake";
            verdictText.textContent = "SYNTHETIC DEEPFAKE";
            verdictHeadline.textContent = "Multi-Modal Splicing / Forgery Flagged";
            verdictSummary.textContent = `Acoustic or visual temporal boundaries exhibit unnatural synthetic artifacts. The neural model detected ${data.top_proposals ? data.top_proposals.length : 0} suspect temporal segments.`;
            gaugeBar.style.stroke = "#E05353";
        } else {
            verdictBadge.className = "verdict-pill-badge authentic";
            verdictText.textContent = "AUTHENTIC VIDEO";
            verdictHeadline.textContent = "Natural Temporal & Audio-Visual Coherence";
            verdictSummary.textContent = "Continuous spatiotemporal lip-sync motions correlate seamlessly with acoustic phonemes. No synthetic alteration boundaries exceeded threshold.";
            gaugeBar.style.stroke = "#40916C";
        }
        
        // Animate circular gauge
        const conf = data.overall_confidence || 0;
        gaugePercentage.textContent = conf.toFixed(1) + "%";
        // circumference of r=50 circle is 2 * PI * 50 ≈ 314.15
        const dashOffset = 314.15 - (314.15 * (conf / 100));
        gaugeBar.style.strokeDashoffset = dashOffset;
        
        // 2. Meta Stats
        metaDuration.textContent = (data.duration || 0).toFixed(2) + "s";
        metaFrames.textContent = data.frames || 0;
        metaLatency.textContent = (data.inference_time_sec || 0.85).toFixed(2) + "s";
        
        // 3. Modality Breakdown
        modVMax.textContent = ((data.v_max || 0) * 100).toFixed(1) + "%";
        modVMean.textContent = ((data.v_mean || 0) * 100).toFixed(2) + "%";
        modAMax.textContent = ((data.a_max || 0) * 100).toFixed(1) + "%";
        modAMean.textContent = ((data.a_mean || 0) * 100).toFixed(2) + "%";
        modBmMax.textContent = ((data.bm_max || 0) * 100).toFixed(1) + "%";
        modProposalsCount.textContent = (data.top_proposals ? data.top_proposals.length : 0);
        
        // 4. Video Player Setup
        if (data.video_url) {
            inspectionVideo.src = data.video_url;
            inspectionVideo.load();
        }
        
        // 5. Proposals list
        proposalsContainer.innerHTML = "";
        if (data.top_proposals && data.top_proposals.length > 0) {
            data.top_proposals.forEach((p, idx) => {
                const chip = document.createElement("div");
                chip.className = "proposal-chip";
                chip.innerHTML = `
                    <span>Seg #${idx + 1}: [${p.start_sec}s - ${p.end_sec}s]</span>
                    <span class="conf-tag">${(p.score * 100).toFixed(1)}%</span>
                `;
                chip.addEventListener("click", () => {
                    inspectionVideo.currentTime = p.start_sec;
                    inspectionVideo.play();
                });
                proposalsContainer.appendChild(chip);
            });
        } else {
            proposalsContainer.innerHTML = `<span style="font-size: 0.85rem; color: var(--text-muted); font-style: italic;">No anomalous boundary segments detected.</span>`;
        }
        
        // 6. Draw Canvas Timeline
        drawTimeline(data);
        
        // Scroll smoothly to results
        resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
        if (window.lucide) lucide.createIcons();
    }
    
    // -------------------------------------------------------------------------
    // 4. Video Player Controls & Timeline Sync
    // -------------------------------------------------------------------------
    playPauseBtn.addEventListener("click", () => {
        if (inspectionVideo.paused || inspectionVideo.ended) {
            inspectionVideo.play();
        } else {
            inspectionVideo.pause();
        }
    });
    
    inspectionVideo.addEventListener("play", () => {
        playIcon.setAttribute("data-lucide", "pause");
        if (window.lucide) lucide.createIcons();
    });
    
    inspectionVideo.addEventListener("pause", () => {
        playIcon.setAttribute("data-lucide", "play");
        if (window.lucide) lucide.createIcons();
    });
    
    inspectionVideo.addEventListener("timeupdate", () => {
        const cur = inspectionVideo.currentTime;
        const dur = inspectionVideo.duration || 1;
        
        // Update time display
        const m = Math.floor(cur / 60).toString().padStart(2, "0");
        const s = (cur % 60).toFixed(2).padStart(5, "0");
        playerTimeDisplay.textContent = `${m}:${s}`;
        
        // Update slider
        seekSlider.value = (cur / dur) * 100;
        
        // Update timeline scrubber line position
        const pct = cur / dur;
        timelineScrubber.style.left = `${pct * 100}%`;
    });
    
    seekSlider.addEventListener("input", (e) => {
        const pct = e.target.value / 100;
        if (inspectionVideo.duration) {
            inspectionVideo.currentTime = pct * inspectionVideo.duration;
        }
    });
    
    muteBtn.addEventListener("click", () => {
        inspectionVideo.muted = !inspectionVideo.muted;
        volumeIcon.setAttribute("data-lucide", inspectionVideo.muted ? "volume-x" : "volume-2");
        if (window.lucide) lucide.createIcons();
    });
    
    // -------------------------------------------------------------------------
    // 5. Interactive Canvas Anomaly Heatmap
    // -------------------------------------------------------------------------
    function drawTimeline(data) {
        const canvas = timelineCanvas;
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        
        const ctx = canvas.getContext("2d");
        ctx.scale(dpr, dpr);
        
        const w = rect.width;
        const h = rect.height;
        const padding = { top: 15, bottom: 25, left: 20, right: 20 };
        const plotW = w - padding.left - padding.right;
        const plotH = h - padding.top - padding.bottom;
        
        // Background
        ctx.fillStyle = "#FAF8F5";
        ctx.fillRect(0, 0, w, h);
        
        // Grid lines
        ctx.strokeStyle = "rgba(28, 25, 23, 0.06)";
        ctx.lineWidth = 1;
        [0.25, 0.5, 0.75, 1.0].forEach(level => {
            const y = padding.top + plotH * (1 - level);
            ctx.beginPath();
            ctx.moveTo(padding.left, y);
            ctx.lineTo(w - padding.right, y);
            ctx.stroke();
        });
        
        // Threshold Line at 0.5
        const threshY = padding.top + plotH * 0.5;
        ctx.strokeStyle = "rgba(200, 90, 50, 0.35)";
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(padding.left, threshY);
        ctx.lineTo(w - padding.right, threshY);
        ctx.stroke();
        ctx.setLineDash([]);
        
        ctx.fillStyle = "rgba(200, 90, 50, 0.6)";
        ctx.font = "9px 'Space Mono', monospace";
        ctx.fillText("50% Splicing Threshold", padding.left + 5, threshY - 4);
        
        const timeline = data.timeline || [];
        if (timeline.length < 2) return;
        
        const maxTime = data.duration || timeline[timeline.length - 1].time_sec || 1;
        
        // Draw Flagged Boundary Regions (shaded red zones)
        if (data.top_proposals) {
            data.top_proposals.forEach(p => {
                const x1 = padding.left + (p.start_sec / maxTime) * plotW;
                const x2 = padding.left + (p.end_sec / maxTime) * plotW;
                ctx.fillStyle = "rgba(224, 83, 83, 0.12)";
                ctx.fillRect(x1, padding.top, Math.max(x2 - x1, 4), plotH);
                
                ctx.strokeStyle = "rgba(224, 83, 83, 0.4)";
                ctx.lineWidth = 1;
                ctx.strokeRect(x1, padding.top, Math.max(x2 - x1, 4), plotH);
            });
        }
        
        // Function to plot a curve
        function plotLine(color, key, fillGradient = false) {
            ctx.beginPath();
            timeline.forEach((pt, i) => {
                const x = padding.left + (pt.time_sec / maxTime) * plotW;
                const val = pt[key] || 0;
                const y = padding.top + plotH * (1 - Math.min(Math.max(val, 0), 1));
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            
            if (fillGradient) {
                ctx.lineTo(padding.left + plotW, padding.top + plotH);
                ctx.lineTo(padding.left, padding.top + plotH);
                ctx.closePath();
                const grad = ctx.createLinearGradient(0, padding.top, 0, padding.top + plotH);
                grad.addColorStop(0, "rgba(42, 157, 143, 0.2)");
                grad.addColorStop(1, "rgba(42, 157, 143, 0.0)");
                ctx.fillStyle = grad;
                ctx.fill();
            } else {
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.stroke();
            }
        }
        
        // Plot boundary fill, then lines
        plotLine(null, "boundary_score", true);
        plotLine("#2A9D8F", "boundary_score", false);
        plotLine("#E76F51", "audio_score", false);
        plotLine("#E05353", "visual_score", false);
        
        // Time labels at bottom
        ctx.fillStyle = "#A8A29E";
        ctx.font = "10px 'Space Mono', monospace";
        ctx.textAlign = "center";
        
        const tickCount = 6;
        for (let i = 0; i <= tickCount; i++) {
            const t = (maxTime * (i / tickCount)).toFixed(1) + "s";
            const x = padding.left + (plotW * (i / tickCount));
            ctx.fillText(t, x, h - 8);
        }
    }
    
    // Canvas Click / Scrubbing
    timelineCanvas.addEventListener("click", (e) => {
        if (!currentAnalysisData) return;
        const rect = timelineCanvas.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const padding = { left: 20, right: 20 };
        const plotW = rect.width - padding.left - padding.right;
        
        let pct = (clickX - padding.left) / plotW;
        pct = Math.max(0, Math.min(pct, 1));
        
        const maxTime = currentAnalysisData.duration || 1;
        const targetTime = pct * maxTime;
        
        if (inspectionVideo.duration) {
            inspectionVideo.currentTime = targetTime;
            timelineScrubber.style.left = `${pct * 100}%`;
        }
    });
    
    // -------------------------------------------------------------------------
    // 6. Curated Benchmark Gallery Loader
    // -------------------------------------------------------------------------
    async function loadSampleGallery() {
        try {
            const resp = await fetch("/api/samples");
            const data = await resp.json();
            if (data.status === "success" && data.samples) {
                renderSampleCards(data.samples);
            }
        } catch (err) {
            console.error("Error loading sample gallery:", err);
        }
    }
    
    function renderSampleCards(samples) {
        sampleCardsGrid.innerHTML = "";
        samples.forEach(s => {
            const isFake = (s.expected === "DEEPFAKE" || s.type === "fake");
            const badgeClass = isFake ? "deepfake" : "authentic";
            const badgeLabel = isFake ? "Deepfake" : "Authentic";
            
            const card = document.createElement("div");
            card.className = "sample-card";
            card.innerHTML = `
                <div class="sample-card-media">
                    <img src="${s.thumb_url || '/api/sample-image/1.jpg'}" alt="${s.title}" class="sample-card-img" loading="lazy">
                    <span class="sample-badge-floating ${badgeClass}">${badgeLabel}</span>
                    <span class="sample-duration-tag">${s.duration ? s.duration.toFixed(1) + 's' : 'Sample'}</span>
                </div>
                <div class="sample-card-body">
                    <h4 class="sample-card-title">${s.title}</h4>
                    <div class="sample-card-meta">
                        <span>Confidence: <strong>${s.confidence ? s.confidence.toFixed(1) + '%' : 'Verified'}</strong></span>
                        <span class="sample-action-cta">Analyze <i data-lucide="arrow-right" style="width: 14px; height: 14px;"></i></span>
                    </div>
                </div>
            `;
            
            card.addEventListener("click", () => {
                runSampleAnalysis(s.id);
            });
            sampleCardsGrid.appendChild(card);
        });
        if (window.lucide) lucide.createIcons();
    }
    
    // Reset Button
    resetBtn.addEventListener("click", () => {
        resultsSection.classList.add("hidden");
        dropzone.classList.remove("hidden");
        filePreviewCard.classList.add("hidden");
        progressCard.classList.add("hidden");
        currentFile = null;
        currentAnalysisData = null;
        fileInput.value = "";
        dropzone.scrollIntoView({ behavior: "smooth" });
    });
    
    // Re-analyze Button (forces live forward pass on sample or file)
    reanalyzeBtn.addEventListener("click", () => {
        if (currentSampleId) {
            runSampleAnalysis(currentSampleId, true);
        } else if (currentFile) {
            runUploadAnalysis(currentFile);
        }
    });
    
    // Export Report
    exportReportBtn.addEventListener("click", () => {
        if (!currentAnalysisData) return;
        const d = currentAnalysisData;
        const text = `=====================================================
CHRONOSYNC AI • MULTIMODAL DEEPFAKE INSPECTION REPORT
=====================================================
Date: ${new Date().toLocaleString()}
File: ${d.file_name || currentSampleId || "Uploaded Clip"}
Duration: ${d.duration}s (${d.frames} frames, ${d.fps} fps)

VERDICT: ${d.verdict}
CONFIDENCE SCORE: ${d.overall_confidence}%

MODALITY CHANNEL BREAKDOWN:
- Visual Spatial Peak: ${(d.v_max * 100).toFixed(2)}% (Mean: ${(d.v_mean * 100).toFixed(2)}%)
- Audio Spectral Peak: ${(d.a_max * 100).toFixed(2)}% (Mean: ${(d.a_mean * 100).toFixed(2)}%)
- Boundary Matching Max: ${(d.bm_max * 100).toFixed(2)}%

SUSPECT TEMPORAL SEGMENTS:
${(d.top_proposals && d.top_proposals.length > 0)
    ? d.top_proposals.map(p => `  • [${p.start_sec}s - ${p.end_sec}s] Score: ${(p.score * 100).toFixed(1)}% (Duration: ${p.duration_sec}s)`).join("\n")
    : "  • None detected"}

Architecture: ChronoSync AI (Boundary-Aware Temporal Dual-Stream Neural Detector)
=====================================================`;

        const blob = new Blob([text], { type: "text/plain" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `ChronoSync_AI_Forensic_Report_${Date.now()}.txt`;
        a.click();
        URL.revokeObjectURL(url);
    });
    
    // Window Resize Handler for Canvas
    window.addEventListener("resize", () => {
        if (currentAnalysisData) {
            drawTimeline(currentAnalysisData);
        }
    });
    
    // Initialize Sample Gallery on load
    loadSampleGallery();
});
