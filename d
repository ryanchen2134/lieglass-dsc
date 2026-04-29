[33mcommit dc224e8a113a738f9e84cd3f4f3fbb001edc1924[m[33m ([m[1;36mHEAD[m[33m -> [m[1;32mTHE-Branch[m[33m)[m
Author: BlingleBlongle <BlorfSmorgus@pm.me>
Date:   Sat Apr 25 15:27:41 2026 -0700

    features

[33mcommit ecd30dcb59aa79af403c32d3f268c7fc378766d2[m[33m ([m[1;31morigin/THE-Branch[m[33m, [m[1;32mgrayscale-training[m[33m)[m
Author: BlingleBlongle <BlorfSmorgus@pm.me>
Date:   Sat Apr 25 10:25:06 2026 -0700

    Working models finally

[33mcommit 8a52795b4c14db13fc9b4e51f8c1fd87e89af336[m[33m ([m[1;31morigin/grayscale-training[m[33m)[m
Author: BlingleBlongle <BlorfSmorgus@pm.me>
Date:   Sun Apr 19 14:23:05 2026 -0700

    Grayscale Training

[33mcommit 38fa8b64588065fa58450ea5fe9ed6aacad08671[m
Author: BlingleBlongle <BlorfSmorgus@pm.me>
Date:   Sun Apr 19 14:20:53 2026 -0700

    Grayscale Training

[33mcommit a3dc204c878cd1a3f92b0a404d1695aab0c3e682[m[33m ([m[1;31morigin/claude/process-all-video-frames-b51PE[m[33m)[m
Author: Claude <noreply@anthropic.com>
Date:   Sun Apr 19 00:38:09 2026 +0000

    Full-frame DOLOS pipeline: all frames, 3× YOLO box, no blacking
    
    Preprocessing (deception_detection/data/preprocessing/preprocess_fullframe.py)
    - Start from Data/DOLOS-all/DOLOS/{id}.mp4 (original videos), not resizedVideosNew.
    - Keep every frame; no uniform sampling, no active-speaker blacking.
    - YOLO face box expanded ×3 (side length) around its centre, squared,
      clamped to frame bounds, resized to 224×224.
    - When a frame has no detection the last-seen box is propagated; the
      very first frames (before any detection) fall back to a centred square
      crop. Frames are never blacked out.
    - Outputs features/{id}/frames_full.npz (compressed uint8 N×224×224×3) +
      audio.wav (16 kHz mono).
    
    Dataset / collate (variable-length)
    - DeceptionDataset now reads frames_full.npz (preferred) or legacy
      frames.npz. Every frame is returned; safety cap via config.max_frames.
    - collate_fn pads the temporal axis to the batch max and emits a
      frame_mask distinguishing real frames from padding.
    
    Visual encoder (CNN_Face + temporal Transformer)
    - Replace HF ViT-B/16 layers (no attention-mask support, fixed-length
      learnable PE) with a native nn.TransformerEncoder using sinusoidal
      positional encoding and src_key_padding_mask. Handles arbitrary N,
      benefits from FlashAttention, removes the full pretrained ViT from VRAM.
    - CNN_Face trunk unchanged; same chunking + gradient checkpointing during
      training to bound peak activation memory on long clips.
    - Masked mean pool excludes padded positions.
    
    Config / training
    - Remove fixed n_frames; introduce max_frames (default 400) and
      legacy_n_frames fallback. vit_n_heads exposed for the temporal encoder.
    - Smaller default batch_size (4) and grad_accum_steps=2 to accommodate
      longer clips; CLI flag renamed --n_frames → --max_frames.
    
    https://claude.ai/code/session_01W4GVmUjy9ufLsuap9a6Ey4

[33mcommit fa13c0e7c0fe6e1c60363c8c13821d8f09e18deb[m
Author: ellakim275 <ellakim@ucsb.edu>
Date:   Sat Apr 18 15:14:18 2026 -0700

    Add files via upload

[33mcommit 71d83afcff9b302184f8ff1d4a5dfd37d9932370[m
Author: Claude <noreply@anthropic.com>
Date:   Fri Apr 17 20:25:53 2026 +0000

    Fix GPU monitor interval (10s → 2s) and show LR per epoch
    
    - 10s poll missed epochs that completed in <10s; 2s captures real utilisation
    - Print current learning rate each epoch so OneCycleLR warmup is visible
      (with batch_size=32 and 39 batches/epoch, 10% warmup = 20 epochs before
      LR reaches max — logit_σ staying low in early epochs is expected)

[33mcommit 09bd41a10e8205a680e28805c7f88a94d51082a0[m
Author: Claude <noreply@anthropic.com>
Date:   Fri Apr 17 20:06:57 2026 +0000

    Optimize dataloader: uint8 transfer + GPU normalization + more workers
    
    Addresses ~25% GPU utilization during training:
    
    1. Switch extract_frames.py from np.savez_compressed to np.savez —
       eliminates decompression during dataset load (~5× faster load).
    2. Dataset returns uint8 frames (n,3,224,224) instead of pre-normalized
       float32 — 4× smaller pinned-memory H2D transfer, skips CPU float work.
    3. FusionModel.forward normalizes uint8 → ImageNet float32 on GPU via
       registered _img_mean / _img_std buffers.
    4. Bump num_workers 4 → 8, add prefetch_factor=4.
    
    Re-run extract_frames.py --force to regenerate uncompressed npz files.

[33mcommit a1122cc86a4648fd40d690408dd44b1ca8c78ab8[m
Author: Claude <noreply@anthropic.com>
Date:   Fri Apr 17 19:49:04 2026 +0000

    Fix torch.compile + DataParallel ordering (compile after DP, not before)
    
    compile-then-DataParallel breaks because DP replicates the compiled
    wrapper to each GPU replica, and the wrapper's __getattr__ intercepts
    attribute access, making 'audio_model' invisible on replicas.
    
    Correct order: DataParallel first, then compile the combined module.
    
    https://claude.ai/code/session_01W3kRKFEBGUAQcbSd5uCNg4

[33mcommit 6a5809402a195649a8a7e02d635bf894661fba5b[m
Author: Claude <noreply@anthropic.com>
Date:   Fri Apr 17 19:31:40 2026 +0000

    Add torch.compile and per-epoch GPU/VRAM monitoring
    
    - torch.compile: applied to FusionModel before DataParallel wrap;
      first epoch is slower (JIT trace), subsequent epochs ~10-20% faster
    - _GPUMonitor: daemon thread polls every 10s via pynvml (GPU util% +
      VRAM); falls back to torch.cuda.memory_reserved VRAM-only if pynvml
      is absent. Averages displayed at end of each epoch line:
        ... logit_σ=0.42 | gpu=87%/91% vram=12.3G/11.8G
    
    https://claude.ai/code/session_01W3kRKFEBGUAQcbSd5uCNg4

[33mcommit 4d6ddbaebc8e06673500bc0a18ebc2b7a65f318d[m
Author: Claude <noreply@anthropic.com>
Date:   Fri Apr 17 19:13:32 2026 +0000

    Print trainable/total parameter count before each fold
    
    https://claude.ai/code/session_01W3kRKFEBGUAQcbSd5uCNg4

[33mcommit 8db88e978b52f66a99b2a3f56115c76711374faf[m
Author: Claude <noreply@anthropic.com>
Date:   Fri Apr 17 16:10:03 2026 +0000

    Freeze pretrained layers to fix overfitting on small dataset
    
    ~26M trainable params for 1255 samples was the root cause of overfitting
    (best val AUC at epoch ~11, then divergence). Freezing all Wav2Vec2 and
    ViT layers reduces trainable params to ~2M (CNN + projection + fusion +
    head). Pretrained representations are used as fixed feature extractors.
    
    - wav2vec2_unfreeze_last_n: 2 → 0
    - vit_unfreeze_last_n: 2 → 0
    - learning_rate: 5e-5 → 1e-4 (safe to raise with no pretrained weights at risk)
    
    https://claude.ai/code/session_01W3kRKFEBGUAQcbSd5uCNg4

[33mcommit 2212255dc98a63cad5e9a75e98c28e27228d8435[m
Author: Claude <noreply@anthropic.com>
Date:   Fri Apr 17 01:18:07 2026 +0000

    Add flush=True to training loop prints for SLURM compatibility
    
    Python stdout uses block buffering under SLURM (non-TTY), so output
    was silently buffered until a full fold completed. flush=True forces
    each line to appear immediately.
    
    https://claude.ai/code/session_01W3kRKFEBGUAQcbSd5uCNg4

[33mcommit 24e07814933c3167188f777d4287305ae35d3630[m
Author: Claude <noreply@anthropic.com>
Date:   Thu Apr 16 23:49:54 2026 +0000

    Print all GPU devices at startup to confirm DataParallel is active
    
    https://claude.ai/code/session_01W3kRKFEBGUAQcbSd5uCNg4

[33mcommit 1dbd75de1214130d37123695c5c29a2c01a13f26[m
Author: Claude <noreply@anthropic.com>
Date:   Thu Apr 16 23:45:57 2026 +0000

    Use both GPUs (DataParallel) and enable async data loading (num_workers=4)
    
    - DataParallel: model splits each batch across all available GPUs;
      checkpoint saving uses model.module.state_dict() to strip the wrapper
    - num_workers=4: frames.npz fast-path has no OpenCV so workers are safe;
      overlaps CPU data loading with GPU compute, eliminating idle GPU time
    
    Expected: epoch time drops from ~40s to ~10-15s on 2×A100.
    
    https://claude.ai/code/session_01W3kRKFEBGUAQcbSd5uCNg4

[33mcommit 2716650d6d2c7325cf1dcba1d03fdd730b85103b[m
Author: Claude <noreply@anthropic.com>
Date:   Thu Apr 16 23:28:29 2026 +0000

    Add label smoothing and restore dropout=0.4 to fix overfitting
    
    Model was reaching train_loss=0.18 / logit_σ=4.8 while val_loss climbed
    to 1.88 — severe overfit from logits being pushed to extreme values.
    
    - dropout 0.2 → 0.4: more regularization needed for ~1255 samples
    - label_smoothing=0.1: soft targets (0→0.05, 1→0.95) prevent the model
      from driving logits to ±∞; applied only during training, not eval
    
    https://claude.ai/code/session_01W3kRKFEBGUAQcbSd5uCNg4

[33mcommit 5239e673e27117a9bdd0c6fe6ebc00c11d2065ad[m
Author: Claude <noreply@anthropic.com>
Date:   Thu Apr 16 18:00:51 2026 +0000

    Fix training collapse: lower LR, dropout, replace batch-level fusion
    
    - learning_rate 5e-4 → 5e-5: previous LR was ~10x too high for
      fine-tuning pretrained transformers, causing the model to overshoot
      into a flat saddle by epoch 9
    - dropout 0.4 → 0.2: less aggressive regularization for ~1255 samples
    - patience 50 → 30: faster iteration while debugging
    - CrossFusionModule: replace batch-level cross-attention (unstable with
      batch_size=8) with per-sample learned gating; same d_cross/d_out interface
    
    https://claude.ai/code/session_01W3kRKFEBGUAQcbSd5uCNg4

[33mcommit b4f94aeac53e9f069fa1aee77422078eea1b92f8[m
Author: Claude <noreply@anthropic.com>
Date:   Thu Apr 16 16:27:26 2026 +0000

    Fix audio extraction in preprocess_resized: use source video as primary
    
    - _ensure_audio now tries src_video (resizedVideosNew file) first, which
      carries the original audio track, before falling back to silence
    - _copy_video: drop -an flag so copied video.mp4 also retains audio
      (encode with -c:a aac instead of stripping the stream)
    
    https://claude.ai/code/session_01W3kRKFEBGUAQcbSd5uCNg4

[33mcommit f140de9c295455efc2f32e8dac0276746a085424[m
Author: Claude <noreply@anthropic.com>
Date:   Thu Apr 16 15:49:07 2026 +0000

    Revert "Gracefully handle corrupt/missing audio.wav in dataset loader"
    
    This reverts commit 609faa61534f3b8874e8959770d7a60f043235fd.

[33mcommit 609faa61534f3b8874e8959770d7a60f043235fd[m
Author: Claude <noreply@anthropic.com>
Date:   Thu Apr 16 15:43:42 2026 +0000

    Gracefully handle corrupt/missing audio.wav in dataset loader
    
    Catch all exceptions from torchaudio.load (covers soundfile errors,
    LibsndfileError, and missing files from git LFS pointers). Returns 1 s
    of silence so training continues even when audio data is unavailable.
    
    https://claude.ai/code/session_01W3kRKFEBGUAQcbSd5uCNg4

[33mcommit 2683e7a2da836f5ab07259314ffa288171f11f3f[m
Author: Claude <noreply@anthropic.com>
Date:   Thu Apr 16 06:47:52 2026 +0000

    Handle corrupt/empty frames.npz gracefully in dataset loader
    
    Catch EOFError (and similar) when loading npz files that are truncated,
    empty, or git LFS pointers. Falls back to OpenCV video decode, then to
    black frames with mask=False if no usable video source exists.
    
    https://claude.ai/code/session_01W3kRKFEBGUAQcbSd5uCNg4

[33mcommit 5664da1381aba2540bc748229f986ec948a4d2af[m
Author: Claude <noreply@anthropic.com>
Date:   Thu Apr 16 06:43:45 2026 +0000

    Remove CUDA memory debug prints; fix n_frames mismatch in dataset
    
    - Remove all _cuda_mem helpers and if-trace blocks from audio_model.py,
      visual_model.py, and train.py
    - Fix RuntimeError: pos_embed size mismatch by slicing [:, :T, :] to
      tolerate any T passed by the dataset
    - Add n_frames param to DeceptionDataset so loaded frames match
      config.n_frames (was hardcoded 64); npz fast-path subsamples uniformly
      when stored N != requested n
    
    https://claude.ai/code/session_01W3kRKFEBGUAQcbSd5uCNg4

[33mcommit 10266f4742c6d9c651088f7060f15267fcdefd6d[m
Author: Claude <noreply@anthropic.com>
Date:   Thu Apr 16 05:48:20 2026 +0000

    Add memory summary prints at each memory-intensive step
    
    Prints torch.cuda.memory_summary(abbreviated=True) before and after:
      - visual_model.py: CNN chunk processing, ViT encoder layers
        (first training forward only, via self._mem_traced flag)
      - audio_model.py: Wav2Vec2 forward pass
        (first training forward only, via self._mem_traced flag)
      - train.py: model forward and loss.backward()
        (step==0 only — once per epoch — to avoid log flooding)
    
    All prints are no-ops when CUDA is not available.
    
    https://claude.ai/code/session_01W3kRKFEBGUAQcbSd5uCNg4

[33mcommit 4eb2edca6109ecf3e58756c9ca2e944fcd62067c[m
Author: Claude <noreply@anthropic.com>
Date:   Thu Apr 16 05:44:55 2026 +0000

    Fix GPU OOM: gradient checkpointing for CNN, n_frames=16, empty_cache between folds
    
    Root cause: chunking CNN forward pass didn't help backward — PyTorch still
    retained ALL chunk activations in the torch.cat computation graph (~1.6 GB
    for B=8, T=64 at stage-1 alone).
    
    Fixes:
    1. visual_model.py: gradient checkpointing on CNN chunks during training.
       Activations are discarded after forward and recomputed during backward,
       trading ~30% extra compute for a large VRAM reduction. Eval path unchanged.
    
    2. config.py: n_frames default 64→16. Four times fewer frames = four times
       less CNN memory. For 1-10s DOLOS clips, 16 uniform frames give adequate
       temporal coverage. Override with --n_frames 64 if GPU allows.
    
    3. train.py: del model/optimizer/scaler + torch.cuda.empty_cache() between
       folds to release freed tensors before next fold's model allocates.
       Added --n_frames CLI flag.
    
    https://claude.ai/code/session_01W3kRKFEBGUAQcbSd5uCNg4

[33mcommit 2d34802930f380114878d6876fd6f9f111409f13[m
Author: Claude <noreply@anthropic.com>
Date:   Thu Apr 16 05:13:18 2026 +0000

    Fix memory/dataloader: extract_frames.py + chunked CNN + AMP + grad accum
    
    Root causes addressed:
    1. CNN OOM: flatten (B=8, T=64) → 512-image mini-batch produced ~1.6 GB
       stage-1 activations. Chunked CNN (cnn_chunk_size=32) caps this at ~102 MB.
    2. Slow/unsafe DataLoader: OpenCV VideoCapture in __getitem__ is fork-unsafe
       and decodes 64 H.264 seeks per sample per epoch. New extract_frames.py
       pre-extracts frames to frames.npz (uint8 compressed); dataset fast-path
       loads it with pure numpy — no OpenCV at training time → num_workers=4 safe.
    3. No mixed precision: added AMP (torch.autocast + GradScaler) to halve
       VRAM for forward/backward on CUDA.
    4. Gradient accumulation: --grad_accum flag allows smaller physical batch
       with larger effective batch (e.g. batch_size=4 grad_accum=2 → effective 8).
    
    New files / changes:
    - extract_frames.py: reads resizedVideosNew, saves features/{id}/frames.npz
    - dataset.py: _load_frames() prefers frames.npz, falls back to OpenCV
    - visual_model.py: chunked CNN loop, stores cnn_chunk_size from config
    - config.py: cnn_chunk_size=32, grad_accum_steps=1
    - train.py: AMP scaler, grad accum in train_one_epoch, --grad_accum flag,
      corrected OneCycleLR total_steps for grad accum, startup config summary
    
    https://claude.ai/code/session_01W3kRKFEBGUAQcbSd5uCNg4

[33mcommit 4257922f4a3b4991daee86abeecc08206db7a0b8[m
Author: Ryan Chen <chen.ryan1234@outlook.com>
Date:   Tue Apr 14 20:20:04 2026 -0700

    model

[33mcommit 4a36746115a173c19bd064868d2b94ac247747fc[m
Author: Ryan Chen <chen.ryan1234@outlook.com>
Date:   Tue Apr 14 20:19:56 2026 -0700

    update model

[33mcommit 2c83c9f19b46439f4d7a07028bc049b132da73c2[m
Author: Claude <noreply@anthropic.com>
Date:   Wed Apr 15 02:01:51 2026 +0000

    Fix DataLoader deadlock: default num_workers=0, pin_memory only on CUDA
    
    OpenCV VideoCapture inside forked/spawned DataLoader workers deadlocks
    silently on Linux, causing training to hang after model load. Fix:
    - config.num_workers defaults to 0 (safe; increase explicitly if needed)
    - pin_memory only enabled when device is CUDA
    - persistent_workers tied to num_workers > 0
    - --num_workers CLI flag added to train.py
    
    https://claude.ai/code/session_01W3kRKFEBGUAQcbSd5uCNg4
