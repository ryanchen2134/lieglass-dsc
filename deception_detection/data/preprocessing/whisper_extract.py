"""
Step 2: Whisper Transcription with Word-Level Timestamps.

For each audio.wav:
1. Run official openai-whisper with word_timestamps=True.
2. Tokenize using the text encoder's tokenizer (all-MiniLM-L6-v2).
3. Align sub-word tokens to parent word timestamps via offset_mapping.
4. Exclude [CLS]/[SEP] special tokens.
5. Save text.pt: {token_ids: LongTensor(n,), timestamps: FloatTensor(n, 2)}

Edge case: empty transcription → empty tensors (collate treats as all-pad).
"""

import torch
from pathlib import Path
from transformers import AutoTokenizer

TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_tokenizer = None  # lazy-loaded


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    return _tokenizer


def extract_text(audio_path: Path, output_dir: Path, whisper_model=None) -> bool:
    """
    Transcribe audio and save text.pt.
    whisper_model: a loaded openai-whisper model (passed in to avoid reloading per sample).
    Returns True on success.
    """
    out_pt = output_dir / "text.pt"
    if out_pt.exists():
        return True

    if not audio_path.exists():
        raise FileNotFoundError(f"audio.wav not found: {audio_path}")

    if whisper_model is None:
        raise ValueError("whisper_model must be provided")

    # --- Transcribe with word-level timestamps ---
    result = whisper_model.transcribe(
        str(audio_path),
        word_timestamps=True,
        language="en",
        verbose=False,
    )

    # Collect (word, t_start, t_end)
    words = []
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            text = w["word"].strip()
            if text:
                words.append((text, float(w["start"]), float(w["end"])))

    if not words:
        # Empty transcription
        data = {
            "token_ids": torch.zeros(0, dtype=torch.long),
            "timestamps": torch.zeros(0, 2, dtype=torch.float32),
        }
        torch.save(data, str(out_pt))
        return True

    # --- Tokenize ---
    tokenizer = _get_tokenizer()
    full_text = " ".join(w[0] for w in words)

    encoding = tokenizer(
        full_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    token_ids = encoding["input_ids"]
    offset_mapping = encoding["offset_mapping"]

    # Build character-level index → word index
    char_to_word = []
    pos = 0
    for i, (word_text, _, _) in enumerate(words):
        # Account for spacing between words
        while pos < len(full_text) and full_text[pos] == " ":
            char_to_word.append(None)  # space
            pos += 1
        for _ in word_text:
            char_to_word.append(i)
            pos += 1

    # Map each token to its parent word's timestamps
    timestamps = []
    valid_ids = []
    for tid, (char_start, char_end) in zip(token_ids, offset_mapping):
        if char_start == char_end:
            continue  # skip special/empty tokens
        # Find word index via first char of token
        word_idx = None
        for c in range(char_start, min(char_end, len(char_to_word))):
            if char_to_word[c] is not None:
                word_idx = char_to_word[c]
                break
        if word_idx is None:
            continue
        t_start = words[word_idx][1]
        t_end = words[word_idx][2]
        valid_ids.append(tid)
        timestamps.append([t_start, t_end])

    if not valid_ids:
        data = {
            "token_ids": torch.zeros(0, dtype=torch.long),
            "timestamps": torch.zeros(0, 2, dtype=torch.float32),
        }
    else:
        data = {
            "token_ids": torch.tensor(valid_ids, dtype=torch.long),
            "timestamps": torch.tensor(timestamps, dtype=torch.float32),
        }

    torch.save(data, str(out_pt))
    return True


def main():
    import argparse
    import csv
    import traceback
    import whisper
    from tqdm import tqdm

    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    parser = argparse.ArgumentParser(description="Step 2: Whisper transcription → text.pt")
    parser.add_argument("--manifest", default="Data/manifest_dolos.csv")
    parser.add_argument("--feature_dir", default="features")
    parser.add_argument("--whisper_model", default="base.en", help="tiny.en/base.en/small.en/medium.en")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample_id", default=None, help="Process a single sample only")
    args = parser.parse_args()

    manifest_path = PROJECT_ROOT / args.manifest
    feature_dir = PROJECT_ROOT / args.feature_dir

    with open(manifest_path, newline="") as f:
        samples = list(csv.DictReader(f))

    if args.sample_id:
        samples = [s for s in samples if s["sample_id"] == args.sample_id]
        if not samples:
            print(f"Sample not found: {args.sample_id}")
            return

    print(f"Loading Whisper ({args.whisper_model}) on {args.device}...")
    whisper_model = whisper.load_model(args.whisper_model, device=args.device)

    errors = []
    for row in tqdm(samples, desc="whisper", unit="sample"):
        sample_id = row["sample_id"]
        sample_dir = feature_dir / sample_id
        audio_path = sample_dir / "audio.wav"
        try:
            extract_text(audio_path, sample_dir, whisper_model=whisper_model)
        except Exception as e:
            errors.append((sample_id, str(e)))
            tqdm.write(f"  [ERROR] {sample_id}: {e}")
            tqdm.write(traceback.format_exc())

    print(f"\nDone. {len(samples) - len(errors)}/{len(samples)} succeeded.")
    if errors:
        for sid, err in errors:
            print(f"  {sid}: {err}")


if __name__ == "__main__":
    main()
