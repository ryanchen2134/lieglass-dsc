# test_detective.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import anthropic
import json
import os
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
app = FastAPI()

DETECTIVE_SYSTEM_PROMPT = """
You are an expert interrogator analyzing a conversation transcript for lies and inconsistencies.

The transcript has NO speaker labels. Infer who is speaking based on context:
- Questions are typically the interviewer
- Answers, explanations, and defensive statements are typically the subject
- The SUBJECT is the one you are analyzing for deception

Respond ONLY in valid JSON matching this exact schema:
{
  "inconsistencies": [
    {
      "description": "what the inconsistency is",
      "severity": "major | minor",
      "score_delta": -0.3 for major or -0.1 for minor,
      "quote": "the exact phrase that triggered this"
    }
  ],
  "suggested_questions": ["question to ask the subject to catch them in an inconsistency"],
  "story_summary": "brief summary of their story"
}
Only populate suggested_questions when you have found at least one inconsistency
If nothing found, return empty arrays. No preamble, only JSON.
"""

class TranscriptLine(BaseModel):
    speaker: str
    text: str
    timestamp: Optional[float] = 0.0

class TestRequest(BaseModel):
    transcript: list[TranscriptLine]
    prior_inconsistencies: Optional[list[dict]] = []

def format_transcript(lines):
    return "\n".join(
        f"[{l.timestamp:.1f}s] {l.text}"
        for l in lines
    )

def parse_stt_output(tokens: list[str], timestamps: list[list[float]], speaker: str) -> TranscriptLine:
    """
    Reassembles wordpiece tokens (with ## continuations) into full words,
    then joins into a sentence with the start timestamp of the first token.
    """
    words = []
    word_start = None

    for token, ts in zip(tokens, timestamps):
        if token in [".", ",", "?", "!", "'"]:
            # punctuation — attach to last word
            if words:
                words[-1] = words[-1] + token
            continue
        
        if token.startswith("##"):
            # continuation of previous word
            if words:
                words[-1] = words[-1] + token[2:]
        else:
            # new word
            words.append(token)
            if word_start is None:
                word_start = ts[0]  # timestamp of first token

    return TranscriptLine(
        speaker="unknown",  # ignored by format_transcript now
        text=" ".join(words),
        timestamp=float(word_start or 0.0)
    )
    
class STTChunk(BaseModel):
    tokens: list[str]
    timestamps: list[list[float]]
    speaker: Optional[str] = None  # no longer required


class STTRequest(BaseModel):
    chunks: list[STTChunk]
    prior_inconsistencies: Optional[list[dict]] = []

@app.post("/test/detective/stt")
async def test_detective_stt(body: STTRequest):
    transcript = [
        parse_stt_output(chunk.tokens, chunk.timestamps, chunk.speaker)
        for chunk in body.chunks
    ]
    return await test_detective(TestRequest(
        transcript=transcript,
        prior_inconsistencies=body.prior_inconsistencies
    ))
    
@app.post("/test/detective")
async def test_detective(body: TestRequest):
    user_message = f"""
TRANSCRIPT:
{format_transcript(body.transcript)}

PREVIOUSLY IDENTIFIED INCONSISTENCIES:
{json.dumps(body.prior_inconsistencies) if body.prior_inconsistencies else "None yet."}

Identify any NEW inconsistencies not already listed above.
"""
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=DETECTIVE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}]
    )

    raw = response.content[0].text
    # Just find the JSON object directly, ignore any markdown wrapping
    start = raw.find('{')
    end = raw.rfind('}') + 1
    try:
        result = json.loads(raw[start:end])
    except json.JSONDecodeError:
        return {"error": "LLM returned malformed JSON", "raw": raw}

    # compute score impact
    total_delta = sum(i["score_delta"] for i in result["inconsistencies"])
    base_score = 0.7  # mock poly score for testing
    
    return {
        "result": result,
        "score_simulation": {
            "poly_base": base_score,
            "llm_delta": total_delta,
            "modded_score": round(base_score + total_delta, 3)
        }
    }