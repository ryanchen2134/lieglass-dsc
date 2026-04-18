import httpx
import time

# Flat stream of tokens as they'd come from STT — no sentence structure
all_tokens = [
    ("Where", 0.0), ("were", 0.5), ("you", 1.0), ("last", 1.5),
    ("Friday", 2.0), ("night", 2.5), ("?", 3.0),
    ("I", 4.0), ("was", 4.5), ("at", 5.0), ("home", 5.5),
    ("all", 6.0), ("night", 6.5), (",", 7.0), ("never", 7.5), ("left", 8.0), (".", 8.5),
    ("Did", 10.0), ("you", 10.5), ("talk", 11.0), ("to", 11.5), ("anyone", 12.0), ("?", 12.5),
    ("No", 13.5), (",", 14.0), ("I", 14.5), ("was", 15.0), ("completely", 15.5), ("alone", 16.0), (".", 16.5),
    ("Your", 18.0), ("neighbor", 18.5), ("says", 19.0), ("they", 19.5), ("saw", 20.0),
    ("your", 20.5), ("car", 21.0), ("leave", 21.5), ("around", 22.0), ("9pm", 22.5), (".", 23.0),
    ("Oh", 24.0), ("right", 24.5), (",", 25.0), ("I", 25.5), ("did", 26.0), ("pop", 26.5),
    ("out", 27.0), ("briefly", 27.5), ("to", 28.0), ("grab", 28.5), ("groceries", 29.0), (".", 29.5),
    ("What", 31.0), ("time", 31.5), ("did", 32.0), ("you", 32.5), ("get", 33.0), ("back", 33.5), ("?", 34.0),
    ("Around", 35.0), ("midnight", 35.5), ("I", 36.0), ("think", 36.5), (",", 37.0),
    ("maybe", 37.5), ("10pm", 38.0), (".", 38.5),
    ("Which", 40.0), ("store", 40.5), ("did", 41.0), ("you", 41.5), ("go", 42.0), ("to", 42.5), ("?", 43.0),
    ("Uh", 44.0), ("Trader", 44.5), ("Joes", 45.0), ("I", 45.5), ("think", 46.0), (".", 46.5),
    ("Or", 47.0), ("maybe", 47.5), ("Ralphs", 48.0), (".", 48.5),
]

DETECTIVE_INTERVAL = 15.0  # fire detective every N seconds of transcript time
SIMULATE_SPEED = 0.1      # real seconds per transcript second (0.1 = 10x speedup)

prior_inconsistencies = []
accumulated_chunks = []
last_detective_time = 0.0

print("=== Starting simulation ===\n")

for token, ts_start in all_tokens:
    ts_end = round(ts_start + 0.5, 2)
    chunk = {"tokens": [token], "timestamps": [[ts_start, ts_end]]}
    accumulated_chunks.append(chunk)

    print(f"  {token:<15} [{ts_start:.2f} - {ts_end:.2f}]")
    time.sleep(SIMULATE_SPEED)

    # fire detective every DETECTIVE_INTERVAL seconds of transcript time
    if ts_start - last_detective_time >= DETECTIVE_INTERVAL:
        last_detective_time = ts_start
        all_so_far = accumulated_chunks.copy()

        print(f"\n  → Detective firing at t={ts_start:.1f}s (mid-stream)...")
        try:
            resp = httpx.post("http://localhost:8000/test/detective/stt", json={
                "chunks": all_so_far,
                "prior_inconsistencies": prior_inconsistencies
            }, timeout=30.0)
            data = resp.json()

            if "error" in data:
                print(f"  ERROR: {data['error']}")
                print(f"  RAW: {data.get('raw', '')[:200]}")
            else:
                score = data["score_simulation"]["modded_score"]
                new_incons = data["result"]["inconsistencies"]
                questions = data["result"]["suggested_questions"]

                print(f"  SCORE: {score}")
                if new_incons:
                    for inc in new_incons:
                        print(f" * [{inc['severity'].upper()}] {inc['description']}")
                    prior_inconsistencies.extend(new_incons)
                else:
                    print(f" * No new inconsistencies")
                if questions:
                    print(f" * Suggested: {questions[0]}")

        except httpx.ReadTimeout:
            print(" * Detective timed out, continuing...")

        print()

print("\n=== Session complete ===")
print(f"Total inconsistencies: {len(prior_inconsistencies)}")