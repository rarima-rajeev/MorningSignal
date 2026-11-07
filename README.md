# Morning Signal

A scheduled AI agent that writes a daily brief of public US drug-safety updates.

| Step | What happens | File |
|---|---|---|
| 1. Wake up | GitHub Actions starts the agent every day at 12:00 UTC | `.github/workflows/daily.yml` |
| 2. Fetch | New drug recalls + boxed-warning label updates from the official openFDA API | `agent/run.py` |
| 3. Summarise | Gemini writes one line per item and one "why it matters" line (retries on errors, logs tokens) | `agent/run.py` |
| 4. Check | Lines that don't point at a fetched record are dropped; firm/drug name must match the source | `agent/run.py` |
| 5. Publish | Issue saved as JSON, committed, and the static site shows it | `site/` |
| 6. Notify | Posts the brief as a GitHub issue that mentions the owner, so GitHub emails it | `.github/workflows/daily.yml` |

Quiet days say so. If the Gemini spend cap is hit, the issue says "Paused: AI limit reached" and still lists the raw records.

## Run it yourself

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
echo "GEMINI_API_KEY=..." > .env.local
.venv/bin/python agent/run.py --backfill   # one-time: last 30 days
.venv/bin/python agent/run.py              # today's issue
python3 -m http.server 3008 --directory site
```

## Results so far
- 30-day backfill (2026-11-07): 76 items over 7 days; total AI cost ₹1.56.
- Spot check of 10 random summaries vs source: 9 supported, 1 partly, 0 not (details on the site).

## Notes on openFDA terms
openFDA data is public domain (CC0). openFDA says its records are unvalidated and not for medical decisions,
and it may rate-limit use; this project makes ~2 calls a day and keeps the disclaimer on every page.
No FDA logo or official styling is used.

Independent project. Not affiliated with the FDA. AI-generated summaries of public data; not medical advice. Always check the linked source.
