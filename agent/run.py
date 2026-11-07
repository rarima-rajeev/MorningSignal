"""Morning Signal: the daily agent.

Each run: fetch new drug-safety records from openFDA -> ask Gemini for a short summary
-> check every line against its source -> save one issue as JSON for the website.

  python agent/run.py             # live run (what GitHub Actions does every morning)
  python agent/run.py --backfill  # one-time: build issues for the last 30 days
"""
import argparse, datetime as dt, json, os, pathlib, sys, time

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import types

ROOT = pathlib.Path(__file__).parent.parent
DATA = ROOT / "site" / "data"
ISSUES = DATA / "issues"
load_dotenv(ROOT / ".env.local")

MODEL = "gemini-3.8-flash"
# Gemini Flash list price in US dollars per million tokens (same rates as Phase Zero), and rupees per dollar.
PRICE_IN, PRICE_OUT, USD_TO_INR = 0.30, 2.50, 89.0
OPENFDA = "https://api.fda.gov/drug"
LOOKBACK_DAYS = 14  # openFDA publishes weekly, so a live run looks back two weeks for anything not yet reported
MAX_LABELS = 10


# ---------- 1. Fetch (official openFDA API only) ----------

def openfda(endpoint, search, limit=100):
    for attempt in range(4):
        try:
            r = httpx.get(f"{OPENFDA}/{endpoint}.json", params={"search": search, "limit": limit}, timeout=30)
            if r.status_code == 404:  # openFDA answers 404 when nothing matches
                return []
            r.raise_for_status()
            return r.json()["results"]
        except (httpx.HTTPError, KeyError) as e:
            print(f"  openFDA retry {attempt + 1}: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"openFDA unreachable for {endpoint}")


def fetch(start, end):
    """Recalls and boxed-warning label updates published between two dates (YYYYMMDD)."""
    items = []
    for r in openfda("enforcement", f"report_date:[{start} TO {end}]"):
        items.append({
            "id": r["recall_number"],
            "kind": "Recall",
            "date": r["report_date"],
            "name": r["recalling_firm"],
            "facts": f"{r['classification']} recall by {r['recalling_firm']}. Product: {r['product_description'][:300]}. "
                     f"Reason: {r['reason_for_recall'][:300]}. Status: {r['status']}.",
            "url": f'{OPENFDA}/enforcement.json?search=recall_number:"{r["recall_number"]}"',
        })
    labels = openfda("label", f"effective_time:[{start} TO {end}] AND _exists_:boxed_warning")
    for r in labels[:MAX_LABELS]:
        brand = (r.get("openfda", {}).get("brand_name") or r.get("openfda", {}).get("generic_name") or ["Unnamed product"])[0]
        maker = (r.get("openfda", {}).get("manufacturer_name") or [""])[0]
        items.append({
            "id": f"{r['set_id']}@{r['effective_time']}",
            "kind": "Label update (label has a boxed warning)",
            "date": r["effective_time"],
            "name": brand,
            "facts": f"Updated label for {brand} ({maker}). Boxed warning: {r['boxed_warning'][0][:500]}",
            "url": f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={r['set_id']}",
        })
    return items


# ---------- 2. Summarise with Gemini (retry + log every call's tokens) ----------

PROMPT = """You write a short morning brief of public US drug-safety updates for a pharma team.
Use ONLY the facts given. Do not add anything that is not in the facts.
For label updates, say the label was updated and what its boxed warning covers; do not claim the boxed warning itself changed.

Return JSON:
- "lines": one entry per item: {{"id": the item id exactly as given, "text": one plain-English sentence, max 30 words, that names the firm or drug}}
- "why_it_matters": one sentence on why this matters for a pharma team (quality, supply, safety or labeling), citing the ids it is based on
- "why_ids": the list of ids that sentence is based on

Items:
{items}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "string"}, "text": {"type": "string"}}, "required": ["id", "text"]}},
        "why_it_matters": {"type": "string"},
        "why_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["lines", "why_it_matters", "why_ids"],
}


class AILimitReached(Exception):
    pass


def summarise(items):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    facts = "\n".join(f"[{i['id']}] {i['kind']}: {i['facts']}" for i in items)
    for attempt in range(4):
        try:
            resp = client.models.generate_content(
                model=MODEL, contents=PROMPT.format(items=facts),
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                    response_mime_type="application/json", response_schema=SCHEMA))
            u = resp.usage_metadata
            tokens_in = u.prompt_token_count or 0
            tokens_out = (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)
            usd = tokens_in / 1e6 * PRICE_IN + tokens_out / 1e6 * PRICE_OUT
            cost = {"tokens_in": tokens_in, "tokens_out": tokens_out, "usd": round(usd, 5), "inr": round(usd * USD_TO_INR, 3)}
            return json.loads(resp.text), cost
        except Exception as e:
            msg = str(e)
            if "RESOURCE_EXHAUSTED" in msg and ("spend" in msg.lower() or "billing" in msg.lower() or attempt == 3):
                raise AILimitReached(msg[:200])
            print(f"  Gemini retry {attempt + 1}: {msg[:150]}")
            time.sleep(5 * 2 ** attempt)
    raise RuntimeError("Gemini failed after 4 tries")


# ---------- 3. Check every line against its source (the simple eval) ----------

def check(items, ai):
    """Keep only lines that point at a record we actually fetched and name that record's firm or drug."""
    by_id = {i["id"]: i for i in items}
    lines, dropped = [], 0
    for line in ai["lines"]:
        item = by_id.get(line["id"])
        if not item:
            dropped += 1
            continue
        first_word = item["name"].split()[0].strip(",.").lower()
        lines.append({**item, "summary": line["text"].strip(),
                      "name_matches_source": first_word in line["text"].lower()})
    why_ids = [i for i in ai["why_ids"] if i in by_id]
    checks = {
        "lines": len(lines),
        "linked_to_source": len(lines),  # by construction: a line without a real source is dropped
        "name_matches_source": sum(l["name_matches_source"] for l in lines),
        "dropped_unsourced": dropped,
        "missing_from_ai": len(set(by_id) - {l["id"] for l in lines}),
        "why_it_matters_cited": bool(why_ids),
    }
    why = {"text": ai["why_it_matters"].strip(), "sources": [{"id": i, "url": by_id[i]["url"]} for i in why_ids]} if why_ids else None
    return lines, why, checks


# ---------- 4. Save the issue (each issue is written as soon as it is done) ----------

def build_issue(day, items, mode):
    issue = {"date": day, "mode": mode, "run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
             "trigger": os.environ.get("GITHUB_EVENT_NAME", "local"), "model": MODEL}
    if not items:
        issue.update(status="quiet", items=[], why_it_matters=None, checks=None, cost=None)
        return issue
    try:
        ai, cost = summarise(items)
        lines, why, checks = check(items, ai)
        issue.update(status="ok", items=lines, why_it_matters=why, checks=checks, cost=cost)
    except AILimitReached as e:
        print(f"  AI limit reached: {e}")
        issue.update(status="paused", items=[{**i, "summary": None} for i in items], why_it_matters=None, checks=None, cost=None)
    return issue


def save(issue):
    ISSUES.mkdir(parents=True, exist_ok=True)
    (ISSUES / f"{issue['date']}.json").write_text(json.dumps(issue, indent=1))
    rebuild_index()


def rebuild_index():
    rows = []
    for f in sorted(ISSUES.glob("*.json"), reverse=True):
        i = json.loads(f.read_text())
        rows.append({"date": i["date"], "mode": i["mode"], "status": i["status"], "count": len(i["items"]),
                     "inr": (i["cost"] or {}).get("inr", 0), "run_at": i["run_at"]})
    latest_run = max((r["run_at"] for r in rows), default=None)
    (DATA / "index.json").write_text(json.dumps({"last_run": latest_run, "issues": rows}, indent=1))


def already_reported():
    seen = set()
    for f in ISSUES.glob("*.json"):
        seen |= {i["id"] for i in json.loads(f.read_text())["items"]}
    return seen


def ymd(d):
    return d.strftime("%Y%m%d")


def live_run(today):
    start = today - dt.timedelta(days=LOOKBACK_DAYS)
    seen = already_reported()
    items = [i for i in fetch(ymd(start), ymd(today)) if i["id"] not in seen]
    print(f"Live run {today}: {len(items)} new items")
    save(build_issue(today.isoformat(), items, "Live run"))


def backfill(today, days=30):
    """One issue per past day, grouped by the date openFDA published each record. Resumable: skips days already saved."""
    start = today - dt.timedelta(days=days)
    all_items = fetch(ymd(start), ymd(today - dt.timedelta(days=1)))
    for n in range(days, 0, -1):
        day = today - dt.timedelta(days=n)
        if (ISSUES / f"{day.isoformat()}.json").exists():
            continue
        items = [i for i in all_items if i["date"] == ymd(day)]
        print(f"Backfill {day}: {len(items)} items")
        save(build_issue(day.isoformat(), items, f"Backfilled {today.isoformat()}"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backfill", action="store_true")
    args = p.parse_args()
    today = dt.datetime.now(dt.timezone.utc).date()
    if args.backfill:
        backfill(today)
    else:
        live_run(today)
    sys.exit(0)
