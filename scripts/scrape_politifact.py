"""One-off scraper: pull a small batch of PolitiFact fact-checks into eval/claims.jsonl.

Not part of the installable package — this is a build-time tool for seeding
and reviewing the ground-truth claim set, not runtime agent code.

Respects politifact.com/robots.txt: Crawl-delay: 10, only /wp-admin/ disallowed.
Run this sparingly; it is meant to produce a small batch for human review
(see README > Ground-truth claim set), not to bulk-harvest the archive.

Usage:
    .venv/bin/python scripts/scrape_politifact.py --n 10 --out misinfo_agent/eval/claims.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from misinfo_agent.eval.schema import POLITIFACT_VERDICT_MAP, Citation, Claim  # noqa: E402

USER_AGENT = "Mozilla/5.0 (research script; portfolio project; contact: brendenrunion@gmail.com)"
CRAWL_DELAY_SECONDS = 10
SITEMAP_INDEX = "https://politifact.com/wp-sitemap.xml"

RATING_ALT_TO_SLUG = {
    "true": "true",
    "mostly true": "mostly-true",
    "half-true": "half-true",
    "half true": "half-true",
    "mostly false": "mostly-false",
    "false": "false",
    "pants on fire!": "pants-fire",
    "pants on fire": "pants-fire",
}


def _get(session: requests.Session, url: str) -> requests.Response:
    resp = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
    resp.raise_for_status()
    return resp


RATING_TO_URL_SLUG = {
    "true": "true",
    "mostly-true": "mostly-true",
    "half-true": "half-true",
    "mostly-false": "mostly-false",
    "false": "false",
    "pants-fire": "pants-fire",
}


def rating_filtered_urls(session: requests.Session, rating: str, limit: int) -> list[str]:
    """Pull fact-check URLs from PolitiFact's own rating-filtered listing page.

    Lets a caller specifically backfill an under-represented verdict bucket
    (e.g. "true", "half-true") instead of only ever getting whatever's most
    recent in the general feed - which is what produced the original
    heavily false-skewed batch.
    """
    slug = RATING_TO_URL_SLUG.get(rating)
    if slug is None:
        raise ValueError(f"Unknown rating {rating!r}; expected one of {sorted(RATING_TO_URL_SLUG)}")

    resp = _get(session, f"https://www.politifact.com/factchecks/list/?ruling={slug}")
    soup = BeautifulSoup(resp.text, "lxml")
    urls, seen = [], set()
    for a in soup.select("a.pf-statement-quote[href]"):
        href = a["href"]
        if href not in seen:
            seen.add(href)
            urls.append(href)
    return urls[:limit]


def latest_factcheck_urls(session: requests.Session, limit: int) -> list[str]:
    index = _get(session, SITEMAP_INDEX)
    time.sleep(CRAWL_DELAY_SECONDS)
    soup = BeautifulSoup(index.text, "xml")
    factcheck_sitemaps = [
        loc.text for loc in soup.find_all("loc") if "posts-factcheck_post" in loc.text
    ]
    if not factcheck_sitemaps:
        raise RuntimeError("No factcheck_post sitemaps found — site structure may have changed")

    latest_sitemap = sorted(
        factcheck_sitemaps, key=lambda u: int(re.search(r"-(\d+)\.xml$", u).group(1))
    )[-1]
    page = _get(session, latest_sitemap)
    time.sleep(CRAWL_DELAY_SECONDS)
    soup = BeautifulSoup(page.text, "xml")
    urls = [loc.text for loc in soup.find_all("loc")]
    return list(reversed(urls))[:limit]


def parse_factcheck(session: requests.Session, url: str, source: str, seq: int) -> Claim | None:
    resp = _get(session, url)
    soup = BeautifulSoup(resp.text, "lxml")

    quote_el = soup.select_one("div.pf-statement-quote p")
    meter_img = soup.select_one("div.pf-statement-meter img[alt]")
    person_el = soup.select_one("a.pf-statement-person")
    date_el = soup.select_one("span.pf-statement-date")
    categories = [a.get_text(strip=True) for a in soup.select("a.pf-category-btn")]
    sources_section = soup.select_one("section#sources")

    if quote_el is None or meter_img is None:
        print(f"  skip (missing claim/rating): {url}", file=sys.stderr)
        return None

    alt = meter_img["alt"].strip().lower()
    rating_slug = RATING_ALT_TO_SLUG.get(alt)
    if rating_slug is None:
        print(f"  skip (unrecognized rating '{alt}'): {url}", file=sys.stderr)
        return None
    verdict = POLITIFACT_VERDICT_MAP[rating_slug]
    verdict_detail = meter_img["alt"].strip()

    date_claimed = None
    if date_el:
        m = re.search(r"stated on (\w+ \d{1,2}, \d{4})", date_el.get_text(strip=True))
        if m:
            from datetime import datetime
            try:
                date_claimed = datetime.strptime(m.group(1), "%B %d, %Y").date().isoformat()
            except ValueError:
                print(f"  skip (invalid date claimed): {url}", file=sys.stderr)
                return None

    date_checked = None
    m = re.search(r"/factchecks/(\d{4})/(\w{3})/(\d{2})/", url)
    if m:
        from datetime import datetime

        year, mon_abbr, day = m.groups()
        try:
            date_checked = datetime.strptime(f"{mon_abbr} {day} {year}", "%b %d %Y").date().isoformat()
        except ValueError:
            print(f"  skip (invalid date): {url}", file=sys.stderr)
            return None

    citations = []
    if sources_section:
        for p in sources_section.select("p"):
            link = p.find("a")
            if not link or not link.get("href"):
                continue
            full_text = p.get_text(" ", strip=True)
            title = link.get_text(strip=True)
            publisher = full_text.split(title)[0].strip(" ,")
            cited_date = full_text.split(title)[-1].strip(" ,")
            citations.append(
                Citation(
                    publisher=publisher or "Unknown",
                    title=title,
                    url=link["href"],
                    cited_date=cited_date or None,
                )
            )

    return Claim(
        id=f"{source}-{seq:04d}",
        source=source,
        claim=quote_el.get_text(" ", strip=True),
        claimant=person_el.get_text(strip=True) if person_el else None,
        date_claimed=date_claimed,
        date_checked=date_checked,
        verdict=verdict,
        verdict_detail=verdict_detail,
        categories=categories,
        fact_check_url=url,
        citations=citations,
        reviewed=False,
    )


def _existing_fact_check_urls_and_max_seq(out_path: Path) -> tuple[set[str], int]:
    """URLs already scraped, and the highest numeric suffix already used in
    an id like "politifact-0100" - NOT just a count of existing claims,
    since a handful of urls get skipped during parsing (missing rating,
    unparseable date), leaving gaps in the sequence.
    """
    if not out_path.exists():
        return set(), 0
    urls, max_seq = set(), 0
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            urls.add(record["fact_check_url"])
            if record["source"] == "politifact":
                max_seq = max(max_seq, int(record["id"].rsplit("-", 1)[1]))
    return urls, max_seq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10, help="number of claims to scrape")
    parser.add_argument(
        "--rating",
        choices=sorted(RATING_TO_URL_SLUG),
        help="pull from PolitiFact's own rating-filtered list instead of the "
        "general latest-fact-checks feed, to backfill a specific verdict bucket",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("misinfo_agent/eval/claims.jsonl")
    )
    args = parser.parse_args()

    session = requests.Session()
    existing_urls, max_seq = _existing_fact_check_urls_and_max_seq(args.out)

    if args.rating:
        print(f"Fetching up to {args.n} '{args.rating}'-rated fact-check URLs from PolitiFact...")
        candidate_urls = rating_filtered_urls(session, args.rating, args.n * 2)
    else:
        print(f"Fetching latest {args.n} fact-check URLs from PolitiFact sitemap...")
        candidate_urls = latest_factcheck_urls(session, args.n * 2)

    urls = [u for u in candidate_urls if u not in existing_urls][: args.n]
    if len(urls) < args.n:
        print(
            f"Note: only found {len(urls)} new (not-already-scraped) urls, "
            f"fewer than the requested {args.n}.",
            file=sys.stderr,
        )

    claims: list[Claim] = []
    for i, url in enumerate(urls, start=1):
        print(f"[{i}/{len(urls)}] {url}")
        claim = parse_factcheck(session, url, source="politifact", seq=max_seq + i)
        if claim:
            claims.append(claim)
        if i < len(urls):
            time.sleep(CRAWL_DELAY_SECONDS)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as f:
        for claim in claims:
            f.write(claim.model_dump_json() + "\n")

    verdict_counts = {}
    for c in claims:
        verdict_counts[c.verdict] = verdict_counts.get(c.verdict, 0) + 1
    print(f"\nWrote {len(claims)} claims to {args.out}")
    print(f"Verdict distribution: {verdict_counts}")
    print("All entries have reviewed=False — review each before using for eval.")


if __name__ == "__main__":
    main()
