"""
Scrape player prices from EuroLeague Fantasy Challenge.

The site is a Flutter WebAssembly app — prices are not in the HTML.
This script opens a real browser, intercepts the JSON API calls the app makes,
and extracts player names + prices automatically.

Setup (one time):
    pip install playwright
    playwright install chromium

Run:
    python scrape_prices.py
"""

import json
import sys
from pathlib import Path
import pandas as pd

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
except ImportError:
    print("Playwright not installed. Run:")
    print("  pip install playwright")
    print("  playwright install chromium")
    sys.exit(1)

FANTASY_URL = "https://euroleaguefantasy.euroleaguebasketball.net/24"
OUTPUT_CSV = "euroleague_prices.csv"

# Keywords that suggest a response contains player/market data
PLAYER_KEYWORDS = {"player", "market", "squad", "roster", "transfer", "credit", "price", "lineup"}


def looks_like_player_list(data) -> list[dict]:
    """
    Walk a JSON blob and return a flat list of dicts that have
    both a name field and a price/credit field.
    """
    candidates = []

    def walk(node):
        if isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            name = (
                node.get("name") or node.get("playerName") or
                node.get("player_name") or node.get("fullName") or
                node.get("displayName")
            )
            price = (
                node.get("price") or node.get("value") or
                node.get("credits") or node.get("credit") or
                node.get("marketValue") or node.get("cost")
            )
            if name and price:
                try:
                    candidates.append({"player.name": str(name).upper(), "price": float(price)})
                except (ValueError, TypeError):
                    pass
            for v in node.values():
                if isinstance(v, (dict, list)):
                    walk(v)

    walk(data)
    return candidates


def scrape() -> list[dict]:
    all_players: list[dict] = []
    seen_urls: set[str] = set()

    # Persistent user data dir so login cookies survive between runs
    USER_DATA_DIR = str(Path.home() / ".euroleague_fantasy_session")

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            USER_DATA_DIR,
            headless=False,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        page = context.new_page()

        def on_response(response):
            url = response.url
            if url in seen_urls:
                return
            seen_urls.add(url)

            if response.status != 200:
                return
            ct = response.headers.get("content-type", "")
            if "json" not in ct:
                return
            # Only process URLs that look player/market related
            url_lower = url.lower()
            if not any(kw in url_lower for kw in PLAYER_KEYWORDS):
                return

            try:
                data = response.json()
            except Exception:
                return

            found = looks_like_player_list(data)
            if found:
                all_players.extend(found)
                print(f"  Captured {len(found)} players from:\n    {url}")

        page.on("response", on_response)

        print("Opening EuroLeague Fantasy…")
        try:
            page.goto(FANTASY_URL, wait_until="networkidle", timeout=30_000)
        except PWTimeout:
            pass  # Flutter app may never reach networkidle

        # Auto-dismiss cookie consent if present
        try:
            page.click("text=Accept All Cookies", timeout=5_000)
            print("✓ Dismissed cookie dialog")
        except PWTimeout:
            pass

        print("\n" + "=" * 60)
        print("ACTION REQUIRED")
        print("=" * 60)
        print("1. Log in to your account if prompted")
        print("2. Navigate to the TRANSFER / PLAYER MARKET section")
        print("3. Browse through every position tab (Guards, Forwards, Centers)")
        print("   — scroll down each list so all players load")
        print("4. Close the browser window when done")
        print("=" * 60)

        # Wait for browser window to be closed by the user
        try:
            page.wait_for_event("close", timeout=300_000)
        except Exception:
            pass

        # Give any in-flight requests a moment to resolve
        try:
            context.pages[0].wait_for_load_state("networkidle", timeout=5_000) if context.pages else None
        except Exception:
            pass

        context.close()

    return all_players


def save(players: list[dict]) -> None:
    if not players:
        print("\nNo player prices captured.")
        print("Tips:")
        print("  • Make sure you are logged in before browsing the market")
        print("  • Scroll through every position tab so all requests fire")
        print("  • If the app never loads, check your internet connection")
        return

    df = (
        pd.DataFrame(players)
        .drop_duplicates("player.name")
        .sort_values("price", ascending=False)
        .reset_index(drop=True)
    )
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✓ Saved {len(df)} players to {OUTPUT_CSV}")
    print("\nTop 10 most expensive:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    print("=" * 60)
    print("EUROLEAGUE FANTASY PRICE SCRAPER")
    print("=" * 60)

    players = scrape()
    save(players)