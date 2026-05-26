## ADDED Requirements

### Requirement: Coach prices scraped from EuroLeague Fantasy market
`scrape_prices.py` SHALL be extended to scrape each EuroLeague coach's current market price from the EuroLeague Fantasy website in the same Playwright browser session used for player prices.

After player prices are collected, the scraper SHALL navigate to the Coach section of the Transfer/Market page, extract each coach row, and write `coach_prices.csv` to the project root with columns:
- `team`: 3–5 character team code matching the codes used in game data (e.g., `OLY`, `MAD`, `ULK`)
- `coach_name`: display name of the coach as shown on the site
- `price`: float, current market price in EuroLeague Fantasy credits

`coach_prices.csv` SHALL be gitignored (same policy as `euroleague_prices.csv`).

If the Coach market tab cannot be located (site layout change), the scraper SHALL raise a clear error identifying the missing selector rather than silently producing an empty file.

#### Scenario: Successful scrape produces coach_prices.csv
- **WHEN** the user runs `python3 scrape_prices.py` and navigates the fantasy market normally
- **THEN** `coach_prices.csv` is written with one row per EL coach, all prices > 0

#### Scenario: Coach tab not found raises clear error
- **WHEN** the Playwright selector for the coach market tab finds no element
- **THEN** the script raises an exception with message identifying the missing selector, before writing any file

#### Scenario: Team codes match game data codes
- **WHEN** `coach_prices.csv` is merged with team game data on the `team` column
- **THEN** all rows match without NaN (assuming the team played that season)

### Requirement: CLAUDE.md updated with coach scraping instructions
The CLAUDE.md pipeline run order and price scraping section SHALL be updated to note that `scrape_prices.py` now also outputs `coach_prices.csv` and must be run before `optimize_team_v2.py` to enable coach co-optimisation.

#### Scenario: Developer knows to run scraper for coach prices
- **WHEN** a developer reads CLAUDE.md's price scraping section
- **THEN** they understand that coach prices are scraped in the same step as player prices