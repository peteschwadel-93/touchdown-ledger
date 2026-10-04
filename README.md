# Touchdown Ledger

NFL anytime-touchdown model and dashboard. Chances come from nflverse play-by-play (who gets the ball where),
priced against sportsbook odds from The Odds API.

The site rebuilds itself: the **Update** workflow runs hourly Thursday to Monday, pulls new games and prices,
and publishes the page with GitHub Pages.

- `index.html` – the dashboard (reads `data.json`, which the workflow builds)
- `build_data.py` – the model and the price fetcher
- `odds.json` – the stored price record (written by the workflow; the Tracker grades from it)
- `.github/workflows/update.yml` – the schedule

**Setup:** add the repository secret `ODDS_API_KEY`, set Settings → Pages → Source to "GitHub Actions",
then run Actions → Update once.

**Past prices:** Actions → Update → Run workflow → choose which seasons under past prices (paid Odds API plan; 10 requests per game).
