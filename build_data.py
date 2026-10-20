#!/usr/bin/env python3
"""Build the data behind the Touchdown Ledger (NFL anytime-touchdown model) from nflverse play-by-play.

  python3 build_data.py --install
      One-time setup. Installs what it needs, then the dashboard lives at http://localhost:8767,
      starts when you log in and refreshes itself every few hours. Undo with --uninstall.

  python3 build_data.py --serve
      Opens the dashboard at http://localhost:8767 with a working Refresh button. Keep
      touchdown_ledger.html in the same folder as this script.

  python3 build_data.py --html touchdown_ledger.html
      Rebuild once and write the fresh numbers into the dashboard file.

  python3 build_data.py --backfill
      Shows what it would cost to pull pre-kickoff prices for every finished game this season (The Odds API's
      historical data, paid plans only). Add --yes to pull them. They land in odds.json and the Tracker grades them.

Prices: put your key from the-odds-api.com in the ODDS_API_KEY environment variable, or in a file called
odds_key.txt next to this script (the dashboard has a box that saves it there for you). Each game costs one
request per pull; listing the games is free. Prices are kept in odds.json so past games can be graded.

Data: github.com/nflverse (play-by-play, snap counts, weekly rosters, injury reports) and nfldata games.csv
(schedule, closing spread and total). Everything is cached in ./nfl_cache.
"""
import argparse, itertools, json, math, os, re, shutil, socket, subprocess, sys, threading, time, unicodedata, urllib.error, urllib.request, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

try:
    import numpy as np, pandas as pd
except ImportError:
    np = pd = None

REL = "https://github.com/nflverse/nflverse-data/releases/download/"
GAMES = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
CACHE = "nfl_cache"
POS = ["QB", "RB", "WR", "TE", "FB"]
ET = ZoneInfo("America/New_York")
UA = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}
# model settings, chosen on a 2025 holdout (see the Model tab)
HL, HLT, SM = 5, 6, 0.6          # half-life in games for players and teams; extra fade across an off-season
FIT_HL = 1.0                     # when fitting the chance curve, a player-game a season older counts half as much (TE scoring has been rising)
K, KT, KD = 1.5, 6.0, 30.0       # shrinkage, in games, for player shares, team run/pass mix and the defence nudge
PBP_COLS = ["game_id", "season", "week", "posteam", "defteam", "play_type", "yardline_100", "air_yards", "two_point_attempt",
            "sack", "touchdown", "td_player_id", "rusher_player_id", "receiver_player_id", "passer_player_id", "pass_touchdown",
            "play_id", "qtr", "fixed_drive", "time", "time_of_day", "td_team", "td_player_name", "rush_touchdown", "complete_pass",
            "rushing_yards", "receiving_yards", "passing_yards", "interception", "fumble_lost"]
HTML = "touchdown_ledger.html"
TAG = "atd-data"
ODDS = "odds.json"
ODDS_API = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
TEAM_NAMES = {"Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF", "Carolina Panthers": "CAR",
              "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL", "Denver Broncos": "DEN",
              "Detroit Lions": "DET", "Green Bay Packers": "GB", "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
              "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
              "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG", "New York Jets": "NYJ",
              "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA",
              "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS"}


# ---------- fetching ----------
def ssl_context():
    import ssl
    ctx = ssl.create_default_context()
    try:
        import certifi
        ctx.load_verify_locations(cafile=certifi.where())
    except Exception:
        for path in (os.environ.get("SSL_CERT_FILE"), "/etc/ssl/cert.pem", "/etc/ssl/certs/ca-certificates.crt"):
            if path and os.path.exists(path):
                try:
                    ctx.load_verify_locations(cafile=path)
                except Exception:
                    pass
    return ctx


CTX = None


def get(url, timeout=180, headers=False):
    global CTX
    CTX = CTX or ssl_context()
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout, context=CTX) as r:
        return (r.read(), r.headers) if headers else r.read()


def cached(name, url, fresh=False):
    """A file from the cache folder; downloaded when missing, or again when fresh is set (current-season files)."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name)
    old = os.path.exists(path) and (time.time() - os.path.getmtime(path)) / 60
    if old is False or (fresh and old > 20):
        try:
            raw = get(url)
            with open(path + ".part", "wb") as f:
                f.write(raw)
            os.replace(path + ".part", path)
        except Exception as e:
            if not os.path.exists(path):
                return None
            print(f"kept cached {name}: {e}", file=sys.stderr)
    return path


def current_season():
    now = datetime.now(ET)
    return now.year if now.month >= 8 else now.year - 1


def load(season):
    years = [season - 3, season - 2, season - 1, season]
    pbp, sn, ros = [], [], []
    for y in years:
        cur = y == season
        p = cached(f"pbp_{y}.parquet", f"{REL}pbp/play_by_play_{y}.parquet", cur)
        if p:
            pbp.append(pd.read_parquet(p, columns=PBP_COLS))
        p = cached(f"snaps_{y}.parquet", f"{REL}snap_counts/snap_counts_{y}.parquet", cur)
        if p:
            sn.append(pd.read_parquet(p))
        p = cached(f"roster_{y}.parquet", f"{REL}weekly_rosters/roster_weekly_{y}.parquet", cur)
        if p:
            ros.append(pd.read_parquet(p, columns=["season", "week", "team", "position", "status", "full_name", "gsis_id", "pfr_id", "espn_id"]))
    if not pbp:
        raise RuntimeError("Could not download nflverse play-by-play")
    g = pd.read_csv(cached("games.csv", GAMES, True))
    inj = None
    p = cached(f"inj_{season}.parquet", f"{REL}injuries/injuries_{season}.parquet", True)
    if p:
        try:
            inj = pd.read_parquet(p)
        except Exception:
            pass
    pl = cached("players.parquet", f"{REL}players/players.parquet")
    pl = pd.read_parquet(pl, columns=["gsis_id", "pfr_id", "display_name"]) if pl else None
    return pd.concat(pbp, ignore_index=True), pd.concat(sn, ignore_index=True), pd.concat(ros, ignore_index=True), g[g.season.isin(years)].copy(), inj, pl


# ---------- expected touchdowns per play ----------
def ybin(y):
    y = int(y)
    return y if y <= 10 else 13 if y <= 15 else 18 if y <= 20 else 25 if y <= 30 else 40 if y <= 50 else 75


def tbin(y):
    y = int(y)
    return 3 if y <= 5 else 8 if y <= 10 else 15 if y <= 20 else 30 if y <= 40 else 70


def player_games(pbp, sn, ros, games, pl):
    op = pbp[(pbp.two_point_attempt != 1) & pbp.play_type.isin(["run", "pass"]) & pbp.posteam.notna() & pbp.yardline_100.notna()]
    rush = op[(op.play_type == "run") & op.rusher_player_id.notna()].copy()
    tgt = op[(op.play_type == "pass") & (op.sack != 1) & op.receiver_player_id.notna()].copy()
    rush["td"] = ((rush.touchdown == 1) & (rush.td_player_id == rush.rusher_player_id)).astype(int)
    tgt["td"] = ((tgt.touchdown == 1) & (tgt.td_player_id == tgt.receiver_player_id)).astype(int)
    tgt["ez"] = (tgt.air_yards >= tgt.yardline_100).astype(int)
    rush["yb"] = rush.yardline_100.map(ybin)
    tgt["tb"] = tgt.yardline_100.map(tbin)
    RT = rush.groupby("yb").td.agg(["mean", "size"])
    TT = tgt.groupby(["ez", "tb"]).td.agg(["mean", "size"])
    rush["x"] = rush.yb.map(RT["mean"])
    tgt["x"] = [TT["mean"][(e, b)] for e, b in zip(tgt.ez, tgt.tb)]
    r = rush.groupby(["game_id", "posteam", "rusher_player_id"]).agg(
        car=("x", "size"), rx=("x", "sum"), rtd=("td", "sum"), i5=("yardline_100", lambda s: int((s <= 5).sum())),
        rzc=("yardline_100", lambda s: int((s <= 20).sum())), ryd=("rushing_yards", "sum")).reset_index().rename(columns={"rusher_player_id": "pid"})
    t = tgt.groupby(["game_id", "posteam", "receiver_player_id"]).agg(
        tg=("x", "size"), tx=("x", "sum"), ctd=("td", "sum"), ez=("ez", "sum"), rec=("complete_pass", "sum"), cyd=("receiving_yards", "sum"),
        rzt=("yardline_100", lambda s: int((s <= 20).sum()))).reset_index().rename(columns={"receiver_player_id": "pid"})
    pas = op[(op.play_type == "pass") & op.passer_player_id.notna()]
    q = pas.groupby(["game_id", "posteam", "passer_player_id"]).agg(
        pyd=("passing_yards", "sum"), ptd=("pass_touchdown", "sum"), pint=("interception", "sum")).reset_index().rename(columns={"passer_player_id": "pid"})
    pg = r.merge(t, on=["game_id", "posteam", "pid"], how="outer").merge(q, on=["game_id", "posteam", "pid"], how="outer").fillna(0)
    # the bet settles on any touchdown the player scores himself (rushing, receiving, return); throwing one does not count
    td = pbp[(pbp.touchdown == 1) & pbp.td_player_id.notna() & (pbp.two_point_attempt != 1)]
    td = td[~((td.pass_touchdown == 1) & (td.td_player_id == td.passer_player_id))]
    a = td.groupby(["game_id", "td_player_id"]).size().rename("tds").reset_index().rename(columns={"td_player_id": "pid"})
    first = td.sort_values(["game_id", "play_id"]).drop_duplicates("game_id").set_index("game_id").td_player_id.to_dict()
    idmap = ros.dropna(subset=["pfr_id", "gsis_id"]).drop_duplicates("pfr_id").set_index("pfr_id").gsis_id
    sn = sn.copy()
    sn["pid"] = sn.pfr_player_id.map(idmap)
    if pl is not None:
        sn["pid"] = sn.pid.fillna(sn.pfr_player_id.map(pl.dropna(subset=["pfr_id", "gsis_id"]).drop_duplicates("pfr_id").set_index("pfr_id").gsis_id))
    # position from the roster where known: the snap file labels some running backs "HB"
    rpos = ros.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id", keep="last").set_index("gsis_id").position
    sn["position"] = sn.pid.map(rpos).where(lambda x: x.isin(POS), sn.position.replace({"HB": "RB"}))
    sn = sn[sn.position.isin(POS) & (sn.offense_snaps > 0) & sn.pid.notna()][["game_id", "team", "pid", "player", "position", "offense_pct"]]
    sn = sn.rename(columns={"team": "posteam", "position": "pos", "offense_pct": "snp", "player": "name"})
    sn["posteam"] = sn.posteam.replace({"LAR": "LA"})
    sn = sn.drop_duplicates(["game_id", "pid"])
    d = sn.merge(pg, on=["game_id", "posteam", "pid"], how="left").merge(a, on=["game_id", "pid"], how="left").fillna(0)
    d["y"] = (d.tds > 0).astype(int)
    d["fdp"] = 0.04 * d.pyd + 4 * d.ptd - d.pint + 0.1 * (d.ryd + d.cyd) + 6 * d.tds + 0.5 * d.rec      # FanDuel points he scored
    d["ftd"] = (d.game_id.map(first) == d.pid).astype(int)
    g = games.set_index("game_id")
    d = d[d.game_id.isin(g.index)]
    for c in ("season", "week", "gameday"):
        d[c] = d.game_id.map(g[c])
    home = d.game_id.map(g.home_team) == d.posteam
    sp, tot = d.game_id.map(g.spread_line), d.game_id.map(g.total_line)
    d["imp"] = np.where(home, tot / 2 + sp / 2, tot / 2 - sp / 2)      # spread_line > 0 means the home team is favoured
    d["opp"] = np.where(home, d.game_id.map(g.away_team), d.game_id.map(g.home_team))
    tm = d.groupby(["game_id", "posteam"]).agg(trx=("rx", "sum"), ttx=("tx", "sum"), trtd=("rtd", "sum"), tctd=("ctd", "sum"), ttg=("tg", "sum")).reset_index()
    d = d.merge(tm, on=["game_id", "posteam"]).sort_values(["gameday", "game_id"]).reset_index(drop=True)
    return d, RT, TT


def td_feed(pbp, d, pl, seasons):
    """Every touchdown in the official play-by-play, in order: [season, week, game, scorer, team, position, quarter, clock, type, time of day]."""
    td = pbp[(pbp.touchdown == 1) & pbp.td_player_id.notna() & (pbp.two_point_attempt != 1) & pbp.season.isin(seasons)]
    td = td[~((td.pass_touchdown == 1) & (td.td_player_id == td.passer_player_id))].sort_values(["season", "week", "game_id", "play_id"])
    who = d.drop_duplicates(["game_id", "pid"]).set_index(["game_id", "pid"])[["name", "pos"]]
    full = pl.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id").set_index("gsis_id").display_name.to_dict() if pl is not None else {}
    last = d.drop_duplicates("pid", keep="last").set_index("pid").name.to_dict()
    out = []
    for r in td.itertuples():
        k = (r.game_id, r.td_player_id)
        name, pos = (who.name[k], who.pos[k]) if k in who.index else (last.get(r.td_player_id) or full.get(r.td_player_id) or r.td_player_name or "", "")
        kind = "Rush" if r.rush_touchdown == 1 and r.td_player_id == r.rusher_player_id else \
               "Rec" if r.pass_touchdown == 1 and r.td_player_id == r.receiver_player_id else "Return"
        clock = str(r.time or "")
        clock = clock[1:] if clock.startswith("0") and len(clock) == 5 else clock
        tod = r.time_of_day if isinstance(r.time_of_day, str) else ""
        team = r.td_team if isinstance(r.td_team, str) else ""
        out.append([int(r.season), int(r.week), r.game_id, name, team, pos, int(r.qtr) if r.qtr == r.qtr else 0, clock, kind, tod])
    return out


# ---------- features ----------
PCOLS = ["rx", "tx", "trx", "ttx", "tds", "car", "tg", "i5", "ez", "snp", "rzc", "rzt", "y", "rec", "ryd", "cyd", "pyd", "ptd", "pint", "fdp"]
TCOLS = ["trx", "ttx", "trtd", "tctd", "ttg"]


def decayed(df, key, cols, hl, final=False):
    """Recency-weighted sums of cols over each entity's EARLIER games (weight halves every hl games, and fades across seasons).
    final=True returns {entity: sums after its last game} instead, for projecting the next game."""
    out = np.zeros((len(df), len(cols) + 1))
    dec = 0.5 ** (1 / hl)
    V, S = df[cols].to_numpy(float), df.season.to_numpy()
    end = {}
    for k, idx in df.groupby(key, sort=False).indices.items():
        acc, last = np.zeros(len(cols) + 1), None
        for i in idx:
            if last is not None and S[i] != last:
                acc = acc * SM
            out[i] = acc
            acc = acc * dec + np.append(V[i], 1.0)
            last = S[i]
        end[k] = (acc, last)
    return end if final else out


def team_games(d):
    return d.groupby(["game_id", "posteam", "opp", "season", "gameday"], sort=False).agg(
        trx=("trx", "first"), ttx=("ttx", "first"), trtd=("trtd", "first"), tctd=("tctd", "first"), ttg=("ttg", "first"),
        imp=("imp", "first")).reset_index().sort_values(["gameday", "game_id"]).reset_index(drop=True)


def add_history(d):
    d = d.copy()
    A = decayed(d, "pid", PCOLS, HL)
    for j, c in enumerate(PCOLS):
        d["p_" + c] = A[:, j]
    d["n"] = A[:, -1]
    tg = team_games(d)
    O, Df = decayed(tg, "posteam", TCOLS, HLT), decayed(tg, "opp", TCOLS, HLT)
    for j, c in enumerate(TCOLS):
        tg["o_" + c], tg["d_" + c] = O[:, j], Df[:, j]
    tg["o_n"], tg["d_n"] = O[:, -1], Df[:, -1]
    return d.merge(tg[["game_id", "posteam"] + [c for c in tg.columns if c[:2] in ("o_", "d_")]], on=["game_id", "posteam"])


def structural(d, lg, pri, tfit):
    """Expected touchdowns for each player: team TDs x (run part x his share of rushing chances + pass part x his share of receiving chances).
    Shares are his slice of the team's expected touchdowns (every carry and target weighted by how often it scores from that spot),
    renormalised over the players dressed for the game."""
    d = d.copy()
    pr, pt = d.pos.map(pri["r"]).fillna(0) * 0.6, d.pos.map(pri["t"]).fillna(0) * 0.6
    d["rsh"] = (d.p_rx + K * lg["rx"] * pr) / (d.p_trx + K * lg["rx"])
    d["tsh"] = (d.p_tx + K * lg["tx"] * pt) / (d.p_ttx + K * lg["tx"])
    d["snw"] = (d.p_snp + 0.25) / (d.n + 1.0)
    grp = d.groupby(["game_id", "posteam"])
    d["rshn"], d["tshn"] = d.rsh / grp.rsh.transform("sum"), d.tsh / grp.tsh.transform("sum")
    d["T"] = np.clip(tfit[0] + tfit[1] * d.imp, 0.5, None)
    o_r, o_t = (d.o_trx + KT * lg["rx"]) / (d.o_n + KT), (d.o_ttx + KT * lg["tx"]) / (d.o_n + KT)
    d["dr"] = (d.d_trtd + KD * lg["rx"]) / (d.d_n + KD) / lg["rx"]
    d["dt"] = (d.d_tctd + KD * lg["tx"]) / (d.d_n + KD) / lg["tx"]
    rr, tt = o_r * d.dr, o_t * d.dt
    d["split"] = rr / (rr + tt)
    d["lam_r"], d["lam_t"] = d["T"] * d.split * d.rshn, d["T"] * (1 - d.split) * d.tshn
    d["lam"] = d.lam_r + d.lam_t
    return d


XC = ["ll", "isQB", "isRB", "isTE", "snw", "qb_ll", "qb_car"]


def design(s):
    """Inputs to the chance curve. Quarterbacks get their own slope on expected TDs and a term for how often they carry the ball, and no
    snap-share term (every starter plays every snap, so it only lifted pocket passers): without these the curve squeezed QBs toward the
    middle, rating statues like Stafford near 16% and running QBs too low. Back-check 2025-26: QB log loss 0.367 -> 0.355."""
    q = (s.pos == "QB").astype(int).to_numpy()
    ll = np.log(s.lam.clip(1e-4)).to_numpy()
    cpg = np.log(s.p_car.to_numpy() / np.clip(s.n.to_numpy(), 1, None) + 0.5)
    return np.c_[ll, q, (s.pos == "RB").astype(int), (s.pos == "TE").astype(int), s.snw.to_numpy() * (1 - q), q * ll, q * cpg]


def recency(s, season):
    """Fitting weights: halve for each season back from the one being predicted (FIT_HL), counting weeks within a season."""
    age = (season - s.season.to_numpy()) + (18 - np.clip(s.week.to_numpy(), None, 18)) / 18
    return 0.5 ** (np.clip(age, 0, None) / FIT_HL)


def fit_logit(X, y, iters=60, l2=1e-3, wt=None):
    """Plain logistic regression by Newton's method (no scikit-learn needed); wt = optional weight per row."""
    X1 = np.c_[np.ones(len(X)), X]
    w = np.zeros(X1.shape[1])
    wt = np.ones(len(X1)) if wt is None else np.asarray(wt, float)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X1 @ w))
        H = (X1 * (wt * p * (1 - p))[:, None]).T @ X1 + l2 * np.eye(len(w))
        step = np.linalg.solve(H, X1.T @ (wt * (y - p)) - l2 * w)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    return w


def predict(w, X):
    return 1 / (1 + np.exp(-(np.c_[np.ones(len(X)), X] @ w)))


# ---------- receptions (for the alt-receptions ladders) ----------
REC_POS = ("WR", "TE", "RB")


def rec_design(s, lgt, vac=False):
    """Inputs for expected catches: recent catches and targets a game, snap share, the team's implied total, position, the team's
    recent targets a game and how many targets the opponent has been facing (a pass funnel). Tested on 2025: matchups by man/zone
    coverage and the spread added almost nothing, so they are left out. vac=True adds the targets left open by teammates who are not
    dressing: it makes the catch numbers more accurate, but on the 2026 back-check the extra edges it found were in spots the books had
    already priced (injury news moves lines) and lost, so the board keeps the plain model and only its weight is used, for the Q tm tag."""
    n1 = np.clip(s.n.to_numpy(float), 1, None)
    cols = [np.ones(len(s)), np.log(s.p_rec.to_numpy() / n1 + 0.3), np.log(s.p_tg.to_numpy() / n1 + 0.3), np.log(s.imp.fillna(22).to_numpy() / 22),
            (s.pos == "TE").to_numpy(float), (s.pos == "RB").to_numpy(float), np.log(np.clip(s.snw.to_numpy(float), 0.05, 1)),
            np.log((s.o_ttg.to_numpy() + 4 * lgt) / (s.o_n.to_numpy() + 4) / lgt), np.log((s.d_ttg.to_numpy() + 4 * lgt) / (s.d_n.to_numpy() + 4) / lgt)]
    if vac:
        cols.append(s.vacr.to_numpy(float))
    return np.column_stack(cols)


FD_LG = {"ypc": 4.3, "ypr": 10.5, "pyd": 195.0, "pint": 0.6, "qtd": 0.85, "sdx": 1.3}


def fd_proj(u):
    """FanDuel points for each player (half-PPR: 0.04 a passing yard, 4 a passing TD, -1 an interception, 0.1 a rushing or receiving yard,
    6 a touchdown, 0.5 a catch). Volume and efficiency come from his recent games (the same recency weights as everything else), yards
    per carry and per catch pulled toward the league average, scaled a little by how many points his team is expected to score against
    its usual. Touchdowns come from the model's own chance to score (expected TDs = -ln(1 - p)), catches from the receptions model.
    The spread (sd) gives tournament ceilings: touchdowns dominate it."""
    n = u.n.clip(lower=1)
    car, rec = u.p_car / n, np.where(u.rm.notna(), u.rm, u.p_rec / n)
    ypc = (u.p_ryd + 20 * FD_LG["ypc"]) / (u.p_car + 20)
    ypr = (u.p_cyd + 8 * FD_LG["ypr"]) / (u.p_rec + 8)
    tdr = (u.o_trtd + u.o_tctd + 4 * 2.4) / (u.o_n + 4)                      # his team's usual touchdowns a game
    env = np.clip((u["T"] / tdr) ** 0.4, 0.8, 1.25)
    ryd, cyd = car * ypc * env, rec * ypr * env
    td = -np.log(1 - np.clip(u.p, 1e-6, 0.95))
    isq = (u.pos == "QB").to_numpy()
    pyd = np.where(isq, (u.p_pyd + 2 * FD_LG["pyd"]) / (n + 2) * env, 0)
    ptd = np.where(isq, u["T"] * (1 - u.split) * FD_LG["qtd"], 0)     # most passing TDs go through the QB; some are trick plays or backups
    pint = np.where(isq, (u.p_pint + 2 * FD_LG["pint"]) / (n + 2), 0)
    mean = 0.04 * pyd + 4 * ptd - pint + 0.1 * (ryd + cyd) + 6 * td + 0.5 * rec
    own = (u.p_fdp + 2 * 14.0) / (n + 2)
    mean = np.where(isq, 0.35 * mean + 0.65 * own, mean)      # quarterbacks: their own recent scoring beat the build-up on 2025-26
    var = 36 * td + 0.25 * 1.4 * rec + (0.055 * (ryd + cyd)) ** 2 + (0.04 * 0.28 * pyd) ** 2 + 16 * ptd + pint
    return u.assign(fd=mean, fdsd=FD_LG["sdx"] * np.sqrt(var),     # sdx: widened so 80% of real games land inside +/-1.28 sd (2025-26 check)
                     f_ryd=ryd, f_cyd=cyd, f_pyd=pyd, f_ptd=ptd, f_rec=rec, f_td=td)


def fd_dst(imp_opp, lg):
    """FanDuel team defense: points-allowed tiers (10 for a shutout down to -4 for 35+), 1 a sack, 2 a takeaway, 6 a defensive or return TD.
    Points allowed are taken as roughly normal around the opponent's implied total; sacks and takeaways at the league rate, scaled by how
    good the opposing offence is expected to be."""
    sd = 9.5
    tiers = [(-99, 0.5, 10), (0.5, 6.5, 7), (6.5, 13.5, 4), (13.5, 20.5, 1), (20.5, 27.5, 0), (27.5, 34.5, -1), (34.5, 99, -4)]
    cdf = lambda x: 0.5 * (1 + math.erf((x - imp_opp) / (sd * math.sqrt(2))))
    pa = sum((cdf(hi) - cdf(lo)) * pts for lo, hi, pts in tiers)
    f = (22.0 / max(imp_opp, 10)) ** 0.6
    mean = pa + f * (lg["sk"] + 2 * lg["to"] + 6 * lg["dtd"])
    return mean, 5.5


def add_vac(s, lgt):
    """Targets left open by teammates who are not dressing: the targets a game the team usually throws against the targets a game the
    players actually dressing usually get (log ratio, capped). History uses who played; this week uses the roster less players listed
    Out or Doubtful. On 2025 and 2026 games the model never saw this cut the receptions error by about 1.5%, and removed most of the
    half-catch-a-game shortfall when a regular receiver was missing."""
    rate = s.p_tg / s.n.clip(lower=1)
    dressed = rate.groupby([s.game_id, s.posteam]).transform("sum")
    team = (s.o_ttg + 4 * lgt) / (s.o_n + 4)
    return s.assign(vacr=np.log(np.clip(team / dressed.clip(lower=5), 0.6, 2.0)), vopen=team - dressed)


def fit_rec(s, season, lgt, vac=False):
    """Poisson regression for expected catches (recent seasons weigh more), and the spread around it: catches vary about 1.4 times
    as much as a Poisson count would, so ladder chances use a negative binomial with that variance (it matched every rung of the
    2025-26 back-check to within a point)."""
    z = s[s.pos.isin(REC_POS) & (s.n >= 2) & (s.p_tg / s.n.clip(lower=1) >= 1.5)]
    X, y, wt = rec_design(z, lgt, vac), z.rec.to_numpy(float), recency(z, season)
    w = np.zeros(X.shape[1]); w[0] = np.log(max(y.mean(), 0.5))
    for _ in range(60):
        mu = np.exp(X @ w)
        step = np.linalg.solve((X * (wt * mu)[:, None]).T @ X + 1e-6 * np.eye(len(w)), X.T @ (wt * (y - mu)))
        w += step
        if np.abs(step).max() < 1e-8:
            break
    mu = np.exp(X @ w)
    phi = float(np.sum(wt * (y - mu) ** 2 / mu) / np.sum(wt))
    return w, max(phi, 1.05)


def scores(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    order = np.argsort(p)
    r = np.empty(len(p)); r[order] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    auc = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else None
    return {"n": int(len(y)), "ll": round(float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()), 4),
            "brier": round(float(((p - y) ** 2).mean()), 4), "auc": round(float(auc), 3) if auc else None,
            "pred": round(float(p.mean()), 4), "hit": round(float(y.mean()), 4)}


def r3(x, n=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def norm_name(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'’`]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"[^a-z]+", " ", s).strip()


# ---------- the build ----------
def make(season=None, old=None):
    season = season or current_season()
    pbp, sn, ros, games, inj, pl = load(season)
    d, RT, TT = player_games(pbp, sn, ros, games, pl)
    if d.empty:
        raise RuntimeError("No games found in the play-by-play")
    lg = {"rx": float(d.groupby(["game_id", "posteam"]).trx.first().mean()), "tx": float(d.groupby(["game_id", "posteam"]).ttx.first().mean())}
    pri = {"r": (d.groupby("pos").rx.sum() / d.groupby("pos").trx.sum()).to_dict(), "t": (d.groupby("pos").tx.sum() / d.groupby("pos").ttx.sum()).to_dict()}
    tg = team_games(d)
    tg = tg[tg.imp.notna()]
    b = np.polyfit(tg.imp, tg.trtd + tg.tctd, 1)
    tfit = (float(b[1]), float(b[0]))
    h = add_history(d)
    s = structural(h, lg, pri, tfit)
    seasons = sorted(s.season.unique())
    first = seasons[0]
    s = s[(s.season > first) | (s.week > 3)]     # the first weeks of the earliest season have no history behind them
    s = s[s.imp.notna()]
    # holdout: fit on everything before last season, score last season
    prev = season - 1
    tr, te = s[s.season < prev], s[s.season == prev]
    model = {"tfit": [r3(tfit[0]), r3(tfit[1], 4)], "lg": {k: r3(v) for k, v in lg.items()}, "hl": HL, "sm": SM, "k": K, "kd": KD,
             "rush": [[int(i), r3(r["mean"]), int(r["size"])] for i, r in RT.iterrows()],
             "tgt": [[int(i[0]), int(i[1]), r3(r["mean"]), int(r["size"])] for i, r in TT.iterrows()], "pos": {}}
    trk = []
    row = lambda r: [int(r.season), int(r.week), r.game_id, r.name, r.posteam, r.opp, r.pos, r3(r.p), int(r.y), int(r.tds), r3(r.pf), int(r.ftd)]
    fc = float(s.groupby("game_id").ftd.sum().mean())

    def with_first(z):      # each player's slice of the game's scoring, scaled to how often the first TD goes to a listed player
        rate = -np.log(1 - z.p)
        return z.assign(pf=rate / rate.groupby(z.game_id).transform("sum") * fc)
    if len(tr) > 2000 and len(te) > 1000:
        w0 = fit_logit(design(tr), tr.y.to_numpy(float), wt=recency(tr, prev))
        te = with_first(te.assign(p=predict(w0, design(te))))
        trk += [row(r) for r in te[te.p >= 0.08].itertuples()]
        bt = {"season": int(prev), "model": scores(te.y, te.p), "base": scores(te.y, np.full(len(te), tr.y.mean())),
              "posbase": scores(te.y, te.pos.map(tr.groupby("pos").y.mean()).fillna(tr.y.mean()).to_numpy())}
        edges = [0, .03, .06, .1, .15, .2, .25, .3, .35, .4, .5, 1]
        te["b"] = pd.cut(te.p, edges)
        bt["cal"] = [[r3(x.p.mean()), r3(x.y.mean()), int(len(x))] for _, x in te.groupby("b", observed=True)]
        bt["bypos"] = {k: [r3(x.p.mean()), r3(x.y.mean()), int(len(x))] for k, x in te.groupby("pos")}
        # how well each single input sorts scorers from non-scorers on its own (rank AUC on the holdout)
        z = te.assign(n1=te.n.clip(lower=1))
        singles = {"Expected TDs per game (carries and targets weighted by field position)": (z.p_rx + z.p_tx) / z.n1,
                   "Red-zone carries + targets per game": (z.p_rzc + z.p_rzt) / z.n1, "Actual TDs per game": z.p_tds / z.n1,
                   "Targets per game": z.p_tg / z.n1, "Snap share": z.snw, "Carries per game": z.p_car / z.n1,
                   "Carries inside the 5 per game": z.p_i5 / z.n1, "End-zone targets per game": z.p_ez / z.n1,
                   "Team implied total": z.imp, "Full model": z.p}
        bt["singles"] = sorted([[k, scores(z.y, (v - v.min()) / (v.max() - v.min() + 1e-9) * 0.98 + 0.01)["auc"]] for k, v in singles.items()], key=lambda x: -x[1])
        model["bt"] = bt
    # tracker: this season's games scored with a model that never saw this season
    pre = s[s.season < season]
    cur = s[s.season == season]
    if len(pre) > 2000 and len(cur):
        w1 = fit_logit(design(pre), pre.y.to_numpy(float), wt=recency(pre, season))
        cur = with_first(cur.assign(p=predict(w1, design(cur))))
        model["trk"] = scores(cur.y, cur.p)
        trk += [row(r) for r in cur[cur.p >= 0.08].itertuples()]
    w = fit_logit(design(s), s.y.to_numpy(float), wt=recency(s, season))
    model["w"] = [r3(x, 4) for x in w]
    # receptions: expected catches for the ladders, fitted on everything; this season graded with a fit that never saw it
    lgt = float(d.groupby(["game_id", "posteam"]).ttg.first().mean())
    try:
        recent = pbp[pbp.season >= season - 1]
        ng = max(recent.groupby(["game_id", "defteam"]).ngroups, 1)
        dst_lg = {"sk": float((recent.sack == 1).sum()) / ng, "to": float(((recent.interception == 1) | (recent.fumble_lost == 1)).sum()) / ng,
                  "dtd": float(((recent.touchdown == 1) & (recent.td_team == recent.defteam)).sum()) / ng}
    except Exception:
        dst_lg = {"sk": 2.4, "to": 1.3, "dtd": 0.15}
    s = add_vac(s, lgt)
    wr, phi = fit_rec(s, season, lgt)
    wv, _ = fit_rec(s, season, lgt, vac=True)        # only its open-targets weight is used, by the page's Q tm tag
    model["rec"] = {"w": [r3(x, 4) for x in wr], "phi": r3(phi), "lgt": r3(lgt, 2), "vw": r3(wv[-1], 4)}
    rtrk = []
    if (s.season < season).sum() > 2000:
        wr0, _ = fit_rec(s[s.season < season], season, lgt)
        z = s[(s.season == season) & s.pos.isin(REC_POS)]
        if len(z):
            z = z.assign(rm=np.exp(rec_design(z, lgt) @ wr0))
            rtrk = [[int(r.season), int(r.week), r.game_id, r.name, r.posteam, r.opp, r.pos, r3(r.rm, 2), int(r.rec)] for r in z[z.rm >= 1].itertuples()]
    model["n"] = int(len(s))
    # first-TD share: a player's slice of everyone's scoring rate in the game, scaled to how often the first TD goes to a listed player
    s = s.assign(p=predict(w, design(s)))
    s["rate"] = -np.log(1 - s.p)
    s["fshare"] = s.rate / s.groupby("game_id").rate.transform("sum")
    fcov = float(s.groupby("game_id").ftd.sum().mean())
    model["fcov"] = r3(fcov)
    s["pf"] = s.fshare * fcov
    s["fb"] = pd.cut(s.pf, [0, .02, .04, .06, .09, .12, .2, 1])
    model["fcal"] = [[r3(x.pf.mean()), r3(x.ftd.mean()), int(len(x))] for _, x in s.groupby("fb", observed=True)]

    # ---- upcoming games
    now = datetime.now(ET)
    today = now.strftime("%Y-%m-%d")
    # upcoming games, plus any from the last three days whose play-by-play has not been published yet (their picks stay on the page)
    played = set(d.game_id)
    since = (now - timedelta(days=3)).strftime("%Y-%m-%d")
    up = games[(games.season == season) & ~games.game_id.isin(played) & (games.gameday >= since)].sort_values(["gameday", "gametime"])
    weeks = sorted(up.week.unique())[:2]
    up = up[up.week.isin(weeks)]
    names = ros.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id", keep="last").set_index("gsis_id").full_name.to_dict()
    cur_ros = ros[(ros.season == season)]
    cur_ros = cur_ros[cur_ros.week == cur_ros.week.max()] if len(cur_ros) else cur_ros
    cur_ros = cur_ros.assign(team=cur_ros.team.replace({"LAR": "LA"}))
    P_end = decayed(d, "pid", PCOLS, HL, final=True)
    tg_all = team_games(d)
    O_end, D_end = decayed(tg_all, "posteam", TCOLS, HLT, final=True), decayed(tg_all, "opp", TCOLS, HLT, final=True)
    dcur = d[d.season == season]
    last2 = {t: list(x.drop_duplicates("game_id").game_id)[-2:] for t, x in dcur.groupby("posteam", sort=False)}
    injw = {}
    if inj is not None and len(inj):
        for r in inj.itertuples():
            injw.setdefault(int(r.week), {})[r.gsis_id] = [r.report_status if isinstance(r.report_status, str) else "",
                                                           r.report_primary_injury if isinstance(r.report_primary_injury, str) else "",
                                                           (r.practice_status if isinstance(r.practice_status, str) else "").replace(" Participation in Practice", "")]
    rows, sched = [], []
    for g in up.itertuples():
        if pd.isna(g.spread_line) or pd.isna(g.total_line):
            continue
        ts = datetime.strptime(f"{g.gameday} {g.gametime}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
        sched.append({"gid": g.game_id, "wk": int(g.week), "d": g.gameday, "t": ts.strftime("%a %-I:%M %p ET") if os.name != "nt" else ts.strftime("%a %I:%M %p ET"),
                      "ts": ts.isoformat(), "a": g.away_team, "h": g.home_team, "sp": float(g.spread_line), "tot": float(g.total_line),
                      "ia": r3(g.total_line / 2 - g.spread_line / 2, 2), "ih": r3(g.total_line / 2 + g.spread_line / 2, 2),
                      "roof": g.roof if isinstance(g.roof, str) else "", "out": {}})
        wkinj = injw.get(int(g.week), {})
        for team, opp, imp in ((g.away_team, g.home_team, sched[-1]["ia"]), (g.home_team, g.away_team, sched[-1]["ih"])):
            rs = cur_ros[(cur_ros.team == team) & cur_ros.position.isin(POS) & (cur_ros.status == "ACT")]
            recent = set(dcur[dcur.game_id.isin(last2.get(team, [])) & (dcur.posteam == team)].pid)
            out = []
            for r in rs.itertuples():
                st = wkinj.get(r.gsis_id, ["", "", ""])
                if st[0] in ("Out", "Doubtful"):
                    if r.gsis_id in P_end:
                        out.append([r.full_name, r.position, st[0], st[1]])
                    continue
                if r.gsis_id not in P_end:
                    continue
                back = r.gsis_id not in recent
                if back and not (r.gsis_id in wkinj and P_end[r.gsis_id][1] >= season - 1):
                    continue          # has not played lately and is not on this week's report: treat as not in the plan
                acc, last = P_end[r.gsis_id]
                acc = acc * (SM if last != season else 1)
                o, dd = O_end.get(team, (np.zeros(len(TCOLS) + 1), season))[0], D_end.get(opp, (np.zeros(len(TCOLS) + 1), season))[0]
                row = {"game_id": g.game_id, "posteam": team, "opp": opp, "pid": r.gsis_id, "name": r.full_name, "pos": r.position,
                       "imp": imp, "inj": st, "back": int(back), "n": acc[-1], "week": int(g.week),
                       "eid": str(int(float(r.espn_id))) if pd.notna(r.espn_id) and str(r.espn_id).strip() else ""}
                row.update({"p_" + c: acc[j] for j, c in enumerate(PCOLS)})
                row.update({"o_" + c: o[j] for j, c in enumerate(TCOLS)}); row["o_n"] = o[-1]
                row.update({"d_" + c: dd[j] for j, c in enumerate(TCOLS)}); row["d_n"] = dd[-1]
                rows.append(row)
            sched[-1]["out"][team] = out
            # one quarterback per team: the listed starter, else whoever has taken the most snaps lately
            qbs = [x for x in rows if x["game_id"] == g.game_id and x["posteam"] == team and x["pos"] == "QB"]
            if len(qbs) > 1:
                want = norm_name(getattr(g, "away_qb_name" if team == g.away_team else "home_qb_name", "") or "")
                keep = next((x for x in qbs if norm_name(x["name"]) == want), None) or max(qbs, key=lambda x: (x["p_snp"] + 0.25) / (x["n"] + 1))
                rows[:] = [x for x in rows if x not in qbs or x is keep]
    picks = []
    if rows:
        u = structural(pd.DataFrame(rows), lg, pri, tfit)
        u["p"] = predict(w, design(u))
        u["rate"] = -np.log(1 - u.p)
        u["pf"] = u.rate / u.groupby("game_id").rate.transform("sum") * fcov
        u = add_vac(u, lgt)
        u = fd_proj(u.assign(rm=np.where(u.pos.isin(REC_POS), np.exp(rec_design(u, lgt) @ wr), np.nan)))
        u["rm"] = np.where(u.pos.isin(REC_POS), np.exp(rec_design(u, lgt) @ wr), np.nan)
        logs, rlog = {}, {}
        for pid, x in d[d.pid.isin(set(u.pid))].groupby("pid", sort=False):
            rlog[pid] = [int(v) for v in x.rec.tail(10)]
            # game log, last ten: [label, opp, snaps, car, tgt, in5, ez, rz, xTD, TD, catches, rush yds, rec yds, game id, team targets]
            logs[pid] = [[f"{int(r.season) % 100}w{int(r.week)}", r.opp, r3(r.snp, 2), int(r.car), int(r.tg), int(r.i5), int(r.ez),
                          int(r.rzc + r.rzt), r3(r.rx + r.tx, 2), int(r.tds), int(r.rec), int(round(r.ryd)), int(round(r.cyd)), r.game_id, int(r.ttg)]
                         for r in x.tail(10).itertuples()]
        for r in u.itertuples():
            n1 = max(r.n, 1e-9)
            picks.append({"g": r.game_id, "id": r.pid, "n": r.name, "t": r.posteam, "o": r.opp, "pos": r.pos, "p": r3(r.p, 4), "pf": r3(r.pf, 4),
                          "T": r3(r.T, 2), "sp": r3(r.split), "rs": r3(r.rshn), "ts": r3(r.tshn), "lr": r3(r.lam_r), "lt": r3(r.lam_t),
                          "sn": r3(r.snw, 2), "dr": r3(r.dr, 2), "dt": r3(r.dt, 2), "ng": r3(r.n, 1), "inj": r.inj, "back": r.back, "eid": r.eid,
                          "x": r3((r.p_rx + r.p_tx) / n1, 2), "td": r3(r.p_tds / n1, 2), "i5": r3(r.p_i5 / n1, 2), "ez": r3(r.p_ez / n1, 2),
                          "rz": r3((r.p_rzc + r.p_rzt) / n1, 2), "log": logs.get(r.pid, []),
                          **({"rm": r3(r.rm, 2), "rg": r3(r.p_rec / n1, 1), "tgg": r3(r.p_tg / n1, 1), "rl": rlog.get(r.pid, []),
                              "ctr": r3(r.p_rec / r.p_tg, 2) if r.p_tg > 0 else None, "tt": r3((r.o_ttg + 4 * lgt) / (r.o_n + 4), 1),
                              "of": r3((r.d_ttg + 4 * lgt) / (r.d_n + 4) / lgt, 2), "vo": r3(r.vopen, 1)} if r.rm == r.rm else {}),
                          "fd": [r3(r.fd, 2), r3(r.fdsd, 2)], "fdp": [r3(r.f_pyd, 1), r3(r.f_ptd, 2), r3(r.f_ryd, 1), r3(r.f_cyd, 1), r3(r.f_rec, 2), r3(r.f_td, 3)]})
    # ---- research tables
    def usage(x):
        gp = x.groupby("pid").agg(n=("name", "last"), t=("posteam", "last"), pos=("pos", "last"), gp=("game_id", "nunique"), snp=("snp", "mean"),
                                  car=("car", "sum"), tg=("tg", "sum"), i5=("i5", "sum"), ez=("ez", "sum"), rzc=("rzc", "sum"), rzt=("rzt", "sum"),
                                  rx=("rx", "sum"), tx=("tx", "sum"), tds=("tds", "sum"), sc=("y", "sum"), trx=("trx", "sum"), ttx=("ttx", "sum")).reset_index()
        gp = gp[(gp.car + gp.tg) >= 3]
        return [[names.get(r.pid, r.n), r.t, r.pos, int(r.gp), r3(r.snp, 2), int(r.car), int(r.tg), int(r.i5), int(r.ez), int(r.rzc + r.rzt),
                 r3(r.rx + r.tx, 2), int(r.tds), int(r.sc), r3(r.rx / r.trx if r.trx else 0), r3(r.tx / r.ttx if r.ttx else 0)] for r in gp.itertuples()]
    players = {str(int(y)): usage(d[d.season == y]) for y in seasons[-2:]}

    def teams(x):
        out = {}
        tgx = team_games(x)
        for t, o in tgx.groupby("posteam"):
            df = tgx[tgx.opp == t]
            xx, xo = x[x.posteam == t], x[x.opp == t]
            ng, nd = len(o), max(len(df), 1)
            by = lambda z, p: r3(z[z.pos == p].tds.sum() / max(z.game_id.nunique(), 1), 2)
            out[t] = {"g": int(ng), "td": r3((o.trtd + o.tctd).mean(), 2), "rtd": r3(o.trtd.mean(), 2), "ctd": r3(o.tctd.mean(), 2),
                      "rx": r3(o.trx.mean(), 2), "tx": r3(o.ttx.mean(), 2), "imp": r3(o.imp.mean(), 1),
                      "i5": r3(xx.i5.sum() / ng, 2), "ez": r3(xx.ez.sum() / ng, 2), "rz": r3((xx.rzc + xx.rzt).sum() / ng, 1),
                      "d": {"td": r3((df.trtd + df.tctd).mean(), 2), "rtd": r3(df.trtd.mean(), 2), "ctd": r3(df.tctd.mean(), 2),
                            "rx": r3(df.trx.mean(), 2), "tx": r3(df.ttx.mean(), 2), "qb": by(xo, "QB"), "rb": by(xo, "RB"), "wr": by(xo, "WR"), "te": by(xo, "TE")}}
        return out
    tms = {str(int(y)): teams(d[d.season == y]) for y in seasons[-2:]}
    out = {"season": int(season), "built": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "through": str(d.gameday.max()),
           "lastwk": int(dcur.week.max()) if len(dcur) else 0, "sched": sched, "picks": picks, "players": players, "teams": tms,
           "model": model, "trk": trk, "feed": td_feed(pbp, d, pl, [season - 1, season]), "rtrk": rtrk}
    out["dst"] = {g["gid"]: {g["a"]: [r3(x, 2) for x in fd_dst(g["ih"], dst_lg)], g["h"]: [r3(x, 2) for x in fd_dst(g["ia"], dst_lg)]} for g in sched}
    model["dst"] = {k: r3(v, 3) for k, v in dst_lg.items()}
    tg_ = {r[2] for r in trk}           # kickoff day and time of every graded game, for slot lines on the Tracker (e.g. the 4:25 TE)
    kk = games[games.game_id.isin(tg_)]
    out["kick"] = {gid: f"{str(wd)[:3]} {gt}" for gid, wd, gt in zip(kk.game_id, kk.weekday, kk.gametime)}
    note = None
    bf = os.environ.get("ODDS_BACKFILL", "").strip().lower()
    if bf in ("1", "2", "3", "4", "5", "6", "7", "true"):
        try:
            backfill(season, int(os.environ.get("ODDS_LEAD", "60")), True, int(os.environ.get("ODDS_BACKFILL_MAX", "0")) or None, back=1 if bf in ("2", "5") else 0, first=bf == "3", two=bf in ("4", "5"), lad=bf == "6", yds=bf == "7")
        except SystemExit as e:
            note = str(e)
            print(note, file=sys.stderr)
    full = remember_odds(sched, old)
    live = {u["gid"] for u in sched}
    out["picks"], locks = apply_locks(sched, picks)
    for row_ in trk:                      # this season's finished games: grade the chances the page showed at kickoff
        lk = (locks.get(row_[2]) or {}).get("lock") if row_[0] == season else None
        if lk and norm_name(row_[3]) in lk:
            v = lk[norm_name(row_[3])]
            row_[7] = v[0]
            if v[1] is not None:
                row_[10] = v[1]
            if len(v) > 3:              # on the Top 8 / long-shot / 2+ TD list at kickoff (1) or not (0)
                row_ += [v[2], v[3], v[4] if len(v) > 4 else None]
    tmap = {(a, b_): r3(c, 2) for a, b_, c in zip(s.game_id, s.posteam, s["T"])}
    for row_ in trk:                      # the team's expected touchdowns, for the 2+ TD chance; then the FanDuel Top 8 / long-shot / 2+ TD flags
        row_ += [None] * (15 - len(row_))
        row_.append(tmap.get((row_[2], row_[4])))
        lk = (locks.get(row_[2]) or {}).get("lock") if row_[0] == season else None
        v = (lk or {}).get(norm_name(row_[3]))
        if v and len(v) > 5 and v[5]:
            row_.append(v[5])
    try:
        # the store apply_locks just saved (prices plus the kickoff locks): saving the earlier copy here wiped the locks (Oct 17-19)
        for k_, v_ in full.items():
            locks.setdefault(k_, v_)
        out["recb"] = rec_board(sched, out["picks"], locks, (model.get("rec") or {}).get("phi", 1.4), season, rtrk)
        with open(ODDS, "w", encoding="utf-8") as f:       # the board's picks are kept per game until kickoff, then frozen
            json.dump(locks, f, separators=(",", ":"), ensure_ascii=False, sort_keys=True)
    except Exception as e:
        print(f"receptions board skipped: {e}", file=sys.stderr)
        out["recb"] = {}
    out["live"] = live_scores(sched)
    out["notes"] = espn_notes({norm_name(x["n"]) for x in out["picks"]})
    out["odds"] = {}
    for k, v in full.items():
        if k == "_meta" or k in live or not v.get("b"):
            out["odds"][k] = {a: b for a, b in v.items() if a not in ("lock", "lockrows")} if k != "_meta" else v
            continue
        best = {}
        for bk, rows_ in v["b"].items():
            for who, price in rows_:
                if who not in best or price > best[who][1]:
                    best[who] = [who, price, bk]
        out["odds"][k] = {"at": v.get("at"), "best": list(best.values())}
        if v.get("b0"):                   # closing-line value: best and FanDuel at the first look, and FanDuel at the last
            b0 = {}
            for bk, rows_ in v["b0"].items():
                for who, price in rows_:
                    if who not in b0 or american(price) > american(b0[who][1]):
                        b0[who] = [who, price, bk]
            out["odds"][k]["best0"] = list(b0.values())
            out["odds"][k]["at0"] = v.get("at0")
            fo = dict((who, price) for who, price in v["b0"].get("FanDuel") or [])
            out["odds"][k]["fd"] = [[who, price, fo.get(who)] for who, price in v["b"].get("FanDuel") or []]
        elif v["b"].get("FanDuel"):       # FanDuel's last anytime price is kept all season, so My bets keeps closing-line value for older bets
            out["odds"][k]["fd"] = [[who, price, None] for who, price in v["b"]["FanDuel"]]
        try:                              # every book's closing prices for recent games, so My bets can show closing-line value
            age = (datetime.now(ET) - datetime.fromisoformat(v.get("at") or "2000-01-01T00:00").replace(tzinfo=ET)).days
            if age <= 14:                 # the page also saves each bet's close on the bet itself once its game starts
                out["odds"][k]["bc"] = v["b"]
                for src, dst in (("t", "bct"), ("f", "bcf"), ("ro", "bco")):
                    if v.get(src):
                        out["odds"][k][dst] = v[src]
                if age <= 5 and v.get("r"):   # full alternate ladders are large: a few days is enough for the page to save them
                    out["odds"][k]["bcr"] = v["r"]
        except Exception:
            pass
        bf_ = {}
        for bk, rows_ in (v.get("f") or {}).items():
            for who, price in rows_:
                if who not in bf_ or price > bf_[who][1]:
                    bf_[who] = [who, price, bk]
        if bf_:
            out["odds"][k]["bestf"] = list(bf_.values())
        bt_ = {}
        for bk, rows_ in (v.get("t") or {}).items():
            for who, price in rows_:
                if who not in bt_ or price > bt_[who][1]:
                    bt_[who] = [who, price, bk]
        if bt_:
            out["odds"][k]["bestt"] = list(bt_.values())
        br_ = {}
        for bk, rows_ in (v.get("r") or {}).items():
            for who, rung, price in rows_:
                kk = (who, rung)
                if kk not in br_ or american(price) > american(br_[kk][2]):
                    br_[kk] = [who, rung, price, bk]
        if br_:
            out["odds"][k]["bestr"] = list(br_.values())
    if note:
        out["odds"]["_meta"]["err"] = note[:200]
    try:
        out["movers"] = movers(out)
    except Exception as e:                # the change log is a nicety: never let it stop a build
        print(f"movers skipped: {e}", file=sys.stderr)
        out["movers"] = []
    return out


# ---------- movers: what changed for each player from one build to the next ----------
MOVERS = "movers.json"


def movers(out, path=MOVERS, keep_days=8):
    """Compares every player in games still to start with the previous build and logs what moved: his chance, the best and FanDuel
    prices, his own injury status, teammates ruled out or back, and the team's betting line (implied points and team touchdowns).
    The last build's numbers live in movers.json (committed with odds.json); the log keeps eight days. Returns the log for the page:
    rows of [time, game, player id, name, team, opponent, pos, before, after, causes] with before/after = [chance, team TDs, best price,
    best book, FanDuel price, status] (None when he was not on the board)."""
    now = datetime.now(timezone.utc)
    try:
        with open(path, encoding="utf-8") as f:
            prev = json.load(f)
    except Exception:
        prev = {}
    psnap, pgame = prev.get("snap") or {}, prev.get("games") or {}
    cut = (now - timedelta(days=keep_days)).strftime("%Y-%m-%dT%H:%MZ")
    log = [r for r in (prev.get("log") or []) if r[0] >= cut]
    games, snap, add = {}, {}, []
    for g in out["sched"]:
        try:
            if datetime.fromisoformat(g["ts"]) <= now:
                continue
        except Exception:
            continue
        outs = g.get("out") or {}
        games[g["gid"]] = {"a": g["a"], "h": g["h"], "imp": {g["a"]: g.get("ia"), g["h"]: g.get("ih")},
                           "out": {t: sorted(x[0] for x in (outs.get(t) or [])) for t in (g["a"], g["h"])}}
    px = {}
    for gid in games:
        best, fd = {}, {}
        for bk, rows_ in (((out.get("odds") or {}).get(gid) or {}).get("b") or {}).items():
            for who, price in rows_:
                k = norm_name(who)
                if k not in best or american(price) > american(best[k][0]):
                    best[k] = [price, bk]
                if bk == "FanDuel":
                    fd[k] = price
        px[gid] = (best, fd)
    rb = {(x["g"], x["id"]): x for x in ((out.get("recb") or {}).get("rows") or [])}
    for r in out["picks"]:
        gid = r["g"]
        if gid not in games:
            continue
        k = norm_name(r["n"])
        b = px[gid][0].get(k)
        x = rb.get((gid, r["id"])) or {}
        snap[gid + "|" + r["id"]] = [r3(r["p"], 4), r3(r.get("T"), 2), b[0] if b else None, b[1] if b else None, px[gid][1].get(k),
                                     (r.get("inj") or [""])[0] or "", r["n"], r["t"], r["o"], r["pos"],
                                     r.get("rm"), x.get("line"), (x.get("ov") or {}).get("a")]

    def causes(gid, a, b):
        if not b:
            return [["gone"]]
        c = [] if a else [["new"]]
        t, gp, gn = (b or a)[7], pgame[gid], games[gid]
        oi, ni = (gp.get("imp") or {}).get(t), gn["imp"].get(t)
        if a and b and (oi != ni or (a[1] is not None and b[1] is not None and abs(a[1] - b[1]) >= 0.01)):
            c.append(["line", [oi, ni], [a[1], b[1]]])
        me = (b or a)[6]
        om, nm = set((gp.get("out") or {}).get(t) or []) - {me}, set(gn["out"].get(t) or []) - {me}
        if om != nm:
            c.append(["mates", sorted(nm - om), sorted(om - nm)])
        if a and b and a[5] != b[5]:
            c.append(["inj", a[5], b[5]])
        if (a and a[2]) != (b and b[2]):
            c.append(["price", a and a[2], b and b[2], a and a[3], b and b[3]])
        if (a and a[4]) != (b and b[4]):
            c.append(["fd", a and a[4], b and b[4]])
        ra, rb_ = (a[10:13] if a and len(a) > 12 else [None] * 3), (b[10:13] if b and len(b) > 12 else [None] * 3)
        if a and b and ra[0] is not None and rb_[0] is not None and abs(ra[0] - rb_[0]) >= 0.1:
            c.append(["rec", ra[0], rb_[0]])
        if a and b and len(a) > 12 and ra[1] != rb_[1] and (ra[1] is not None or rb_[1] is not None):
            c.append(["rline", ra[1], rb_[1]])
        elif a and b and ra[2] != rb_[2] and rb_[1] is not None and len(a) > 12:
            c.append(["rov", ra[2], rb_[2], rb_[1]])
        if a and b and not c and abs((a[0] or 0) - (b[0] or 0)) >= 0.002:
            c.append(["model"])
        return c

    for key in sorted(set(snap) | set(psnap)):
        gid = key.rsplit("|", 1)[0]
        if gid not in games or gid not in pgame:      # only games both builds saw, and only before kickoff
            continue
        a, b = psnap.get(key), snap.get(key)
        if a == b:
            continue
        c = causes(gid, a, b)
        if not c:
            continue
        z = b or a
        cut_ = lambda s: s[:6] + list(s[10:13]) if len(s) > 12 else s[:6]
        add.append([out["built"], gid, key.rsplit("|", 1)[1], z[6], z[7], z[8], z[9], cut_(a) if a else None, cut_(b) if b else None, c])
    log += add
    if add or snap != psnap or games != pgame:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"at": out["built"], "snap": snap, "games": games, "log": log}, f, separators=(",", ":"), ensure_ascii=False)
    return log


# ---------- the receptions board: picks, ladder watch and parlay ideas ----------
# Starting guesses, like the other rules before a record exists. Chances are the model's averaged half and half with the market's own
# (each book's over and under on the main line with the margin removed); ladder rungs use the expected catches averaged the same way.
# Retuned on the first six weeks of 2026 prices (80 games): the biggest pick margins and the smallest ladder edges both lost, so picks now
# need a margin of 1 to 4 points and go to the likeliest, and ladder rungs need an edge of 10% to 40% within two rungs of the over rung.
# Picks are tight ends and backs only, at -150 or longer: on those six weeks WR picks lost in five of six weeks (the most-bet, sharpest
# lines) and heavy favourites gave back most of the edge.
RB = {"pos": ["TE", "RB"], "pmin": 1.667, "m": 0.01, "mx": 0.04, "books": 3, "gap": 0.12, "lp": 0.20, "llo": 0.10, "lhi": 0.40, "lmax": 15.0, "pm": 0.45, "pr": 0.30, "plo": 0.10, "phi_": 0.40}


def nb_ge(mu, k, phi):
    """Chance of k or more catches when catches average mu and vary phi times as much as a Poisson count."""
    if mu is None or mu <= 0:
        return None
    if k <= 0:
        return 1.0
    r = mu / (phi - 1)
    q = r / (r + mu)
    pk = q ** r
    c = pk
    for i in range(1, k):
        pk = pk * (i - 1 + r) / i * (1 - q)
        c += pk
    return min(max(1 - c, 0.0), 1.0)


def mu_for(prob, k, phi):
    lo, hi = 0.05, 20.0
    for _ in range(40):
        m = (lo + hi) / 2
        if nb_ge(m, k, phi) < prob:
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


def rec_board(sched, picks, store, phi, season, rtrk=None):
    """Picks, ladder watch and parlay ideas for every week: this week's and next week's live (each game's picks frozen at its kickoff),
    and this season's earlier weeks as they would have looked, rebuilt from the stored prices an hour before kickoff and the catches the
    model expected (a fit that never saw this season), unless picks were recorded live for that week."""
    now = datetime.now(ET)
    games = {g["gid"]: g for g in sched}
    started = {gid: datetime.fromisoformat(g["ts"]) <= now for gid, g in games.items()}
    actual = {}
    past = {}
    items = list(picks)
    for x in rtrk or []:
        actual[(x[2], norm_name(x[3]))] = x[8]
        rec = store.get(x[2]) or {}
        if x[2] not in games and (rec.get("ro") or rec.get("r")):
            past.setdefault(x[1], set()).add(x[2])
            items.append({"g": x[2], "id": "x:" + norm_name(x[3]), "n": x[3], "t": x[4], "o": x[5], "pos": x[6], "rm": x[7], "_past": 1})
    pastmode = {}
    for wk, gids in past.items():
        live = any((store.get(g_) or {}).get("rxp") is not None or (store.get(g_) or {}).get("rxl") is not None for g_ in gids)
        pastmode[wk] = "frozen" if live else "retro"
        for g_ in gids:
            started[g_] = live
    rows, legs = [], []
    for r in items:
        mu = r.get("rm")
        if mu is None or mu < 0.8 or (r["g"] not in games and not r.get("_past")):
            continue
        gid, k = r["g"], norm_name(r["n"])
        rec = store.get(gid) or {}
        ou = {}
        for bk, lst in (rec.get("ro") or {}).items():
            for who, pt, ov, un in lst:
                if norm_name(who) == k:
                    ou.setdefault(bk, []).append((pt, ov, un))
        row = {"g": gid, "id": r["id"], "n": r["n"], "t": r["t"], "o": r["o"], "pos": r["pos"], "mu": mu, "line": None}
        kk = max(2, int(round(mu)))
        mu_b = mu
        if ou:
            pts = [x[0] for v in ou.values() for x in v]
            line = max(set(pts), key=lambda x: (pts.count(x), -abs(x - mu)))
            same = {bk: next(x for x in v if x[0] == line) for bk, v in ou.items() if any(x[0] == line for x in v)}
            fair = [(1 / american(ov)) / (1 / american(ov) + 1 / american(un)) for _, ov, un in same.values() if ov and un]
            mk = sum(fair) / len(fair) if fair else None
            kk = int(math.floor(line)) + 1
            pm = nb_ge(mu, kk, phi)
            pb = (pm + mk) / 2 if mk is not None else pm
            if mk is not None:
                mu_b = (mu + mu_for(mk, kk, phi)) / 2
            row.update({"line": line, "books": len(same), "mk": r3(mk), "pm": r3(pm), "pb": r3(pb), "check": int(mk is not None and abs(pm - mk) > RB["gap"])})
            for side, idx, c in (("ov", 1, pb), ("un", 2, 1 - pb)):
                pr = [(x[idx], bk) for bk, x in same.items() if x[idx]]
                if not pr:
                    continue
                a, bk = max(pr, key=lambda z: american(z[0]))
                dd = american(a)
                row[side] = {"a": a, "bk": bk, "n": len(pr), "p": r3(c), "e": r3(c * dd - 1), "m": r3(c - 1 / dd),
                             "by": {b_: x[idx] for b_, x in same.items() if x[idx]}}
        row["mub"] = r3(mu_b, 2)
        lad = {}
        for bk, lst in (rec.get("r") or {}).items():
            for who, rung, price in lst:
                if norm_name(who) == k:
                    lad.setdefault(rung, {})[bk] = price
        row["lad"] = {}
        for rung, by in sorted(lad.items()):
            a, bk = max(((v, b_) for b_, v in by.items()), key=lambda z: american(z[0]))
            c = nb_ge(mu_b, rung, phi)
            if c < 0.03 or c > 0.97:
                continue
            row["lad"][str(rung)] = {"a": a, "bk": bk, "n": len(by), "p": r3(c), "e": r3(c * american(a) - 1), "by": by}
        row["kk"] = kk
        if r.get("_past"):
            row["act"] = actual.get((gid, k))
            row["wk"] = int(gid.split("_")[1])
        rows.append(row)
    # ---- weekly selections, frozen game by game at kickoff
    out = {"rows": rows, "picks": [], "lad": [], "parl": [], "rule": RB, "phi": r3(phi)}
    weeks = sorted({g["wk"] for g in sched})
    byg = {}
    for x in rows:
        byg.setdefault(x["g"], []).append(x)
    # a live week keeps its games that already finished and left the schedule (Sunday's games on a Monday), with their locked picks
    allw = [(wk, [g for g in sched if g["wk"] == wk] + [{"gid": g_} for g_ in sorted(past.get(wk, ())) if g_ not in games], "live") for wk in weeks] + [(wk, [{"gid": g_} for g_ in sorted(gids)], pastmode[wk]) for wk, gids in sorted(past.items()) if wk not in weeks]
    out["weeks"] = {str(wk): mode for wk, _, mode in allw}
    for wk, gw, mode in allw:
        n_pick = min(5, max(1, math.ceil(len(gw) / 3)))
        # picks: main-line overs and unders
        frozen = [p_ for g in gw if started[g["gid"]] for p_ in ((store.get(g["gid"]) or {}).get("rxp") or [])]
        cands = []
        for g in gw:
            if started[g["gid"]]:
                continue
            for x in byg.get(g["gid"], []):
                if not x.get("line") or x.get("check") or x.get("books", 0) < RB["books"] or x["pos"] not in RB["pos"]:
                    continue
                for side in ("ov", "un"):
                    s = x.get(side)
                    if s and RB["m"] <= s["m"] <= RB["mx"] and american(s["a"]) >= RB["pmin"]:
                        dd = american(s["a"])
                        kel = max(s["e"], 0) / (dd - 1)
                        cands.append({"g": x["g"], "id": x["id"], "n": x["n"], "t": x["t"], "o": x["o"], "pos": x["pos"], "side": side[0], "line": x["line"],
                                      "a": s["a"], "bk": s["bk"], "p": s["p"], "e": s["e"], "m": s["m"], "pm": x["pm"], "mk": x["mk"], "mu": x["mu"],
                                      "u": min(1.5, max(0.25, round(kel * 25 * 4) / 4))})
        cands.sort(key=lambda c: -c["p"])          # likeliest first: the biggest margins were mostly model error
        teams = {p_["t"] for p_ in frozen}
        chosen = []
        for c in cands:
            if len(frozen) + len(chosen) >= n_pick:
                break
            if c["t"] in teams or any(o["id"] == c["id"] for o in chosen):
                continue
            teams.add(c["t"])
            chosen.append(c)
        for g in gw:
            if not started[g["gid"]] and mode == "live":
                store.setdefault(g["gid"], {})["rxp"] = [dict(c, wk=wk) for c in chosen if c["g"] == g["gid"]]
        rt = {"rt": 1} if mode == "retro" else {}
        out["picks"] += [dict(p_, wk=wk, lk=1) for p_ in frozen] + [dict(c, wk=wk, **rt) for c in chosen]
        # ladder watch
        frozen_l = [p_ for g in gw if started[g["gid"]] for p_ in ((store.get(g["gid"]) or {}).get("rxl") or [])]
        lc = []
        for g in gw:
            if started[g["gid"]]:
                continue
            for x in byg.get(g["gid"], []):
                if x.get("check"):
                    continue
                kk = x["kk"]
                good = [(int(k_), v) for k_, v in x["lad"].items() if max(2, kk) <= int(k_) <= kk + 2 and v["p"] >= RB["lp"] and RB["llo"] <= v["e"] <= RB["lhi"]
                        and american(v["a"]) <= RB["lmax"]]
                if len(good) < 2 or max(k_ for k_, _ in good) < kk:
                    continue
                wts = [v["p"] * v["e"] for _, v in good]
                tot = sum(wts)
                lc.append({"g": x["g"], "id": x["id"], "n": x["n"], "t": x["t"], "o": x["o"], "pos": x["pos"], "line": x["line"], "mu": x["mu"], "mub": x["mub"],
                           "te": r3(sum(v["e"] for _, v in good)), "rungs": [[k_, v["a"], v["bk"], v["p"], v["e"], r3(w_ / tot, 2) if tot > 0 else r3(1 / len(good), 2)] for (k_, v), w_ in zip(good, wts)]})
        lc.sort(key=lambda c: -c["te"])
        n_l = min(5, max(1, math.ceil(len(gw) / 3)))
        chosen_l = [c for c in lc if c["id"] not in {p_["id"] for p_ in frozen_l}][:max(0, n_l - len(frozen_l))]
        for g in gw:
            if not started[g["gid"]] and mode == "live":
                store.setdefault(g["gid"], {})["rxl"] = [dict(c, wk=wk) for c in chosen_l if c["g"] == g["gid"]]
        out["lad"] += [dict(p_, wk=wk, lk=1) for p_ in frozen_l] + [dict(c, wk=wk, **rt) for c in chosen_l]
        # parlay ideas: legs that each beat their price at one book, one per player, different games
        key = f"{season}|{wk}"
        pz = store.setdefault("_rxparl", {})
        kept = [i for i in (pz.get(key) or []) if any(started.get(l_["g"]) for l_ in i["legs"])]
        pool = []
        for g in gw:
            if started[g["gid"]]:
                continue
            for x in byg.get(g["gid"], []):
                if x.get("check") or not x.get("line"):
                    continue
                for side in ("ov", "un"):
                    s = x.get(side)
                    if s and s["p"] >= RB["pm"]:
                        for bk, a in s["by"].items():
                            e = s["p"] * american(a) - 1
                            if RB["m"] <= s["p"] - 1 / american(a) <= RB["mx"]:
                                pool.append({"g": x["g"], "id": x["id"], "n": x["n"], "t": x["t"], "lab": ("Over " if side == "ov" else "Under ") + f"{x['line']:g}", "bk": bk, "a": a, "p": s["p"], "e": r3(e)})
                for k_, v in x["lad"].items():
                    if x["kk"] + 1 <= int(k_) <= x["kk"] + 2 and v["p"] >= RB["pr"]:
                        for bk, a in v["by"].items():
                            e = v["p"] * american(a) - 1
                            if RB["plo"] <= e <= RB["phi_"]:
                                pool.append({"g": x["g"], "id": x["id"], "n": x["n"], "t": x["t"], "lab": f"{k_}+ catches", "bk": bk, "a": a, "p": v["p"], "e": r3(e)})
        ideas = []
        for size in (2, 3):
            best = None
            for bk in {l_["bk"] for l_ in pool}:
                L = sorted([l_ for l_ in pool if l_["bk"] == bk], key=lambda z: -z["e"])[:12]
                for combo in itertools.combinations(L, size):
                    if len({c["g"] for c in combo}) < size or len({c["id"] for c in combo}) < size:
                        continue
                    q = math.prod(c["p"] for c in combo)
                    D_ = math.prod(american(c["a"]) for c in combo)
                    ev = q * D_ - 1
                    if best is None or ev > best["ev"]:
                        best = {"bk": bk, "legs": list(combo), "q": r3(q, 4), "d": r3(D_, 3), "ev": r3(ev, 3)}
            if best:
                kel = max(best["ev"], 0) / (best["d"] - 1)
                best["u"] = round(min(0.5, max(0.1, kel * 100 / 8)) * 20) / 20
                ideas.append(best)
        new = [i for i in ideas if len(i["legs"]) not in {len(k_["legs"]) for k_ in kept}]
        if mode == "live":
            pz[key] = [dict(i, wk=wk) for i in kept + new]
        out["parl"] += [dict(i, wk=wk, lk=1 if i in kept else 0, **rt) for i in kept + new]
    for x in rows:                      # past weeks only need the best prices on the page
        if "act" in x:
            for s in ("ov", "un"):
                if x.get(s):
                    x[s].pop("by", None)
            for v in x["lad"].values():
                v.pop("by", None)
    return out

# ---------- kickoff locks ----------
def american(a):
    return 1 + a / 100 if a > 0 else 1 + 100 / -a


# Week 6 of 2026 lost its Sunday kickoff locks (a later save of odds.json wrote over them, Oct 17-19). Rebuilt from movers.json, which
# kept every player's chance and best and FanDuel prices at each run: each game from the last run before its kickoff, in kickoff order,
# by the same Top 8 / long-shot rule. {game: {player: [chance, Top 8, long shot, FanDuel Top 8, FanDuel long shot]}}. Fills only what is missing.
LOCK_RESTORE = json.loads('''{"2026_06_ARI_LA":{"alex bachman":[0.0461,0,0,0,0],"bam knight":[0.1514,0,0,0,0],"blake corum":[0.2437,0,0,0,0],"colby parkinson":[0.2443,0,0,0,1],"davante adams":[0.4711,0,0,0,0],"davis allen":[0.0948,0,0,0,0],"devin duvernay":[0.0413,0,0,0,0],"elijah higgins":[0.0668,0,0,0,0],"jacoby brissett":[0.155,0,0,0,0],"jalen brooks":[0.0663,0,0,0,0],"jordan whittington":[0.0387,0,0,0,0],"kendrick bourne":[0.0779,0,0,0,0],"konata mumpfield":[0.0965,0,0,0,0],"kyren williams":[0.5836,0,0,0,0],"matthew stafford":[0.0682,0,0,0,0],"max klare":[0.0521,0,0,0,0],"michael wilson":[0.3579,1,0,1,0],"puka nacua":[0.4888,0,0,0,0],"reggie virgil":[0.0987,0,0,0,0],"simi fehoko":[0.0204,0,0,0,0],"teagan quitoriano":[0.1009,0,0,0,0],"trey mcbride":[0.3867,0,0,0,0],"tutu atwell":[0.0455,0,0,0,0],"tyler allgeier":[0.2111,0,0,0,0],"tyler higbee":[0.1736,0,0,0,0]},"2026_06_BAL_CLE":{"blake whiteheart":[0.0233,0,0,0,0],"carsen ryan":[0.0199,0,0,0,0],"denzel boston":[0.3127,0,0,0,0],"derrick henry":[0.4932,0,0,0,0],"deshaun watson":[0.2351,0,1,0,0],"devontez walker":[0.084,0,0,0,0],"elijah sarratt":[0.0554,0,0,0,0],"harold fannin":[0.2984,0,0,0,0],"isaiah bond":[0.0736,0,0,0,0],"jakobi lane":[0.1875,0,1,0,1],"jaleel mclaughlin":[0.0618,0,0,0,0],"jerry jeudy":[0.1073,0,0,0,0],"josh cuevas":[0.0208,0,0,0,0],"justice hill":[0.0902,0,0,0,0],"kc concepcion":[0.1082,0,0,0,0],"lajohntay wester":[0.0205,0,0,0,0],"malachi corley":[0.0422,0,0,0,0],"mark andrews":[0.2774,0,0,0,0],"michael burton":[0.0662,0,0,0,0],"nick vannett":[0.0237,0,0,0,0],"quinshon judkins":[0.3868,1,0,0,0],"rashod bateman":[0.1904,0,0,0,0],"tylan wallace":[0.0315,0,0,0,0],"tyler huntley":[0.1834,0,0,0,0],"zay flowers":[0.1981,0,0,0,0]},"2026_06_BUF_LV":{"ashton jeanty":[0.5288,0,0,0,0],"brock bowers":[0.4804,0,0,0,0],"connor heyward":[0.0893,0,0,0,0],"dalton kincaid":[0.1365,0,0,0,0],"dareke young":[0.0291,0,0,0,0],"dawson knox":[0.2682,0,0,0,0],"dj moore":[0.2463,0,0,0,0],"dylan laube":[0.1142,0,0,0,0],"greg dortch":[0.0611,0,0,0,0],"ian thomas":[0.0338,0,0,0,0],"jackson hawes":[0.0343,0,0,0,0],"jalen nailor":[0.1574,0,0,0,0],"james cook":[0.4908,0,0,0,0],"josh allen":[0.4222,0,0,0,0],"josh palmer":[0.1212,0,0,0,0],"keleki latu":[0.0168,0,0,0,0],"keon coleman":[0.2377,0,0,0,0],"khalil shakir":[0.2225,0,0,0,0],"kirk cousins":[0.0397,0,0,0,0],"malik benson":[0.0658,0,0,0,0],"michael mayer":[0.2353,0,0,0,0],"mike washington":[0.0923,0,0,0,0],"ray davis":[0.0723,0,0,0,0],"skyler bell":[0.1829,0,0,0,0],"tre tucker":[0.3016,0,0,0,0],"ty johnson":[0.2092,0,0,0,0]},"2026_06_CAR_PHI":{"anthony tyus":[0.0867,0,0,0,0],"bryce young":[0.0879,0,0,0,0],"brycen tremayne":[0.0717,0,0,0,0],"chuba hubbard":[0.4453,0,0,0,0],"dallas goedert":[0.348,0,0,0,0],"dameon pierce":[0.3535,0,0,0,0],"darius cooper":[0.1339,0,0,0,0],"darren waller":[0.196,0,0,0,0],"david moore":[0.0834,0,0,0,0],"devonta smith":[0.2502,0,0,0,0],"dontayvion wicks":[0.1707,0,0,0,0],"ej jenkins":[0.0135,0,0,0,0],"elijah moore":[0.0525,0,0,0,0],"jalen coker":[0.2448,0,0,0,0],"john metchie":[0.1258,0,0,0,0],"johnny mundt":[0.0211,0,0,0,0],"makai lemon":[0.0784,0,0,0,0],"marquise brown":[0.0929,0,0,0,0],"mitchell evans":[0.0923,0,0,0,0],"tetairoa mcmillan":[0.3578,1,0,1,0],"tommy tremble":[0.0734,0,0,0,0],"trevor etienne":[0.0677,0,0,0,0]},"2026_06_CHI_ATL":{"austin hooper":[0.119,0,0,0,0],"bijan robinson":[0.6015,0,0,0,0],"brian robinson":[0.2175,0,0,0,0],"caleb williams":[0.2685,0,0,0,0],"charlie woerner":[0.0583,0,0,0,0],"chris blair":[0.0336,0,0,0,0],"cole kmet":[0.1252,0,0,0,0],"colston loveland":[0.3253,0,0,0,0],"dandre swift":[0.5171,0,0,0,0],"drake london":[0.3254,0,0,0,0],"jahan dotson":[0.2566,0,0,0,0],"jahdae walker":[0.081,0,0,0,0],"kalif raymond":[0.1114,0,0,0,0],"kyle pitts":[0.3023,0,0,1,0],"luther burden":[0.293,0,0,0,0],"michael penix":[0.0552,0,0,0,0],"nick muse":[0.019,0,0,0,0],"olamide zaccheaus":[0.1065,0,0,0,0],"rome odunze":[0.3682,1,0,1,0],"roschon johnson":[0.2115,0,0,0,0],"sam roush":[0.0367,0,0,0,0],"zachariah branch":[0.0491,0,0,0,0],"zavion thomas":[0.0315,0,0,0,0]},"2026_06_DAL_GB":{"bo melton":[0.0661,0,0,0,0],"brevyn spann ford":[0.0363,0,0,0,0],"ceedee lamb":[0.3734,0,0,0,0],"christian watson":[0.421,0,0,0,0],"dak prescott":[0.1177,0,0,0,0],"george pickens":[0.4002,0,0,0,0],"hunter luepke":[0.0601,0,0,0,0],"j michael sturdivant":[0.0376,0,0,0,0],"jake ferguson":[0.2765,0,0,0,0],"javonte williams":[0.6277,0,0,0,0],"jonathan mingo":[0.0688,0,0,0,0],"jonnu smith":[0.0965,0,0,0,0],"jordan love":[0.0655,0,0,0,0],"josh whyle":[0.0377,0,0,0,0],"kaleb johnson":[0.1647,0,0,0,0],"kavontae turpin":[0.0652,0,0,0,0],"luke schoonmaker":[0.0257,0,0,0,0],"mark redman":[0.0261,0,0,0,0],"marshawn lloyd":[0.4012,0,0,0,0],"matthew golden":[0.3357,0,0,0,0],"pierre strong":[0.1433,0,0,0,0],"ryan flournoy":[0.2217,0,0,0,0],"savion williams":[0.0657,0,0,0,0],"tucker kraft":[0.3537,0,0,0,0],"tyler goodson":[0.0659,0,0,0,0]},"2026_06_HOU_JAX":{"ameer abdullah":[0.0532,0,0,0,0],"bhayshul tuten":[0.3113,0,0,0,0],"brenton strange":[0.2132,0,0,0,0],"brevin jordan":[0.0318,0,0,0,0],"brian thomas":[0.2207,0,0,0,0],"cade stover":[0.0413,0,0,0,0],"chris rodriguez":[0.2254,0,0,0,0],"cj stroud":[0.1432,0,0,0,0],"dalton schultz":[0.1705,0,0,0,0],"dare ogunbowale":[0.1141,0,0,0,0],"foster moreau":[0.1417,0,0,0,0],"jakobi meyers":[0.3035,0,0,0,0],"jaylin noel":[0.0928,0,0,0,0],"josh cameron":[0.1251,0,0,0,0],"kayshon boutte":[0.1381,0,0,0,0],"lequint allen":[0.0681,0,0,0,0],"nate boerkircher":[0.0568,0,0,0,0],"nico collins":[0.3189,0,0,0,0],"parker washington":[0.2814,0,0,0,0],"quintin morris":[0.0289,0,0,0,0],"tanner koziol":[0.0175,0,0,0,0],"travis hunter":[0.076,0,0,0,0],"trevor lawrence":[0.1564,0,0,0,0],"woody marks":[0.4034,0,0,0,0],"xavier hutchinson":[0.1376,0,0,0,0]},"2026_06_LAC_KC":{"ben vansumeren":[0.0505,0,0,0,0],"brashard smith":[0.1453,0,0,0,0],"brenen thompson":[0.039,0,0,0,0],"charlie kolar":[0.0391,0,0,0,0],"cyrus allen":[0.039,0,0,0,0],"derius davis":[0.0203,0,0,0,0],"emmett johnson":[0.0972,0,0,0,0],"hayden rucci":[0.0575,0,0,0,0],"jake briningstool":[0.0352,0,0,0,0],"jalen royals":[0.1271,0,0,0,0],"justin herbert":[0.2317,0,1,0,1],"keandre lambert smith":[0.0386,0,0,0,0],"keaton mitchell":[0.1078,0,0,0,0],"kenneth walker":[0.5966,0,0,0,0],"kimani vidal":[0.13,0,0,0,0],"ladd mcconkey":[0.1616,0,0,0,0],"marquez valdes scantling":[0.0833,0,0,0,0],"noah gray":[0.1473,0,0,0,0],"omarion hampton":[0.287,0,0,0,0],"oronde gadsden":[0.1107,0,0,0,0],"patrick mahomes":[0.1225,0,0,0,0],"quentin johnston":[0.1498,0,0,0,0],"scott matlock":[0.0425,0,0,0,0],"theo wease":[0.0708,0,0,0,0],"travis kelce":[0.3205,0,0,0,0],"xavier worthy":[0.3304,0,0,0,0]},"2026_06_NO_NYG":{"alvin kamara":[0.2601,0,0,0,0],"barion brown":[0.0368,0,0,0,0],"bryce lance":[0.0974,0,0,0,0],"cam skattebo":[0.3685,0,0,0,0],"cedric tillman":[0.142,0,0,0,0],"charlie jones":[0.0321,0,0,0,0],"chris manhertz":[0.0194,0,0,0,0],"chris olave":[0.3518,0,0,0,0],"cj donaldson":[0.096,0,0,0,0],"darnell mooney":[0.2082,0,1,0,1],"devaughn vele":[0.3292,0,0,0,0],"isaiah likely":[0.3216,0,0,0,0],"jameis winston":[0.0889,0,0,0,0],"juwan johnson":[0.2837,0,0,0,0],"kendre miller":[0.2772,0,0,0,0],"malachi fields":[0.1038,0,0,0,0],"malik nabers":[0.4177,1,0,1,0],"najee harris":[0.2344,0,0,0,0],"noah fant":[0.1502,0,0,0,0],"oscar delp":[0.023,0,0,0,0],"patrick ricard":[0.0558,0,0,0,0],"treyton welch":[0.0172,0,0,0,0],"tyler shough":[0.1853,0,0,0,0],"tyrone tracy":[0.1658,0,0,0,0]},"2026_06_NYJ_NE":{"aj brown":[0.2834,0,0,0,0],"andrew beck":[0.0754,0,0,0,0],"braelon allen":[0.305,0,0,0,0],"cameron latu":[0.0131,0,0,0,0],"corey kiner":[0.1032,0,0,0,0],"demario douglas":[0.2002,0,0,0,0],"drake maye":[0.2579,0,0,0,0],"efton chism":[0.1242,0,0,0,0],"eli raridon":[0.1892,0,0,0,0],"garrett wilson":[0.2782,0,0,0,0],"geno smith":[0.115,0,0,0,0],"hunter henry":[0.2368,0,0,0,0],"isaiah davis":[0.1214,0,0,0,0],"isaiah williams":[0.0936,0,0,0,0],"jelani woods":[0.0342,0,0,0,0],"jeremy ruckert":[0.1183,0,0,0,0],"kenyon sadiq":[0.1642,0,0,0,0],"kyle williams":[0.0704,0,0,0,0],"malik mcclain":[0.0351,0,0,0,0],"mason taylor":[0.124,0,0,0,0],"omar cooper":[0.059,0,0,0,0],"rhamondre stevenson":[0.4027,0,0,0,0],"romeo doubs":[0.3368,1,0,1,0],"tanner arkin":[0.0253,0,0,0,0],"tim patrick":[0.0873,0,0,0,0],"treveyon henderson":[0.211,0,0,0,0]},"2026_06_PIT_TB":{"aaron rodgers":[0.0345,0,0,0,0],"bauer sharp":[0.0147,0,0,0,0],"brandon smith":[0.0474,0,0,0,0],"bucky irving":[0.3312,0,0,0,0],"cade otton":[0.1582,0,0,0,0],"chris godwin":[0.1619,0,0,0,0],"darnell washington":[0.2357,0,0,0,0],"dk metcalf":[0.3403,1,0,0,0],"emeka egbuka":[0.306,0,0,1,0],"germie bernard":[0.1222,0,0,0,0],"jalon daniels":[0.1642,0,0,0,0],"jaylen warren":[0.3351,0,0,0,0],"kameron johnson":[0.0202,0,0,0,0],"kenny gainwell":[0.2176,0,0,0,0],"ko kieft":[0.0122,0,0,0,0],"lew nichols":[0.0969,0,0,0,0],"pat freiermuth":[0.2981,0,0,0,0],"payne durham":[0.0273,0,0,0,0],"rico dowdle":[0.2423,0,0,0,0],"riley nowakowski":[0.0513,0,0,0,0],"robert tonyan":[0.0219,0,0,0,0],"sean tucker":[0.1754,0,0,0,0],"ted hurst":[0.1345,0,0,0,0],"tez johnson":[0.081,0,0,0,0],"travis homer":[0.0564,0,0,0,0]},"2026_06_SEA_DEN":{"adam prentice":[0.0414,0,0,0,0],"adam trautman":[0.0257,0,0,0,0],"aj barner":[0.3327,0,0,0,0],"bo nix":[0.173,0,0,0,0],"brady russell":[0.0402,0,0,0,0],"cooper kupp":[0.1967,0,1,0,1],"courtland sutton":[0.3218,1,0,1,0],"drew lock":[0.0312,0,0,0,0],"elijah arroyo":[0.0335,0,0,0,0],"emanuel wilson":[0.3721,0,0,0,0],"eric saubert":[0.0397,0,0,0,0],"evan engram":[0.1091,0,0,0,0],"george holani":[0.2186,1,0,1,0],"jaxon smith njigba":[0.4103,0,0,0,0],"jaylen waddle":[0.1609,0,0,0,0],"jk dobbins":[0.2919,0,0,0,0],"liljordan humphrey":[0.0924,0,0,0,0],"marvin mims":[0.058,0,0,0,0],"nate adkins":[0.0842,0,0,0,0],"rashid shaheed":[0.1017,0,0,0,0],"rj harvey":[0.2366,0,0,0,0],"tory horton":[0.1114,0,0,0,0],"troy franklin":[0.1033,0,0,0,0],"tyler badie":[0.0638,0,0,0,0]},"2026_06_TEN_IND":{"andrew ogletree":[0.0355,0,0,0,0],"calvin ridley":[0.0647,0,0,0,0],"cam ward":[0.1403,0,0,0,0],"carnell tate":[0.1851,0,0,0,0],"chimere dike":[0.0716,0,0,0,0],"daniel bellinger":[0.0493,0,0,0,0],"daniel jones":[0.1571,0,0,0,0],"david martin robinson":[0.0169,0,0,0,0],"deion burks":[0.0786,0,0,0,0],"elic ayomanor":[0.223,0,1,0,0],"gunnar helm":[0.1528,0,0,0,0],"jonathan taylor":[0.6066,0,0,0,0],"josh downs":[0.2182,0,0,0,0],"keenan allen":[0.2765,0,0,0,0],"kylen granson":[0.0229,0,0,0,0],"laquon treadwell":[0.1809,0,0,0,0],"mo alie cox":[0.0645,0,0,0,0],"nicholas singleton":[0.0488,0,0,0,0],"seth mcgowan":[0.1096,0,0,0,0],"tony pollard":[0.3585,0,0,0,0],"tyjae spears":[0.137,0,0,0,0],"tyler warren":[0.4351,1,0,0,0],"wandale robinson":[0.2579,0,0,0,0],"will mallory":[0.0292,0,0,0,0]}}''')


def restore_locks(store):
    for gid, rows in LOCK_RESTORE.items():
        rec = store.get(gid)
        if not isinstance(rec, dict):
            continue
        lk = rec.get("lock")
        if not lk:
            rec["lock"] = {n: [v[0], None, v[1], v[2], None, [v[3], v[4], None]] for n, v in rows.items()}
        else:
            for n, v in rows.items():
                if n in lk and len(lk[n]) == 5:
                    lk[n].append([v[3], v[4], None])


def apply_locks(sched, picks, now=None):
    """Freeze each game's numbers at kickoff. In the two hours before a game its rows are saved (the last save wins); once it has
    kicked off the saved rows are shown instead of fresh ones, so picks do not move during the game. The saved chances are kept
    for good in odds.json, and the Tracker grades this season's games on them: the record is what the page actually showed."""
    now = now or datetime.now(ET)
    try:
        store = json.load(open(ODDS, encoding="utf-8"))
    except Exception:
        store = {}
    restore_locks(store)
    by = {}
    for r in picks:
        by.setdefault(r["g"], []).append(r)
    live = {u["gid"] for u in sched}
    hours = {u["gid"]: (datetime.fromisoformat(u["ts"]) - now).total_seconds() / 3600 for u in sched}
    # who is on the Top 8 and long-shot lists right now, with every game already under way held as it stood at its kickoff
    member = {}
    for sfx, only in (("", None), ("f", "FanDuel")):      # held at the best price, and again at FanDuel's own prices
      for wk in {u["wk"] for u in sched}:
        cand = []
        for u in sched:
            if u["wk"] != wk:
                continue
            rec = store.get(u["gid"]) or {}
            started = hours[u["gid"]] <= 0 and rec.get("lockrows")
            best, best2 = {}, {}
            for src, dst in ((rec.get("b") or {}, best), (rec.get("t") or {}, best2)):
                for bk_, rows_ in src.items():
                    if only and bk_ != only:
                        continue
                    for x in rows_:
                        k = norm_name(x[0])
                        if isinstance(x[1], (int, float)) and x[1] and (k not in dst or american(x[1]) > dst[k]):
                            dst[k] = american(x[1])
            for r in (rec["lockrows"] if started else by.get(u["gid"], [])):
                d = best.get(norm_name(r["n"]))
                ev = r["p"] * d - 1 if d else None
                pick = bool(d and r["p"] >= 0.2 and 0.03 <= ev <= 0.25)
                long_ = bool(d and not pick and 0.10 <= r["p"] < 0.25 and 5 <= d < 10 and ev > 0)
                d2, p2 = best2.get(norm_name(r["n"])), two_chance(r["p"], r.get("pos"), r.get("T"))
                ev2 = p2 * d2 - 1 if d2 else None
                cand.append({"key": (u["gid"], r["id"]), "p": r["p"], "ev": ev, "pick": pick, "long": long_,
                             "two": bool(d2 and p2 >= 0.15), "k2": p2,          # the 2+ TD picks: the four likeliest with a price, whatever the edge
                             "t8": r.get("t8" + sfx) if started else None, "l4": r.get("l4" + sfx) if started else None, "d2": r.get("d2" + sfx) if started else None})
        for flag, kind, n, rank in (("t8", "pick", 8, "p"), ("l4", "long", 4, "ev"), ("d2", "two", 4, "k2")):
            # held players keep their place; a player whose game is still to come joins when he ranks in the best n of
            # everyone still eligible, so the list can grow past n but never drops anybody
            best_n = sorted((c for c in cand if c[flag] != 0 and c[kind]), key=lambda c: -c[rank])[:n]
            for c in cand:
                if c[flag] == 1 or (c[flag] is None and any(c is x for x in best_n)):
                    member.setdefault(c["key"], {})[flag + sfx] = 1
    out = []
    for u in sched:
        hrs = hours[u["gid"]]
        rows = by.get(u["gid"], [])
        rec = store.get(u["gid"]) or {}
        if 0 < hrs <= 2 and rows:
            flags = lambda r, f: int((member.get((u["gid"], r["id"])) or {}).get(f, 0))
            rec["lockrows"] = [dict(r, t8=flags(r, "t8"), l4=flags(r, "l4"), d2=flags(r, "d2"), t8f=flags(r, "t8f"), l4f=flags(r, "l4f"), d2f=flags(r, "d2f")) for r in rows]
            rec["lock"] = {norm_name(r["n"]): [r["p"], r["pf"], flags(r, "t8"), flags(r, "l4"), flags(r, "d2"), [flags(r, "t8f"), flags(r, "l4f"), flags(r, "d2f")]] for r in rows}
            store[u["gid"]] = rec
            u["lk"] = 1          # saved: the page can say "frozen" the moment the game kicks off
        elif hrs <= 0 and rec.get("lockrows"):
            rows = rec["lockrows"]
            u["locked"] = 1
        out += rows
    for k, v in store.items():
        if k != "_meta" and k not in live and isinstance(v, dict):
            v.pop("lockrows", None)
    with open(ODDS, "w", encoding="utf-8") as f:
        json.dump(store, f, separators=(",", ":"), ensure_ascii=False, sort_keys=True)
    return out, store


# ---------- live results ----------
ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/"
ESPN_ABBR = {"WSH": "WAS", "LAR": "LA"}


def parse_summary(summ):
    """Touchdowns in an ESPN game summary: ({player: count}, first scorer, [[player, quarter, clock, kind, wallclock], ...],
    {player: [full name, team]}). Counts come from the box score (full names); each touchdown play is matched back to those names."""
    td, kinds = {}, {}
    for team in ((summ.get("boxscore") or {}).get("players") or []):
        ab = (team.get("team") or {}).get("abbreviation") or ""
        ab = ESPN_ABBR.get(ab, ab)
        for grp in team.get("statistics") or []:
            labels = grp.get("labels") or []
            if grp.get("name") == "passing" or "TD" not in labels:
                continue
            i = labels.index("TD")
            for a in grp.get("athletes") or []:
                try:
                    n = int(float((a.get("stats") or [])[i]))
                except Exception:
                    n = 0
                if n > 0:
                    full = (a.get("athlete") or {}).get("displayName") or ""
                    k = norm_name(full)
                    td[k] = td.get(k, 0) + n
                    kinds.setdefault(k, {"full": full, "team": ab, "g": set()})["g"].add(grp.get("name"))
    plays = []
    dr = summ.get("drives") or {}
    for d in (dr.get("previous") or []) + ([dr["current"]] if isinstance(dr.get("current"), dict) else []):
        plays += d.get("plays") or []
    feed, seen = [], set()
    key = lambda x: f"{(x.get('period') or {}).get('number')}|{(x.get('clock') or {}).get('displayValue')}"      # one touchdown per quarter-and-clock
    wall = {key(x): x.get("wallclock") for x in plays if x.get("wallclock")}      # the scoring summary has no time of day; the drive plays do
    for sp in (summ.get("scoringPlays") or []) + plays:
        kind = ((sp.get("scoringType") or {}).get("abbreviation") or "") + " " + ((sp.get("type") or {}).get("text") or "")
        if "TD" not in kind and "touchdown" not in kind.lower():
            continue
        if key(sp) in seen:
            continue
        seen.add(key(sp))
        text = sp.get("text") or ""
        who = None
        m = re.match(r"^(.*?)\s+\d+\s+(?:Yd|Yard)", text)          # "Jonathan Taylor 3 Yd Run" style
        if m and norm_name(m.group(1)) in td:
            who = norm_name(m.group(1))
        else:
            ret = bool(re.search("return|interception|fumble|block|kickoff|punt", kind, re.I))
            want = "ret" if ret else "receiving" if "pass" in kind.lower() else "rushing" if "rush" in kind.lower() else None
            best = None
            for k, v in kinds.items():                              # "J.Williams left guard ..." style: initial + surname
                parts = v["full"].split()
                if len(parts) < 2:
                    continue
                if want == "ret":                                   # a return: the scorer is credited outside rushing/receiving
                    if not (v["g"] - {"rushing", "receiving"}):
                        continue
                elif want and want not in v["g"]:
                    continue
                pos = text.rfind(parts[0][0] + "." + parts[1]) if want in ("ret", "receiving") else text.find(parts[0][0] + "." + parts[1])
                if pos >= 0 and (best is None or (pos > best[0] if want in ("ret", "receiving") else pos < best[0])):
                    best = (pos, k)
            who = best[1] if best else None
        label = "Return" if re.search("return|interception|fumble|block|kickoff|punt", kind, re.I) else "Rec" if "pass" in kind.lower() else "Rush" if "rush" in kind.lower() else "TD"
        feed.append([who, (sp.get("period") or {}).get("number"), (sp.get("clock") or {}).get("displayValue") or "", label, sp.get("wallclock") or wall.get(key(sp)) or ""])
    first = feed[0][0] if feed else None
    return td, first, feed, {k: [v["full"], v["team"]] for k, v in kinds.items()}


BOX = {"passing": ("p", ["C/ATT", "YDS", "TD", "INT"]), "rushing": ("r", ["CAR", "YDS", "TD", "LONG"]),
       "receiving": ("c", ["TGTS", "REC", "YDS", "TD", "LONG"])}


def parse_box(summ):
    """Offensive box score from an ESPN game summary: {team: {"p": [[name, C/ATT, yards, TD, INT]], "r": [[name, carries, yards, TD, long]],
    "c": [[name, targets, catches, yards, TD, long]]}}."""
    out = {}
    for team in ((summ.get("boxscore") or {}).get("players") or []):
        ab = (team.get("team") or {}).get("abbreviation") or ""
        ab = ESPN_ABBR.get(ab, ab)
        for grp in team.get("statistics") or []:
            if grp.get("name") not in BOX:
                continue
            key, want = BOX[grp["name"]]
            labels = grp.get("labels") or []
            for a in grp.get("athletes") or []:
                st = a.get("stats") or []
                row = [(a.get("athlete") or {}).get("displayName") or ""] + [st[labels.index(w)] if w in labels and labels.index(w) < len(st) else "" for w in want]
                out.setdefault(ab, {}).setdefault(key, []).append(row)
    return out


def espn_notes(names):
    """Latest injury note for each listed player from ESPN's injury feed: {player: [status, one-line note, date]}."""
    out = {}
    try:
        data = json.loads(get(ESPN + "injuries", 30))
        for team in data.get("injuries") or []:
            for it in team.get("injuries") or []:
                k = norm_name((it.get("athlete") or {}).get("displayName") or "")
                if k in names:
                    out[k] = [it.get("status") or "", (it.get("shortComment") or "")[:240], (it.get("date") or "")[:10]]
    except Exception as ex:
        print(f"injury notes unavailable: {ex}", file=sys.stderr)
    return out


def live_scores(sched):
    """Touchdown scorers for listed games that have kicked off, from ESPN, so picks can be marked before the play-by-play arrives.
    {game: {"st": "in" or "post", "td": {player: touchdowns}, "first": player}}. Empty if ESPN cannot be read."""
    now = datetime.now(ET)
    started = [u for u in sched if datetime.fromisoformat(u["ts"]) <= now]
    out = {}
    try:
        for day in sorted({u["d"] for u in started}):
            sb = json.loads(get(f"{ESPN}scoreboard?dates={day.replace('-', '')}", 30))
            for e in sb.get("events") or []:
                comp = (e.get("competitions") or [{}])[0]
                tm = {c.get("homeAway"): ESPN_ABBR.get((c.get("team") or {}).get("abbreviation"), (c.get("team") or {}).get("abbreviation")) for c in comp.get("competitors") or []}
                u = next((x for x in started if x["a"] == tm.get("away") and x["h"] == tm.get("home")), None)
                state = ((e.get("status") or {}).get("type") or {}).get("state")
                if not u or state not in ("in", "post"):
                    continue
                rec = {"st": state, "td": {}, "first": None,
                       "sc": {ESPN_ABBR.get((c.get("team") or {}).get("abbreviation"), (c.get("team") or {}).get("abbreviation")): c.get("score") for c in comp.get("competitors") or []},
                       "clk": ((e.get("status") or {}).get("type") or {}).get("shortDetail") or ""}
                try:
                    summ = json.loads(get(f"{ESPN}summary?event={e['id']}", 30))
                    rec["td"], rec["first"], rec["ev"], rec["nm"] = parse_summary(summ)
                    rec["box"] = parse_box(summ)
                except Exception as ex:
                    print(f"live: {u['gid']}: {ex}", file=sys.stderr)
                out[u["gid"]] = rec
    except Exception as ex:
        print(f"live results unavailable: {ex}", file=sys.stderr)
    return out


# ---------- prices ----------
def odds_key():
    k = os.environ.get("ODDS_API_KEY", "").strip()
    if not k:
        try:
            k = open("odds_key.txt", encoding="utf-8").read().strip()
        except Exception:
            pass
    return k


def parse_td(doc, market="player_anytime_td"):
    """{book: [[player, american price], ...]} for one event."""
    out = {}
    for bk in doc.get("bookmakers") or []:
        for mk in bk.get("markets") or []:
            if mk.get("key") != market:
                continue
            rows = []
            for o in mk.get("outcomes") or []:
                who, price = o.get("description") or o.get("name"), o.get("price")
                if not who or str(o.get("name")) == "No" or who in ("Yes", "No") or not isinstance(price, (int, float)):
                    continue
                rows.append([who, int(price)])
            if rows:
                out[bk.get("title") or bk.get("key")] = rows
    return out


def parse_two(doc):
    """{book: [[player, american price], ...]} for two or more touchdowns (the over on a 1.5 line) in one event."""
    out = {}
    for bk in doc.get("bookmakers") or []:
        rows = {}
        for mk in bk.get("markets") or []:
            if mk.get("key") not in ("player_tds_over", "player_tds"):
                continue
            for o in mk.get("outcomes") or []:
                who, price, pt = o.get("description"), o.get("price"), o.get("point")
                if who and str(o.get("name")) == "Over" and isinstance(price, (int, float)) and isinstance(pt, (int, float)) and abs(pt - 1.5) < 0.01:
                    rows[who] = int(price)
        if rows:
            out[bk.get("title") or bk.get("key")] = [[k, v] for k, v in rows.items()]
    return out


def parse_ladder(doc, market="player_receptions_alternate"):
    """{book: [[player, rung, american price], ...]} for the alt-receptions ladder: the over on 4.5 is stored as rung 5 (5+ catches)."""
    out = {}
    for bk in doc.get("bookmakers") or []:
        rows = {}
        for mk in bk.get("markets") or []:
            if mk.get("key") != market:
                continue
            for o in mk.get("outcomes") or []:
                who, price, pt = o.get("description"), o.get("price"), o.get("point")
                if who and str(o.get("name")) == "Over" and isinstance(price, (int, float)) and isinstance(pt, (int, float)):
                    rows[(who, int(math.floor(pt)) + 1)] = int(price)
        if rows:
            out[bk.get("title") or bk.get("key")] = [[k[0], k[1], v] for k, v in sorted(rows.items())]
    return out


def parse_ou(doc, market="player_receptions"):
    """{book: [[player, line, over price, under price], ...]} for a main over/under player market (a side can be None)."""
    out = {}
    for bk in doc.get("bookmakers") or []:
        rows = {}
        for mk in bk.get("markets") or []:
            if mk.get("key") != market:
                continue
            for o in mk.get("outcomes") or []:
                who, price, pt, side = o.get("description"), o.get("price"), o.get("point"), str(o.get("name"))
                if not who or not isinstance(price, (int, float)) or not isinstance(pt, (int, float)) or side not in ("Over", "Under"):
                    continue
                x = rows.setdefault((who, float(pt)), [None, None])
                x[0 if side == "Over" else 1] = int(price)
        if rows:
            out[bk.get("title") or bk.get("key")] = [[k[0], k[1], v[0], v[1]] for k, v in sorted(rows.items())]
    return out


TWO = {"p": 0.20, "lo": -0.10, "hi": 0.50}      # the 2+ TD picks: chance of two 20%+, edge -10% to +50%, best four by edge (the page uses the same numbers)


TWO_W = {"c": 0.362, "s": 1.035, "RB": -0.138, "WR": -0.54, "QB": 0.409, "T": 0.257}


def two_chance(p, pos=None, T=None):
    """Chance of two or more touchdowns. Starts from what the chance of one implies if touchdowns arrived independently, then
    corrects it: second touchdowns cluster with lead backs, running quarterbacks and high-scoring teams, and are rarer for
    receivers. Weights fitted on 7,693 priced player-games (2025 and early 2026); the page uses the same numbers."""
    p = min(max(p, 1e-6), 0.999)
    lam = -math.log(1 - p)
    q = min(max(1 - (1 - p) * (1 + lam), 1e-6), 0.99)
    z = TWO_W["c"] + TWO_W["s"] * math.log(q / (1 - q)) + TWO_W.get(pos, 0.0) + TWO_W["T"] * ((T if T is not None else 2.5) - 2.5)
    return 1 / (1 + math.exp(-z))


def remember_odds(sched, old, force=False):
    """Anytime-TD prices for the listed games from The Odds API, kept in odds.json.

    Two requests per game per look (anytime and first touchdown, one region). ODDS_LOOKS sets the looks, in hours before
    kickoff (default "24,3,1": the day before, after inactives are close, and just before kickoff). About 420 requests a month.
    Pulling stops when fewer than 15 requests remain on the key."""
    store = dict((old or {}).get("odds") or {})
    try:
        with open(ODDS, encoding="utf-8") as f:
            store.update(json.load(f))
    except Exception:
        pass
    key = odds_key()
    meta = store.get("_meta") or {}
    if not key:
        meta["key"] = 0
        store["_meta"] = meta
        return store
    meta["key"] = 1
    now = datetime.now(ET)
    looks = sorted((float(x) for x in os.environ.get("ODDS_LOOKS", "24,3,1").split(",") if x.strip()), reverse=True)
    want = []
    for u in sched:
        hrs = (datetime.fromisoformat(u["ts"]) - now).total_seconds() / 3600
        rec = store.get(u["gid"]) or {}
        due = sum(1 for h in looks if hrs <= h)          # looks whose time has come
        nof = rec.get("b") and not rec.get("f") and rec.get("ftries", 0) < 3 and hrs <= looks[0]     # one catch-up for first-TD prices
        nof = nof or (rec.get("b") and not rec.get("t") and rec.get("ttries", 0) < 2 and hrs <= looks[0])      # and for 2+ TD prices
        nof = nof or (rec.get("b") and not rec.get("r") and rec.get("rtries", 0) < 2 and hrs <= looks[0])      # and for receptions ladders
        if 0 < hrs and ((due > rec.get("n", 0) and rec.get("tries", 0) < 8) or nof or (force and hrs <= 72)):
            want.append((u, due))
    if want:
        try:
            events = json.loads(get(f"{ODDS_API}/events?apiKey={key}", 60))     # listing events is free
            ids = {(TEAM_NAMES.get(e.get("away_team")), TEAM_NAMES.get(e.get("home_team"))): e["id"] for e in events}
            for u, due in want:
                eid = ids.get((u["a"], u["h"]))
                if not eid:
                    continue
                raw, hd = get(f"{ODDS_API}/events/{eid}/odds?apiKey={key}&regions=us&markets=player_anytime_td,player_1st_td,player_tds_over,player_receptions,player_receptions_alternate&oddsFormat=american", 60, headers=True)
                doc = json.loads(raw)
                books, first, two, lad, ou = parse_td(doc), parse_td(doc, "player_1st_td"), parse_two(doc), parse_ladder(doc), parse_ou(doc)
                left = hd.get("x-requests-remaining")
                if left is not None:
                    meta["left"] = int(float(left))
                rec = store.get(u["gid"]) or {}
                if books:
                    if "b0" not in rec:             # the first look is kept for closing-line value: did the price shorten by kickoff?
                        rec["b0"], rec["at0"] = (rec["b"], rec.get("at")) if rec.get("b") else (books, now.strftime("%Y-%m-%dT%H:%M"))
                    rec.update({"at": now.strftime("%Y-%m-%dT%H:%M"), "b": books, "n": max(due, rec.get("n", 0))})
                    if first:
                        rec["f"] = first
                    else:
                        rec["ftries"] = rec.get("ftries", 0) + 1
                    if two:
                        rec["t"] = two
                    else:
                        rec["ttries"] = rec.get("ttries", 0) + 1
                    if lad:
                        rec["r"] = lad
                    else:
                        rec["rtries"] = rec.get("rtries", 0) + 1
                    if ou:
                        rec["ro"] = ou
                else:
                    rec["tries"] = rec.get("tries", 0) + 1      # market not posted yet; try again next run
                store[u["gid"]] = rec
                if left is not None and float(left) < 15:
                    print("odds: request allowance nearly used; stopping", file=sys.stderr)
                    break
            meta.pop("err", None)
        except Exception as e:
            meta["err"] = str(e).replace(key, "***")[:200]
            print(f"odds unavailable: {meta['err']}", file=sys.stderr)
    store["_meta"] = meta
    with open(ODDS, "w", encoding="utf-8") as f:
        json.dump(store, f, separators=(",", ":"), ensure_ascii=False, sort_keys=True)
    return store


def backfill(season, lead=60, go=False, limit=None, back=0, first=False, two=False, lad=False, yds=False):
    """Closing-ish anytime-TD prices for this season's finished games, from The Odds API's historical endpoints (paid plans only).

    For each finished game without stored prices it asks for the snapshot `lead` minutes before kickoff. Cost: 10 requests per
    game, plus 1 per distinct kickoff slot to look up event ids. Nothing is spent unless go is set."""
    season = season or current_season()
    key = odds_key()
    if not key:
        sys.exit("No Odds API key. Put it in odds_key.txt next to this script or in the ODDS_API_KEY environment variable.")
    g = pd.read_csv(cached("games.csv", GAMES, True))
    g = g[g.season.isin(range(season - back, season + 1)) & g.home_score.notna()].sort_values(["gameday", "gametime"], ascending=False)
    try:
        store = json.load(open(ODDS, encoding="utf-8"))
    except Exception:
        store = {}
    if yds:        # alt receiving-yards ladders (back-test only for now): one market, 10 requests a game
        todo = [r for r in g.itertuples() if not (store.get(r.game_id) or {}).get("ya") and not (store.get(r.game_id) or {}).get("yatry")]
    elif lad:        # alt-receptions ladders for games that do not have them yet
        todo = [r for r in g.itertuples() if not (store.get(r.game_id) or {}).get("ro") and not (store.get(r.game_id) or {}).get("rotry")]
    elif two:        # 2+ TD prices for games that do not have them yet
        todo = [r for r in g.itertuples() if not (store.get(r.game_id) or {}).get("t") and not (store.get(r.game_id) or {}).get("ttry")]
    elif first:    # first-touchdown prices for games that do not have them yet (and anytime too where that is missing)
        todo = [r for r in g.itertuples() if not (store.get(r.game_id) or {}).get("f") and not (store.get(r.game_id) or {}).get("ftry")]
    else:
        todo = [r for r in g.itertuples() if not (store.get(r.game_id) or {}).get("b")]
    if limit:
        todo = todo[:limit]
    slots = {}
    for r in todo:
        ko = datetime.strptime(f"{r.gameday} {r.gametime}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
        slots.setdefault(ko - timedelta(minutes=lead), []).append(r)
    cost = sum(10 if yds or (lad and (store.get(r.game_id) or {}).get("r")) else 20 if lad else 10 for r in todo) + len(slots)
    print(f"{len(g)} finished games ({season - back}-{season}), {len(g) - len(todo)} already have prices. To fetch: {len(todo)} games in {len(slots)} kickoff slots.")
    print(f"Estimated cost: about {cost} requests (10 per game + 1 per slot).")
    if not todo:
        return
    if not go:
        print("Nothing spent yet. Run again with --backfill --yes to pull them.")
        return
    got = spent = 0
    left = None
    for when, rows in sorted(slots.items()):
        stamp = when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            raw, hd = get(f"{ODDS_API.replace('/v4/', '/v4/historical/')}/events?apiKey={key}&date={stamp}", 60, headers=True)
            events = json.loads(raw).get("data") or []
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")[:300].replace(key, "***")
            sys.exit(f"The Odds API refused the historical request ({e.code}). Historical prices need a paid plan.\n{body}")
        except Exception as e:
            sys.exit(f"Could not reach The Odds API: {str(e).replace(key, '***')}")
        ids = {(TEAM_NAMES.get(e.get("away_team")), TEAM_NAMES.get(e.get("home_team"))): e["id"] for e in events}
        for r in rows:
            eid = ids.get((r.away_team, r.home_team))
            if not eid:
                print(f"  {r.game_id}: not listed at {stamp}", file=sys.stderr)
                continue
            try:
                old_rec = store.get(r.game_id) or {}
                mk = "player_anytime_td" if not first else "player_1st_td" if old_rec.get("b") else "player_anytime_td,player_1st_td"
                if two:
                    mk = "player_tds_over" if old_rec.get("b") else "player_anytime_td,player_tds_over"
                if yds:
                    mk = "player_reception_yds_alternate"
                elif lad:      # games that already have ladders only need the main over/under lines
                    mk = "player_receptions" if old_rec.get("r") else "player_receptions,player_receptions_alternate"
                raw, hd = get(f"{ODDS_API.replace('/v4/', '/v4/historical/')}/events/{eid}/odds?apiKey={key}&regions=us&markets={mk}"
                              f"&oddsFormat=american&date={stamp}", 60, headers=True)
            except urllib.error.HTTPError as e:
                print(f"  {r.game_id}: {e.code} {e.read().decode('utf-8', 'ignore')[:200].replace(key, '***')}", file=sys.stderr)
                continue
            except Exception as e:
                print(f"  {r.game_id}: {str(e).replace(key, '***')}", file=sys.stderr)
                continue
            doc = json.loads(raw)
            books = parse_td(doc.get("data") or {})
            if yds:        # {book: [[player, yards (the over on 39.5 is stored as 40), price], ...]}
                ya = parse_ladder(doc.get("data") or {}, "player_reception_yds_alternate")
                rec = dict(old_rec)
                rec.update({"ya": ya, "yat": doc.get("timestamp") or stamp} if ya else {"yatry": 1})
                store[r.game_id] = rec
                got += 1 if ya else 0
                with open(ODDS, "w", encoding="utf-8") as f:
                    json.dump(store, f, separators=(",", ":"), ensure_ascii=False, sort_keys=True)
                left = hd.get("x-requests-remaining")
                print(f"  {r.game_id}: {len(ya)} books with yards ladders" + (f"  ({left} requests left)" if left else ""), flush=True)
                if left is not None and float(left) < 2500:      # keep room for the NBA and NHL dashboards on the same key
                    print("Request allowance down to 2,500; stopping so the other dashboards keep enough.")
                    return
                time.sleep(0.3)
                continue
            if lad:
                ladder, ou = parse_ladder(doc.get("data") or {}), parse_ou(doc.get("data") or {})
                rec = dict(old_rec)
                if ladder:
                    rec.update({"r": ladder, "rat": doc.get("timestamp") or stamp})
                rec.update({"ro": ou} if ou else {"rotry": 1})
                store[r.game_id] = rec
                got += 1 if (ladder or ou) else 0
                with open(ODDS, "w", encoding="utf-8") as f:
                    json.dump(store, f, separators=(",", ":"), ensure_ascii=False, sort_keys=True)
                left = hd.get("x-requests-remaining")
                print(f"  {r.game_id}: {len(ladder)} books with ladders" + (f"  ({left} requests left)" if left else ""), flush=True)
                if left is not None and float(left) < 50:
                    print("Request allowance nearly used; stopping.")
                    return
                time.sleep(0.3)
                continue
            ftd = parse_td(doc.get("data") or {}, "player_1st_td") if first else {}
            if two:
                first, ftd = True, parse_two(doc.get("data") or {})      # stored below under "t" in place of "f"
            left = hd.get("x-requests-remaining")
            snap = doc.get("timestamp") or stamp
            at = datetime.fromisoformat(snap.replace("Z", "+00:00")).astimezone(ET).strftime("%Y-%m-%dT%H:%M")
            rec = dict(old_rec)
            if books and not rec.get("b"):
                rec.update({"at": at, "b": books, "hist": 1})
            if two:
                rec.update({"t": ftd} if ftd else {"ttry": 1})
            elif first:
                rec.update({"f": ftd} if ftd else {"ftry": 1})      # ftry: asked once, market was not there
            if books or ftd or first:
                store[r.game_id] = rec
                got += 1 if (books or ftd) else 0
                books = ftd if first else books
                with open(ODDS, "w", encoding="utf-8") as f:      # save as we go, so a stop loses nothing
                    json.dump(store, f, separators=(",", ":"), ensure_ascii=False, sort_keys=True)
            print(f"  {r.game_id}: {len(books)} books" + (f"  ({left} requests left)" if left else ""), flush=True)
            if left is not None and float(left) < 50:
                print("Request allowance nearly used; stopping. Run again later to continue where this left off.")
                return
            time.sleep(0.3)
    print(f"Stored prices for {got} games in {ODDS}. Rebuild the dashboard (Refresh, or --html {HTML}) and the Tracker will grade them.")


# ---------- writing and serving ----------
def embedded(path):
    try:
        m = re.search(r'<script id="%s" type="application/json">(.*?)</script>' % TAG, open(path, encoding="utf-8").read(), re.S)
        return json.loads(m.group(1))
    except Exception:
        return None


def write_html(path, out):
    page = open(path, encoding="utf-8").read()
    blob = json.dumps(out, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    pat = re.compile(r'(<script id="%s" type="application/json">).*?(</script>)' % TAG, re.S)
    if not pat.search(page):
        raise RuntimeError(f"{path} has no {TAG} block to refresh")
    with open(path, "w", encoding="utf-8") as f:
        f.write(pat.sub(lambda m: m.group(1) + blob + m.group(2), page, count=1))


def serve(a):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    html = a.html or HTML
    if not os.path.exists(html):
        sys.exit(f"Put {html} in the same folder as this script, then run it again.")
    lock = threading.Lock()

    def refresh(prices_only=False):
        old = embedded(html)
        if prices_only and old:
            old["odds"] = remember_odds(old.get("sched") or [], old, force=True)
            out = old
        else:
            out = make(a.season, old)
        write_html(html, out)
        print(f"{datetime.now(ET):%b %d %H:%M} refreshed: through {out['through']}, {len(out['sched'])} games listed, "
              f"{sum(1 for k, v in out['odds'].items() if k != '_meta' and v.get('b'))} with prices", flush=True)
        return out

    def keep_fresh():
        while True:
            with lock:
                try:
                    refresh()
                except Exception as e:
                    print(f"{datetime.now(ET):%b %d %H:%M} refresh failed: {e}", file=sys.stderr, flush=True)
            time.sleep(max(a.every, 0.25) * 3600)

    class H(BaseHTTPRequestHandler):
        def send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.split("?")[0].split("#")[0] in ("/", "/index.html"):
                self.send(200, open(html, "rb").read(), "text/html; charset=utf-8")
            else:
                self.send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path == "/key":
                k = self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode().strip()
                if not re.fullmatch(r"[A-Za-z0-9]{16,64}", k):
                    return self.send(200, json.dumps({"error": "That does not look like an Odds API key"}).encode(), "application/json")
                with open("odds_key.txt", "w", encoding="utf-8") as f:
                    f.write(k)
            elif self.path not in ("/refresh", "/prices"):
                return self.send(404, b"not found", "text/plain")
            if not lock.acquire(blocking=False):
                return self.send(200, json.dumps({"error": "A refresh is already running"}).encode(), "application/json")
            try:
                body = json.dumps(refresh(self.path != "/refresh"), separators=(",", ":"), ensure_ascii=False).encode()
            except Exception as e:
                print(f"refresh failed: {e}", file=sys.stderr)
                body = json.dumps({"error": str(e)}).encode()
            finally:
                lock.release()
            self.send(200, body, "application/json")

        def log_message(self, *args):
            pass

    url = f"http://localhost:{a.port}/"
    if port_open(a.port):
        sys.exit(f"Port {a.port} is already in use (another ledger?). Try  --serve --port {a.port + 1}")
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"Touchdown Ledger is at {url}  (Ctrl+C to stop)", flush=True)
    if a.every:
        threading.Thread(target=keep_fresh, daemon=True).start()
    if not a.no_open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


HOME_DIR = os.path.join(os.path.expanduser("~"), "TouchdownLedger")
PLIST = os.path.join(os.path.expanduser("~"), "Library", "LaunchAgents", "com.touchdownledger.plist")
WIN_START = os.path.join(os.environ.get("APPDATA", ""), "Microsoft", "Windows", "Start Menu", "Programs", "Startup", "TouchdownLedger.bat")


def port_open(port):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def install(a):
    """Copy the dashboard to ~/TouchdownLedger and have it start at login and refresh itself."""
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(HOME_DIR, exist_ok=True)
    for name in (os.path.basename(__file__), HTML, "odds_key.txt", "odds.json"):
        src, dst = os.path.join(here, name), os.path.join(HOME_DIR, name)
        if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)
    script = os.path.join(HOME_DIR, os.path.basename(__file__))
    if not os.path.exists(os.path.join(HOME_DIR, HTML)):
        sys.exit(f"{HTML} needs to be in the same folder as this script. Put it there and run this again.")
    need = []
    for mod, pkg in (("pandas", "pandas"), ("pyarrow", "pyarrow"), ("certifi", "certifi")):
        try:
            __import__(mod)
        except ImportError:
            need.append(pkg)
    if need:
        print(f"Installing {' and '.join(need)} (one time)...")
        base = [sys.executable, "-m", "pip", "install", "--user", "--quiet"] + need
        if subprocess.call(base) != 0 and subprocess.call(base + ["--break-system-packages"]) != 0:
            sys.exit(f"Could not install {' '.join(need)}. Run:  python3 -m pip install {' '.join(need)}   then run this again.")
    url = f"http://localhost:{a.port}/"
    cmd = [sys.executable, script, "--serve", "--no-open", "--port", str(a.port)]
    if sys.platform == "darwin" and os.path.exists(PLIST):     # stop an earlier copy of this ledger before checking the port
        subprocess.call(["launchctl", "unload", PLIST], stderr=subprocess.DEVNULL)
        time.sleep(1)
    if port_open(a.port):
        sys.exit(f"Something else is already using port {a.port} (another ledger?). Run this again with a free one, e.g.  --install --port {a.port + 1}")
    if sys.platform == "darwin":
        os.makedirs(os.path.dirname(PLIST), exist_ok=True)
        args = "".join(f"<string>{c}</string>" for c in cmd)
        log = os.path.join(HOME_DIR, "ledger.log")
        with open(PLIST, "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
                    '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n<plist version="1.0"><dict>'
                    f'<key>Label</key><string>com.touchdownledger</string><key>ProgramArguments</key><array>{args}</array>'
                    '<key>RunAtLoad</key><true/><key>KeepAlive</key><true/>'
                    f'<key>WorkingDirectory</key><string>{HOME_DIR}</string>'
                    f'<key>StandardOutPath</key><string>{log}</string><key>StandardErrorPath</key><string>{log}</string>'
                    '</dict></plist>\n')
        subprocess.call(["launchctl", "unload", PLIST], stderr=subprocess.DEVNULL)
        subprocess.call(["launchctl", "load", "-w", PLIST])
    elif os.name == "nt":
        pyw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
        pyw = pyw if os.path.exists(pyw) else sys.executable
        with open(WIN_START, "w") as f:
            f.write(f'@echo off\nstart "" "{pyw}" "{script}" --serve --no-open --port {a.port}\n')
        subprocess.Popen([pyw] + cmd[1:], cwd=HOME_DIR, creationflags=0x00000008)
    else:
        subprocess.Popen(cmd, cwd=HOME_DIR, start_new_session=True, stdout=open(os.path.join(HOME_DIR, "ledger.log"), "a"), stderr=subprocess.STDOUT)
        print("Started for this session. Add this to your login items to keep it:\n  " + " ".join(cmd))
    for _ in range(40):
        if port_open(a.port):
            break
        time.sleep(0.5)
    else:
        sys.exit(f"Set up, but the dashboard did not start. See {os.path.join(HOME_DIR, 'ledger.log')}")
    print(f"\nDone. Touchdown Ledger is at {url}\nBookmark that address. It starts when you log in and refreshes itself every {a.every:g} hours.\n"
          f"The first refresh downloads about 90 MB of play-by-play; reload the page in a few minutes.")
    if not a.no_open:
        webbrowser.open(url)


def uninstall(a):
    if sys.platform == "darwin" and os.path.exists(PLIST):
        subprocess.call(["launchctl", "unload", PLIST], stderr=subprocess.DEVNULL)
        os.remove(PLIST)
    if os.name == "nt" and os.path.exists(WIN_START):
        os.remove(WIN_START)
    print(f"Removed the login item. Delete {HOME_DIR} to remove the files. If the page still opens, restart your computer.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, help="year the season starts; default is the season in progress")
    ap.add_argument("--out", default="data.json")
    ap.add_argument("--html", help="dashboard file to refresh in place")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--install", action="store_true")
    ap.add_argument("--uninstall", action="store_true")
    ap.add_argument("--every", type=float, default=4, help="hours between automatic refreshes while serving (0 = off)")
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--backfill", action="store_true", help="pull pre-kickoff prices for this season's finished games (paid Odds API plan)")
    ap.add_argument("--yes", action="store_true", help="with --backfill: actually spend the requests")
    ap.add_argument("--lead", type=int, default=60, help="with --backfill: minutes before kickoff to take the prices from")
    ap.add_argument("--limit", type=int, help="with --backfill: only the first N games (to test cheaply)")
    ap.add_argument("--last-season", action="store_true", help="with --backfill: also last season's games")
    ap.add_argument("--first-td", action="store_true", help="with --backfill: first-touchdown prices for this season's games")
    a = ap.parse_args()
    if a.install:
        return install(a)
    if a.uninstall:
        return uninstall(a)
    if pd is None:
        sys.exit("pandas is missing. Run:  python3 build_data.py --install")
    if a.backfill:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        return backfill(a.season, a.lead, a.yes, a.limit, back=1 if a.last_season else 0, first=a.first_td)
    if a.serve:
        return serve(a)
    try:
        out = make(a.season, embedded(a.html) if a.html else None)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, separators=(",", ":"), ensure_ascii=False)
        if a.html:
            write_html(a.html, out)
    except RuntimeError as e:
        sys.exit(str(e))
    print(f"through {out['through']}: {len(out['sched'])} upcoming games, {len(out['picks'])} players priced -> {a.out} ({os.path.getsize(a.out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
