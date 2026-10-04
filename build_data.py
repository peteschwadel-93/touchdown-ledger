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
import argparse, json, os, re, shutil, socket, subprocess, sys, threading, time, unicodedata, urllib.error, urllib.request, webbrowser
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
HL, HLT, SM = 6, 6, 0.6          # half-life in games for players and teams; extra fade across an off-season
K, KT, KD = 1.5, 6.0, 30.0       # shrinkage, in games, for player shares, team run/pass mix and the defence nudge
PBP_COLS = ["game_id", "season", "week", "posteam", "defteam", "play_type", "yardline_100", "air_yards", "two_point_attempt",
            "sack", "touchdown", "td_player_id", "rusher_player_id", "receiver_player_id", "passer_player_id", "pass_touchdown",
            "play_id", "qtr", "fixed_drive"]
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
            ros.append(pd.read_parquet(p, columns=["season", "week", "team", "position", "status", "full_name", "gsis_id", "pfr_id"]))
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
        rzc=("yardline_100", lambda s: int((s <= 20).sum()))).reset_index().rename(columns={"rusher_player_id": "pid"})
    t = tgt.groupby(["game_id", "posteam", "receiver_player_id"]).agg(
        tg=("x", "size"), tx=("x", "sum"), ctd=("td", "sum"), ez=("ez", "sum"),
        rzt=("yardline_100", lambda s: int((s <= 20).sum()))).reset_index().rename(columns={"receiver_player_id": "pid"})
    pg = r.merge(t, on=["game_id", "posteam", "pid"], how="outer").fillna(0)
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
    d["ftd"] = (d.game_id.map(first) == d.pid).astype(int)
    g = games.set_index("game_id")
    d = d[d.game_id.isin(g.index)]
    for c in ("season", "week", "gameday"):
        d[c] = d.game_id.map(g[c])
    home = d.game_id.map(g.home_team) == d.posteam
    sp, tot = d.game_id.map(g.spread_line), d.game_id.map(g.total_line)
    d["imp"] = np.where(home, tot / 2 + sp / 2, tot / 2 - sp / 2)      # spread_line > 0 means the home team is favoured
    d["opp"] = np.where(home, d.game_id.map(g.away_team), d.game_id.map(g.home_team))
    tm = d.groupby(["game_id", "posteam"]).agg(trx=("rx", "sum"), ttx=("tx", "sum"), trtd=("rtd", "sum"), tctd=("ctd", "sum")).reset_index()
    d = d.merge(tm, on=["game_id", "posteam"]).sort_values(["gameday", "game_id"]).reset_index(drop=True)
    return d, RT, TT


# ---------- features ----------
PCOLS = ["rx", "tx", "trx", "ttx", "tds", "car", "tg", "i5", "ez", "snp", "rzc", "rzt", "y"]
TCOLS = ["trx", "ttx", "trtd", "tctd"]


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
        trx=("trx", "first"), ttx=("ttx", "first"), trtd=("trtd", "first"), tctd=("tctd", "first"),
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


XC = ["ll", "isQB", "isRB", "isTE", "snw"]


def design(s):
    return np.c_[np.log(s.lam.clip(1e-4)), (s.pos == "QB").astype(int), (s.pos == "RB").astype(int), (s.pos == "TE").astype(int), s.snw]


def fit_logit(X, y, iters=60, l2=1e-3):
    """Plain logistic regression by Newton's method (no scikit-learn needed)."""
    X1 = np.c_[np.ones(len(X)), X]
    w = np.zeros(X1.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X1 @ w))
        H = (X1 * (p * (1 - p))[:, None]).T @ X1 + l2 * np.eye(len(w))
        step = np.linalg.solve(H, X1.T @ (y - p) - l2 * w)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    return w


def predict(w, X):
    return 1 / (1 + np.exp(-(np.c_[np.ones(len(X)), X] @ w)))


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
    if len(tr) > 2000 and len(te) > 1000:
        w0 = fit_logit(design(tr), tr.y.to_numpy(float))
        te = te.assign(p=predict(w0, design(te)))
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
    trk = []
    pre = s[s.season < season]
    cur = s[s.season == season]
    if len(pre) > 2000 and len(cur):
        w1 = fit_logit(design(pre), pre.y.to_numpy(float))
        cur = cur.assign(p=predict(w1, design(cur)))
        model["trk"] = scores(cur.y, cur.p)
        for r in cur[cur.p >= 0.08].itertuples():
            trk.append([int(r.week), r.game_id, r.name, r.posteam, r.opp, r.pos, r3(r.p), int(r.y), int(r.tds)])
    w = fit_logit(design(s), s.y.to_numpy(float))
    model["w"] = [r3(x, 4) for x in w]
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
    up = games[(games.season == season) & games.home_score.isna() & (games.gameday >= today)].sort_values(["gameday", "gametime"])
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
                o, dd = O_end.get(team, (np.zeros(5), season))[0], D_end.get(opp, (np.zeros(5), season))[0]
                row = {"game_id": g.game_id, "posteam": team, "opp": opp, "pid": r.gsis_id, "name": r.full_name, "pos": r.position,
                       "imp": imp, "inj": st, "back": int(back), "n": acc[-1], "week": int(g.week)}
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
        logs = {}
        for pid, x in d[d.pid.isin(set(u.pid))].groupby("pid", sort=False):
            logs[pid] = [[f"{int(r.season) % 100}w{int(r.week)}", r.opp, r3(r.snp, 2), int(r.car), int(r.tg), int(r.i5), int(r.ez),
                          int(r.rzc + r.rzt), r3(r.rx + r.tx, 2), int(r.tds)] for r in x.tail(6).itertuples()]
        for r in u.itertuples():
            n1 = max(r.n, 1e-9)
            picks.append({"g": r.game_id, "id": r.pid, "n": r.name, "t": r.posteam, "o": r.opp, "pos": r.pos, "p": r3(r.p, 4), "pf": r3(r.pf, 4),
                          "T": r3(r.T, 2), "sp": r3(r.split), "rs": r3(r.rshn), "ts": r3(r.tshn), "lr": r3(r.lam_r), "lt": r3(r.lam_t),
                          "sn": r3(r.snw, 2), "dr": r3(r.dr, 2), "dt": r3(r.dt, 2), "ng": r3(r.n, 1), "inj": r.inj, "back": r.back,
                          "x": r3((r.p_rx + r.p_tx) / n1, 2), "td": r3(r.p_tds / n1, 2), "i5": r3(r.p_i5 / n1, 2), "ez": r3(r.p_ez / n1, 2),
                          "rz": r3((r.p_rzc + r.p_rzt) / n1, 2), "log": logs.get(r.pid, [])})
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
           "model": model, "trk": trk}
    note = None
    if os.environ.get("ODDS_BACKFILL", "").strip().lower() in ("1", "true"):
        try:
            backfill(season, int(os.environ.get("ODDS_LEAD", "60")), True, int(os.environ.get("ODDS_BACKFILL_MAX", "0")) or None)
        except SystemExit as e:
            note = str(e)
            print(note, file=sys.stderr)
    out["odds"] = remember_odds(sched, old)
    if note:
        out["odds"]["_meta"]["err"] = note[:200]
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


def remember_odds(sched, old, force=False):
    """Anytime-TD prices for the listed games from The Odds API, kept in odds.json.

    One request per game per look (one market, one region). ODDS_LOOKS sets the looks, in hours before kickoff
    (default "24,3,1": the day before, after inactives are close, and just before kickoff). About 210 requests a month.
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
        if 0 < hrs and ((due > rec.get("n", 0) and rec.get("tries", 0) < 8) or (force and hrs <= 72)):
            want.append((u, due))
    if want:
        try:
            events = json.loads(get(f"{ODDS_API}/events?apiKey={key}", 60))     # listing events is free
            ids = {(TEAM_NAMES.get(e.get("away_team")), TEAM_NAMES.get(e.get("home_team"))): e["id"] for e in events}
            for u, due in want:
                eid = ids.get((u["a"], u["h"]))
                if not eid:
                    continue
                raw, hd = get(f"{ODDS_API}/events/{eid}/odds?apiKey={key}&regions=us&markets=player_anytime_td&oddsFormat=american", 60, headers=True)
                books = parse_td(json.loads(raw))
                left = hd.get("x-requests-remaining")
                if left is not None:
                    meta["left"] = int(float(left))
                rec = store.get(u["gid"]) or {}
                if books:
                    rec.update({"at": now.strftime("%Y-%m-%dT%H:%M"), "b": books, "n": max(due, rec.get("n", 0))})
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


def backfill(season, lead=60, go=False, limit=None):
    """Closing-ish anytime-TD prices for this season's finished games, from The Odds API's historical endpoints (paid plans only).

    For each finished game without stored prices it asks for the snapshot `lead` minutes before kickoff. Cost: 10 requests per
    game, plus 1 per distinct kickoff slot to look up event ids. Nothing is spent unless go is set."""
    season = season or current_season()
    key = odds_key()
    if not key:
        sys.exit("No Odds API key. Put it in odds_key.txt next to this script or in the ODDS_API_KEY environment variable.")
    g = pd.read_csv(cached("games.csv", GAMES, True))
    g = g[(g.season == season) & g.home_score.notna()].sort_values(["gameday", "gametime"])
    try:
        store = json.load(open(ODDS, encoding="utf-8"))
    except Exception:
        store = {}
    todo = [r for r in g.itertuples() if not (store.get(r.game_id) or {}).get("b")]
    if limit:
        todo = todo[:limit]
    slots = {}
    for r in todo:
        ko = datetime.strptime(f"{r.gameday} {r.gametime}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
        slots.setdefault(ko - timedelta(minutes=lead), []).append(r)
    cost = 10 * len(todo) + len(slots)
    print(f"{len(g)} finished {season} games, {len(g) - len(todo)} already have prices. To fetch: {len(todo)} games in {len(slots)} kickoff slots.")
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
                raw, hd = get(f"{ODDS_API.replace('/v4/', '/v4/historical/')}/events/{eid}/odds?apiKey={key}&regions=us&markets=player_anytime_td"
                              f"&oddsFormat=american&date={stamp}", 60, headers=True)
            except urllib.error.HTTPError as e:
                print(f"  {r.game_id}: {e.code} {e.read().decode('utf-8', 'ignore')[:200].replace(key, '***')}", file=sys.stderr)
                continue
            except Exception as e:
                print(f"  {r.game_id}: {str(e).replace(key, '***')}", file=sys.stderr)
                continue
            doc = json.loads(raw)
            books = parse_td(doc.get("data") or {})
            left = hd.get("x-requests-remaining")
            if books:
                snap = doc.get("timestamp") or stamp
                at = datetime.fromisoformat(snap.replace("Z", "+00:00")).astimezone(ET).strftime("%Y-%m-%dT%H:%M")
                store[r.game_id] = {"at": at, "b": books, "hist": 1}
                got += 1
                with open(ODDS, "w", encoding="utf-8") as f:      # save as we go, so a stop loses nothing
                    json.dump(store, f, separators=(",", ":"), ensure_ascii=False, sort_keys=True)
            print(f"  {r.game_id}: {len(books)} books" + (f"  ({left} requests left)" if left else ""), flush=True)
            if left is not None and float(left) < 15:
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
    a = ap.parse_args()
    if a.install:
        return install(a)
    if a.uninstall:
        return uninstall(a)
    if pd is None:
        sys.exit("pandas is missing. Run:  python3 build_data.py --install")
    if a.backfill:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        return backfill(a.season, a.lead, a.yes, a.limit)
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
