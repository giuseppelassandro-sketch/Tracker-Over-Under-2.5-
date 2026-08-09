#!/usr/bin/env python3
"""
Analisi Over/Under 2.5 gol — script auto-contenuto.

Scarica i dati da football-data.co.uk (gratuiti, senza chiave), calcola per ogni
partita in arrivo una stima della probabilita' di Over 2.5 basata sullo storico
delle squadre, la confronta con la quota dei bookmaker e genera index.html.

Nessun file locale richiesto, nessun input manuale.
Uso:  python3 analisi_over_under.py [--output index.html]
"""

import argparse
import io
import ssl
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

import pandas as pd

# --------------------------------------------------------------------------
# Configurazione
# --------------------------------------------------------------------------

BASE = "https://www.football-data.co.uk"
FIXTURES_URL = f"{BASE}/fixtures.csv"
EXTRA_FIXTURES_URL = f"{BASE}/new_league_fixtures.csv"

# Campionati "principali": cartelle mmz4281 + fixtures.csv.
# Le chiavi sono i codici usati nella colonna Div di football-data.co.uk.
LEAGUES = {
    "I1": "Serie A",
    "E0": "Premier League",
    "SP1": "LaLiga",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
    "N1": "Eredivisie",
    "P1": "Primeira Liga",
}

# Campionati "extra": stanno in /new/{CODICE}.csv e in new_league_fixtures.csv,
# con uno schema diverso (Home/Away invece di HomeTeam/AwayTeam, HG/AG invece di
# FTHG/FTAG, stagioni per anno solare) e SENZA quote Over/Under 2.5: per questi
# la stima resta, mentre le colonne quota e scostamento restano vuote.
# codice -> (nome paese in new_league_fixtures.csv, nome leggibile)
EXTRA_LEAGUES = {
    "SWE": ("Sweden", "Allsvenskan"),
    "NOR": ("Norway", "Eliteserien"),
}


def league_name(code):
    """Nome leggibile di un campionato, principale o extra."""
    if code in LEAGUES:
        return LEAGUES[code]
    if code in EXTRA_LEAGUES:
        return EXTRA_LEAGUES[code][1]
    return code

# Quante stagioni di storico includere (stagione corrente + N-1 precedenti)
N_SEASONS = 3

# Finestra delle partite da mostrare (giorni da oggi)
DAYS_AHEAD = 8

# Forza dello "shrinkage" bayesiano verso la media del campionato, espressa in
# numero di partite equivalenti. Serve per le squadre con poco storico
# (neopromosse): con 4 partite giocate il loro tasso grezzo e' inaffidabile,
# quindi lo si tira verso la media di lega. 0 = nessuna correzione.
PRIOR_MATCHES = 8

USER_AGENT = "Mozilla/5.0 (compatible; over-under-25-bot/1.0)"
TIMEOUT = 45


def _ssl_context():
    """Contesto SSL con i certificati di certifi quando disponibili.

    Su GitHub Actions i certificati di sistema bastano; su alcune installazioni
    Python di macOS mancano e la verifica fallirebbe.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


SSL_CONTEXT = _ssl_context()


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------

def fetch_csv(url, require_div=True):
    """Scarica un CSV. Ritorna un DataFrame, oppure None se manca (404) o e' illeggibile.

    `require_div` distingue i file principali (che hanno la colonna Div) da quelli
    dei campionati extra, che usano invece Country/League.

    football-data.co.uk risponde alle risorse mancanti con una pagina HTML e
    status 404: senza controllare lo status, pandas proverebbe a interpretare
    l'HTML come CSV producendo dati insensati.
    """
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT, context=SSL_CONTEXT) as resp:
            if resp.status != 200:
                print(f"  [skip] {url} -> HTTP {resp.status}", file=sys.stderr)
                return None
            raw = resp.read()
    except urllib.error.HTTPError as e:
        print(f"  [skip] {url} -> HTTP {e.code}", file=sys.stderr)
        return None
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"  [skip] {url} -> errore di rete: {e}", file=sys.stderr)
        return None

    # utf-8-sig: i file di football-data.co.uk hanno il BOM sulla prima colonna
    try:
        df = pd.read_csv(
            io.StringIO(raw.decode("utf-8-sig", errors="replace")),
            on_bad_lines="skip",
        )
    except Exception as e:  # CSV malformato o pagina di errore
        print(f"  [skip] {url} -> CSV illeggibile: {e}", file=sys.stderr)
        return None

    if df.empty or (require_div and "Div" not in df.columns):
        print(f"  [skip] {url} -> contenuto inatteso", file=sys.stderr)
        return None
    return df


def season_codes(today, n=N_SEASONS):
    """Codici cartella football-data ('2526', '2425', ...) piu' recenti.

    La stagione europea inizia a luglio: da luglio in poi l'anno solare corrente
    apre la nuova stagione.
    """
    start_year = today.year if today.month >= 7 else today.year - 1
    codes = []
    for i in range(n):
        y = start_year - i
        codes.append(f"{y % 100:02d}{(y + 1) % 100:02d}")
    return codes


def load_extra_history(today, n=N_SEASONS):
    """Storico dei campionati extra, normalizzato nello schema principale.

    I file /new/{CODICE}.csv contengono tutte le stagioni dal 2012 in poi, con
    stagioni per anno solare (2026) perche' questi campionati vanno da primavera
    ad autunno. Si tengono solo le ultime `n`.
    """
    frames = []
    for code in EXTRA_LEAGUES:
        url = f"{BASE}/new/{code}.csv"
        df = fetch_csv(url, require_div=False)
        if df is None:
            continue

        needed = {"Season", "Home", "Away", "HG", "AG"}
        if not needed.issubset(df.columns):
            print(f"  [skip] {url} -> colonne mancanti", file=sys.stderr)
            continue

        df["Season"] = pd.to_numeric(df["Season"], errors="coerce")
        df = df.dropna(subset=["Season"])
        recent = sorted(df["Season"].unique())[-n:]
        df = df[df["Season"].isin(recent)]

        out = pd.DataFrame(
            {
                "Div": code,
                "HomeTeam": df["Home"],
                "AwayTeam": df["Away"],
                "FTHG": pd.to_numeric(df["HG"], errors="coerce"),
                "FTAG": pd.to_numeric(df["AG"], errors="coerce"),
                "Season": df["Season"].astype(int).astype(str),
            }
        )
        out["SeasonLabel"] = out["Season"]
        frames.append(out)
        print(
            f"  [ok]   {code}: {len(out)} righe "
            f"(stagioni {', '.join(str(int(s)) for s in recent)})",
            file=sys.stderr,
        )
    return frames


def load_history(today):
    """Scarica lo storico dei campionati configurati per le ultime N stagioni."""
    frames = []
    for season in season_codes(today):
        for code in LEAGUES:
            url = f"{BASE}/mmz4281/{season}/{code}.csv"
            df = fetch_csv(url)
            if df is None:
                continue
            needed = {"Div", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
            if not needed.issubset(df.columns):
                print(f"  [skip] {url} -> colonne mancanti", file=sys.stderr)
                continue
            df = df[list(needed)].copy()
            df["Season"] = season
            df["SeasonLabel"] = f"{season[:2]}/{season[2:]}"
            frames.append(df)
            print(f"  [ok]   {season}/{code}: {len(df)} righe", file=sys.stderr)

    frames.extend(load_extra_history(today))

    if not frames:
        # Stesse colonne del caso popolato, cosi' il resto della pipeline
        # (groupby su IsOver, media di lega, render) funziona anche a vuoto.
        return pd.DataFrame(
            {
                "Div": pd.Series(dtype="object"),
                "HomeTeam": pd.Series(dtype="object"),
                "AwayTeam": pd.Series(dtype="object"),
                "FTHG": pd.Series(dtype="float"),
                "FTAG": pd.Series(dtype="float"),
                "Season": pd.Series(dtype="object"),
                "SeasonLabel": pd.Series(dtype="object"),
                "TotGoals": pd.Series(dtype="float"),
                "IsOver": pd.Series(dtype="int"),
            }
        )

    hist = pd.concat(frames, ignore_index=True)
    hist["FTHG"] = pd.to_numeric(hist["FTHG"], errors="coerce")
    hist["FTAG"] = pd.to_numeric(hist["FTAG"], errors="coerce")
    hist = hist.dropna(subset=["FTHG", "FTAG", "HomeTeam", "AwayTeam"])
    hist["TotGoals"] = hist["FTHG"] + hist["FTAG"]
    hist["IsOver"] = (hist["TotGoals"] >= 3).astype(int)
    return hist


# --------------------------------------------------------------------------
# Calcolo dei tassi Over per squadra
# --------------------------------------------------------------------------

def build_team_rates(hist):
    """Tassi Over 2.5 per squadra, separati fra partite in casa e in trasferta.

    Ritorna (rates, league_avg):
      rates[(div, team)] = {
          "home_rate", "home_n", "away_rate", "away_n",   # su tutto lo storico
          "seasons": {etichetta: {"home_raw", "home_n", "away_raw", "away_n"}},
      }
      league_avg[div] = tasso Over medio del campionato

    I valori per stagione sono grezzi, non "smussati": servono a mostrare la
    tendenza reale di una squadra stagione per stagione, e applicare lo shrinkage
    la appiattirebbe proprio dove si vuole leggerla. Lo shrinkage resta invece sui
    tassi complessivi, che alimentano la stima usata per la classifica.
    """
    league_avg = {}
    for div, grp in hist.groupby("Div"):
        league_avg[div] = float(grp["IsOver"].mean()) if len(grp) else 0.5

    rates = {}

    def blend(raw_rate, n, div):
        """Shrinkage verso la media di lega, proporzionale alla poca esperienza."""
        avg = league_avg.get(div, 0.5)
        if PRIOR_MATCHES <= 0:
            return raw_rate if n > 0 else avg
        return (raw_rate * n + avg * PRIOR_MATCHES) / (n + PRIOR_MATCHES)

    home = hist.groupby(["Div", "HomeTeam"])["IsOver"].agg(["mean", "count"])
    for (div, team), row in home.iterrows():
        rec = rates.setdefault((div, team), {})
        rec["home_raw"] = float(row["mean"])
        rec["home_n"] = int(row["count"])
        rec["home_rate"] = blend(float(row["mean"]), int(row["count"]), div)

    away = hist.groupby(["Div", "AwayTeam"])["IsOver"].agg(["mean", "count"])
    for (div, team), row in away.iterrows():
        rec = rates.setdefault((div, team), {})
        rec["away_raw"] = float(row["mean"])
        rec["away_n"] = int(row["count"])
        rec["away_rate"] = blend(float(row["mean"]), int(row["count"]), div)

    # Squadre viste solo in casa o solo in trasferta: completa col dato di lega
    for (div, _team), rec in rates.items():
        avg = league_avg.get(div, 0.5)
        rec.setdefault("home_rate", avg)
        rec.setdefault("home_n", 0)
        rec.setdefault("away_rate", avg)
        rec.setdefault("away_n", 0)

    # --- dettaglio per stagione (valori grezzi) ---
    if len(hist):
        hs = hist.groupby(["Div", "SeasonLabel", "HomeTeam"])["IsOver"].agg(["mean", "count"])
        for (div, label, team), row in hs.iterrows():
            s = rates.setdefault((div, team), {}).setdefault("seasons", {}).setdefault(label, {})
            s["home_raw"] = float(row["mean"])
            s["home_n"] = int(row["count"])

        as_ = hist.groupby(["Div", "SeasonLabel", "AwayTeam"])["IsOver"].agg(["mean", "count"])
        for (div, label, team), row in as_.iterrows():
            s = rates.setdefault((div, team), {}).setdefault("seasons", {}).setdefault(label, {})
            s["away_raw"] = float(row["mean"])
            s["away_n"] = int(row["count"])

    return rates, league_avg


def season_labels_by_div(hist):
    """Etichette di stagione disponibili per ciascun campionato, in ordine."""
    out = {}
    if not len(hist):
        return out
    for div, grp in hist.groupby("Div"):
        out[div] = sorted(grp["SeasonLabel"].dropna().unique())
    return out


# --------------------------------------------------------------------------
# Partite in arrivo
# --------------------------------------------------------------------------

def to_float(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) or f <= 1.0 else f


def load_extra_fixtures(extra=None):
    """Partite in arrivo dei campionati extra, normalizzate nello schema principale.

    Questo file non contiene quote Over/Under 2.5 (solo 1X2): le colonne relative
    restano assenti e a valle vengono trattate come mancanti.
    """
    extra = EXTRA_LEAGUES if extra is None else extra
    if not extra:
        return None

    df = fetch_csv(EXTRA_FIXTURES_URL, require_div=False)
    if df is None or df.empty or "Country" not in df.columns:
        return None

    paesi = {paese: code for code, (paese, _) in extra.items()}
    df = df[df["Country"].isin(paesi)].copy()
    if df.empty:
        return None

    df["Div"] = df["Country"].map(paesi)
    df = df.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam"})
    keep = [c for c in ["Div", "Date", "Time", "HomeTeam", "AwayTeam"] if c in df.columns]
    return df[keep]


def load_fixtures(today, leagues=None, extra=None):
    """Partite in arrivo nei prossimi DAYS_AHEAD giorni per i campionati scelti."""
    leagues = leagues or LEAGUES
    parts = []

    main = fetch_csv(FIXTURES_URL)
    if main is not None and not main.empty:
        sel = main[main["Div"].isin(leagues.keys())]
        if not sel.empty:
            parts.append(sel.copy())

    ex = load_extra_fixtures(extra)
    if ex is not None and not ex.empty:
        parts.append(ex)

    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)
    df["MatchDate"] = pd.to_datetime(
        df["Date"], format="%d/%m/%Y", errors="coerce"
    )
    df = df.dropna(subset=["MatchDate"])

    start = pd.Timestamp(today.date())
    end = start + pd.Timedelta(days=DAYS_AHEAD)
    df = df[(df["MatchDate"] >= start) & (df["MatchDate"] <= end)]
    if df.empty:
        return df
    return df.sort_values(["MatchDate", "Time"])


def analyse(fixtures, rates, league_avg, leagues=None):
    """Per ogni partita: probabilita' stimata, quota e confronto col bookmaker."""
    leagues = leagues or LEAGUES
    rows = []
    for _, fx in fixtures.iterrows():
        div = fx["Div"]
        home, away = fx["HomeTeam"], fx["AwayTeam"]
        avg = league_avg.get(div, 0.5)

        h = rates.get((div, home), {})
        a = rates.get((div, away), {})
        home_rate = h.get("home_rate", avg)
        away_rate = a.get("away_rate", avg)
        model_prob = (home_rate + away_rate) / 2

        odds_over = to_float(fx.get("Avg>2.5")) or to_float(fx.get("B365>2.5"))
        odds_under = to_float(fx.get("Avg<2.5")) or to_float(fx.get("B365<2.5"))

        implied = 1 / odds_over if odds_over else None
        # Probabilita' "equa": rimuove il margine del bookmaker normalizzando
        # le due probabilita' implicite, che sommate superano il 100%.
        fair = None
        if odds_over and odds_under:
            io_, iu = 1 / odds_over, 1 / odds_under
            fair = io_ / (io_ + iu)

        edge = (model_prob - fair) if fair is not None else None

        # Dettaglio stagione per stagione: valore in casa della squadra di casa e
        # valore in trasferta dell'ospite, tenuti distinti per leggere la tendenza.
        per_season = {}
        h_seasons = h.get("seasons", {})
        a_seasons = a.get("seasons", {})
        for label in set(h_seasons) | set(a_seasons):
            hs = h_seasons.get(label, {})
            as_ = a_seasons.get(label, {})
            per_season[label] = {
                "home": hs.get("home_raw"),
                "home_n": hs.get("home_n", 0),
                "away": as_.get("away_raw"),
                "away_n": as_.get("away_n", 0),
            }

        rows.append(
            {
                "div": div,
                "league": league_name(div),
                "per_season": per_season,
                "date": fx["MatchDate"],
                "time": fx.get("Time", "") if pd.notna(fx.get("Time", "")) else "",
                "home": home,
                "away": away,
                "home_rate": home_rate,
                "home_n": h.get("home_n", 0),
                "away_rate": away_rate,
                "away_n": a.get("away_n", 0),
                "model_prob": model_prob,
                "odds_over": odds_over,
                "odds_under": odds_under,
                "implied": implied,
                "fair": fair,
                "edge": edge,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("model_prob", ascending=False).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------
# Output HTML
# --------------------------------------------------------------------------

def pct(x, decimals=1):
    return "&mdash;" if x is None or pd.isna(x) else f"{x * 100:.{decimals}f}%"


def num(x, decimals=2):
    return "&mdash;" if x is None or pd.isna(x) else f"{x:.{decimals}f}"


GIORNI = ["lun", "mar", "mer", "gio", "ven", "sab", "dom"]


def data_it(ts):
    """Data in italiano: strftime userebbe il locale C anche sul runner CI."""
    return f"{GIORNI[ts.weekday()]} {ts.day:02d}/{ts.month:02d}"


def esc(s):
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


CSS = """
:root{--bg:#E4E8E0;--surface:#F5F7F2;--surface-2:#EBEEE6;--ink:#16211C;
--ink-soft:#4B584F;--line:#C7CFC2;--accent:#B8722A;--good:#2E7D4F;
--good-bg:#DEEEE3;--risk:#B23B2E;--risk-bg:#F4DFDA;--neutral:#767E74;
--neutral-bg:#E7EAE3;color-scheme:light}
@media(prefers-color-scheme:dark){:root{--bg:#10160E;--surface:#1A2118;
--surface-2:#202A1D;--ink:#E8ECE3;--ink-soft:#A6B09E;--line:#33402E;
--accent:#E0913D;--good:#57B67F;--good-bg:#1E3327;--risk:#E1685A;
--risk-bg:#3C221D;--neutral:#9AA396;--neutral-bg:#232B1F;color-scheme:dark}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
-webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:2.25rem 1.25rem 4rem;
display:flex;flex-direction:column;gap:1.5rem}
.eyebrow{font-size:.72rem;font-weight:600;letter-spacing:.12em;
text-transform:uppercase;color:var(--accent)}
h1{font-size:clamp(1.9rem,4.5vw,2.8rem);line-height:1.05;margin:.25rem 0 0;
letter-spacing:-.02em}
.sub{color:var(--ink-soft);font-size:.95rem;max-width:70ch;line-height:1.55;margin:0}
.head{border-bottom:1px solid var(--line);padding-bottom:1.1rem}
.meta{color:var(--ink-soft);font-size:.8rem;margin-top:.5rem}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.75rem}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;
padding:.85rem 1rem}
.card .k{font-size:.68rem;font-weight:600;letter-spacing:.07em;
text-transform:uppercase;color:var(--ink-soft)}
.card .v{font-size:1.5rem;font-weight:700;font-variant-numeric:tabular-nums;
margin-top:.15rem}
.shell{background:var(--surface);border:1px solid var(--line);border-radius:10px;
overflow-x:auto}
table{width:100%;border-collapse:collapse;min-width:900px}
td .match{overflow-wrap:normal}
tbody td:nth-child(2){min-width:190px}
thead th{background:var(--surface-2);text-align:left;font-size:.67rem;
font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:var(--ink-soft);
padding:.6rem .55rem;border-bottom:1px solid var(--line);white-space:nowrap}
tbody td{padding:.55rem .55rem;border-bottom:1px solid var(--line);font-size:.84rem}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover{background:var(--surface-2)}
.num{font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
.rank{font-weight:700;color:var(--accent);width:2.2em;font-variant-numeric:tabular-nums}
.match{font-weight:600}
.when{font-size:.74rem;color:var(--ink-soft)}
.badge{display:inline-block;font-size:.68rem;font-weight:700;padding:.15rem .5rem;
border-radius:999px;background:var(--neutral-bg);color:var(--neutral);white-space:nowrap}
.chip{display:inline-block;font-size:.72rem;font-weight:700;padding:.18rem .55rem;
border-radius:999px;white-space:nowrap}
.chip.over{background:var(--good-bg);color:var(--good)}
.chip.under{background:var(--risk-bg);color:var(--risk)}
.chip.flat{background:var(--neutral-bg);color:var(--neutral)}
.bar-wrap{display:flex;align-items:center;gap:.5rem;min-width:120px}
.bar{flex:1;height:6px;border-radius:4px;background:var(--surface-2);
border:1px solid var(--line);overflow:hidden}
.bar>span{display:block;height:100%;background:var(--accent)}
details{background:var(--surface);border:1px solid var(--line);border-radius:10px}
summary{cursor:pointer;padding:.8rem 1.1rem;font-weight:700;font-size:.85rem}
.legend{padding:0 1.1rem 1.1rem;color:var(--ink-soft);font-size:.85rem;
line-height:1.6;display:grid;gap:.5rem}
code{background:var(--surface-2);border:1px solid var(--line);border-radius:4px;
padding:.05rem .3rem;font-size:.85em}
.empty{padding:2.5rem 1rem;text-align:center;color:var(--ink-soft);font-size:.9rem}
.league{display:flex;flex-direction:column;gap:.6rem}
.league-head{display:flex;align-items:baseline;gap:.75rem;flex-wrap:wrap;
border-left:3px solid var(--accent);padding-left:.7rem}
.league-head h2{margin:0;font-size:1.25rem;letter-spacing:-.01em}
.league-meta{color:var(--ink-soft);font-size:.78rem}
th.season,td.season{background:color-mix(in srgb,var(--accent) 7%,transparent)}
td.season .sv{font-weight:600}
td.season .sep{color:var(--ink-soft);margin:0 .15rem;font-weight:400}
td.season{white-space:nowrap}
footer{color:var(--ink-soft);font-size:.78rem;text-align:center;line-height:1.6}
@media(max-width:640px){.wrap{padding:1.5rem 1rem 3rem}}
"""


def render_html(results, hist, league_avg, today, leagues=None):
    leagues = leagues or LEAGUES
    n_leagues = len(leagues) + len(EXTRA_LEAGUES)
    generated = today.strftime("%d/%m/%Y %H:%M UTC")
    n_matches = len(results)
    n_hist = len(hist)
    hist_over = float(hist["IsOver"].mean()) if n_hist else None
    seasons = (
        ", ".join(sorted(hist["SeasonLabel"].dropna().unique()))
        if n_hist
        else "&mdash;"
    )

    season_cols = season_labels_by_div(hist)

    def season_cell(info, label):
        """Cella 'casa / trasferta' per una singola stagione."""
        if not info:
            return '<td class="num season">&mdash;</td>'
        h, a = info.get("home"), info.get("away")
        if h is None and a is None:
            return '<td class="num season">&mdash;</td>'
        hs = f"{h * 100:.0f}%" if h is not None else "&ndash;"
        as_ = f"{a * 100:.0f}%" if a is not None else "&ndash;"
        n = f"{info.get('home_n', 0)}/{info.get('away_n', 0)}"
        return (
            f'<td class="num season"><span class="sv">{hs}</span>'
            f'<span class="sep">/</span><span class="sv">{as_}</span>'
            f'<div class="when">{n} gare</div></td>'
        )

    def league_section(div, block):
        labels = season_cols.get(div, [])
        rows_html = []
        for pos, (_, r) in enumerate(block.iterrows(), start=1):
            if r["edge"] is None or pd.isna(r["edge"]):
                chip = '<span class="chip flat">no quota</span>'
            elif r["edge"] >= 0.05:
                chip = f'<span class="chip over">Over +{r["edge"] * 100:.1f}pt</span>'
            elif r["edge"] <= -0.05:
                chip = f'<span class="chip under">Under {r["edge"] * 100:.1f}pt</span>'
            else:
                chip = '<span class="chip flat">in linea</span>'

            when = data_it(r["date"])
            if r["time"]:
                when += f" &middot; {esc(r['time'])}"

            # Con 0 partite nello storico il valore mostrato e' la media del
            # campionato, non una statistica della squadra: va detto.
            home_note = f"{r['home_n']} gare" if r["home_n"] else "media lega"
            away_note = f"{r['away_n']} gare" if r["away_n"] else "media lega"

            season_cells = "".join(
                season_cell(r["per_season"].get(lb), lb) for lb in labels
            )

            rows_html.append(
                f"""<tr>
<td class="rank">{pos}</td>
<td><div class="match">{esc(r['home'])} &ndash; {esc(r['away'])}</div>
<div class="when">{when}</div></td>
<td class="num">{pct(r['home_rate'])}<div class="when">{home_note}</div></td>
<td class="num">{pct(r['away_rate'])}<div class="when">{away_note}</div></td>
{season_cells}
<td><div class="bar-wrap"><span class="num">{pct(r['model_prob'])}</span>
<div class="bar"><span style="width:{max(0, min(100, r['model_prob'] * 100)):.1f}%"></span></div>
</div></td>
<td class="num">{num(r['odds_over'])}</td>
<td class="num">{pct(r['fair'])}</td>
<td class="num">{chip}</td>
</tr>"""
            )

        season_heads = "".join(
            f'<th class="num season">{esc(lb)}</th>' for lb in labels
        )
        avg_txt = pct(league_avg.get(div))
        n_block = len(block)
        return f"""<section class="league">
<div class="league-head">
  <h2>{esc(league_name(div))}</h2>
  <span class="league-meta">{n_block} {'partita' if n_block == 1 else 'partite'}
  &middot; media storica Over 2.5 {avg_txt}</span>
</div>
<div class="shell"><table>
<thead><tr>
<th>#</th><th>Partita</th>
<th class="num">Over casa</th><th class="num">Over trasf.</th>
{season_heads}
<th class="num">Stima Over 2.5</th><th class="num">Quota Over</th>
<th class="num">Prob. book</th><th class="num">Scostamento</th>
</tr></thead>
<tbody>{''.join(rows_html)}</tbody></table></div>
</section>"""

    if n_matches:
        # Ordine fisso: prima i campionati principali nell'ordine dichiarato,
        # poi gli extra. Compaiono solo quelli con partite in programma.
        ordine = list(LEAGUES.keys()) + list(EXTRA_LEAGUES.keys())
        presenti = [d for d in ordine if d in set(results["div"])]
        presenti += [d for d in results["div"].unique() if d not in ordine]
        table = "".join(
            league_section(d, results[results["div"] == d]) for d in presenti
        )
    else:
        table = f"""<div class="shell"><div class="empty">
Nessuna partita dei campionati monitorati nei prossimi {DAYS_AHEAD} giorni.<br>
Il file <code>fixtures.csv</code> di football-data.co.uk elenca solo le giornate
imminenti: fuori stagione, o quando i campionati seguiti non giocano, la tabella
resta vuota e si ripopola da sola alla prossima esecuzione.
</div></div>"""

    league_rows = "".join(
        f"<div><strong>{esc(league_name(d))}</strong>: media Over 2.5 storica "
        f"{pct(v)}</div>"
        for d, v in sorted(league_avg.items(), key=lambda kv: league_name(kv[0]))
    ) or "<div>Nessuno storico disponibile.</div>"

    return f"""<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Analisi Over/Under 2.5 &mdash; aggiornamento settimanale</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">

<div class="head">
  <div class="eyebrow">Aggiornamento automatico settimanale</div>
  <h1>Analisi Over/Under 2.5 gol</h1>
  <p class="sub">Partite dei prossimi {DAYS_AHEAD} giorni in {n_leagues} campionati europei,
  ordinate dalla piu' propensa all'Over 2.5 alla piu' propensa all'Under. La stima nasce
  dallo storico delle squadre; la quota affiancata mostra cosa ne pensa il mercato.</p>
  <div class="meta">Generato il {generated} &middot; dati:
  <a href="https://www.football-data.co.uk">football-data.co.uk</a></div>
</div>

<div class="cards">
  <div class="card"><div class="k">Partite analizzate</div><div class="v">{n_matches}</div></div>
  <div class="card"><div class="k">Gare nello storico</div><div class="v">{n_hist:,}</div></div>
  <div class="card"><div class="k">Over 2.5 storico</div><div class="v">{pct(hist_over, 1)}</div></div>
  <div class="card"><div class="k">Stagioni incluse</div><div class="v" style="font-size:.95rem">{seasons}</div></div>
</div>

{table}

<details>
<summary>Come si legge la tabella</summary>
<div class="legend">
  <div><strong>Over casa / Over trasf.</strong> &mdash; quota parte di partite chiuse con
  3+ gol dalla squadra di casa nelle sue gare interne, e dall'ospite nelle sue gare esterne,
  sullo storico caricato. Sotto, il numero di partite su cui e' calcolata: la dicitura
  <em>media lega</em> segnala una squadra assente dallo storico (neopromossa o appena salita
  di categoria), per la quale si usa la media del campionato al posto di un dato suo.</div>
  <div><strong>Colonne per stagione</strong> (sfondo colorato) &mdash; lo stesso dato
  stagione per stagione, per vedere se una squadra sta cambiando tendenza invece di
  leggere un unico numero medio. Ogni cella riporta <em>squadra di casa / squadra
  ospite</em>, e sotto il numero di partite di ciascuna in quella stagione. Sono valori
  grezzi, non corretti verso la media di lega: a inizio stagione poggiano su pochissime
  gare e vanno letti come indizio, non come misura.</div>
  <div><strong>Stima Over 2.5</strong> &mdash; media dei due tassi. Le squadre con poco
  storico (neopromosse) vengono avvicinate alla media del loro campionato con un peso
  pari a {PRIOR_MATCHES} partite, per non dare fiducia eccessiva a percentuali basate su
  pochissime gare.</div>
  <div><strong>Quota Over</strong> &mdash; <code>Avg&gt;2.5</code>, media dei bookmaker
  (ripiego su <code>B365&gt;2.5</code> se assente). Per Allsvenskan ed Eliteserien la fonte
  pubblica solo le quote 1X2: per quelle partite quota, probabilita' del mercato e
  scostamento restano vuoti e compare l'etichetta <em>no quota</em>, mentre la stima
  statistica resta valida.</div>
  <div><strong>Prob. book</strong> &mdash; probabilita' implicita nella quota, ripulita dal
  margine del bookmaker normalizzando Over e Under (la somma grezza di
  <code>1/quota</code> supera sempre il 100%).</div>
  <div><strong>Scostamento</strong> &mdash; differenza fra la stima e la probabilita' del
  mercato, in punti percentuali. Oltre &plusmn;5pt viene evidenziato. Uno scostamento ampio
  segnala che il modello e il mercato la vedono diversamente, non che il modello abbia
  ragione: il mercato incorpora infortuni, formazioni e motivazioni che questo calcolo
  ignora del tutto.</div>
  <div>{league_rows}</div>
</div>
</details>

<footer>
Analisi statistica automatica a scopo informativo &mdash; non costituisce consulenza di
scommessa ne' garanzia di risultato. Il gioco puo' causare dipendenza.
</footer>

</div>
</body>
</html>"""


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analisi Over/Under 2.5 gol")
    parser.add_argument("--output", default="index.html", help="file HTML di destinazione")
    args = parser.parse_args()

    today = datetime.now(timezone.utc)
    print(f"Analisi Over/Under 2.5 - {today:%Y-%m-%d %H:%M UTC}", file=sys.stderr)

    print("Scarico lo storico...", file=sys.stderr)
    hist = load_history(today)
    print(f"Storico: {len(hist)} partite", file=sys.stderr)

    rates, league_avg = build_team_rates(hist)
    print(f"Squadre con statistiche: {len(rates)}", file=sys.stderr)

    print("Scarico le partite in arrivo...", file=sys.stderr)
    fixtures = load_fixtures(today)
    print(f"Partite nei prossimi {DAYS_AHEAD} giorni: {len(fixtures)}", file=sys.stderr)

    results = analyse(fixtures, rates, league_avg) if len(fixtures) else pd.DataFrame()

    html = render_html(results, hist, league_avg, today)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Scritto {args.output} ({len(html):,} byte)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
