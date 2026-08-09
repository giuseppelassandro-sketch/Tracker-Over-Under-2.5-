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

# Codici football-data.co.uk -> nome leggibile (stessi codici della colonna Div)
LEAGUES = {
    "I1": "Serie A",
    "E0": "Premier League",
    "SP1": "LaLiga",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
}

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

def fetch_csv(url):
    """Scarica un CSV. Ritorna un DataFrame, oppure None se manca (404) o e' illeggibile.

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

    if df.empty or "Div" not in df.columns:
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
            frames.append(df)
            print(f"  [ok]   {season}/{code}: {len(df)} righe", file=sys.stderr)

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
      rates[(div, team)] = {"home_rate", "home_n", "away_rate", "away_n"}
      league_avg[div]    = tasso Over medio del campionato
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

    return rates, league_avg


# --------------------------------------------------------------------------
# Partite in arrivo
# --------------------------------------------------------------------------

def to_float(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) or f <= 1.0 else f


def load_fixtures(today, leagues=None):
    """Partite in arrivo nei prossimi DAYS_AHEAD giorni per i campionati scelti."""
    leagues = leagues or LEAGUES
    df = fetch_csv(FIXTURES_URL)
    if df is None or df.empty:
        return pd.DataFrame()

    df = df[df["Div"].isin(leagues.keys())].copy()
    if df.empty:
        return df

    df["MatchDate"] = pd.to_datetime(
        df["Date"], format="%d/%m/%Y", errors="coerce"
    )
    df = df.dropna(subset=["MatchDate"])

    start = pd.Timestamp(today.date())
    end = start + pd.Timedelta(days=DAYS_AHEAD)
    df = df[(df["MatchDate"] >= start) & (df["MatchDate"] <= end)]
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

        rows.append(
            {
                "div": div,
                "league": leagues.get(div, div),
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


def stagione_it(code):
    """'2526' -> '25/26'."""
    return f"{code[:2]}/{code[2:]}" if len(code) == 4 else code


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
table{width:100%;border-collapse:collapse;min-width:940px}
thead th{background:var(--surface-2);text-align:left;font-size:.67rem;
font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:var(--ink-soft);
padding:.7rem .8rem;border-bottom:1px solid var(--line);white-space:nowrap}
tbody td{padding:.62rem .8rem;border-bottom:1px solid var(--line);font-size:.86rem}
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
footer{color:var(--ink-soft);font-size:.78rem;text-align:center;line-height:1.6}
@media(max-width:640px){.wrap{padding:1.5rem 1rem 3rem}}
"""


def render_html(results, hist, league_avg, today, leagues=None):
    leagues = leagues or LEAGUES
    generated = today.strftime("%d/%m/%Y %H:%M UTC")
    n_matches = len(results)
    n_hist = len(hist)
    hist_over = float(hist["IsOver"].mean()) if n_hist else None
    seasons = (
        ", ".join(stagione_it(s) for s in sorted(hist["Season"].unique()))
        if n_hist
        else "&mdash;"
    )

    if n_matches:
        rows_html = []
        for i, r in results.iterrows():
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

            rows_html.append(
                f"""<tr>
<td class="rank">{i + 1}</td>
<td><div class="match">{esc(r['home'])} &ndash; {esc(r['away'])}</div>
<div class="when">{when}</div></td>
<td><span class="badge">{esc(r['league'])}</span></td>
<td class="num">{pct(r['home_rate'])}<div class="when">{home_note}</div></td>
<td class="num">{pct(r['away_rate'])}<div class="when">{away_note}</div></td>
<td><div class="bar-wrap"><span class="num">{pct(r['model_prob'])}</span>
<div class="bar"><span style="width:{max(0, min(100, r['model_prob'] * 100)):.1f}%"></span></div>
</div></td>
<td class="num">{num(r['odds_over'])}</td>
<td class="num">{pct(r['fair'])}</td>
<td class="num">{chip}</td>
</tr>"""
            )

        table = f"""<div class="shell"><table>
<thead><tr>
<th>#</th><th>Partita</th><th>Camp.</th>
<th class="num">Over casa</th><th class="num">Over trasf.</th>
<th class="num">Stima Over 2.5</th><th class="num">Quota Over</th>
<th class="num">Prob. book</th><th class="num">Scostamento</th>
</tr></thead>
<tbody>{''.join(rows_html)}</tbody></table></div>"""
    else:
        table = f"""<div class="shell"><div class="empty">
Nessuna partita dei campionati monitorati nei prossimi {DAYS_AHEAD} giorni.<br>
Il file <code>fixtures.csv</code> di football-data.co.uk elenca solo le giornate
imminenti: fuori stagione, o quando i campionati seguiti non giocano, la tabella
resta vuota e si ripopola da sola alla prossima esecuzione.
</div></div>"""

    league_rows = "".join(
        f"<div><strong>{esc(leagues.get(d, d))}</strong>: media Over 2.5 storica "
        f"{pct(v)}</div>"
        for d, v in sorted(league_avg.items())
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
  <p class="sub">Partite dei prossimi {DAYS_AHEAD} giorni nei 5 maggiori campionati europei,
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
  <div><strong>Stima Over 2.5</strong> &mdash; media dei due tassi. Le squadre con poco
  storico (neopromosse) vengono avvicinate alla media del loro campionato con un peso
  pari a {PRIOR_MATCHES} partite, per non dare fiducia eccessiva a percentuali basate su
  pochissime gare.</div>
  <div><strong>Quota Over</strong> &mdash; <code>Avg&gt;2.5</code>, media dei bookmaker
  (ripiego su <code>B365&gt;2.5</code> se assente).</div>
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
