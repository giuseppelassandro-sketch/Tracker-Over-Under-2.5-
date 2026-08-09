# Analisi Over/Under 2.5 gol

Analisi automatica settimanale delle partite dei 5 maggiori campionati europei
(Serie A, Premier League, LaLiga, Bundesliga, Ligue 1), pubblicata su GitHub Pages.

Ogni giovedì mattina lo script scarica i dati da
[football-data.co.uk](https://www.football-data.co.uk) (gratuiti, senza chiave API),
calcola per ogni partita in arrivo una stima della probabilità di Over 2.5 gol,
la confronta con la quota dei bookmaker e rigenera la pagina.

## Come funziona

1. **Storico** — scarica la stagione in corso di ogni campionato
   (`mmz4281/{stagione}/{codice}.csv`) e calcola per ogni squadra la quota parte
   di partite chiuse con 3+ gol, separando gare in casa e in trasferta.
2. **Partite in arrivo** — legge `fixtures.csv`, che contiene le gare imminenti
   con le quote Over/Under 2.5 (`Avg>2.5`, `Avg<2.5`).
3. **Stima** — per ogni partita, media fra il tasso Over-in-casa della squadra
   di casa e il tasso Over-in-trasferta dell'ospite.
4. **Confronto col mercato** — affianca la probabilità implicita nella quota,
   ripulita dal margine del bookmaker, e ne evidenzia lo scostamento.

Le squadre con poco storico (neopromosse) vengono avvicinate alla media del loro
campionato con un peso di 8 partite equivalenti, per non dare fiducia eccessiva a
percentuali basate su pochissime gare. Le squadre del tutto assenti dallo storico
usano la media di lega, indicata esplicitamente in tabella.

## Uso locale

```bash
pip install pandas certifi
python3 analisi_over_under.py --output index.html
```

## Limiti

La stima usa **solo** la frequenza storica dei gol. Non considera infortuni,
formazioni, calendario, motivazioni o cambi di allenatore — tutti fattori che il
mercato incorpora. Uno scostamento ampio segnala che modello e mercato la vedono
diversamente, non che il modello abbia ragione.

Analisi statistica a scopo informativo: non è consulenza di scommessa né garanzia
di risultato.

## Struttura

```
analisi_over_under.py        script auto-contenuto (pandas)
.github/workflows/main.yml   cron settimanale + deploy su GitHub Pages
```
