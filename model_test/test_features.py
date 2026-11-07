# Tests candidate features for the lobby forecast: each one is added on top of the current features
# (balloon/forecast.py feature_matrix) and scored on the same rolling day-by-day test. Nothing is saved or pushed;
# outside data is cached in data/ (gitignored).
# Run from the repo root: .venv/bin/python model_test/test_features.py
# 2026-11-07, 30 days of data, 14 test days, 11 lobbies on checks with no lobby yet: only "players on rotating
# community servers" helped consistently (better on 10/14 days, 99% of bootstraps). Hours since the last TF2 update
# had the biggest gain but on only 7/14 days, so a few updates. Acceleration, TF2 players worldwide and the long
# shots (earthquakes, moon, day of month) did nothing or hurt. Recheck in December with more lobbies.
import os
import sys, json, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv = ['x', '--no-push', '--no-readme']
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, requests
from sklearn.metrics import log_loss, roc_auc_score
import generate_report as gr
from balloon import forecast as fc
from balloon.voters import voters_waiting, read_snapshots, read_changes

TF2_COUNTS_URL = 'https://steamcharts.com/app/440/chart-data.json'
TF2_COUNTS_CACHE = os.path.join(gr.DATA_DIR, 'tf2_player_counts.json')
TF2_NEWS_URL = 'https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/?appid=440&count=100&feeds=tf2_blog'
TF2_NEWS_CACHE = os.path.join(gr.DATA_DIR, 'tf2_news.json')
BOOTSTRAPS = 2000


def cached_json(url, path):
    try:
        r = requests.get(url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
        r.raise_for_status()
        with open(path, 'w') as f:
            f.write(r.text)
    except Exception as e:
        print(f'fetch failed ({e}), using cache')
    with open(path) as f:
        return json.load(f)


# Value of `series` (UTC index) at or before each check, within `tolerance`
def as_of(ts, series, tolerance):
    s = series.sort_index()
    right = pd.DataFrame({'t': s.index.tz_convert('UTC').as_unit('ns'), 'v': s.to_numpy(dtype=float)})
    left = pd.DataFrame({'t': ts.dt.tz_convert('UTC').dt.as_unit('ns')})
    return pd.merge_asof(left, right, on='t', direction='backward', tolerance=tolerance)['v'].to_numpy()


# Busiest-server players `minutes` before each check (no check then: today's value, like change_15)
def lagged(f, col, minutes, tolerance):
    hist = pd.DataFrame({'then': f['ts'], 'v': f[col]})
    then = pd.merge_asof(pd.DataFrame({'then': f['ts'] - pd.Timedelta(minutes=minutes)}), hist, on='then',
                         direction='backward', tolerance=tolerance)['v']
    return then.fillna(f[col]).to_numpy(dtype=float)


# 1 if any check in [t - back, t - back + width] had a lobby
def lobby_in_window(f, threshold, back, width):
    ts = f['ts'].values
    busy = np.r_[0, np.cumsum(f['online_players'].to_numpy() >= threshold)]
    lo = np.searchsorted(ts, ts - np.timedelta64(back, 'm'), side='left')
    hi = np.searchsorted(ts, ts - np.timedelta64(back - width, 'm'), side='right')
    return (busy[hi] - busy[lo] > 0).astype(float)


def moon_illumination(ts):
    days = (ts.dt.tz_convert('UTC') - pd.Timestamp('2000-01-06 18:14', tz='UTC')).dt.total_seconds() / 86400
    phase = (days / 29.530588853) % 1
    return ((1 - np.cos(2 * np.pi * phase)) / 2).to_numpy()


def candidates(f, thr, cm):
    tol = pd.Timedelta(minutes=cm)
    p = f['online_players'].to_numpy(dtype=float)
    out = {}

    # --- Acceleration / momentum shape ---
    out['Change, last 5 min'] = p - lagged(f, 'online_players', 5, tol)
    f['_c15_then'] = lagged(f, 'change_15', 15, tol)
    out['Acceleration (15-min change vs the 15 before)'] = f['change_15'].to_numpy() - f['_c15_then'].to_numpy()
    out['Change on all servers, last 15 min'] = f['total_players'].to_numpy() - lagged(f, 'total_players', 15, tol)
    peak = f.set_index('ts')['online_players'].rolling('60min').max().to_numpy()
    out['Peak on the busiest server, last hour'] = np.log1p(peak)
    out['Below last hour\'s peak'] = peak - p

    # --- TF2 as a whole ---
    counts = cached_json(TF2_COUNTS_URL, TF2_COUNTS_CACHE)
    tf2 = pd.Series([c for _, c in counts], index=pd.to_datetime([t for t, _ in counts], unit='ms', utc=True))
    # A few samples read 0 (SteamCharts glitches), so they're dropped
    tf2 = tf2[(tf2.index >= f['ts'].min() - pd.Timedelta(days=10)) & (tf2 > 0)]
    now = as_of(f['ts'], tf2, pd.Timedelta(hours=3))
    out['TF2 players worldwide'] = np.log(np.nan_to_num(now, nan=np.nanmedian(now)))
    # Against the trailing 7-day average, so it isn't just the time of day again
    usual = as_of(f['ts'], tf2.rolling('7D').mean(), pd.Timedelta(hours=3))
    rel = now / usual - 1
    out['TF2 players vs their 7-day average'] = np.nan_to_num(rel, nan=0.0)
    hour_ago = as_of(f['ts'] - pd.Timedelta(hours=1), tf2, pd.Timedelta(hours=3))
    out['TF2 players, change over the last hour'] = np.nan_to_num(np.log(now / hour_ago), nan=0.0)

    # Humans on the rotating servers the tracker watches. The model has used it since 2026-11-07; this test's
    # baseline cfg leaves it out, so it's tested here as an addition like the rest
    fc.add_rotation(f, fc.read_rotation())
    out['Players on rotating community servers'] = [f['has_rotation_data'].to_numpy(), np.log1p(f['rotation_players'].to_numpy())]

    # --- Long shots ---
    out['Lobby in this same hour yesterday'] = lobby_in_window(f, thr, 24 * 60, 60)
    out['Lobby in this same hour last week'] = lobby_in_window(f, thr, 7 * 24 * 60, 60)
    news = cached_json(TF2_NEWS_URL, TF2_NEWS_CACHE)['appnews']['newsitems']
    updates = sorted(pd.Timestamp(n['date'], unit='s', tz='UTC') for n in news if 'Update' in n['title'])
    upd = pd.Series(np.arange(len(updates), dtype=float), index=pd.DatetimeIndex(updates))
    utc = f['ts'].dt.tz_convert('UTC').dt.as_unit('ns')
    last = pd.merge_asof(pd.DataFrame({'t': utc}), pd.DataFrame({'t': upd.index.as_unit('ns'), 'u': upd.index.as_unit('ns')}),
                         on='t', direction='backward')['u']
    hrs = (utc - last).dt.total_seconds().to_numpy() / 3600
    out['Hours since the last TF2 update'] = np.log1p(np.nan_to_num(hrs, nan=30 * 24).clip(0, 30 * 24))
    with open(gr.QUAKE_CACHE_PATH) as fh:
        quakes = json.load(fh)
    # Count for hour h is known once h ends
    q = pd.Series(list(quakes.values()), index=pd.to_datetime(list(quakes), utc=True) + pd.Timedelta(hours=1), dtype=float)
    out['Earthquakes worldwide, last hour'] = np.nan_to_num(as_of(f['ts'], q, pd.Timedelta(hours=3)), nan=q.median())
    out['Moon brightness'] = moon_illumination(f['ts'])
    out['Day of the month'] = f['ts'].dt.day.to_numpy(dtype=float)
    return out


# Same as fc.rolling_evaluation, but keeps each check's probability so days can be resampled
def rolling(ts, X, y):
    dates = ts.dt.date.to_numpy()
    test_days = sorted(set(dates))[fc.EVAL_MIN_TRAIN_DAYS:][-fc.EVAL_MAX_DAYS:]
    probs, idx = np.full(len(y), np.nan), []
    for day in test_days:
        start = pd.Timestamp(day).tz_localize(ts.dt.tz)
        train = (ts < start - pd.Timedelta(minutes=fc.FORECAST_MINUTES)).to_numpy()
        test = dates == day
        if train.sum() < 100 or len(set(y[train])) < 2 or not test.any():
            continue
        probs[test] = fc.new_forecast_model().fit(X[train], y[train]).predict_proba(X[test])[:, 1]
    return probs


def row_loss(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    df, server_df = gr.load_and_preprocess_data(gr.LOG_PATH)
    years = range(df['ts'].min().year, df['ts'].max().year + 2)
    hol = {d for ds in gr.holiday_calendar(years).values() for d in ds}
    thr, cm = gr.LOBBY_PLAYERS, gr.CHECK_MINUTES
    vr = voters_waiting(read_snapshots(), read_changes())
    server_df = fc.merge_relay_addresses(server_df, fc.relay_targets(server_df))
    f = fc.check_frame(df, server_df, thr, cm, vr)
    dates = f['ts'].dt.date.drop_duplicates()
    wd = pd.to_datetime(dates).dt.dayofweek.value_counts().reindex(range(7), fill_value=0)
    cfg = {'threshold': thr, 'check_minutes': cm, 'servers': fc.pick_servers(f['addr'], f['online_players']),
           'use_days': bool(wd.min() >= fc.FORECAST_MIN_DATES_PER_WEEKDAY),
           'use_holidays': bool(dates.isin(hol).sum() >= fc.FORECAST_MIN_HOLIDAY_DATES), 'holiday_dates': hol,
           'use_voters': len({r['ts'][:10] for r in vr}) >= fc.FORECAST_MIN_VOTER_DATES
                         and int((f['voters_waiting'] > 0).sum()) >= fc.FORECAST_MIN_VOTER_CHECKS}
    cands = candidates(f, thr, cm)
    keep, y = fc.lobby_labels(f, thr, cm)
    ex = f.iloc[keep].reset_index(drop=True)
    X0 = fc.feature_matrix(ex, cfg)
    cols = {k: np.column_stack(v if isinstance(v, list) else [v])[keep] for k, v in cands.items()}
    is_open = ex['online_players'].to_numpy() < thr
    test_dates = ex['ts'].dt.date.to_numpy()

    base = rolling(ex['ts'], X0, y)
    tested = ~np.isnan(base)
    days = sorted(set(test_dates[tested]))
    print(f"{len(dates)} days of data, {len(y)} examples, {y.sum()} lobby-hour positives; "
          f"test: {len(days)} days ({days[0]} .. {days[-1]}), "
          f"{fc.scores(y[tested & is_open], base[tested & is_open], base[tested & is_open])['lobbies']} lobbies on open checks\n")

    groups = {**{k: [k] for k in cols},
              'ALL acceleration': list(cols)[:5],
              'ALL TF2-wide': ['TF2 players worldwide', 'TF2 players vs their 7-day average',
                               'TF2 players, change over the last hour', 'Players on rotating community servers'],
              'ALL candidates': list(cols)}
    rng = np.random.default_rng(0)
    boot = rng.integers(0, len(days), size=(BOOTSTRAPS, len(days)))

    def summarize(name, p, mask_extra=None):
        m = tested if mask_extra is None else tested & mask_extra
        d = row_loss(y[m], p[m]) - row_loss(y[m], base[m])
        per_day = np.array([d[test_dates[m] == day].sum() for day in days])
        n = np.array([(test_dates[m] == day).sum() for day in days])
        sims = per_day[boot].sum(1) / n[boot].sum(1)
        auc = roc_auc_score(y[m], p[m]) if len(set(y[m])) == 2 else np.nan
        return {'LL': log_loss(y[m], p[m]), 'dLL': d.mean(), 'AUC': auc,
                'days better': f"{int((per_day < 0).sum())}/{len(days)}", 'P(helps)': (sims < 0).mean()}

    rows = [{'feature': '(current model)', **{f'{k}': v for k, v in summarize('', base).items()},
             **{f'open {k}': v for k, v in summarize('', base, is_open).items()}}]
    for name, members in groups.items():
        X = np.column_stack([X0, *[cols[k] for k in members]])
        p = rolling(ex['ts'], X, y)
        full = fc.new_forecast_model().fit(X, y)
        w = full[-1].coef_[0][X0.shape[1]:]
        rows.append({'feature': name, **summarize(name, p), **{f'open {k}': v for k, v in summarize(name, p, is_open).items()},
                     'weight': np.sign(w[-1]) * np.sqrt((w ** 2).sum()) if len(members) == 1 else np.nan})
    out = pd.DataFrame(rows)
    pd.set_option('display.width', 250)
    print(out[['feature', 'LL', 'dLL', 'AUC', 'days better', 'P(helps)', 'open LL', 'open dLL', 'open AUC',
               'open days better', 'open P(helps)', 'weight']].to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print("\ndLL = change in log loss vs the current model (negative is better). P(helps) = share of day-resampled "
          "bootstraps where it lowers log loss; days better = test days it beat the current model on. "
          "weight = standardized coefficient in a model fit on everything (sign = direction).")


if __name__ == '__main__':
    main()
