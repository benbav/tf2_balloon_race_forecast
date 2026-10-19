import datetime as dt
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from balloon.log import LOG_TIMEZONE, parse_log, tail_lines
from balloon.voters import (SNAPSHOT_LOG_PATH, SNAPSHOT_STALE_MINUTES, read_changes, read_snapshots,
                            recent_snapshots, voters_waiting)

# "Will a lobby form?" model: chance some server hits `threshold`+ players within the next hour.
# The daily report trains it on the whole log and saves it; the 5-minute poller (main.py) loads it and
# scores the latest check, so features can look back at recent checks. Both build features with
# check_frame + feature_matrix, so training and live scoring always see the same inputs.
FORECAST_MINUTES = 60
# Servers that were the busiest most often get their own feature; the rest share one "other server" bucket
FORECAST_SERVERS = 6
FORECAST_MIN_SERVER_CHECKS = 20
# A feature that only a day or two of the log can teach just memorizes those days, so day of week and
# holidays are left out until there's enough of each: until then it's weekend vs weekday, and no holiday flag
FORECAST_MIN_DATES_PER_WEEKDAY = 2
FORECAST_MIN_HOLIDAY_DATES = 3
# Momentum: how much the busiest server's count changed over these lookbacks
MOMENTUM_MINUTES = (15, 30)
# "Minutes since the last lobby" stops counting here
SINCE_BUSY_CAP_MINUTES = 24 * 60
# Live scoring reads this much of the end of the log, enough to cover the longest lookback
TAIL_BYTES = 2_000_000
# Voters waiting: the most Balloon Race voters (balloon/voters.py) on one rotating server that isn't on
# Balloon Race yet. Left out until the tracker's snapshots cover this many days with a voter seen in them.
FORECAST_MIN_VOTER_DATES = 7
# Rolling evaluation: each of the latest days (up to this many) is predicted by a model trained only
# on the days before it, and at least this many earlier days are needed before a day is tested
EVAL_MAX_DAYS = 14
EVAL_MIN_TRAIN_DAYS = 2
CALIBRATION_BINS = np.linspace(0, 1, 11)


# Address of the busiest server at each check. Older checks were logged before per-server lines
# existed, so their address is guessed from the server's name.
def busiest_addrs(df, server_df):
    top = server_df.sort_values('players').groupby('ts').last()['addr']
    by_name = server_df.groupby('name')['addr'].agg(lambda a: a.mode().iloc[0])
    return df['ts'].map(top).fillna(df['server_name'].map(by_name))


# Most voters waiting on any one server at each check, and whether the tracker was running yet then
def add_voters(f, voter_rows):
    f['voters_waiting'], f['has_voter_data'] = 0.0, 0.0
    if not voter_rows:
        return
    v = pd.DataFrame(voter_rows)
    v['ts'] = pd.to_datetime(v['ts']).dt.tz_localize('UTC').dt.tz_convert(f['ts'].dt.tz).astype(f['ts'].dtype)
    v = v.sort_values('ts')
    tolerance = pd.Timedelta(minutes=SNAPSHOT_STALE_MINUTES)
    for _, rows in v.groupby('addr'):
        on = pd.merge_asof(f[['ts']], rows[['ts', 'voters']], on='ts', direction='backward', tolerance=tolerance)
        f['voters_waiting'] = np.maximum(f['voters_waiting'], on['voters'].fillna(0).to_numpy())
    f['has_voter_data'] = (f['ts'] >= v['ts'].iloc[0]).astype(float)


# One row per check with everything the features need, each computed only from that check and earlier ones
def check_frame(df, server_df, threshold, check_minutes, voter_rows=None):
    f = df.sort_values('ts').reset_index(drop=True)
    f['addr'] = busiest_addrs(f, server_df)
    per_check = server_df.groupby('ts')['players'].agg(['sum', 'size'])
    # Older checks only know the busiest server
    f['total_players'] = f['ts'].map(per_check['sum']).fillna(f['online_players'])
    f['servers_on'] = f['ts'].map(per_check['size']).fillna((f['online_players'] > 0).astype(int))
    history = f[['ts', 'online_players']].rename(columns={'ts': 'then', 'online_players': 'players_then'})
    tolerance = pd.Timedelta(minutes=check_minutes)
    for minutes in MOMENTUM_MINUTES:
        then = pd.merge_asof(pd.DataFrame({'then': f['ts'] - pd.Timedelta(minutes=minutes)}), history,
                             on='then', direction='backward', tolerance=tolerance)
        # No check back then (start of the log, or the tracker was down): count it as no change
        f[f'change_{minutes}'] = f['online_players'] - then['players_then'].fillna(f['online_players']).values
    busy = f.loc[f['online_players'] >= threshold, ['ts']].rename(columns={'ts': 'last_busy'})
    last = pd.merge_asof(f[['ts']], busy, left_on='ts', right_on='last_busy', direction='backward')
    since = (f['ts'] - last['last_busy']).dt.total_seconds() / 60
    f['minutes_since_busy'] = since.fillna(SINCE_BUSY_CAP_MINUTES).clip(upper=SINCE_BUSY_CAP_MINUTES)
    add_voters(f, voter_rows)
    return f


def feature_matrix(f, cfg):
    players = f['online_players'].to_numpy(dtype=float)
    hour = (f['ts'].dt.hour + f['ts'].dt.minute / 60).to_numpy(dtype=float)
    angle = 2 * np.pi * hour / 24
    day = f['ts'].dt.dayofweek.to_numpy()
    days = [(day == d).astype(float) for d in range(7)] if cfg['use_days'] else [(day >= 5).astype(float)]
    # Nobody on means no busiest server, so it falls in the "other" bucket (all zeros)
    addr, on = f['addr'].to_numpy(dtype=object), players > 0
    servers = [((addr == s) & on).astype(float) for s in cfg['servers']]
    holidays = [f['ts'].dt.date.isin(cfg['holiday_dates']).to_numpy(dtype=float)] if cfg['use_holidays'] else []
    # Checks from before the tracker started have no count, so the flag keeps them from reading as "0 voters on".
    # Models saved before this feature existed have no use_voters.
    voters = ([f['has_voter_data'].to_numpy(dtype=float), f['voters_waiting'].to_numpy(dtype=float)]
              if cfg.get('use_voters') else [])
    return np.column_stack([
        np.log1p(players), (players >= cfg['threshold']).astype(float),
        np.log1p(f['total_players'].to_numpy(dtype=float)), f['servers_on'].to_numpy(dtype=float),
        *[f[f'change_{m}'].to_numpy(dtype=float) for m in MOMENTUM_MINUTES],
        np.log1p(f['minutes_since_busy'].to_numpy(dtype=float)),
        np.sin(angle), np.cos(angle), np.sin(2 * angle), np.cos(2 * angle),
        *days, *servers, *holidays, *voters,
    ])


# 1 if a later check within the next hour is busy; checks whose next hour is mostly missing (end of the
# log, or the tracker was down) are dropped
def lobby_labels(f, threshold, check_minutes):
    ts, busy = f['ts'].values, (f['online_players'] >= threshold).values
    horizon = np.timedelta64(FORECAST_MINUTES, 'm')
    labels, keep = [], []
    for i in range(len(f)):
        end = np.searchsorted(ts, ts[i] + horizon, side='right')
        if end - i - 1 < FORECAST_MINUTES // check_minutes * 0.75:
            continue
        labels.append(busy[i + 1:end].any())
        keep.append(i)
    return np.array(keep, dtype=int), np.array(labels, dtype=int)


def pick_servers(addrs, players):
    counts = addrs[players > 0].value_counts()
    return counts[counts >= FORECAST_MIN_SERVER_CHECKS].index[:FORECAST_SERVERS].tolist()


# Strong regularization: with only a handful of lobbies to learn from, weaker settings (C=0.5) were
# overconfident and scored worse than guessing the base rate on the rolling test
def new_forecast_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=1000))


def scores(y, probs, guess):
    out = {'examples': int(len(y)), 'positives': int(y.sum()),
           # Neighboring checks share most of their next hour, so separate lobbies are the real sample size
           'lobbies': int(((y == 1) & (np.r_[0, y[:-1]] == 0)).sum())}
    if not len(y):
        return out
    out.update(log_loss=round(log_loss(y, probs, labels=[0, 1]), 4),
               baseline_log_loss=round(log_loss(y, guess, labels=[0, 1]), 4),
               brier=round(float(brier_score_loss(y, probs)), 4),
               baseline_brier=round(float(brier_score_loss(y, guess)), 4))
    # AUC needs both outcomes
    if len(set(y)) == 2:
        auc = float(roc_auc_score(y, probs))
        out.update(auc=round(auc, 3), gini=round(2 * auc - 1, 3))
    return out


def calibration(y, probs):
    bins = np.clip(np.digitize(probs, CALIBRATION_BINS[1:-1]), 0, len(CALIBRATION_BINS) - 2)
    return [{'from': round(CALIBRATION_BINS[b], 1), 'to': round(CALIBRATION_BINS[b + 1], 1), 'checks': int((bins == b).sum()),
             'predicted': round(float(probs[bins == b].mean()), 3), 'actual': round(float(y[bins == b].mean()), 3)}
            for b in range(len(CALIBRATION_BINS) - 1) if (bins == b).any()]


# Rolling-origin test: each test day is predicted by a model fit only on examples whose next hour ended
# before that day started, so no test lobby leaks into training. Scored on every check, and on just the
# checks without a lobby yet (the question that matters; checks already busy are easy).
def rolling_evaluation(ts, X, y, players, threshold):
    dates = ts.dt.date.to_numpy()
    test_days = sorted(set(dates))[EVAL_MIN_TRAIN_DAYS:][-EVAL_MAX_DAYS:]
    probs, guess, idx, used = [], [], [], []
    for day in test_days:
        start = pd.Timestamp(day).tz_localize(ts.dt.tz)
        train = (ts < start - pd.Timedelta(minutes=FORECAST_MINUTES)).to_numpy()
        test = dates == day
        if train.sum() < 100 or len(set(y[train])) < 2 or not test.any():
            continue
        model = new_forecast_model().fit(X[train], y[train])
        probs.append(model.predict_proba(X[test])[:, 1])
        guess.append(np.full(test.sum(), y[train].mean()))
        idx.append(np.flatnonzero(test))
        used.append(day.isoformat())
    if not used:
        return None
    probs, guess, idx = np.concatenate(probs), np.concatenate(guess), np.concatenate(idx)
    test_y, is_open = y[idx], players[idx] < threshold
    return {'method': 'rolling by day', 'test_days': used, 'all': scores(test_y, probs, guess),
            'no_lobby_yet': scores(test_y[is_open], probs[is_open], guess[is_open]),
            'calibration': calibration(test_y, probs)}


# Returns (info for the website, fit for the regression chart or None)
def train_lobby_forecast(df, server_df, holiday_dates, threshold, check_minutes, timezone, model_path):
    voter_rows = voters_waiting(read_snapshots(), read_changes())
    f = check_frame(df, server_df, threshold, check_minutes, voter_rows)
    dates = f['ts'].dt.date.drop_duplicates()
    weekday_dates = pd.to_datetime(dates).dt.dayofweek.value_counts().reindex(range(7), fill_value=0)
    cfg = {'threshold': threshold, 'check_minutes': check_minutes, 'timezone': timezone,
           'servers': pick_servers(f['addr'], f['online_players']),
           'use_days': bool(len(dates)) and bool(weekday_dates.min() >= FORECAST_MIN_DATES_PER_WEEKDAY),
           'use_holidays': bool(dates.isin(holiday_dates).sum() >= FORECAST_MIN_HOLIDAY_DATES),
           'holiday_dates': set(holiday_dates),
           'use_voters': len({r['ts'][:10] for r in voter_rows}) >= FORECAST_MIN_VOTER_DATES
                         and bool(f['voters_waiting'].max() > 0)}
    keep, y = lobby_labels(f, threshold, check_minutes)
    examples = f.iloc[keep].reset_index(drop=True)
    X = feature_matrix(examples, cfg)
    days = (f['ts'].max() - f['ts'].min()).days if len(f) else 0
    info = {'model': 'Logistic regression', 'horizon_minutes': FORECAST_MINUTES, 'threshold': threshold, 'timezone': timezone,
            'days': days, 'examples': int(len(y)), 'positives': int(y.sum()), 'servers': cfg['servers'],
            'uses_day_of_week': cfg['use_days'], 'uses_holidays': cfg['use_holidays'], 'uses_voters': cfg['use_voters']}
    if len(y) < 100 or y.sum() < 3 or y.sum() == len(y):
        print('Lobby forecast: not enough data to train')
        return dict(info, ready=False), None
    evaluation = rolling_evaluation(examples['ts'], X, y, examples['online_players'].to_numpy(), threshold)
    if evaluation:
        info['evaluation'] = evaluation
        a, o = evaluation['all'], evaluation['no_lobby_yet']
        print(f"Lobby forecast, {len(evaluation['test_days'])} test days: log loss {a['log_loss']:.3f} "
              f"(base rate guess {a['baseline_log_loss']:.3f}), AUC {a.get('auc')}, Gini {a.get('gini')}; "
              f"no lobby yet: AUC {o.get('auc')}, Gini {o.get('gini')}, {o['lobbies']} lobbies")
    model = new_forecast_model().fit(X, y)
    trained = dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')
    joblib.dump({'model': model, 'cfg': cfg, 'trained': trained}, model_path)
    # Each training check's score (log-odds of a lobby) and what happened, for the regression chart
    fit = {'score': model.decision_function(X), 'y': y}
    return dict(info, ready=True, trained=trained), fit


# Chance of a lobby in the next hour as of the latest check in the log, or None if there's no model yet
def predict_now(log_path, model_path, snapshot_path=SNAPSHOT_LOG_PATH):
    try:
        saved = joblib.load(model_path)
    except FileNotFoundError:
        return None
    cfg = saved['cfg']
    checks, servers = parse_log(tail_lines(log_path, TAIL_BYTES))
    if checks.empty:
        return None
    for frame in (checks, servers):
        frame['ts'] = pd.to_datetime(frame['ts']).dt.tz_localize(LOG_TIMEZONE).dt.tz_convert(cfg['timezone'])
    voter_rows = None
    if cfg.get('use_voters'):
        voter_rows = voters_waiting(recent_snapshots(snapshot_path), read_changes())
    f = check_frame(checks, servers, cfg['threshold'], cfg['check_minutes'], voter_rows).tail(1)
    chance = float(saved['model'].predict_proba(feature_matrix(f, cfg))[0, 1])
    row = f.iloc[0]
    return {'chance': round(chance, 3), 'as_of': row['ts'].isoformat(), 'players': int(row['online_players']),
            'change_15': int(row['change_15']), 'voters_waiting': int(row['voters_waiting']), 'trained': saved['trained']}
