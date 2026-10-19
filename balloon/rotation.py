import os
from collections import Counter
from datetime import timedelta

# Map rotation tracker: rotation_tracker.py logs which map each proven rotating server is on every
# 5 minutes and on every map change, to its own log apart from the main one. This turns that log into the
# website's "Map rotation" section, to see whether Balloon Race comes up in a fixed spot in each server's cycle.
ROTATION_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data',
                                 'rotation_log.csv')
ROTATION_COLUMNS = ['ts', 'addr', 'name', 'map', 'players', 'bots', 'max_players', 'next_map']
BALLOON_MAP = 'balloon_race_v2b'
# A gap longer than this between checks means the tracker was down, so a map change across it wasn't seen
MAX_GAP = timedelta(minutes=15)
# The tracker checks every 30 seconds but only logs a map every 5 minutes, and on every map change
CHECK_MINUTES = 5
# Balloon Race has to follow the same map this many times before the cycle is called fixed
MIN_REPEATS = 3
RECENT_STINTS = 30
RECENT_GAPS = 6


# Runs of consecutive checks on the same map. An empty map means the server was offline.
def map_stints(rows):
    stints = []
    for r in rows.itertuples():
        last = stints[-1] if stints else None
        if last and r.map == last['map'] and r.ts - last['end'] <= MAX_GAP:
            last['end'] = r.ts
            last['peak'] = max(last['peak'], r.players)
            last['next_map'] = r.next_map or last['next_map']
        else:
            stints.append({'map': r.map, 'start': r.ts, 'end': r.ts, 'peak': r.players, 'next_map': r.next_map,
                           # Started right after the previous stint, so the change between them was seen
                           'follows': bool(last) and r.ts - last['end'] <= MAX_GAP})
    return stints


def verdict(before, visits):
    seen = sum(before.values())
    if not seen:
        return 'waiting', ('Waiting for Balloon Race to come up' if not visits
                           else 'Waiting to see which map comes before Balloon Race')
    top, count = before.most_common(1)[0]
    if count < seen:
        return 'varies', f'Not fixed: after {top} {count} of {seen} times'
    if seen < MIN_REPEATS:
        return 'waiting', f'After {top} so far ({seen} of {seen}), need {MIN_REPEATS} to call it'
    return 'fixed', f'Looks fixed: always after {top} ({seen} of {seen})'


def server_summary(rows):
    stints = map_stints(rows)
    visits = [i for i, s in enumerate(stints) if s['map'] == BALLOON_MAP]
    before = Counter(stints[i - 1]['map'] for i in visits if stints[i]['follows'] and stints[i - 1]['map'])
    after = Counter(stints[i + 1]['map'] for i in visits
                    if i + 1 < len(stints) and stints[i + 1]['follows'] and stints[i + 1]['map'])
    # Start to start; maps hidden by a tracker outage aren't counted
    gaps = [{'hours': round((stints[b]['start'] - stints[a]['start']).total_seconds() / 3600, 1),
             'maps_between': sum(1 for s in stints[a + 1:b] if s['map'])}
            for a, b in zip(visits, visits[1:])]
    status, text = verdict(before, visits)
    last = rows.iloc[-1]
    names = rows['name'][rows['name'] != '']
    return {
        'addr': last['addr'], 'name': names.iloc[-1] if len(names) else last['addr'],
        'checks': int(len(rows)),
        'current': {'map': last['map'], 'players': int(last['players']), 'next_map': last['next_map'],
                    'as_of': last['ts'].isoformat(), 'since': stints[-1]['start'].isoformat()},
        'visits': len(visits),
        'before': before.most_common(), 'after': after.most_common(),
        'gaps': gaps[-RECENT_GAPS:],
        'verdict': status, 'verdict_text': text,
        'recent': [{'map': s['map'], 'start': s['start'].isoformat(), 'peak': int(s['peak']), 'next_map': s['next_map'],
                    'minutes': int((s['end'] - s['start']).total_seconds() // 60) + CHECK_MINUTES}
                   for s in stints[-RECENT_STINTS:]],
    }


# Summary of every tracked server for the website, or None before the tracker has logged anything
def rotation_summary(log_path=ROTATION_LOG_PATH):
    # Imported here so rotation_tracker.py, which runs all the time and only needs the constants, stays small
    import pandas as pd
    try:
        df = pd.read_csv(log_path, dtype={'map': str, 'name': str, 'next_map': str}, keep_default_na=False)
    except FileNotFoundError:
        return None
    if df.empty:
        return None
    df['ts'] = pd.to_datetime(df['ts']).dt.tz_localize('UTC')
    df['players'] = pd.to_numeric(df['players'], errors='coerce').fillna(0).astype(int)
    df = df.sort_values('ts')
    servers = [server_summary(rows.reset_index(drop=True)) for _, rows in df.groupby('addr')]
    # Most decided first, then the ones that have hit Balloon Race most
    order = {'fixed': 0, 'varies': 1, 'waiting': 2}
    servers.sort(key=lambda s: (order[s['verdict']], -s['visits']))
    return {'since': df['ts'].min().isoformat(), 'last_check': df['ts'].max().isoformat(),
            'balloon_map': BALLOON_MAP, 'min_repeats': MIN_REPEATS, 'servers': servers}
