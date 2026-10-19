from datetime import datetime, timedelta, timezone
import json
import os
from balloon.log import tail_lines
from balloon.rotation import BALLOON_MAP

# Balloon Race voters: at every map change on a rotating server, rotation_tracker.py notes who was on right
# before it and who was still there a few minutes after. Players who stay for Balloon Race much more often
# than they stay for other maps are probably the ones voting for it. Names only - the server query that
# lists players doesn't give SteamIDs - so a player who changes their name starts over.
VOTER_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data',
                              'rotation_players.jsonl')
# Who's on each tracked server every 5 minutes, for counting the voters on at any past check (a lobby
# forecast input). Empty or offline servers aren't written. It grows a few MB a day, so it stays on the server.
SNAPSHOT_LOG_PATH = os.path.join(os.path.dirname(VOTER_LOG_PATH), 'rotation_snapshots.jsonl')
# Enough of the end of the snapshot log to cover the last few minutes on every server
SNAPSHOT_TAIL_BYTES = 300_000
# A server's latest snapshot older than this is stale (it emptied out or the tracker was down)
SNAPSHOT_STALE_MINUTES = 10
# Map changes a player has to have been there for, to Balloon Race and to anything else, before they're scored
MIN_BALLOON_CHANGES = 5
MIN_OTHER_CHANGES = 3
# Stay rate for Balloon Race minus stay rate for other maps; this or more counts as a voter
VOTER_MIN_SCORE = 0.25
LEADERBOARD_SIZE = 10


def read_jsonl(path):
    try:
        with open(path, encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


# Every map change logged so far: {ts, addr, from_map, to_map, before: [names], stayed: [names]}
def read_changes(path=VOTER_LOG_PATH):
    return read_jsonl(path)


# Every snapshot logged so far: {ts, addr, name, map, players: [names]} (older ones have no name)
def read_snapshots(path=SNAPSHOT_LOG_PATH):
    return read_jsonl(path)


# The last few minutes of snapshots, without reading the whole log
def recent_snapshots(path=SNAPSHOT_LOG_PATH):
    try:
        return [json.loads(line) for line in tail_lines(path, SNAPSHOT_TAIL_BYTES) if line.strip()]
    except FileNotFoundError:
        return []


# Every player seen at enough map changes, best voter first
def player_scores(changes):
    # name -> [Balloon Race stays, Balloon Race changes, other stays, other changes]
    tally = {}
    for c in changes:
        stayed = set(c['stayed'])
        i = 0 if c['to_map'] == BALLOON_MAP else 2
        for name in set(c['before']):
            t = tally.setdefault(name, [0, 0, 0, 0])
            t[i] += name in stayed
            t[i + 1] += 1
    scores = []
    for name, (balloon_stays, balloon_changes, other_stays, other_changes) in tally.items():
        if balloon_changes < MIN_BALLOON_CHANGES or other_changes < MIN_OTHER_CHANGES:
            continue
        # The +1/+2 keeps someone seen only a few times from scoring a perfect 100% or 0%
        balloon_rate = (balloon_stays + 1) / (balloon_changes + 2)
        other_rate = (other_stays + 1) / (other_changes + 2)
        scores.append({'name': name, 'score': round(balloon_rate - other_rate, 3),
                       'balloon_stays': balloon_stays, 'balloon_changes': balloon_changes,
                       'other_stays': other_stays, 'other_changes': other_changes})
    return sorted(scores, key=lambda s: (-s['score'], -s['balloon_changes']))


# The website's voter leaderboard (top LEADERBOARD_SIZE only)
def voter_summary(path=VOTER_LOG_PATH):
    changes = read_changes(path)
    voters = [s for s in player_scores(changes) if s['score'] >= VOTER_MIN_SCORE]
    return {'changes': len(changes), 'balloon_changes': sum(c['to_map'] == BALLOON_MAP for c in changes),
            'voters': len(voters), 'min_balloon_changes': MIN_BALLOON_CHANGES,
            'min_other_changes': MIN_OTHER_CHANGES, 'top': voters[:LEADERBOARD_SIZE]}


# Names scoring as voters from the map changes before `before` (a 'YYYY-MM-DD HH:MM:SS' UTC string), or all
def voter_names(changes, before=None):
    known = [c for c in changes if before is None or c['ts'] < before]
    return {s['name'] for s in player_scores(known) if s['score'] >= VOTER_MIN_SCORE}


# Voters on each snapshot's server, for the lobby forecast. Each day's snapshots are counted with the voters
# known when that day started, so a check is never helped by a Balloon Race vote that came after it.
# Servers already on Balloon Race are left out: the main log already sees that lobby.
def voters_waiting(snapshots, changes):
    rows, voters_by_day = [], {}
    for snap in snapshots:
        if snap['map'] == BALLOON_MAP:
            continue
        day = snap['ts'][:10]
        if day not in voters_by_day:
            voters_by_day[day] = voter_names(changes, before=day)
        rows.append({'ts': snap['ts'], 'addr': snap['addr'], 'name': snap.get('name', ''), 'map': snap['map'],
                     'voters': len(voters_by_day[day].intersection(snap['players']))})
    return rows


# Voters on each rotating server right now (not on Balloon Race, latest snapshot still fresh), most first
def voters_on_now(path=SNAPSHOT_LOG_PATH):
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=SNAPSHOT_STALE_MINUTES)).strftime('%Y-%m-%d %H:%M:%S')
    # Every server's latest snapshot, including any on Balloon Race, so a server that just switched to it drops out
    latest = {}
    for snap in recent_snapshots(path):
        latest[snap['addr']] = snap
    fresh = [s for s in latest.values() if s['ts'] >= cutoff]
    return sorted(voters_waiting(fresh, read_changes()), key=lambda r: -r['voters'])
