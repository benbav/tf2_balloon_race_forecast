import json
import os
from balloon.rotation import BALLOON_MAP

# Balloon Race voters: at every map change on a rotating server, rotation_tracker.py notes who was on right
# before it and who was still there a few minutes after. Players who stay for Balloon Race much more often
# than they stay for other maps are probably the ones voting for it. Names only - the server query that
# lists players doesn't give SteamIDs - so a player who changes their name starts over.
VOTER_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data',
                              'rotation_players.jsonl')
# Who's on each tracked server every 5 minutes, so the voters on at any past check can be counted later
# (e.g. as an input to the lobby forecast). Empty or offline servers aren't written.
SNAPSHOT_LOG_PATH = os.path.join(os.path.dirname(VOTER_LOG_PATH), 'rotation_snapshots.jsonl')
# Map changes a player has to have been there for, to Balloon Race and to anything else, before they're scored
MIN_BALLOON_CHANGES = 3
MIN_OTHER_CHANGES = 3
# Stay rate for Balloon Race minus stay rate for other maps; this or more counts as a voter
VOTER_MIN_SCORE = 0.25
LEADERBOARD_SIZE = 10


# Every map change logged so far: {ts, addr, from_map, to_map, before: [names], after: [names]}
def read_changes(path=VOTER_LOG_PATH):
    try:
        with open(path, encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


# Every player seen at enough map changes, best voter first
def player_scores(changes):
    # name -> [Balloon Race stays, Balloon Race changes, other stays, other changes]
    tally = {}
    for c in changes:
        after = set(c['after'])
        i = 0 if c['to_map'] == BALLOON_MAP else 2
        for name in set(c['before']):
            t = tally.setdefault(name, [0, 0, 0, 0])
            t[i] += name in after
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
