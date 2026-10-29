from datetime import datetime, timezone
import json
import os

# Balloon Race top scorers: rotation_tracker.py reads every player's score from the server query on each server
# that's on Balloon Race. A game is one player's continuous stay on the map; it ends when they leave, the map
# changes or their score resets, and is logged with the best score they reached. Names only (the query has no
# SteamIDs), so players sharing a name are counted together.
SCORE_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data',
                              'balloon_scores.jsonl')
# A game not seen for this long (server stopped answering, tracker down) is over
STALE_SECONDS = 300
LEADERBOARD_SIZE = 10


def utc(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime('%Y-%m-%d %H:%M:%S')


class ScoreKeeper:
    """Follows the games in progress on every Balloon Race server and logs each one as it ends."""

    def __init__(self, path=SCORE_LOG_PATH):
        self.path = path
        # (addr, player) -> {server, best, last, start, seen}
        self.games = {}

    def update(self, addr, server, players, now):
        """players: {name: score} from one query of a server on Balloon Race."""
        for name, score in players.items():
            g = self.games.get((addr, name))
            # A lower score than last time means the map restarted or they reconnected
            if g and score < g['last']:
                self.finish((addr, name))
                g = None
            if not g:
                g = self.games[(addr, name)] = {'server': server, 'best': 0, 'last': 0, 'start': now, 'seen': now}
            g.update(best=max(g['best'], score), last=score, seen=now)
        # Anyone on this server who wasn't in the reply has left
        for key in [k for k in self.games if k[0] == addr and k[1] not in players]:
            self.finish(key)

    def leave(self, addr):
        """The server left Balloon Race or emptied out: every game on it is over."""
        for key in [k for k in self.games if k[0] == addr]:
            self.finish(key)

    def expire(self, now):
        for key in [k for k, g in self.games.items() if now - g['seen'] > STALE_SECONDS]:
            self.finish(key)

    def finish(self, key):
        g = self.games.pop(key)
        # Someone who never scored isn't worth a line
        if g['best'] <= 0:
            return
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'ts': utc(g['seen']), 'addr': key[0], 'server': g['server'], 'player': key[1],
                                'score': g['best'], 'minutes': round((g['seen'] - g['start']) / 60)},
                               ensure_ascii=False) + '\n')


def read_games(path=SCORE_LOG_PATH):
    try:
        with open(path, encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


# The website's top scorers: best score in a single game, with each player's total over every game
def score_summary(path=SCORE_LOG_PATH):
    games = read_games(path)
    players = {}
    for g in games:
        p = players.setdefault(g['player'], {'name': g['player'], 'best': 0, 'best_server': '', 'total': 0,
                                             'games': 0})
        if g['score'] > p['best']:
            p.update(best=g['score'], best_server=g['server'])
        p['total'] += g['score']
        p['games'] += 1
    top = sorted(players.values(), key=lambda p: (-p['best'], -p['total']))[:LEADERBOARD_SIZE]
    return {'games': len(games), 'players': len(players), 'since': games[0]['ts'] if games else None, 'top': top}
