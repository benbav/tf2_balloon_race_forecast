from datetime import datetime, timezone
from balloon.log import LINE_RE, SERVER_RE
from balloon.rotation import ROTATION_COLUMNS, ROTATION_LOG_PATH
from balloon.servers import clean_name, is_dedicated
from balloon.voters import VOTER_LOG_PATH
import requests
import socket
import struct
import dotenv
import json
import time
import csv
import sys
import os

# Map rotation tracker, kept apart from main.py and its log. Runs all the time (balloon-rotation.service)
# and checks each proven rotating server every 30 seconds, whatever map it's on:
# - logs the map to data/rotation_log.csv, to see if their map cycles are fixed
# - at every map change, logs who was on right before and who stayed, to data/rotation_players.jsonl,
#   to find the players who vote for Balloon Race (balloon/voters.py)
# `python rotation_tracker.py --once` does a single check, for testing.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(SCRIPT_DIR, 'data', 'tf2_balloon_log.txt')
# "Proven" = has had a lobby this big on Balloon Race in the main log
PROVEN_PLAYERS = 10
# Steam Datagram Relay addresses change, so those servers can't be found again by address
RELAY_PREFIX = '169.254.'
A2S_TIMEOUT = 2
CHECK_SECONDS = 30
# A map is logged at least this often per server, plus on every map change
MAP_LOG_SECONDS = 300
# Who stayed after a map change: anyone seen this many seconds after it, once they've had time to load in
AFTER_WINDOW = (60, 240)
# A map change across a longer gap than this (server offline, tracker down) isn't used for voters
MAX_CHANGE_GAP = 300
# How often the proven server list is re-read from the main log
WATCH_LIST_SECONDS = 3600

def proven_servers():
    """Rotating servers with a fixed address that have had a PROVEN_PLAYERS+ lobby on Balloon Race."""
    peaks, names = {}, {}
    with open(LOG_PATH, encoding='utf-8', errors='replace') as f:
        for line in f:
            if 'SERVER ' not in line:
                continue
            m = LINE_RE.match(line.rstrip('\n'))
            s = m and SERVER_RE.match(m.group(2))
            if s:
                addr = s.group(1)
                peaks[addr] = max(peaks.get(addr, 0), int(s.group(3)))
                names[addr] = clean_name(s.group(2))
    return sorted(a for a, p in peaks.items()
                  if p >= PROVEN_PLAYERS and not a.startswith(RELAY_PREFIX) and not is_dedicated(names[a], a))

def server_info(api_key, addr):
    """The server as Steam lists it right now, whatever map it's on, or None if it's offline."""
    url = ('https://api.steampowered.com/IGameServersService/GetServerList/v1/'
           f'?key={api_key}&filter=\\appid\\440\\gameaddr\\{addr}')
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return next((s for s in r.json()['response'].get('servers', []) if s['addr'] == addr), None)

def a2s_query(addr, code, reply_code):
    """Reply body to an A2S query sent straight to the server (no Steam API), or None if it doesn't answer."""
    host, port = addr.rsplit(':', 1)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(A2S_TIMEOUT)
        try:
            sock.sendto(b'\xff\xff\xff\xff' + code + b'\xff\xff\xff\xff', (host, int(port)))
            reply = sock.recv(65535)
            # Replies start with a challenge to send back; some servers send a fresh one a few times over
            for _ in range(5):
                if reply[4:5] != b'A':
                    break
                sock.sendto(b'\xff\xff\xff\xff' + code + reply[5:9], (host, int(port)))
                reply = sock.recv(65535)
            # Long replies come split over several packets: 12-byte header, then the piece
            if reply[:4] == b'\xfe\xff\xff\xff':
                total, parts = reply[8], {reply[9]: reply[12:]}
                while len(parts) < total:
                    p = sock.recv(65535)
                    parts[p[9]] = p[12:]
                reply = b''.join(parts[i] for i in sorted(parts))
        except (OSError, IndexError):
            return None
    return reply[5:] if reply[4:5] == reply_code else None

def next_map(addr):
    """SourceMod's sm_nextmap from an A2S_RULES query, or '' if the server hides its rules or doesn't answer."""
    body = a2s_query(addr, b'V', b'E')
    if body is None:
        return ''
    fields = body[2:].split(b'\x00')
    rules = dict(zip(fields[0::2], fields[1::2]))
    return rules.get(b'sm_nextmap', b'').decode('utf-8', errors='replace')

def player_names(addr):
    """Names of everyone on the server from an A2S_PLAYER query, or None if it doesn't answer.
    Players still connecting have no name yet and are left out."""
    body = a2s_query(addr, b'U', b'D')
    if body is None:
        return None
    names, i = set(), 1
    try:
        for _ in range(body[0]):
            end = body.index(b'\x00', i + 1)
            name = body[i + 1:end].decode('utf-8', errors='replace').strip()
            # Then a 4-byte score and 4-byte seconds connected
            struct.unpack_from('<lf', body, end + 1)
            i = end + 9
            if name:
                names.add(name)
    except (ValueError, struct.error):
        return None
    return names

def append_jsonl(path, record):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

class Tracker:
    def __init__(self, api_key):
        self.api_key = api_key
        self.watch, self.watch_at = [], 0
        # addr -> what the last checks saw
        self.servers = {}

    def check_all(self):
        if time.monotonic() - self.watch_at > WATCH_LIST_SECONDS or not self.watch:
            self.watch, self.watch_at = proven_servers(), time.monotonic()
        for addr in self.watch:
            try:
                self.check(addr)
            except Exception as e:
                # One bad server or reply shouldn't stop the others or the loop
                print(f'Rotation tracker: error checking {addr}: {e!r}')

    def check(self, addr):
        try:
            s = server_info(self.api_key, addr)
        except (requests.RequestException, ValueError, KeyError) as e:
            # Unknown, not offline, so nothing is logged for it
            print(f'Rotation tracker: error looking up {addr}: {e!r}')
            return
        now = time.time()
        st = self.servers.setdefault(addr, {'map': None, 'since': now, 'seen': 0, 'players': set(),
                                            'logged_map': None, 'logged_at': 0, 'change': None})
        current = s.get('map', '') if s else ''
        humans = s['players'] - s.get('bots', 0) if s else 0
        names = (player_names(addr) if humans else set()) if s else None
        changed = bool(current) and current != st['map']

        # Who's still on after the last map change
        change = st['change']
        if change:
            age = now - change['at']
            if names and AFTER_WINDOW[0] <= age <= AFTER_WINDOW[1]:
                change['after'] |= names
            if age > AFTER_WINDOW[1] or changed or not s:
                self.finish_change(change)
                st['change'] = None

        # The server isn't listed during the switch, so the old map is the last one it was online with
        if changed:
            if st['map'] and st['players'] and now - st['seen'] <= MAX_CHANGE_GAP:
                st['change'] = {'at': now, 'addr': addr, 'from_map': st['map'], 'to_map': current,
                                'before': sorted(st['players']), 'after': set()}
            st['map'], st['since'], st['players'] = current, now, set()
        if s:
            st['seen'] = now
            if names is not None:
                st['players'] = names

        if current != st['logged_map'] or now - st['logged_at'] >= MAP_LOG_SECONDS:
            self.log_map(addr, s, current, humans)
            st['logged_map'], st['logged_at'] = current, now

    def finish_change(self, change):
        # Nobody seen after it (empty server, or it went offline) says nothing about who stayed
        if not change['after']:
            return
        append_jsonl(VOTER_LOG_PATH, {
            'ts': datetime.fromtimestamp(change['at'], timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
            'addr': change['addr'], 'from_map': change['from_map'], 'to_map': change['to_map'],
            # Only who stayed: anyone who joined after the change says nothing about voting
            'before': change['before'], 'stayed': sorted(change['after'] & set(change['before']))})

    def log_map(self, addr, s, current, humans):
        row = {'ts': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'), 'addr': addr,
               'name': '', 'map': '', 'players': 0, 'bots': 0, 'max_players': '', 'next_map': ''}
        if s:
            row.update(name=clean_name(s['name']), map=current, players=humans, bots=s.get('bots', 0),
                       max_players=s.get('max_players', ''), next_map=next_map(addr))
        new_file = not os.path.exists(ROTATION_LOG_PATH)
        with open(ROTATION_LOG_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=ROTATION_COLUMNS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)

def main():
    dotenv.load_dotenv(os.path.join(SCRIPT_DIR, '.env'))
    tracker = Tracker(os.getenv('steam_api_key'))
    while True:
        started = time.monotonic()
        tracker.check_all()
        if '--once' in sys.argv:
            return
        time.sleep(max(0, CHECK_SECONDS - (time.monotonic() - started)))

if __name__ == "__main__":
    main()
