import re
from balloon.servers import clean_name

# Log timestamps are written in the server's clock (UTC)
LOG_TIMEZONE = 'UTC'

LINE_RE = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - INFO - (.*)$')
SERVER_RE = re.compile(r'^SERVER (\S+) \| (.*) : (\d+)$')
PLAYERS_RE = re.compile(r'^(.*) : (\d+)$')


# One row per 5-minute check (busiest server and its count), plus one row per active server per check.
# Timestamps are left as logged strings.
def parse_log(lines):
    # Imported here so rotation_tracker.py, which runs all the time and only needs the regexes, stays small
    import pandas as pd
    checks, servers = [], []
    for line in lines:
        m = LINE_RE.match(line.rstrip('\n'))
        if not m:
            continue
        ts, msg = m.groups()
        if msg.startswith('No one playing') or msg.startswith('Found people playing'):
            checks.append({'ts': ts, 'server_name': None, 'online_players': 0})
        elif not checks or checks[-1]['ts'] != ts:
            continue
        elif s := SERVER_RE.match(msg):
            servers.append({'ts': ts, 'addr': s.group(1), 'name': clean_name(s.group(2)),
                            'players': int(s.group(3))})
        elif p := PLAYERS_RE.match(msg):
            checks[-1]['server_name'] = clean_name(p.group(1))
            checks[-1]['online_players'] = int(p.group(2))
    return (pd.DataFrame(checks, columns=['ts', 'server_name', 'online_players']),
            pd.DataFrame(servers, columns=['ts', 'addr', 'name', 'players']))


# The end of the log, without reading the whole file
def tail_lines(path, max_bytes):
    with open(path, 'rb') as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - max_bytes))
        data = f.read().decode('utf-8', errors='replace')
    lines = data.splitlines()
    # The first line is probably cut off partway
    return lines[1:] if size > max_bytes else lines
