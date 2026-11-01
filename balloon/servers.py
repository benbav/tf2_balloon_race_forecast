import re

# Servers that run Balloon Race all the time; everything else only rotates onto the map now and then.
# A server counts as dedicated if its name matches, or its address is listed here.
DEDICATED_NAME_RE = re.compile(r'balloon|bloon|balloon race 24/7', re.I)

DEDICATED_ADDRS = set()

# Steam Datagram Relay addresses change, so those servers can't be found again by address
RELAY_PREFIX = '169.254.'


# Some servers pad their names with control characters, which browsers and plotly.js can't render
def clean_name(name):
    return re.sub(r'[\x00-\x1f\x7f]', '', name).strip()


def is_dedicated(name, addr=None):
    return addr in DEDICATED_ADDRS or bool(DEDICATED_NAME_RE.search(name))


# Relay servers show up under a new address now and then, splitting one server's history across several.
# {name: address} to fold each such name's relay addresses into: its fixed address if it has one, otherwise
# the relay address seen last
def relay_targets(server_df):
    latest = server_df.sort_values('ts').groupby('addr').agg(name=('name', 'last'), last_seen=('ts', 'max'))
    latest['relay'] = latest.index.str.startswith(RELAY_PREFIX)
    targets = {}
    for name, group in latest.groupby('name'):
        if len(group) > 1 and group['relay'].any():
            fixed = group[~group['relay']]
            targets[name] = (fixed if len(fixed) else group)['last_seen'].idxmax()
    return targets


# server_df with relay addresses folded in by name, using relay_targets() unless given a saved mapping
def merge_relay_addresses(server_df, targets=None):
    targets = relay_targets(server_df) if targets is None else targets
    merged = server_df['name'].map(targets)
    use = server_df['addr'].str.startswith(RELAY_PREFIX) & merged.notna()
    return server_df.assign(addr=server_df['addr'].where(~use, merged))
