from datetime import datetime, timezone
from balloon.forecast import predict_now
from balloon.servers import clean_name, is_dedicated
from balloon.voters import voters_on_now
import subprocess
import requests
import logging
import dotenv
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Runtime files: the log, forecast model and caches
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
LOG_PATH = os.path.join(DATA_DIR, 'tf2_balloon_log.txt')
# Trained daily by generate_report.py
FORECAST_MODEL_PATH = os.path.join(DATA_DIR, 'forecast_model.joblib')
# Branch the website reads "who's on now" from; force-pushed every check so it stays one commit
LIVE_BRANCH = 'live'
MIN_PLAYER_THRESHOLD = 10
# Servers that had MIN_PLAYER_THRESHOLD+ players last check, so Discord is only pinged when a lobby fills up
DISCORD_STATE_FILE = os.path.join(DATA_DIR, 'discord_notified.json')
# Discord is also pinged when this many Balloon Race voters (balloon/voters.py) are on a rotating server that
# isn't on it yet, since they may vote it in next. Servers already pinged, so it's once per crowd.
VOTER_ALERT_MIN = 8
DISCORD_VOTERS_STATE_FILE = os.path.join(DATA_DIR, 'discord_voters_notified.json')

def get_server_list():
    api_key = os.getenv('steam_api_key')
    url = f'https://api.steampowered.com/IGameServersService/GetServerList/v1/?key={api_key}&limit=50&filter=\\appid\\440\\map\\balloon_race_v2b'
    r = requests.get(url, timeout=30)
    servers = r.json()['response'].get('servers', [])
    # Human players only - some idle servers fill up with bots
    return [
        {'addr': s['addr'], 'name': s['name'], 'players': s['players'] - s.get('bots', 0),
         'max_players': s.get('max_players')}
        for s in servers
    ]

def notify_discord(servers):
    """Post to the Discord webhook for each server that just reached MIN_PLAYER_THRESHOLD or more players."""
    webhook_url = os.getenv('discord_webhook_url')
    if not webhook_url:
        return
    try:
        with open(DISCORD_STATE_FILE) as f:
            already_notified = set(json.load(f))
    except (FileNotFoundError, ValueError):
        already_notified = set()

    busy = [s for s in servers if s['players'] >= MIN_PLAYER_THRESHOLD]
    for s in busy:
        if s['addr'] in already_notified:
            continue
        content = (f"**PEOPLE ON BALLOON RACE:** {s['players']}"
                   f"{'/' + str(s['max_players']) if s.get('max_players') else ''} players\n"
                   f"Server: **{clean_name(s['name'])}**\n"
                   f"IP: `{s['addr']}`\n"
                   f"Console: `connect {s['addr']}`")
        try:
            requests.post(webhook_url, json={'content': content}, timeout=15).raise_for_status()
            logging.info(f"Sent Discord notification for {s['name']}")
        except requests.RequestException as e:
            print(f'Error sending Discord notification: {e}')
            # Leave it out of the state file so the next check retries
            busy = [b for b in busy if b is not s]

    with open(DISCORD_STATE_FILE, 'w') as f:
        json.dump([s['addr'] for s in busy], f)

def notify_voters():
    """Post to the Discord webhook for each rotating server that just reached VOTER_ALERT_MIN voters waiting."""
    webhook_url = os.getenv('discord_webhook_url')
    if not webhook_url:
        return
    try:
        with open(DISCORD_VOTERS_STATE_FILE) as f:
            already_notified = set(json.load(f))
    except (FileNotFoundError, ValueError):
        already_notified = set()

    waiting = [s for s in voters_on_now() if s['voters'] >= VOTER_ALERT_MIN]
    for s in waiting:
        if s['addr'] in already_notified:
            continue
        content = (f"**BALLOON RACE VOTERS WAITING:** {s['voters']} players who usually vote for Balloon Race are on "
                   f"**{s['name'] or s['addr']}** (on `{s['map']}`), so it might come up next\n"
                   f"IP: `{s['addr']}`\n"
                   f"Console: `connect {s['addr']}`")
        try:
            requests.post(webhook_url, json={'content': content}, timeout=15).raise_for_status()
            logging.info(f"Sent Discord voter notification for {s['name'] or s['addr']}")
        except requests.RequestException as e:
            print(f'Error sending Discord voter notification: {e}')
            # Leave it out of the state file so the next check retries
            waiting = [w for w in waiting if w is not s]

    with open(DISCORD_VOTERS_STATE_FILE, 'w') as f:
        json.dump([s['addr'] for s in waiting], f)

def lobby_forecast():
    """Chance of a lobby in the next hour from the checks logged so far, or None if it can't be worked out."""
    try:
        return predict_now(LOG_PATH, FORECAST_MODEL_PATH)
    except Exception as e:
        print(f'Error computing lobby forecast: {e!r}')
        return None

def publish_status(servers, forecast):
    """Push the current server list to the live branch without touching the working tree."""
    status = {
        'checked': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'servers': [dict(s, name=clean_name(s['name']), dedicated=is_dedicated(s['name'], s['addr'])) for s in servers],
        'forecast': forecast,
    }

    def git(*args, stdin=None):
        return subprocess.run(['git', *args], cwd=SCRIPT_DIR, input=stdin, capture_output=True,
                              text=True, check=True, timeout=60).stdout.strip()

    try:
        blob = git('hash-object', '-w', '--stdin', stdin=json.dumps(status, ensure_ascii=False))
        tree = git('mktree', stdin=f'100644 blob {blob}\tstatus.json\n')
        commit = git('commit-tree', tree, '-m', 'live status')
        git('push', '-fq', 'origin', f'{commit}:refs/heads/{LIVE_BRANCH}')
    except (subprocess.SubprocessError, OSError) as e:
        # Printed to cron.log rather than logged, so failures don't clutter the activity log
        print(f'Error publishing live status: {getattr(e, "stderr", None) or e}')

def main():
    # Load environment variables
    dotenv.load_dotenv()

    # Set up logging
    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Get the server list and check for players
    servers = get_server_list()
    output_dict = {s['name']: s['players'] for s in servers}
    if any(value > 0 for value in output_dict.values()):
        # Log server activity
        max_server = max(output_dict, key=output_dict.get)
        logging.info('Found people playing Balloon Race')
        logging.info(f'{max_server} : {output_dict[max_server]}')
        # Log every active server so the report can rank them
        for s in servers:
            if s['players'] > 0:
                logging.info(f"SERVER {s['addr']} | {s['name']} : {s['players']}")
    else:
        logging.info('No one playing balloon race :(')

    notify_discord(servers)
    try:
        notify_voters()
    except Exception as e:
        # The voter logs are the rotation tracker's; a problem there shouldn't stop the live status
        print(f'Error checking Balloon Race voters: {e!r}')

    # After logging, so the forecast includes this check
    publish_status(servers, lobby_forecast())

if __name__ == "__main__":
    main()
