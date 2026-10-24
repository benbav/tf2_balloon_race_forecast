### WHEN ARE PEOPLE PLAYING BALLOON RACE????
#### No more randomly checking and hoping people are on - here is the cold hard data

##### Checks every 5 minutes, charts update every day (times in Mountain Time)

**Get pinged when a 10+ player server is going: [join the Discord](https://discord.gg/gdDVEeH7Y)**

**Website with a live "who's on now" list and server filters: https://benbav.github.io/tf2_balloon_race_forecast/**

Example of the live lobby forecast (a static screenshot - on the website it updates every 5 minutes):

![Lobby forecast example](site/assets/forecast_example.png)

### Top 5 Balloon Race servers

Ranked by how often they have a lobby going (as many players as the Discord ping), then by how often anyone's on. Favorite these so they're one click away.

<!-- TOP_SERVERS_START -->
| # | Server | Connect (paste in TF2 console) | Best time for a 10+ lobby | Best day | Has 10+ players | Has players | Avg players when on | Peak |
|---|---|---|---|---|---|---|---|---|
| 1 | [TR]  Türk Ninjalar sunucusu | `connect 88.233.138.82:27017` | Sun · 6 AM – 8 AM MT (24%) | Friday | 1% of checks | 1% of checks | 13.3 | 18 |
| 2 | shounic trenches (USA Chicago) | `connect 193.243.190.25:27015` | Fri · 6 PM – 8 PM MT (5%) | Saturday | 0% of checks | 1% of checks | 17.1 | 84 |
| 3 | Server hosted by Physgun.com | `connect 45.45.238.134:27105` | Fri · 6 PM – 8 PM MT (17%) | Friday | 0% of checks | 0% of checks | 21.8 | 23 |
| 4 | Aar's Server \| Map Testing \| Virginia | `connect 169.254.112.56:12184` | Sat · 2 PM – 4 PM MT (18%) | Saturday | 0% of checks | 0% of checks | 24.7 | 26 |
| 5 | Spanakopita | `connect 99.110.11.58:27015` | Mon · 9 PM – 11 PM MT (8%) | Monday | 0% of checks | 1% of checks | 5.3 | 10 |

To favorite one: Steam → View → Game Servers → Favorites → **+** and paste the address.
<!-- TOP_SERVERS_END -->

---

### How it works

- `main.py` runs every 5 minutes. It asks the Steam API which TF2 servers are on `balloon_race_v2b`, logs every server with human players (bots excluded) to `data/tf2_balloon_log.txt`, and scores the lobby forecast. If `discord_webhook_url` is set, it posts to Discord when a server reaches `LOBBY_PLAYERS` (10) or more players.
- `site/index.html` is the website, served free by GitHub Pages. It shows the charts, the top servers from `site/charts/site_data.json`, and the live server list.
- `generate_report.py` runs once a day. It turns the log into the website's charts, retrains the lobby forecast, rewrites the top servers table, and pushes the README and log to `main`. Everything in `site/` goes to the `site` branch as a single commit that's replaced daily, so ~16MB of charts a day doesn't pile up in the history. The log lives in `main` so its history is never lost.
- `main.py` also pushes the current server list to the `live` branch every check, the same single-commit way, for the website's "who's on now" list.
- `rotation_tracker.py` runs all the time as a systemd service (`balloon-rotation.service`), separately from everything else. Every 30 seconds it looks up the rotating servers that have had a `LOBBY_PLAYERS`+ player lobby on Balloon Race (fixed addresses only):
  - It logs whichever map each one is on, plus its `sm_nextmap` when the server shares it, to `data/rotation_log.csv` (every 5 minutes and on every map change), so we can see whether Balloon Race comes up at a fixed spot in their map cycles.
  - At every map change it logs who was on right before and who was still there a few minutes later to `data/rotation_players.jsonl`, from the server's own player list (names only, no SteamIDs). Players who stay for Balloon Race far more often than for other maps are probably voting for it (`balloon/voters.py`).
  - Every 5 minutes it logs who's on each one to `data/rotation_snapshots.jsonl` (on the server only), so the lobby forecast can count how many of those voters are sitting on a rotating server that isn't on Balloon Race yet. The forecast starts using that count once the snapshots cover a week with a voter seen in them.
  - `generate_report.py` summarizes both logs daily (`balloon/rotation.py`, `balloon/voters.py`) into the "Map rotation tracker" section and the top 10 voter leaderboard at the bottom of the website.
- `balloon/` is the code both scripts share: reading the log (`log.py`), which servers are dedicated (`servers.py`) and the lobby forecast model (`forecast.py`).
- `data/` holds the log and the voter log (tracked in git) plus files that aren't: the rotation log and player snapshots (they grow too fast for git, so they stay on the server), the trained model, the Discord ping state and the cached unemployment and earthquake data.

### First-time setup

**1. Install**

```bash
git clone git@github.com:benbav/tf2_balloon_race_forecast.git
cd tf2_balloon_race_forecast
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/plotly_get_chrome   # headless Chrome that plotly uses to save the chart PNGs
```

**2. Create a `.env` file** in the project folder (it's gitignored, never commit it):

```bash
# Steam Web API key
steam_api_key=

# Optional: Discord channel webhook for "lobby is up" pings
discord_webhook_url=
```

Where to get each value:

| Key | Where to get it |
|---|---|
| `steam_api_key` | Sign in at https://steamcommunity.com/dev/apikey, enter any domain name (e.g. `localhost`), and copy the key. |
| `discord_webhook_url` | Optional. In Discord: channel settings (gear icon) → Integrations → Webhooks → New Webhook → Copy Webhook URL. Posts once when a server reaches `LOBBY_PLAYERS` or more players, and again only after it empties out and refills. |

**3. Test it**

```bash
.venv/bin/python main.py && tail data/tf2_balloon_log.txt
.venv/bin/python generate_report.py --no-push   # draws charts without pushing
```

**4. Schedule it** with `crontab -e` (cron runs in the server's clock, UTC here):

```cron
*/5 * * * * cd /path/to/tf2_balloon_race_forecast && .venv/bin/python main.py >> cron.log 2>&1
0 6 * * * cd /path/to/tf2_balloon_race_forecast && .venv/bin/python generate_report.py >> cron.log 2>&1
```

Then start the rotation tracker, which runs all the time rather than from cron (edit the paths and user in the file first if yours differ):

```bash
sudo cp balloon-rotation.service /etc/systemd/system/ && sudo systemctl enable --now balloon-rotation
journalctl -u balloon-rotation -f   # its output
```

The daily report pushes to GitHub, so the repo remote needs to use SSH (`git remote set-url origin git@github.com:benbav/tf2_balloon_race_forecast.git`) with a key added to your GitHub account.

**5. Turn on the website**: on GitHub, go to **Settings → Pages → Build and deployment → Deploy from a branch**, pick the `site` branch and `/ (root)` (the branch appears after the first `generate_report.py` run), and save. Free Pages needs a public repo. The site is live at `https://<user>.github.io/tf2_balloon_race_forecast/` a minute later.

### Settings

- `balloon/config.py`: `LOBBY_PLAYERS` (how many players make a lobby: Discord pings, the website's alarm, the forecast, the charts and which servers the rotation tracker watches; the website picks it up on the next daily report)
- `balloon/servers.py`: `DEDICATED_NAME_RE` / `DEDICATED_ADDRS` (which servers count as dedicated 24/7)
- `balloon/log.py`: `LOG_TIMEZONE` (clock the log was written in)
- `balloon/forecast.py`: `FORECAST_MINUTES` and the model's other settings
- `rotation_tracker.py`: `CHECK_SECONDS`
- `balloon/voters.py`: `MIN_BALLOON_CHANGES`, `MIN_OTHER_CHANGES`, `VOTER_MIN_SCORE`, `LEADERBOARD_SIZE`
- `generate_report.py`: `ZONES` (chart timezones; the first is the default), `MIN_DAYS_FOR_SCORECARD`, `SCORECARD_RECENT_DAYS`, `TOP_SERVER_COUNT`
