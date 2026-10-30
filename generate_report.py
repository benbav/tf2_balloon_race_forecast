from plotly.subplots import make_subplots
import datetime as dt
from zoneinfo import ZoneInfo
import holidays
import requests
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from balloon.config import LOBBY_PLAYERS
from balloon.forecast import train_lobby_forecast
from balloon.log import LOG_TIMEZONE, parse_log
from balloon.rotation import rotation_summary
from balloon.scores import score_summary
from balloon.voters import voter_summary
from balloon.servers import is_dedicated
import subprocess
import json
import io
import re
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Runtime files: the log, forecast model and caches
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
# Everything GitHub Pages serves; published as the root of the site branch
SITE_DIR = os.path.join(SCRIPT_DIR, 'site')
OUTPUT_FOLDER = os.path.join(SITE_DIR, 'charts')
LOG_PATH = os.path.join(DATA_DIR, 'tf2_balloon_log.txt')
README_PATH = os.path.join(SCRIPT_DIR, 'README.md')
# Lobby forecast model, trained here daily and used by main.py on every check
FORECAST_MODEL_PATH = os.path.join(DATA_DIR, 'forecast_model.joblib')
# One line per day: the forecast's rolling test scores, so the website can chart them over time
FORECAST_HISTORY_PATH = os.path.join(DATA_DIR, 'forecast_history.jsonl')
# BLS v1 has a tight daily request limit, so the series is cached and refetched at most once a day
BLS_URL = 'https://api.bls.gov/publicAPI/v1/timeseries/data/LNS14000000'
BLS_CACHE_PATH = os.path.join(DATA_DIR, 'bls_unemployment.json')
# Every earthquake USGS knows of worldwide, counted per hour and cached so only recent hours are refetched
USGS_URL = 'https://earthquake.usgs.gov/fdsnws/event/1/query'
QUAKE_CACHE_PATH = os.path.join(DATA_DIR, 'earthquakes_hourly.json')
# Late reports keep trickling in, so this much of the end of the cache is refetched each run
QUAKE_REFETCH = pd.Timedelta(days=2)

# Charts are shown in these zones (the log itself is in LOG_TIMEZONE).
# The first zone is the default (README, all charts); the rest only redraw the charts that depend on the hour.
ZONES = [
    # (timezone, label, file suffix)
    ('America/Denver', 'MT', ''),
    ('America/New_York', 'ET', '_et'),
    ('America/Chicago', 'CT', '_ct'),
    ('America/Los_Angeles', 'PT', '_pt'),
]
DISPLAY_TIMEZONE, TZ_LABEL, ZONE_SUFFIX = ZONES[0]

# Scorecard needs at least this many days of checks before it names a winner
MIN_DAYS_FOR_SCORECARD = 7
# Scorecard looks at this many recent days so it follows changing habits
SCORECARD_RECENT_DAYS = 28
TOP_SERVER_COUNT = 5
# The website's server heatmap has a dropdown with this many servers
HEATMAP_SERVER_COUNT = 10
# Windows the website's "next likely lobby" countdown picks from
LOBBY_WINDOW_COUNT = 5
# The website's server table has room for more
SITE_SERVER_COUNT = 10
CHECK_MINUTES = 5
# A day needs at least this many checks to count in the holiday comparison
MIN_CHECKS_PER_DAY = 144

DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Colors (validated dataviz reference palette)
SURFACE = '#fcfcfb'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#e6e5e0'
BAR = '#2a78d6'
SERIES = ['#2a78d6', '#eb6834']
SEQUENTIAL = ['#f3f2ef', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
COLORSCALE = [[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)]
FONT = 'Inter, "Helvetica Neue", Helvetica, Arial, sans-serif'

# Every chart is drawn for all servers, then again for each server type: (file suffix, label, color)
ALL_SERVERS = ('', None, None)
DEDICATED = ('_dedicated', 'Dedicated 24/7 servers', SERIES[0])
ROTATING = ('_rotating', 'Rotating servers', SERIES[1])

README_START = '<!-- TOP_SERVERS_START -->'
README_END = '<!-- TOP_SERVERS_END -->'


def hour_label(h):
    return f"{h % 12 or 12} {'AM' if h % 24 < 12 else 'PM'}"


def window_label(hour_of_week, short=False):
    day = DAYS[hour_of_week // 24]
    start = hour_of_week % 24
    return f"{day[:3] if short else day} · {hour_label(start)} – {hour_label(start + 2)}"


def add_time_columns(df):
    df['ts'] = pd.to_datetime(df['ts']).dt.tz_localize(LOG_TIMEZONE)
    return in_zone(df)


# Recompute the clock columns in the current DISPLAY_TIMEZONE
def in_zone(df):
    df = df.copy()
    df['ts'] = df['ts'].dt.tz_convert(DISPLAY_TIMEZONE)
    df['day_of_week'] = pd.Categorical(df['ts'].dt.day_name(), categories=DAYS, ordered=True)
    df['hour'] = df['ts'].dt.hour
    df['hour_of_week'] = df['ts'].dt.dayofweek * 24 + df['hour']
    df['month'] = df['ts'].dt.strftime('%Y-%m')
    return df


# Load and preprocess data: one row per 5-minute check, plus one row per active server per check
def load_and_preprocess_data(file_path):
    with open(file_path, encoding='utf-8', errors='replace') as f:
        checks, servers = parse_log(f)
    df = add_time_columns(checks)
    server_df = add_time_columns(servers)
    server_df['dedicated'] = [is_dedicated(n, a) for n, a in zip(server_df['name'], server_df['addr'])]
    return df, server_df


# The checks again, but only counting one type of server: each check keeps its busiest server of that type
def split_checks(df, server_df, dedicated):
    top = server_df[server_df['dedicated'] == dedicated].sort_values('players').groupby('ts').last()
    # Older checks were logged before per-server lines existed, so only the busiest server is known
    legacy = ~df['ts'].isin(server_df['ts']) & (df['online_players'] > 0)
    legacy &= df['server_name'].fillna('').map(is_dedicated) == dedicated
    out = df.copy()
    out['online_players'] = df['ts'].map(top['players']).where(~legacy, df['online_players']).fillna(0).astype(int)
    out['server_name'] = df['ts'].map(top['name']).where(~legacy, df['server_name'])
    return out


def base_layout(title, subtitle=None, variant=ALL_SERVERS, **kwargs):
    text = f"<b>{title}</b>"
    if variant[1]:
        text += f" <span style='color:{variant[2]}'><b>· {variant[1]}</b></span>"
    if subtitle:
        text += f"<br><span style='font-size:14px;color:{TEXT_SECONDARY}'>{subtitle}</span>"
    layout = dict(
        title=dict(text=text, x=0.02, xanchor='left', y=0.95, font=dict(size=22, color=TEXT_PRIMARY)),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=13, color=TEXT_SECONDARY),
        margin=dict(l=24, r=32, t=96, b=48),
        width=1000,
        height=600,
        showlegend=False,
    )
    layout.update(kwargs)
    return layout


def empty_state(fig, message='Not enough data yet'):
    fig.add_annotation(x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False,
                       text=message, font=dict(size=20, color=TEXT_SECONDARY))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)


def save(fig, name, variant=ALL_SERVERS):
    fig.write_image(os.path.join(OUTPUT_FOLDER, name.replace('.png', f'{variant[0]}{ZONE_SUFFIX}.png')), scale=2)


def days_of_data(df):
    return (df['ts'].max() - df['ts'].min()).days if len(df) else 0


# % of checks with someone on, for every 2-hour window of the week (wrapping Sunday night into Monday)
def window_rates(df, active):
    rate = (
        df.assign(active=active)
        .groupby('hour_of_week')['active']
        .agg(['sum', 'count'])
        .reindex(range(168), fill_value=0)
    )
    nxt = rate.shift(-1)
    nxt.iloc[-1] = rate.iloc[0]
    window = rate + nxt
    window['pct'] = (window['sum'] / window['count'].where(window['count'] > 0)).fillna(0)
    return window


# Best non-overlapping 2-hour windows, best first
def top_windows(window, n):
    picked = []
    for start in window['pct'].sort_values(ascending=False).index:
        if len(picked) == n or window.loc[start, 'pct'] == 0:
            break
        if all(min((start - p) % 168, (p - start) % 168) >= 2 for p in picked):
            picked.append(int(start))
    return picked


# The last SCORECARD_RECENT_DAYS of checks, so the best times follow changing habits
def recent_checks(df):
    return df[df['ts'] >= df['ts'].max() - pd.Timedelta(days=SCORECARD_RECENT_DAYS)] if len(df) else df


# The website's "next likely lobby" countdown: the best 2-hour windows for a LOBBY_PLAYERS+ lobby,
# as hours of the week in the current DISPLAY_TIMEZONE
def lobby_windows(df):
    if days_of_data(df) < MIN_DAYS_FOR_SCORECARD:
        return []
    recent = recent_checks(df)
    window = window_rates(recent, recent['online_players'] >= LOBBY_PLAYERS)
    return [{'start': p, 'pct': round(float(window.loc[p, 'pct']), 3)} for p in top_windows(window, LOBBY_WINDOW_COUNT)]


# The day's rolling test scores from the forecast info, or None if it couldn't be tested
def forecast_history_row(forecast, date):
    e = forecast.get('evaluation')
    if not e:
        return None
    a, o = e['all'], e['no_lobby_yet']
    return {'date': date.isoformat(), 'test_days': len(e['test_days']), 'days': forecast['days'],
            'auc': a.get('auc'), 'log_loss': a['log_loss'], 'baseline_log_loss': a['baseline_log_loss'],
            'no_lobby_yet_auc': o.get('auc'), 'no_lobby_yet_log_loss': o.get('log_loss'),
            'no_lobby_yet_baseline_log_loss': o.get('baseline_log_loss')}


def read_forecast_history():
    if not os.path.exists(FORECAST_HISTORY_PATH):
        return []
    with open(FORECAST_HISTORY_PATH, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


# Adds (or replaces) the row for its date, keeping the file in date order
def save_forecast_history(row):
    rows = {r['date']: r for r in read_forecast_history()}
    rows[row['date']] = row
    with open(FORECAST_HISTORY_PATH, 'w', encoding='utf-8') as f:
        for date in sorted(rows):
            f.write(json.dumps(rows[date]) + '\n')


# AUC and log loss of the forecast's rolling test, as scored each day, to see whether it improves with more data
def generate_forecast_history(history):
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1, subplot_titles=(
        '<b>AUC</b> · higher is better, 0.5 is a coin flip', '<b>Log loss</b> · lower is better'))
    fig.update_layout(base_layout(
        'Is the forecast getting better?',
        'Each day, the last 14 days are each predicted by a model trained only on the days before',
        height=460, showlegend=True,
        legend=dict(orientation='h', x=0, y=-0.18, font=dict(size=13, color=TEXT_PRIMARY)),
    ))
    fig.update_annotations(font=dict(size=14, color=TEXT_PRIMARY))
    if not history:
        empty_state(fig, 'No scores yet')
        save(fig, 'forecast_history.png')
        return
    dates = [r['date'] for r in history]
    line = lambda key, name, color, dash, legend=True: go.Scatter(
        x=dates, y=[r.get(key) for r in history], name=name, mode='lines+markers', showlegend=legend,
        line=dict(color=color, width=3, dash=dash), marker=dict(size=6), hoverinfo='skip')
    fig.add_trace(line('auc', 'All checks', SERIES[0], 'solid'), row=1, col=1)
    fig.add_trace(line('no_lobby_yet_auc', 'Checks with no lobby up yet', SERIES[1], 'solid'), row=1, col=1)
    fig.add_trace(line('log_loss', 'All checks', SERIES[0], 'solid', False), row=1, col=2)
    fig.add_trace(line('no_lobby_yet_log_loss', 'No lobby yet', SERIES[1], 'solid', False), row=1, col=2)
    fig.add_trace(line('baseline_log_loss', 'Always guessing the average', TEXT_SECONDARY, 'dot'), row=1, col=2)
    fig.add_hline(y=0.5, line=dict(color=GRID, width=2), row=1, col=1)
    fig.update_xaxes(showgrid=False, tickformat='%b %d', ticks='')
    fig.update_yaxes(gridcolor=GRID, zeroline=False, ticks='')
    fig.update_yaxes(range=[0.4, 1], row=1, col=1)
    save(fig, 'forecast_history.png')


# The forecast's logistic regression, drawn like the classic best-fit-line chart: every training check is a
# dot at 0% or 100% (did a lobby form in the next hour?) placed by the model's score, with the fitted S-curve
# through them and the actual share of lobbies in score bins to show how well the curve fits
def generate_forecast_regression(fit, threshold):
    fig = go.Figure()
    fig.update_layout(base_layout('How the PhD candidate forecast is fit', 'Logistic regression on every 5-minute check',
                                  width=640, height=480, margin=dict(l=24, r=24, t=88, b=48)))
    if fit is None:
        empty_state(fig)
        save(fig, 'forecast_regression.png')
        return
    score, y = fit['score'], fit['y']
    rng = np.random.default_rng(0)
    fig.add_trace(go.Scatter(
        x=score, y=y * 100 + rng.uniform(-3, 3, len(y)), mode='markers',
        marker=dict(size=5, color=TEXT_SECONDARY, opacity=0.25), hoverinfo='skip',
    ))
    bins = pd.qcut(score, 12, duplicates='drop')
    binned = pd.DataFrame({'score': score, 'y': y}).groupby(bins, observed=True).mean()
    fig.add_trace(go.Scatter(
        x=binned['score'], y=binned['y'] * 100, mode='markers',
        marker=dict(size=10, color=BAR, line=dict(color=SURFACE, width=2)),
    ))
    xs = np.linspace(score.min(), score.max(), 200)
    fig.add_trace(go.Scatter(x=xs, y=100 / (1 + np.exp(-xs)), mode='lines', line=dict(color=BAR, width=3)))
    # Label the line where it crosses 50%, or at its top end if it never gets there
    label_x = min(0.0, xs[-1])
    fig.add_annotation(x=label_x, y=100 / (1 + np.exp(-label_x)), text='Best fit line', ax=-70, ay=-40,
                       arrowcolor=TEXT_SECONDARY, font=dict(size=15, color=TEXT_PRIMARY))
    fig.add_annotation(x=0.01, y=0.86, xref='paper', yref='paper', xanchor='left', showarrow=False, align='left',
                       text=f"<span style='color:{BAR}'>●</span> actual share of lobbies<br>"
                            f"<span style='color:{TEXT_SECONDARY}'>●</span> one check",
                       font=dict(size=14, color=TEXT_SECONDARY))
    fig.update_xaxes(title=f'Model score (log-odds of a {threshold}+ lobby)', gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title='Lobby in the next hour', range=[-8, 108], tickvals=[0, 25, 50, 75, 100],
                     ticktext=['0%', '25%', '50%', '75%', '100%'], gridcolor=GRID, zeroline=False)
    fig.update_layout(font_size=15)
    save(fig, 'forecast_regression.png')


# Ranked bars: the 5 best non-overlapping 2-hour windows
def generate_top_windows_plot(df, variant=ALL_SERVERS):
    window = window_rates(df, df['online_players'] >= LOBBY_PLAYERS)
    picked = top_windows(window, 5) if days_of_data(df) >= MIN_DAYS_FOR_SCORECARD else []
    picked = picked[::-1]
    pcts = [window.loc[p, 'pct'] * 100 for p in picked]

    fig = go.Figure(go.Bar(
        x=pcts, y=[f"{window_label(p, short=True)} {TZ_LABEL}" for p in picked], orientation='h',
        marker_color=BAR, text=[f"{v:.0f}%" for v in pcts], textposition='outside', cliponaxis=False,
        textfont=dict(color=TEXT_PRIMARY, size=15),
    ))
    fig.update_layout(base_layout(
        'Top 5 times to play',
        f'Best 2-hour windows (no overlaps) · % of checks with a server at {LOBBY_PLAYERS}+ players',
        height=480, barcornerradius=4, bargap=0.35,
        variant=variant,
    ))
    if not picked:
        empty_state(fig)
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, ticksuffix='%', rangemode='tozero')
    fig.update_yaxes(showgrid=False, ticks='', tickfont=dict(size=15, color=TEXT_PRIMARY))
    save(fig, 'top_windows.png', variant)


# Day x hour heatmap of how often a condition held
def generate_heatmap(df, active, title, subtitle, filename, variant=ALL_SERVERS):
    heatmap_data = (
        df.assign(active=active * 100)
        .pivot_table(index='hour', columns='day_of_week', values='active',
                     aggfunc='mean', observed=False)
        .reindex(index=range(24), columns=DAYS)
        .fillna(0)
    )
    z = heatmap_data.values
    zmax = max(float(z.max()), 1)

    # Hours that round to 0% are left blank, and there are no gaps between cells, so only the busy hours show
    fig = go.Figure(go.Heatmap(
        z=np.where(z.round() > 0, z, np.nan),
        x=DAYS,
        y=[hour_label(h) for h in range(24)],
        colorscale=COLORSCALE,
        zmin=0, zmax=zmax,
        hoverinfo='skip',
        colorbar=dict(title=dict(text='% of checks', side='top'), ticksuffix='%',
                      thickness=12, outlinewidth=0, len=0.8),
    ))
    # Label non-zero cells, with ink that stays readable on light and dark steps
    for yi, h in enumerate(range(24)):
        for xi, d in enumerate(DAYS):
            v = z[yi, xi]
            if round(v) > 0:
                fig.add_annotation(x=d, y=hour_label(h), text=f"{v:.0f}%", showarrow=False,
                                   font=dict(size=11, color='#ffffff' if v / zmax > 0.45 else TEXT_PRIMARY))

    # Close to square so it fills the card under the daily forecast at the top of the website
    fig.update_layout(base_layout(title, subtitle, variant=variant, height=860))
    fig.update_yaxes(showgrid=False, ticks='')
    fig.update_xaxes(side='top', showgrid=False, ticks='')
    # With every cell blank the chart would be empty, so say why
    if not (z.round() > 0).any():
        fig.update_traces(showscale=False)
        empty_state(fig, f'No {LOBBY_PLAYERS}+ player lobbies seen yet')
    # Room on the right for the color bar's title
    fig.update_layout(margin=dict(l=24, r=72, t=130, b=24))
    save(fig, filename, variant)


# How long a lobby keeps going once it fills up
def generate_session_length_plot(df, variant=ALL_SERVERS):
    active = (df.sort_values('ts')['online_players'] >= LOBBY_PLAYERS).values
    ts = df.sort_values('ts')['ts'].values
    sessions, run = [], 0
    for i, a in enumerate(active):
        contiguous = i > 0 and (ts[i] - ts[i - 1]) <= pd.Timedelta(minutes=CHECK_MINUTES + 2)
        if a and run and contiguous:
            run += 1
        else:
            if run:
                sessions.append(run * CHECK_MINUTES)
            run = 1 if a else 0
    if run:
        sessions.append(run * CHECK_MINUTES)

    buckets = [('Under 15 min', 0, 15), ('15–30 min', 15, 30), ('30–60 min', 30, 60),
               ('1–2 hours', 60, 120), ('2+ hours', 120, float('inf'))]
    counts = [sum(lo <= s < hi for s in sessions) for _, lo, hi in buckets]
    median = pd.Series(sessions).median() if sessions else None

    fig = go.Figure(go.Bar(
        x=[b[0] for b in buckets], y=counts, marker_color=BAR,
        text=counts, textposition='outside', cliponaxis=False, textfont=dict(color=TEXT_PRIMARY),
    ))
    subtitle = f'A lobby = back-to-back 5-minute checks with a server at {LOBBY_PLAYERS}+ players'
    if median is not None:
        subtitle += f' · median lobby: {median:.0f} min'
    fig.update_layout(base_layout(f'How long {LOBBY_PLAYERS}+ player lobbies last', subtitle,
                                  barcornerradius=4, bargap=0.4, variant=variant))
    if not sessions:
        empty_state(fig, f'No {LOBBY_PLAYERS}+ player lobbies seen yet')
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, title='Lobbies', rangemode='tozero')
    fig.update_xaxes(showgrid=False, ticks='')
    save(fig, 'session_length.png', variant)


# % of checks each week (starting Monday) with a LOBBY_PLAYERS+ server
def weekly_lobby_share(df):
    week = df['ts'].dt.tz_localize(None).dt.to_period('W-SUN').dt.start_time
    grouped = (df['online_players'] >= LOBBY_PLAYERS).groupby(week)
    # Skip weeks with under 3 days of checks (e.g. the week logging started) so they don't skew the line
    min_checks = 3 * 24 * 60 // CHECK_MINUTES
    keep = grouped.size() >= min_checks
    if len(keep):
        keep.iloc[-1] = True  # always show the current week
    return (grouped.mean() * 100)[keep]


# Generate bar plot for monthly max
def generate_monthly_avg_plot(df, variant=ALL_SERVERS):
    monthly = df.groupby('month')['online_players'].max().sort_index()
    labels = [pd.Timestamp(m + '-01').strftime('%b %Y') for m in monthly.index]

    fig = go.Figure(go.Bar(
        x=labels, y=monthly.values, marker_color=BAR,
        text=monthly.values, textposition='outside', cliponaxis=False,
        textfont=dict(color=TEXT_PRIMARY),
    ))
    fig.update_layout(base_layout(
        'Max Balloon Race players by month',
        'Most players seen at once on any server',
        barcornerradius=4, bargap=0.45,
        variant=variant,
    ))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, title='Players', rangemode='tozero')
    fig.update_xaxes(showgrid=False, ticks='')
    save(fig, 'monthly_avg.png', variant)


# Monthly US unemployment rate (seasonally adjusted) from BLS, as a Series indexed by month start
def load_unemployment():
    cached = None
    if os.path.exists(BLS_CACHE_PATH):
        with open(BLS_CACHE_PATH, encoding='utf-8') as f:
            cached = json.load(f)
    fresh = cached and dt.datetime.now().timestamp() - os.path.getmtime(BLS_CACHE_PATH) < 86400
    if not fresh:
        try:
            r = requests.get(BLS_URL, timeout=30)
            r.raise_for_status()
            body = r.json()
            if body.get('status') != 'REQUEST_SUCCEEDED':
                raise ValueError(body.get('message'))
            cached = body['Results']['series'][0]['data']
            with open(BLS_CACHE_PATH, 'w', encoding='utf-8') as f:
                json.dump(cached, f)
        except Exception as e:
            print(f'Error fetching BLS unemployment data: {e}')
    rows = [(pd.Timestamp(f"{d['year']}-{d['period'][1:]}-01"), float(d['value']))
            for d in cached or []
            if d['period'].startswith('M') and d['period'] != 'M13' and d['value'] != '-']  # '-' = not published
    return pd.Series(dict(rows), dtype=float).sort_index()


# Earthquakes worldwide per UTC hour from `start` up to the last full hour, as a Series indexed by hour
def load_earthquakes(start):
    counts = {}
    if os.path.exists(QUAKE_CACHE_PATH):
        with open(QUAKE_CACHE_PATH, encoding='utf-8') as f:
            counts = json.load(f)
    start = start.tz_convert('UTC').floor('h')
    end = pd.Timestamp.now(tz='UTC').floor('h')
    fetch_from = max(start, pd.Timestamp(max(counts)) - QUAKE_REFETCH) if counts else start
    try:
        # In two-week chunks to stay well under USGS's 20,000 events per request
        while fetch_from < end:
            chunk_end = min(fetch_from + pd.Timedelta(days=14), end)
            r = requests.get(USGS_URL, timeout=60, params={
                'format': 'csv', 'eventtype': 'earthquake', 'orderby': 'time-asc',
                'starttime': fetch_from.strftime('%Y-%m-%dT%H:%M:%S'),
                'endtime': chunk_end.strftime('%Y-%m-%dT%H:%M:%S'),
            })
            r.raise_for_status()
            times = pd.read_csv(io.StringIO(r.text), usecols=['time'])['time'] if r.text.strip() else pd.Series()
            per_hour = pd.to_datetime(times, utc=True).dt.floor('h').value_counts()
            # Hours with no earthquakes count as 0
            for hour in pd.date_range(fetch_from, chunk_end, freq='h', inclusive='left'):
                counts[hour.isoformat()] = int(per_hour.get(hour, 0))
            fetch_from = chunk_end
    except Exception as e:
        print(f'Error fetching USGS earthquake data: {e}')
    with open(QUAKE_CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(counts, f)
    quakes = pd.Series(list(counts.values()), index=pd.to_datetime(list(counts), utc=True), dtype=float).sort_index()
    return quakes[quakes.index >= start]


# Balloon Race players against something that has nothing to do with them,
# with a dotted least-squares line and Pearson's r to prove it
def generate_correlation_plot(x, y, title, subtitle, x_title, y_title, filename, variant=ALL_SERVERS, x_suffix=''):
    pairs = pd.DataFrame({'x': x, 'y': y}).dropna()
    fits = len(pairs) >= 3 and pairs['x'].nunique() > 1 and pairs['y'].nunique() > 1
    r = np.corrcoef(pairs['x'], pairs['y'])[0, 1] if fits else None

    fig = go.Figure(go.Scatter(
        x=pairs['x'], y=pairs['y'], mode='markers',
        marker=dict(color=BAR, size=9, opacity=0.45, line=dict(width=0)),
    ))
    if fits:
        slope, intercept = np.polyfit(pairs['x'], pairs['y'], 1)
        ends = np.array([pairs['x'].min(), pairs['x'].max()])
        fig.add_trace(go.Scatter(x=ends, y=slope * ends + intercept, mode='lines',
                                 line=dict(color=SERIES[1], width=3, dash='dot')))
        fig.add_annotation(x=ends[1], y=slope * ends[1] + intercept, text=f'<b>r = {r:+.2f}</b>',
                           showarrow=False, xanchor='left', xshift=10, font=dict(color=TEXT_PRIMARY, size=15))
    fig.update_layout(base_layout(
        f"{title} (r = {f'{r:+.2f}' if fits else 'undefined'})",
        f"{subtitle} · {len(pairs):,} points" + ('' if fits else f' · {x_title.lower()} hasn\'t moved yet'),
        variant=variant, margin=dict(l=24, r=90, t=96, b=48),
    ))
    if not len(pairs):
        empty_state(fig)
    fig.update_xaxes(title=x_title, showgrid=False, zeroline=False, ticksuffix=x_suffix)
    fig.update_yaxes(title=y_title, showgrid=True, gridcolor=GRID, zeroline=False, rangemode='tozero')
    save(fig, filename, variant)


# The weekly trend line with earthquakes worldwide that week overlaid on their own axis, and Pearson's r
# between the two. Earthquakes per day, so the partial latest week lines up with the full ones.
def generate_earthquake_correlation_plot(df, quakes, variant=ALL_SERVERS):
    weekly = weekly_lobby_share(df)
    week = quakes.index.tz_localize(None).to_period('W-SUN').start_time
    quakes_per_day = (quakes.groupby(week).mean() * 24).reindex(weekly.index)
    pairs = pd.DataFrame({'lobby': weekly, 'quakes': quakes_per_day}).dropna()
    fits = len(pairs) >= 3 and pairs['lobby'].nunique() > 1 and pairs['quakes'].nunique() > 1
    r = np.corrcoef(pairs['lobby'], pairs['quakes'])[0, 1] if fits else None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weekly.index, y=weekly.values, mode='lines+markers', name=f'% of checks with a {LOBBY_PLAYERS}+ lobby',
        line=dict(color=BAR, width=3), marker=dict(size=8, color=BAR, line=dict(color=SURFACE, width=2)),
    ))
    fig.add_trace(go.Scatter(
        x=quakes_per_day.index, y=quakes_per_day.values, mode='lines+markers', name='Earthquakes per day',
        yaxis='y2', line=dict(color=SERIES[1], width=3, dash='dot'),
        marker=dict(size=8, color=SERIES[1], line=dict(color=SURFACE, width=2)),
    ))
    fig.update_layout(base_layout(
        f"Earthquakes vs Balloon Race players (r = {f'{r:+.2f}' if fits else 'undefined'})",
        f'Each week: % of checks with a server at {LOBBY_PLAYERS}+ players vs earthquakes per day worldwide (USGS)'
        + ('' if fits else ' · needs 3 weeks to correlate'),
        variant=variant, showlegend=True, margin=dict(l=24, r=80, t=96, b=80),
        legend=dict(orientation='h', x=0.02, y=-0.1, yanchor='top', bgcolor='rgba(0,0,0,0)'),
        yaxis2=dict(overlaying='y', side='right', showgrid=False, zeroline=False, rangemode='tozero', tickformat=',.0f',
                    title=dict(text='Earthquakes per day', font=dict(color=SERIES[1])), tickfont=dict(color=SERIES[1])),
    ))
    if not len(weekly):
        empty_state(fig)
    fig.update_layout(yaxis=dict(title=dict(text=f'% of checks with {LOBBY_PLAYERS}+', font=dict(color=BAR)),
                                 tickfont=dict(color=BAR), showgrid=True, gridcolor=GRID, zeroline=False,
                                 ticksuffix='%', rangemode='tozero'))
    fig.update_xaxes(showgrid=False, ticks='', tickformat='%b %d')
    if len(weekly):
        fig.update_xaxes(range=[weekly.index[0] - pd.Timedelta(days=3), weekly.index[-1] + pd.Timedelta(days=3)])
    save(fig, 'earthquake_correlation.png', variant)


def generate_unemployment_correlation_plot(df, unemployment, variant=ALL_SERVERS):
    daily_peak = df.groupby(df['ts'].dt.date)['online_players'].max()
    daily_peak.index = pd.to_datetime(daily_peak.index)
    # Each day gets the latest rate BLS had published for it, since the current month comes out a month late
    rate = unemployment.reindex(daily_peak.index, method='ffill') if len(unemployment) else daily_peak * np.nan
    generate_correlation_plot(
        rate, daily_peak, 'Unemployment vs Balloon Race players',
        'Each dot is one day: US unemployment rate (BLS) vs most Balloon Race players seen at once',
        'Unemployment rate', 'Peak players that day', 'unemployment_correlation.png', variant, x_suffix='%',
    )


# US holidays plus a few unofficial gaming days and school-break periods
def holiday_calendar(years):
    rename = {"Washington's Birthday": "Presidents' Day",
              'Juneteenth National Independence Day': 'Juneteenth'}
    days = {}
    for d, name in holidays.US(years=years).items():
        name = name.replace(' (observed)', '')
        days.setdefault(rename.get(name, name), []).append(d)
    for y in years:
        days.setdefault('Halloween', []).append(dt.date(y, 10, 31))
        days.setdefault('Christmas Eve', []).append(dt.date(y, 12, 24))
        days.setdefault("New Year's Eve", []).append(dt.date(y, 12, 31))
        thanksgiving = next(d for d, n in holidays.US(years=y).items() if n == 'Thanksgiving Day')
        days.setdefault('Thanksgiving weekend (Thu–Sun)', []).extend(
            thanksgiving + dt.timedelta(days=i) for i in range(4))
        days.setdefault('Winter break (Dec 20 – Jan 3)', []).extend(
            dt.date(y, 12, 20) + dt.timedelta(days=i) for i in range(15))
    return days


# Table: how much busier each holiday is than a normal day of the same weekday
def generate_holiday_table(df, variant=ALL_SERVERS):
    daily = (
        df.assign(active=df['online_players'] >= LOBBY_PLAYERS, date=df['ts'].dt.date)
        .groupby('date')
        .agg(rate=('active', 'mean'), checks=('active', 'size'), peak=('online_players', 'max'))
    )
    daily = daily[daily['checks'] >= MIN_CHECKS_PER_DAY]
    years = sorted({d.year for d in daily.index} | {dt.date.today().year}) if len(daily) else [dt.date.today().year]
    calendar = holiday_calendar(range(years[0] - 1, years[-1] + 2))
    holiday_dates = {d for dates in calendar.values() for d in dates}

    # Baseline: average day of the same weekday that isn't a holiday
    normal = daily[[d not in holiday_dates for d in daily.index]]
    baseline = (normal.groupby([d.weekday() for d in normal.index])['rate'].mean()
                if len(normal) else pd.Series(dtype=float))

    rows = []
    for name, dates in calendar.items():
        seen = [d for d in dates if d in daily.index]
        if not seen or baseline.empty:
            continue
        rate = daily.loc[seen, 'rate'].mean()
        base = pd.Series([baseline.get(d.weekday()) for d in seen]).mean()
        if pd.isna(base):
            continue
        rows.append({
            'name': name,
            'last': max(seen),
            'days': len(seen),
            'rate': rate,
            'base': base,
            'lift': rate / base if base > 0 else float('inf') if rate > 0 else 1.0,
            'peak': int(daily.loc[seen, 'peak'].max()),
        })
    rows.sort(key=lambda r: r['lift'], reverse=True)

    today = dt.date.today()
    upcoming = sorted((min(d for d in dates if d >= today), name)
                      for name, dates in calendar.items() if any(d >= today for d in dates))[:3]
    next_up = ' · '.join(f"{name} ({d.strftime('%b %d')})" for d, name in upcoming)

    fig = go.Figure()
    if rows:
        def lift_text(r):
            if r['lift'] == float('inf'):
                return '<b>New activity</b>'
            lift = round(r['lift'], 1)
            arrow = '▲' if lift >= 1.1 else '▼' if lift <= 0.9 else '–'
            return f"<b>{r['lift']:.1f}×</b> {arrow}"

        def lift_fill(r):
            if r['lift'] >= 2:
                return SEQUENTIAL[2]
            if r['lift'] >= 1.25:
                return SEQUENTIAL[1]
            return SURFACE

        fig.add_trace(go.Table(
            columnwidth=[3.2, 1.6, 1.4, 1.4, 1.3, 0.9],
            header=dict(
                values=['<b>Holiday</b>', '<b>Last seen</b>', f'<b>{LOBBY_PLAYERS}+ lobby</b>', '<b>Normal day</b>',
                        '<b>vs normal</b>', '<b>Peak</b>'],
                fill_color=SURFACE, line_color=GRID, align=['left'] + ['right'] * 5, height=40,
                font=dict(size=14, color=TEXT_SECONDARY, family=FONT),
            ),
            cells=dict(
                values=[
                    [f"<b>{r['name']}</b>" for r in rows],
                    [r['last'].strftime('%b %d, %Y') for r in rows],
                    [f"{r['rate']:.1%}" for r in rows],
                    [f"{r['base']:.1%}" for r in rows],
                    [lift_text(r) for r in rows],
                    [r['peak'] for r in rows],
                ],
                fill_color=[[SURFACE] * len(rows)] * 4 + [[lift_fill(r) for r in rows], [SURFACE] * len(rows)],
                line_color=GRID, align=['left'] + ['right'] * 5, height=38,
                font=dict(size=14, color=TEXT_PRIMARY, family=FONT),
            ),
        ))
    fig.update_layout(base_layout(
        'Holidays vs normal days',
        f'"{LOBBY_PLAYERS}+ lobby" = % of checks with a server at {LOBBY_PLAYERS}+ players'
        ' · compared with non-holiday days of the same weekday'
        + f'<br>Next up: {next_up}',
        height=max(600, 200 + 38 * len(rows)),  # at least as tall as the chart beside it on the site
        margin=dict(l=24, r=24, t=130, b=16),
        variant=variant,
    ))
    fig.update_layout(title=dict(y=1, yanchor='top', pad=dict(t=24)))
    if not rows:
        empty_state(fig, 'No holidays in the data yet')
    save(fig, 'holidays.png', variant)


# Rank servers by how often they have a LOBBY_PLAYERS+ lobby (then by how often anyone's on), with each one's
# best times for a lobby. A server that's never had one has no best time or day.
def top_server_stats(df, server_df):
    stats = []
    for addr, rows in server_df.groupby('addr'):
        # Only count checks since the server was first seen, so new servers aren't penalized
        checks = df[df['ts'] >= rows['ts'].min()]
        active_ts = set(rows.loc[rows['players'] > 0, 'ts'])
        busy_ts = set(rows.loc[rows['players'] >= LOBBY_PLAYERS, 'ts'])
        active = checks['ts'].isin(active_ts)
        busy = checks['ts'].isin(busy_ts)
        window = window_rates(checks, busy)
        best = int(window['pct'].idxmax()) if busy.any() else None
        by_day = busy.groupby(checks['day_of_week'], observed=False).mean()
        stats.append({
            'addr': addr,
            'name': rows.sort_values('ts')['name'].iloc[-1],
            'pct_active': active.mean(),
            'pct_busy': busy.mean(),
            'best_window': best,
            'best_window_pct': window.loc[best, 'pct'] if busy.any() else 0.0,
            'best_day': by_day.idxmax() if busy.any() else None,
            'avg_players': rows.loc[rows['players'] > 0, 'players'].mean(),
            'peak': rows['players'].max(),
            'checks': checks,
            'active': active,
            'busy': busy,
        })
    stats.sort(key=lambda s: (s['pct_busy'], s['pct_active']), reverse=True)
    return stats


# One day x hour heatmap per top server, on a shared scale, picked from a dropdown on the website.
# Returns the servers drawn, in order, so the dropdown can name them
def generate_server_heatmaps(stats, variant=ALL_SERVERS):
    # Pick the servers that most often have a busy lobby, not just someone on
    stats = sorted((s for s in stats if s['busy'].any()), key=lambda s: s['busy'].mean(), reverse=True)
    stats = stats[:HEATMAP_SERVER_COUNT]
    if not stats:
        fig = go.Figure()
        fig.update_layout(base_layout('When the top servers are busy', height=300, variant=variant))
        empty_state(fig, 'No busy lobbies seen yet')
        save(fig, 'server_heatmap_1.png', variant)
        return []

    grids = []
    for s in stats:
        grids.append(
            (s['busy'] * 100).groupby([s['checks']['day_of_week'], s['checks']['hour']], observed=False)
            .mean().unstack().reindex(index=DAYS, columns=range(24)).fillna(0)
        )
    zmax = max(max(float(g.values.max()) for g in grids), 1)

    for i, (s, g) in enumerate(zip(stats, grids)):
        fig = go.Figure(go.Heatmap(
            z=g.values, x=[hour_label(h) for h in range(24)], y=[d[:3] for d in DAYS],
            colorscale=COLORSCALE, zmin=0, zmax=zmax, xgap=2, ygap=2, hoverinfo='skip',
            colorbar=dict(title=dict(text='% of checks', side='top'), ticksuffix='%', thickness=12, outlinewidth=0),
        ))
        fig.update_layout(base_layout(
            f"#{i + 1} {s['name'][:50]}",
            f"{s['addr']} · % of checks with {LOBBY_PLAYERS}+ human players · times in {TZ_LABEL}",
            height=440,
            variant=variant,
        ))
        fig.update_xaxes(showgrid=False, ticks='', tickangle=-45)
        fig.update_yaxes(showgrid=False, ticks='', autorange='reversed')
        save(fig, f'server_heatmap_{i + 1}.png', variant)
    return [{'name': s['name'], 'addr': s['addr']} for s in stats]


# Write the top servers table into the README between the marker comments
def update_readme(stats):
    if stats:
        lines = [
            f'| # | Server | Connect (paste in TF2 console) | Best time for a {LOBBY_PLAYERS}+ lobby | Best day '
            f'| Has {LOBBY_PLAYERS}+ players | Has players | Avg players when on | Peak |',
            '|---|---|---|---|---|---|---|---|---|',
        ]
        for i, s in enumerate(stats):
            name = s['name'].replace('|', '\\|')
            best = ('–' if s['best_window'] is None
                    else f"{window_label(s['best_window'], short=True)} {TZ_LABEL} ({s['best_window_pct']:.0%})")
            lines.append(
                f"| {i + 1} | {name} | `connect {s['addr']}` | {best} | {s['best_day'] or '–'} "
                f"| {s['pct_busy']:.0%} of checks | {s['pct_active']:.0%} of checks | {s['avg_players']:.1f} | {s['peak']} |"
            )
        lines.append('')
        lines.append('To favorite one: Steam → View → Game Servers → Favorites → **+** and paste the address.')
    else:
        lines = ['_No server data yet - check back after a few days._']
    section = f"{README_START}\n" + '\n'.join(lines) + f"\n{README_END}"

    with open(README_PATH, encoding='utf-8') as f:
        readme = f.read()
    pattern = re.compile(re.escape(README_START) + '.*?' + re.escape(README_END), re.S)
    if not pattern.search(readme):
        print('README markers not found; skipping top servers table')
        return
    readme = re.sub(r'pinged when a \d+\+ player', f'pinged when a {LOBBY_PLAYERS}+ player', readme)
    with open(README_PATH, 'w', encoding='utf-8') as f:
        f.write(pattern.sub(lambda _: section, readme))


# Top servers for each timezone and filter on the website
def write_site_data(stats_by_zone, windows, forecast, heatmap_servers):
    data = {'updated': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'), 'lobby_players': LOBBY_PLAYERS,
            'top_servers': {}, 'heatmap_servers': heatmap_servers,
            'lobby_windows': {'timezone': ZONES[0][0], 'recent_days': SCORECARD_RECENT_DAYS, **windows}, 'forecast': forecast}
    for (zone_key, key), (stats, label) in stats_by_zone.items():
        data['top_servers'].setdefault(zone_key, {})[key] = [{
            'name': s['name'],
            'addr': s['addr'],
            'best_time': None if s['best_window'] is None else f"{window_label(s['best_window'], short=True)} {label}",
            'best_time_pct': round(float(s['best_window_pct']), 3),
            'best_day': s['best_day'],
            'pct_busy': round(float(s['pct_busy']), 3),
            'pct_active': round(float(s['pct_active']), 3),
            'avg_players': round(float(s['avg_players']), 1),
            'peak': int(s['peak']),
        } for s in stats]
    with open(os.path.join(OUTPUT_FOLDER, 'site_data.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


# The map rotation tracker's own section of the website (with the voter and top scorer leaderboards),
# from its own logs
def write_rotation_data():
    summary = rotation_summary()
    with open(os.path.join(OUTPUT_FOLDER, 'rotation.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(summary or {}, voters=voter_summary(), scorers=score_summary(),
                       updated=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')),
                  f, ensure_ascii=False, indent=1)


# Push the log and README to main, and the website (page + charts) to its own branch
def push_to_github():
    os.chdir(SCRIPT_DIR)
    os.system("git add README.md data/tf2_balloon_log.txt")
    # On their own, so a missing voter log can't stop the main log from being added
    os.system("git add data/rotation_players.jsonl")
    os.system("git add data/forecast_history.jsonl")
    os.system("git commit -m 'update log'")
    os.system("git push origin main")
    publish_site()


# GitHub Pages serves SITE_BRANCH (the contents of site/ at its root), which is replaced by a single
# fresh commit each day so the ~16MB of charts never pile up in the repo history
SITE_BRANCH = 'site'


def publish_site():
    # A throwaway index, so the main branch's staging area is never touched
    env = dict(os.environ, GIT_INDEX_FILE=os.path.join(SCRIPT_DIR, '.git', 'site-index'))

    def git(*args):
        return subprocess.run(['git', *args], cwd=SCRIPT_DIR, env=env, capture_output=True,
                              text=True, check=True).stdout.strip()

    try:
        git('read-tree', '--empty')
        git('add', '-f', 'site')
        commit = git('commit-tree', git('write-tree', '--prefix=site/'), '-m', 'update site')
        git('push', '-fq', 'origin', f'{commit}:refs/heads/{SITE_BRANCH}')
    except subprocess.CalledProcessError as e:
        print(f'Error publishing site: {e.stderr}')
    finally:
        if os.path.exists(env['GIT_INDEX_FILE']):
            os.remove(env['GIT_INDEX_FILE'])


# Main workflow
if __name__ == "__main__":
    # Optional: generate_report.py [log_path [output_folder]] [--no-push] [--no-readme]
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    log_path = args[0] if args else LOG_PATH
    if len(args) > 1:
        OUTPUT_FOLDER = args[1]
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    graph_df, server_df = load_and_preprocess_data(log_path)
    unemployment = load_unemployment()
    earthquakes = load_earthquakes(graph_df['ts'].min())
    stats_by_zone = {}
    windows = {}
    heatmap_servers = {}
    for DISPLAY_TIMEZONE, TZ_LABEL, ZONE_SUFFIX in ZONES:
        default_zone = ZONE_SUFFIX == ZONES[0][2]
        zone_df, zone_servers = in_zone(graph_df), in_zone(server_df)
        variants = [
            (ALL_SERVERS, zone_df, zone_servers),
            (DEDICATED, split_checks(zone_df, zone_servers, True), zone_servers[zone_servers['dedicated']]),
            (ROTATING, split_checks(zone_df, zone_servers, False), zone_servers[~zone_servers['dedicated']]),
        ]
        for variant, df, servers in variants:
            generate_top_windows_plot(df, variant)
            generate_heatmap(
                df, df['online_players'] >= LOBBY_PLAYERS,
                f'How often there is a busy lobby ({LOBBY_PLAYERS}+ players)',
                f'% of 5-minute checks with a server at {LOBBY_PLAYERS}+ human players · times in {TZ_LABEL}',
                'busy_heatmap.png', variant,
            )
            all_server_stats = top_server_stats(zone_df, servers)
            drawn = generate_server_heatmaps(all_server_stats, variant)
            stats_by_zone[(TZ_LABEL.lower(), variant[0].lstrip('_') or 'all')] = (all_server_stats[:SITE_SERVER_COUNT],
                                                                                    TZ_LABEL)
            # These don't depend on the time of day, so they're only drawn once
            if default_zone:
                windows[variant[0].lstrip('_') or 'all'] = lobby_windows(df)
                heatmap_servers[variant[0].lstrip('_') or 'all'] = drawn
                generate_session_length_plot(df, variant)
                generate_monthly_avg_plot(df, variant)
                generate_holiday_table(df, variant)
                generate_earthquake_correlation_plot(df, earthquakes, variant)
                generate_unemployment_correlation_plot(df, unemployment, variant)
    DISPLAY_TIMEZONE, TZ_LABEL, ZONE_SUFFIX = ZONES[0]
    stats = stats_by_zone[(TZ_LABEL.lower(), 'all')][0][:TOP_SERVER_COUNT]
    years = range(graph_df['ts'].min().year, graph_df['ts'].max().year + 2)
    holiday_dates = {d for dates in holiday_calendar(years).values() for d in dates}
    forecast, fit = train_lobby_forecast(graph_df, server_df, holiday_dates, LOBBY_PLAYERS, CHECK_MINUTES,
                                         ZONES[0][0], FORECAST_MODEL_PATH)
    generate_forecast_regression(fit, LOBBY_PLAYERS)
    history_row = forecast_history_row(forecast, dt.datetime.now(ZoneInfo(ZONES[0][0])).date())
    if history_row:
        save_forecast_history(history_row)
    generate_forecast_history(read_forecast_history())
    write_site_data(stats_by_zone, windows, forecast, heatmap_servers)
    write_rotation_data()
    if '--no-readme' not in sys.argv:
        update_readme(stats)
    if '--no-push' not in sys.argv:
        push_to_github()
