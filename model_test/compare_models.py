# Compares tree models against the current logistic regression on the forecast's rolling test
# (balloon/forecast.py rolling_evaluation). Nothing is saved or pushed.
# Run from the repo root: .venv/bin/python model_test/compare_models.py
# 2026-10-29, 21 days of data: trees had similar AUC but worse log loss than the logistic regression, even on
# checks with no lobby yet. Worth retrying with 8+ weeks of data.
import os
import sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv = ['x', '--no-push', '--no-readme']
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import generate_report as gr
from balloon import forecast as fc
from balloon.voters import voters_waiting, read_snapshots, read_changes
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

df, server_df = gr.load_and_preprocess_data(gr.LOG_PATH)
years = range(df['ts'].min().year, df['ts'].max().year + 2)
hol = {d for ds in gr.holiday_calendar(years).values() for d in ds}
thr, cm = gr.LOBBY_PLAYERS, gr.CHECK_MINUTES
vr = voters_waiting(read_snapshots(), read_changes())
f = fc.check_frame(df, server_df, thr, cm, vr)
dates = f['ts'].dt.date.drop_duplicates()
wd = pd.to_datetime(dates).dt.dayofweek.value_counts().reindex(range(7), fill_value=0)
cfg = {'threshold': thr, 'check_minutes': cm, 'servers': fc.pick_servers(f['addr'], f['online_players']),
       'use_days': bool(wd.min() >= fc.FORECAST_MIN_DATES_PER_WEEKDAY),
       'use_holidays': bool(dates.isin(hol).sum() >= fc.FORECAST_MIN_HOLIDAY_DATES), 'holiday_dates': hol,
       'use_voters': len({r['ts'][:10] for r in vr}) >= fc.FORECAST_MIN_VOTER_DATES
                     and int((f['voters_waiting'] > 0).sum()) >= fc.FORECAST_MIN_VOTER_CHECKS}
keep, y = fc.lobby_labels(f, thr, cm)
ex = f.iloc[keep].reset_index(drop=True)
X = fc.feature_matrix(ex, cfg)
# Trees can split a raw clock hour and day number directly instead of sin/cos + one-hots
hour = (ex['ts'].dt.hour + ex['ts'].dt.minute / 60).to_numpy()
Xt = np.column_stack([X, hour, ex['ts'].dt.dayofweek.to_numpy()])
players = ex['online_players'].to_numpy()
print(f"days {len(dates)}, examples {len(y)}, positives {y.sum()}, cfg use_days={cfg['use_days']} "
      f"servers={len(cfg['servers'])} voters={cfg['use_voters']}, features {X.shape[1]}")

models = {
    'LogReg C=0.01 (current)': (lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=1000)), X),
    'LogReg + pairwise interactions': (lambda: make_pipeline(StandardScaler(), PolynomialFeatures(2, interaction_only=True, include_bias=False),
                                                         StandardScaler(), LogisticRegression(C=0.001, max_iter=3000)), X),
    'HistGB shallow (depth 2)': (lambda: HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=150,
                                    min_samples_leaf=100, l2_regularization=1.0, random_state=0), Xt),
    'HistGB depth 3': (lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                    min_samples_leaf=50, l2_regularization=1.0, random_state=0), Xt),
    'RandomForest': (lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=50, max_features=0.5,
                                                   n_jobs=-1, random_state=0), Xt),
    'ExtraTrees': (lambda: ExtraTreesClassifier(n_estimators=300, min_samples_leaf=50, max_features=0.5,
                                               n_jobs=-1, random_state=0), Xt),
}
rows = []
for name, (make, M) in models.items():
    fc.new_forecast_model = make
    ev = fc.rolling_evaluation(ex['ts'], M, y, players, thr)
    a, o = ev['all'], ev['no_lobby_yet']
    rows.append({'model': name, 'LL': a['log_loss'], 'base LL': a['baseline_log_loss'], 'brier': a['brier'],
                 'AUC': a.get('auc'), 'open LL': o['log_loss'], 'open base LL': o['baseline_log_loss'],
                 'open brier': o['brier'], 'open AUC': o.get('auc'), 'open lobbies': o['lobbies']})
print(f"test days: {len(ev['test_days'])} ({ev['test_days'][0]} .. {ev['test_days'][-1]})")
print(pd.DataFrame(rows).to_string(index=False))
