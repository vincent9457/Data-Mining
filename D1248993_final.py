import yfinance as yf
import pandas as pd
import numpy as np
import os
import time
import random
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import mplfinance as mpf
import warnings
import requests
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, f1_score
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras import layers, models
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from scipy import stats

warnings.filterwarnings("ignore")
TICKERS = ["1326.TW", "2317.TW", "2330.TW", "2439.TW", "2615.TW", "3006.TW", "3019.TW", "3026.TW", "3481.TW", "6209.TW"]
DATA_FOLDER = "stock_cache"
RESULT_FOLDER = "stock_results"
START = "2016-01-01"
END = "2025-12-31"

MIN_FREQ = 3
ROUND_DECIMALS = 2
SIM_THRESHOLD = 0.5
TEST_YEAR = 2025
UP_THRESHOLD = 5
DOWN_THRESHOLD = -5
RANDOM_SLEEP = (5, 15)
GA_POP_SIZE = 40
GA_GENERATIONS = 30
GA_CROSSOVER_RATE = 0.8
GA_MUTATION_RATE = 0.1
GA_SIM_THRESHOLD = 0.5      
GA_MIN_FREQ = 5             
GA_MIN_AVG_RETURN = 3.0     
GA_MAX_DD = 0.03            

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def safe_download(ticker, start=START, end=END, retries=3):
    cache_file = os.path.join(DATA_FOLDER, f"{ticker}.csv")

    if os.path.exists(cache_file):
        print(f"使用快取資料：{ticker}")
        df = pd.read_csv(cache_file, index_col=0)
        df.index = pd.to_datetime(df.index, errors="coerce")
        return df

    for i in range(retries):
        try:
            print(f"下載 {ticker} ... (嘗試 {i+1})")
            df = yf.download(ticker, start=start, end=end, auto_adjust=False)

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            expected_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

            if "Adj Close" not in df.columns:
                print("Adj Close 不存在，使用 Close 當作 Adj Close")
                df["Adj Close"] = df["Close"]

            if "Volume" not in df.columns:
                print("Volume 不存在，自動填 0")
                df["Volume"] = 0

            df = df[expected_cols]

            for c in expected_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            df = df.dropna()

            df.to_csv(cache_file)
            print(f"{ticker} 下載成功並已快取。")
            return df

        except Exception as e:
            print(f"{ticker} 第 {i+1} 次失敗：{e}")
            time.sleep(5)

    print(f"{ticker} 多次失敗，略過")
    return pd.DataFrame()

def run_xgb_classification_binary(train_df, test_df, features):
    train_df = train_df.dropna(subset=["label_binary"])
    test_df = test_df.dropna(subset=["label_binary"])

    X_train = train_df[features]
    y_train = train_df["label_binary"]

    X_test = test_df[features]
    y_test = test_df["label_binary"]

    class_counts = y_train.value_counts().to_dict()
    total = len(y_train)
    class_weight = {cls: total/count for cls, count in class_counts.items()}
    sample_weight = y_train.map(class_weight)

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss", 
        tree_method="hist",
        # random_state=42
        random_state=None
    )

    model.fit(X_train, y_train, sample_weight=sample_weight)

    y_pred_prob = model.predict_proba(X_test)[:,1]
    y_pred = (y_pred_prob > 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return model, cm, f1, y_pred


def run_xgb_regression(train_df, test_df, features):
    train_df = train_df.dropna(subset=["future_return"])
    test_df = test_df.dropna(subset=["future_return"])

    X_train = train_df[features]
    y_train = train_df["future_return"]

    X_test = test_df[features]
    y_test = test_df["future_return"]

    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        # random_state=42
        random_state=None
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return model, rmse, r2, y_pred

def run_rf_classification_binary(train_df, test_df, features):
    train_df = train_df.dropna(subset=["label_binary"])
    test_df = test_df.dropna(subset=["label_binary"])

    X_train = train_df[features]
    y_train = train_df["label_binary"]
    X_test = test_df[features]
    y_test = test_df["label_binary"]

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",
        # random_state=42
        random_state=None
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return clf, cm, f1, y_pred


def run_rf_regression(train_df, test_df, features):
    train_df = train_df.dropna(subset=["future_return"])
    test_df = test_df.dropna(subset=["future_return"])

    X_train = train_df[features]
    y_train = train_df["future_return"]

    X_test = test_df[features]
    y_test = test_df["future_return"]

    reg = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        # random_state=42
        random_state=None
    )
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return reg, rmse, r2, y_pred

def get_xgb_feature_weights_from_classifier(model, features):
    booster = model.get_booster()
    score = booster.get_score(importance_type="gain")

    gains = np.zeros(len(features), dtype=float)
    for i in range(len(features)):
        gains[i] = score.get(f"f{i}", 0.0)

    if gains.sum() <= 0:
        gains = np.ones(len(features), dtype=float)

    weights = gains / gains.sum()
    return weights

def similarity_weighted_scaled(a, b, w):
    diff2 = (a - b) ** 2
    d = np.sqrt(np.sum(w * diff2))
    return 1.0 / (1.0 + d)

def plot_confusion_matrix(cm, labels, title, save_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def build_autoencoder(input_dim=10, latent_dim=4):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(8, activation='relu')(inputs)
    latent = layers.Dense(latent_dim, activation='relu')(x)

    x = layers.Dense(8, activation='relu')(latent)
    outputs = layers.Dense(input_dim, activation='linear')(x)

    autoencoder = models.Model(inputs, outputs)
    encoder = models.Model(inputs, latent)

    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

def remove_ex_dividend_days_yf(df, ticker, start=START, end=END):
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    cache_dir = "dividend_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{ticker}_dividends.csv")

    if os.path.exists(cache_path):
        dividends = pd.read_csv(cache_path, index_col=0, parse_dates=True).iloc[:, 0]
        print(f"使用 dividends 快取：{ticker}")
    else:
        print(f"下載 dividends：{ticker}")
        stock = yf.Ticker(ticker)
        dividends = stock.dividends
        dividends.to_csv(cache_path)
        time.sleep(1.5)

    dividends = dividends.loc[start:end]
    ex_div_dates = dividends[dividends > 0].index
    df_clean = df.drop(ex_div_dates, errors="ignore")

    print("===== 除權息處理（yfinance）=====")
    print(f"{ticker} 除權息日數量：{len(ex_div_dates)}")
    print("================================")

    return df_clean, ex_div_dates

def calc_k_features(df):
    df = df.copy()
    for c in ["Open","High","Low","Close","Volume","Adj Close"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Open","High","Low","Close","Volume"])
    df = df[(df["Close"] != 0) & (df["Volume"] != 0)]

    df["upper_shadow"] = (df["High"] - np.maximum(df["Open"], df["Close"])) / df["Close"] * 100
    df["lower_shadow"] = (np.minimum(df["Open"], df["Close"]) - df["Low"]) / df["Close"] * 100
    df["body"] = (df["Close"] - df["Open"]) / df["Close"] * 100
    df["prev_upper_shadow"] = df["upper_shadow"].shift(1)
    df["prev_lower_shadow"] = df["lower_shadow"].shift(1)
    df["prev_body"] = df["body"].shift(1)
    df["open_pattern"] = (df["Open"] - df["Close"].shift(1)) / df["Close"] * 100
    df["close_pattern"] = (df["Close"] - df["Close"].shift(1)) / df["Close"] * 100
    df["vol_ratio"] = np.where(df["Volume"]==0, 0, (df["Volume"] - df["Volume"].rolling(5).mean()) / df["Volume"])
    df["trend_5d"] = (df["Close"].shift(2) - df["Close"].shift(7)) / df["Close"].shift(7)
    return df.replace([np.inf, -np.inf], np.nan).dropna()

def similarity_euclid_scaled(a, b):
    d = euclidean(a, b)
    return 1.0 / (1.0 + d)

def extract_and_evaluate_candidates(train_df, features, min_freq, round_decimals, sim_threshold):
    candidates = train_df[train_df["label"] != 0].copy()
    if candidates.empty:
        return pd.DataFrame(), None

    train_clean = train_df.replace([np.inf, -np.inf], np.nan).dropna(subset=features)
    if train_clean.empty:
        return pd.DataFrame(), None

    scaler = StandardScaler().fit(train_clean[features])
    train_scaled = pd.DataFrame(scaler.transform(train_clean[features]), index=train_clean.index, columns=features)

    grouped = candidates[features].round(round_decimals).value_counts().reset_index()
    grouped.columns = features + ["count"]
    grouped = grouped[grouped["count"] >= min_freq]
    if grouped.empty:
        return pd.DataFrame(), scaler

    records = []
    for _, row in grouped.iterrows():
        key_vec = row[features].values.astype(float)
        key_scaled = scaler.transform(key_vec.reshape(1,-1))[0]
        sims = train_scaled.apply(lambda r: similarity_weighted_scaled(r.values, key_scaled, weights), axis=1)
        matched = train_clean[sims > sim_threshold]
        freq = len(matched)
        if freq == 0:
            continue
        wins = (matched["future_return"] > 5).sum()
        losses = (matched["future_return"] < -5).sum()
        win_rate = wins / freq
        avg_return = matched["future_return"].mean()
        pos = matched[matched["future_return"] > 0]["future_return"]
        neg = matched[matched["future_return"] < 0]["future_return"]
        avg_gain = pos.mean() if len(pos)>0 else 0.0
        avg_loss = neg.mean() if len(neg)>0 else 0.0
        expectancy = win_rate * avg_gain + (1 - win_rate) * avg_loss
        rec = {f: row[f] for f in features}
        rec.update({
            "count": int(row["count"]),
            "replay_freq": int(freq),
            "win_rate": float(win_rate),
            "avg_return": float(avg_return),
            "expectancy": float(expectancy)
        })
        records.append(rec)
    return pd.DataFrame(records).sort_values("expectancy", ascending=False), scaler

def benjamini_hochberg(pvals):
    pvals = np.array(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(1, n+1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = q
    return out

def collect_candidate_patterns(train_df, features, weights,
                               round_list=[2,1,0],
                               min_freq_list=[3,2,1],
                               sim_thresholds=[0.85,0.75,0.65,0.55,0.5],
                               min_total_required=20):
    train_clean = train_df.replace([np.inf, -np.inf], np.nan).dropna(subset=features)
    if train_clean.empty:
        return pd.DataFrame(), None

    scaler = StandardScaler().fit(train_clean[features])
    train_scaled = pd.DataFrame(scaler.transform(train_clean[features]), index=train_clean.index, columns=features)
    pool = {}
    labeled = train_clean[train_clean["label"] != 0]

    for rnd in round_list:
        grouped = labeled[features].round(rnd).value_counts().reset_index()
        grouped.columns = features + ["count"]
        for _, grow in grouped.iterrows():
            key_tuple = tuple([float(grow[f]) for f in features])
            pool.setdefault(key_tuple, {"count_raw": int(grow["count"])})

    for pattern_tuple in list(pool.keys()):
        raw_vec = np.array(pattern_tuple, dtype=float)
        raw_scaled = scaler.transform(raw_vec.reshape(1, -1))[0]
        found_any = False
        for sim_th in sim_thresholds:
            sims = train_scaled.apply(
                lambda r: similarity_weighted_scaled(r.values, raw_scaled, weights),
                axis=1
            )
            matched = train_clean[sims > sim_th]
            freq = len(matched)

            if freq == 0:
                continue

            pattern_ret = matched["future_return"].values
            all_ret = train_clean["future_return"].values

            t_stat, p_value = stats.ttest_ind(
                pattern_ret, all_ret,
                equal_var=False,
                nan_policy="omit"
            )

            lift = np.nanmean(pattern_ret) - np.nanmean(all_ret)

            wins = (matched["future_return"] > 5).sum()
            win_rate = wins / freq
            avg_return = matched["future_return"].mean()

            pool[pattern_tuple].update({
                "replay_freq": int(freq),
                "win_rate": float(win_rate),
                "avg_return": float(avg_return),
                "expectancy": float(avg_return),
                "sim_threshold_used": float(sim_th),
                "t_pvalue": float(p_value),
                "lift_vs_all": float(lift)
            })

            break 

        if not found_any:
            pool[pattern_tuple].update({
                "replay_freq": 0,
                "win_rate": 0.0,
                "avg_return": 0.0,
                "avg_gain": 0.0,
                "avg_loss": 0.0,
                "expectancy": 0.0,
                "sim_threshold_used": None
            })

    records = []
    for pat, info in pool.items():
        rec = {f: pat[i] for i, f in enumerate(features)}
        rec.update(info)
        records.append(rec)
    patterns_df = pd.DataFrame(records)

    if len(patterns_df) < min_total_required:
        extra_grouped = train_clean[features].round(0).value_counts().reset_index()
        extra_grouped.columns = features + ["count"]
        for _, grow in extra_grouped.iterrows():
            key_tuple = tuple([float(grow[f]) for f in features])
            if key_tuple not in pool:
                pool.setdefault(key_tuple, {"count_raw": int(grow["count"])})
        new_records = []
        for pat in pool.keys():
            if "replay_freq" in pool[pat]:
                continue
            raw_vec = np.array(pat, dtype=float)
            raw_scaled = scaler.transform(raw_vec.reshape(1, -1))[0]
            sims = train_scaled.apply(lambda r: similarity_weighted_scaled(r.values, raw_scaled, weights), axis=1)
            matched = train_clean[sims > 0.5]  
            freq = len(matched)
            wins = (matched["future_return"] > 5).sum() if freq>0 else 0
            win_rate = wins / freq if freq>0 else 0.0
            avg_return = matched["future_return"].mean() if freq>0 else 0.0
            pos = matched[matched["future_return"] > 0]["future_return"]
            neg = matched[matched["future_return"] < 0]["future_return"]
            avg_gain = pos.mean() if len(pos) > 0 else 0.0
            avg_loss = neg.mean() if len(neg) > 0 else 0.0
            expectancy = win_rate * avg_gain + (1 - win_rate) * avg_loss
            pool[pat].update({
                "replay_freq": int(freq),
                "win_rate": float(win_rate),
                "avg_return": float(avg_return),
                "avg_gain": float(avg_gain),
                "avg_loss": float(avg_loss),
                "expectancy": float(expectancy),
                "sim_threshold_used": 0.5
            })

        records = []
        for pat, info in pool.items():
            rec = {f: pat[i] for i, f in enumerate(features)}
            rec.update(info)
            records.append(rec)
        patterns_df = pd.DataFrame(records)

    patterns_df = patterns_df[patterns_df["replay_freq"] >= MIN_FREQ]
    patterns_df = patterns_df[
        (patterns_df["avg_return"] >= 2.0) |
        (patterns_df["avg_return"] <= -2.0)
    ]

    patterns_df = patterns_df.replace([np.inf, -np.inf], np.nan).dropna()

    patterns_df = patterns_df.sort_values(
        ["expectancy", "replay_freq"],
        ascending=[False, False]
    )
    if "t_pvalue" in patterns_df.columns:
        patterns_df["t_qvalue"] = benjamini_hochberg(patterns_df["t_pvalue"].values)
    return patterns_df.reset_index(drop=True), scaler

def evaluate(trades):
    if len(trades) == 0:
        return 0, 0, 0
    equity = np.cumprod([1 + r for r in trades])
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1
    max_dd = drawdown.min()   
    total_return = equity[-1] - 1
    winrate = sum(r > 0 for r in trades) / len(trades)
    return total_return, winrate, abs(max_dd)

def simple_backtest(signals_df, price_df):
    trades = []

    for date, row in signals_df.iterrows():
        if date not in price_df.index:
            continue

        entry_idx = price_df.index.get_loc(date)

        if entry_idx + 3 >= len(price_df):
            continue

        entry_price = price_df.iloc[entry_idx]["Close"]
        exit_price  = price_df.iloc[entry_idx + 3]["Close"]

        if row["signal"] == 1:
            ret = (exit_price - entry_price) / entry_price
        elif row["signal"] == -1:
            ret = (entry_price - exit_price) / entry_price
        else:
            continue

        trades.append(ret)

    return trades

def get_top_patterns(train_df, features, weights, need_each=10):
    patterns_df, scaler = collect_candidate_patterns(
        train_df, features, weights,
        round_list=[2,1,0],
        min_freq_list=[3,2,1],
        sim_thresholds=[0.9,0.8,0.7,0.6,0.5],
        min_total_required=need_each*4
    )
    if patterns_df.empty or scaler is None:
        return pd.DataFrame(), pd.DataFrame(), scaler

    
    bull_df = patterns_df[patterns_df["expectancy"] > 0].copy()
    bear_df = patterns_df[patterns_df["expectancy"] < 0].copy()

    if len(bull_df) < need_each:
        more = patterns_df[patterns_df["expectancy"] >= 0].sort_values(["expectancy", "replay_freq"], ascending=[False, False])
        bull_df = more.head(max(need_each, len(bull_df)) ).head(need_each)
    if len(bear_df) < need_each:
        more = patterns_df[patterns_df["expectancy"] <= 0].sort_values(["expectancy", "replay_freq"], ascending=[True, False])
        
        bear_df = more.head(max(need_each, len(bear_df))).head(need_each)

    if len(bull_df) < need_each:
        need = need_each - len(bull_df)
        candidate = patterns_df[~patterns_df.index.isin(bull_df.index)].sort_values(["replay_freq", "expectancy"], ascending=[False, False]).head(need)
        bull_df = pd.concat([bull_df, candidate]).head(need_each)
    if len(bear_df) < need_each:
        need = need_each - len(bear_df)
        candidate = patterns_df[~patterns_df.index.isin(bear_df.index)].sort_values(["replay_freq", "expectancy"], ascending=[False, True]).head(need)
        bear_df = pd.concat([bear_df, candidate]).head(need_each)

    bull_df = bull_df.drop_duplicates(subset=features).head(need_each).reset_index(drop=True)
    bear_df = bear_df.drop_duplicates(subset=features).head(need_each).reset_index(drop=True)

    return bull_df, bear_df, scaler

def evaluate_patterns_on_test(patterns_df, scaler, features, test_df, threshold):
    if patterns_df.empty or scaler is None:
        return pd.DataFrame()
    test_clean = test_df.replace([np.inf, -np.inf], np.nan).dropna(subset=features)
    if test_clean.empty:
        return pd.DataFrame()
    test_scaled = pd.DataFrame(scaler.transform(test_clean[features]), index=test_clean.index, columns=features)
    rows = []
    for _, row in patterns_df.iterrows():
        vec = row[features].values.astype(float)
        vec_scaled = scaler.transform(vec.reshape(1,-1))[0]
        sims = test_scaled.apply(lambda r: similarity_weighted_scaled(r.values, vec_scaled, weights), axis=1)
        matched = test_clean[sims > threshold]
        freq = len(matched)
        if freq == 0:
            continue
        win_rate = (matched["future_return"] > 5).mean()
        avg_return = matched["future_return"].mean()
        rec = {f: row[f] for f in features}
        rec.update({"test_matches": freq, "test_win_rate": float(win_rate), "test_avg_return": float(avg_return)})
        rows.append(rec)
    return pd.DataFrame(rows)

def generate_pattern_signals_daily(bull_df, bear_df, scaler, features, test_df, weights, threshold=0.6):
    if scaler is None:
        return pd.DataFrame(columns=["signal"])

    test_clean = test_df.replace([np.inf, -np.inf], np.nan).dropna(subset=features).copy()
    if test_clean.empty:
        return pd.DataFrame(columns=["signal"])

    test_scaled = pd.DataFrame(
        scaler.transform(test_clean[features]),
        index=test_clean.index,
        columns=features
    )

    bull_vecs = []
    for _, r in bull_df.iterrows():
        v = r[features].values.astype(float)
        bull_vecs.append(scaler.transform(v.reshape(1, -1))[0])

    bear_vecs = []
    for _, r in bear_df.iterrows():
        v = r[features].values.astype(float)
        bear_vecs.append(scaler.transform(v.reshape(1, -1))[0])

    signals = []
    for dt, row in test_scaled.iterrows():
        x = row.values

        best_sim = -1
        best_sig = 0

        for pv in bull_vecs:
            sim = similarity_weighted_scaled(x, pv, weights)
            if sim > best_sim:
                best_sim = sim
                best_sig = 1

        for pv in bear_vecs:
            sim = similarity_weighted_scaled(x, pv, weights)
            if sim > best_sim:
                best_sim = sim
                best_sig = -1

        if best_sim >= threshold:
            signals.append((dt, best_sig))

    if not signals:
        return pd.DataFrame(columns=["signal"])

    return pd.DataFrame(signals, columns=["date", "signal"]).set_index("date")

def random_chromosome(train_df):
    row = train_df.sample(1)
    return row[FEATURES].values.flatten()

def ga_fitness(chromosome, train_df, scaler):
    df = train_df.dropna(subset=FEATURES + ["future_return"]).copy()
    if df.empty:
        return 0.0

    X = scaler.transform(df[FEATURES])
    pattern_vec = scaler.transform(np.array(chromosome).reshape(1, -1))[0]

    sims = np.array([similarity_weighted_scaled(x, pattern_vec, weights) for x in X])
    matched = df[sims >= GA_SIM_THRESHOLD]

    freq = len(matched)
    if freq == 0:
        return 0.0

    avg_ret = matched["future_return"].mean()

    rets = matched["future_return"].values / 100.0
    equity = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(equity)
    max_dd = abs((equity / peak - 1).min())
    score = 0.0
    score += freq * 1.0
    score += avg_ret * 2.0

    if max_dd > GA_MAX_DD:
        score -= (max_dd - GA_MAX_DD) * 100

    return score

def ga_tournament_selection(population, fitnesses, k=3):
    idx = np.random.choice(len(population), k)
    best = idx[np.argmax([fitnesses[i] for i in idx])]
    return population[best]

def ga_crossover(p1, p2):
    if random.random() > GA_CROSSOVER_RATE:
        return p1.copy(), p2.copy()
    alpha = random.random()
    c1 = alpha * p1 + (1 - alpha) * p2
    c2 = alpha * p2 + (1 - alpha) * p1
    return c1, c2

def ga_mutate(chrom):
    for i in range(len(chrom)):
        if random.random() < GA_MUTATION_RATE:
            chrom[i] += np.random.normal(0, 0.5)
    return chrom

def run_ga(train_df):
    scaler = StandardScaler().fit(train_df[FEATURES])
    population = [random_chromosome(train_df) for _ in range(GA_POP_SIZE)]

    best_chrom = None
    best_fit = -np.inf
    history = []
    for gen in range(GA_GENERATIONS):
        fitnesses = [ga_fitness(ch, train_df, scaler) for ch in population]

        gen_best = np.max(fitnesses)
        gen_avg = np.mean(fitnesses)
        if gen_best > best_fit:
            best_fit = gen_best
            best_chrom = population[np.argmax(fitnesses)]
            best_gen = gen + 1
        history.append({
            "generation": gen + 1,
            "best_fitness": gen_best,
            "avg_fitness": gen_avg
        })

        print(
            f"GA Gen {gen+1}/{GA_GENERATIONS} | "
            f"best={gen_best:.2f}, avg={gen_avg:.2f}"
        )
        new_pop = []
        while len(new_pop) < GA_POP_SIZE:
            p1 = ga_tournament_selection(population, fitnesses)
            p2 = ga_tournament_selection(population, fitnesses)
            c1, c2 = ga_crossover(p1, p2)
            new_pop.append(ga_mutate(c1))
            new_pop.append(ga_mutate(c2))

        population = new_pop[:GA_POP_SIZE]
        print(f"GA Gen {gen+1}/{GA_GENERATIONS}, best fitness = {best_fit:.2f}")
    history_df = pd.DataFrame(history)
    return best_chrom, scaler, history_df, best_fit, best_gen

def plot_candlestick_patterns(df_patterns, title, savepath=None, scale=2.0):
    if df_patterns.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, len(df_patterns)), 4))
    ax.set_title(title)

    ax.set_facecolor('#f9f9f9')
    ax.grid(True, linestyle='--', color='#A0A0A0', alpha=0.9, linewidth=0.8)

    ax.set_xlim(-1, len(df_patterns))
    ax.axhline(0, color='#666666', linestyle='--', linewidth=0.8)
    for i, (_, row) in enumerate(df_patterns.iterrows()):
        o = 0
        c = row["body"] * scale / 100.0
        h = max(o, c) + row["upper_shadow"] * scale / 100.0
        l = min(o, c) - row["lower_shadow"] * scale / 100.0
        color = "red" if c > o else "green"
        ax.plot([i, i], [l, h], color="black")
        ax.add_patch(plt.Rectangle((i-0.3, min(o,c)), 0.6, abs(c-o), color=color))
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
        plt.close(fig)

ALL_TRAIN_FEATURES = []
ALL_LATENT_RESULTS = {}  
rf_f1_scores = {}
xgb_f1_scores = {}

FEATURES = ["upper_shadow","lower_shadow","body",
            "prev_upper_shadow","prev_lower_shadow","prev_body",
            "open_pattern","close_pattern","vol_ratio","trend_5d"]

ALL_BINARY = {}  
ALL_DATES = None

summaries = []
for ticker in TICKERS:
    print("\n" + "="*60)
    print(f"處理 {ticker}")
    out_dir = os.path.join(RESULT_FOLDER, ticker)
    os.makedirs(out_dir, exist_ok=True)

    df_raw = safe_download(ticker)
    if df_raw.empty:
        continue

    df_clean, ex_div_dates = remove_ex_dividend_days_yf(df_raw, ticker)
    df_feat = calc_k_features(df_clean)
    if df_feat.empty:
        continue

    df_feat["future_return"] = (df_feat["Close"].shift(-3) - df_feat["Close"]) / df_feat["Close"] * 100
    df_feat["label"] = 0
    df_feat.loc[df_feat["future_return"] > UP_THRESHOLD, "label"] = 1    
    df_feat.loc[df_feat["future_return"] < DOWN_THRESHOLD, "label"] = -1
    df_feat.index = pd.to_datetime(df_feat.index, errors="coerce")
    
    df_feat["label_binary"] = 0
    df_feat.loc[df_feat["future_return"] > 1, "label_binary"] = 1

    train = df_feat[(df_feat.index.year >= 2016) & (df_feat.index.year <= 2024)]
    test = df_feat[df_feat.index.year == TEST_YEAR]
    ALL_TRAIN_FEATURES.append(train[FEATURES])
    ALL_LATENT_RESULTS[ticker] = df_feat.copy()
    if ALL_DATES is None:
        ALL_DATES = df_feat.index
    else:
        ALL_DATES = ALL_DATES.union(df_feat.index)

    ALL_BINARY[ticker] = df_feat["label_binary"]

    if train.empty or test.empty:
        print("訓練或測試資料不足，略過")
        continue
    print(f"訓練資料: {train.shape}, 測試資料: {test.shape}")

    xgb_clf, xgb_cm, xgb_f1, xgb_pred = run_xgb_classification_binary(train, test, FEATURES)
    weights = get_xgb_feature_weights_from_classifier(xgb_clf, FEATURES)
    pd.DataFrame({"feature": FEATURES, "weight": weights}).to_csv(os.path.join(out_dir, "xgb_feature_weights.csv"), index=False)

    bull_df, bear_df, scaler = get_top_patterns(train, FEATURES, weights, need_each=10)

    all_candidates = pd.concat([bull_df, bear_df]).reset_index(drop=True)
    if not all_candidates.empty:
        all_candidates.to_csv(os.path.join(out_dir, "candidates_all.csv"), index=False)

    bull_df.to_csv(os.path.join(out_dir, "top10_bull.csv"), index=False)
    bear_df.to_csv(os.path.join(out_dir, "top10_bear.csv"), index=False)

    pattern_signal_df = generate_pattern_signals_daily(
        bull_df, bear_df, scaler, FEATURES, test, weights, threshold=SIM_THRESHOLD
    )

    pattern_trades = simple_backtest(pattern_signal_df, test)
    p_ret, p_wr, p_dd = evaluate(pattern_trades)
    print(f"Method1 Pattern trades: {len(pattern_trades)} (threshold={SIM_THRESHOLD})")

    pd.DataFrame([{
        "return": p_ret,
        "winrate": p_wr,
        "maxdd": p_dd
    }]).to_csv(os.path.join(out_dir, "method1_pattern_metrics.csv"), index=False)

    if not bull_df.empty:
        plot_candlestick_patterns(bull_df, f"{ticker} Bull Top10", os.path.join(out_dir,"bull_top10.png"))
    if not bear_df.empty:
        plot_candlestick_patterns(bear_df, f"{ticker} Bear Top10", os.path.join(out_dir,"bear_top10.png"))

    mc = mpf.make_marketcolors(
        up='r', down='g',
        edge='inherit',
        wick='inherit',
        volume={'up': 'r', 'down': 'g'}
    )

    s = mpf.make_mpf_style(
        base_mpf_style='classic', 
        marketcolors=mc,
        gridstyle='--',
        gridcolor='#A0A0A0',
        facecolor='#f9f9f9',
        y_on_right=True,
        rc={
            'axes.grid': True,
            'grid.alpha': 0.9,
            'grid.linewidth': 0.8
        },
    )
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    df_clean = df_clean.dropna(subset=["Open","High","Low","Close"])

    END_DATE = df_clean.index.max()
    START_3Y = END_DATE - pd.DateOffset(years=3)

    df_last_3y = df_clean.loc[START_3Y:END_DATE].copy()
    try:
        fig, axes = mpf.plot(
            df_last_3y,
            type='candle',
            volume=True,
            style=s,
            figratio=(16,9),
            figscale=1.2,
            title=f"{ticker} Recent 3 Years K-line",
            savefig=os.path.join(out_dir, "recent_3years.png"),
            warn_too_much_data=0,
            show_nontrading=False
        )

        ax_price = axes[0]
        ax_vol = axes[2]

        for ax in [ax_price, ax_vol]:
            ax.set_facecolor('#f9f9f9') 
            ax.grid(True, linestyle='--', color='#A0A0A0', alpha=0.9, linewidth=0.8)
            ax.yaxis.tick_right()

        ax_price.set_ylabel("Price", fontsize=12)
        ax_vol.set_ylabel("Volume", fontsize=12)

        for idx, (date, row) in enumerate(df_clean.iterrows()):
            o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
            color = "r" if c > o else "g"
            ax_price.plot([date, date], [l, h], color=color, linewidth=0.8) 
            ax_price.add_patch(plt.Rectangle(
                (date, min(o, c)), width=pd.Timedelta(days=0.6),
                height=abs(c - o), color=color
            ))

        colors = ['r' if c > o else 'g' for o, c in zip(df_clean['Open'], df_clean['Close'])]
        ax_vol.clear()
        ax_vol.bar(df_clean.index, df_clean['Volume'], color=colors, width=1.0)
        ax_vol.set_xlim(df_clean.index[0], df_clean.index[-1])

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "full_history.png"), dpi=200)
        plt.close(fig)

    except Exception as e:
        print("畫全歷史圖失敗:", e)

    try:
        mpf.plot(
            df_feat.tail(60),
            type='candle',
            volume=True,
            style=s,
            figratio=(16,9),
            figscale=1.2,
            title=f"{ticker} Recent 60 Days K-line",
            savefig=os.path.join(out_dir, "recent_60days.png"),
            block=False,
            warn_too_much_data=0,
            show_nontrading=False
        )
    except Exception as e:
        print("畫近60天圖失敗:", e)

    summaries.append({
        "ticker": ticker,
        "train_rows": len(train),
        "test_rows": len(test),
        "found_patterns": len(all_candidates),
        "bull_top10": len(bull_df),
        "bear_top10": len(bear_df)
    })

    rf_clf, cm, f1, y_pred_cls = run_rf_classification_binary(train, test, FEATURES)
    plot_confusion_matrix(
        cm,
        labels=["非漲(0)", "漲(1)"],
        title=f"RF Confusion Matrix - {ticker}",
        save_path=os.path.join(out_dir, "rf_confusion_matrix.png")
    )
    pd.DataFrame(cm).to_csv(os.path.join(out_dir, "rf_confusion_matrix.csv"), index=False)
    with open(os.path.join(out_dir, "rf_f1_score.txt"), "w") as f:
        f.write(str(f1))

    print(f"RF 分類 F1-score: {f1:.4f}")

    rf_reg, rmse, r2, y_pred_reg = run_rf_regression(train, test, FEATURES)

    with open(os.path.join(out_dir, "rf_regression_scores.txt"), "w") as f:
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"R2: {r2:.4f}\n")

    print(f"RF 迴歸 RMSE: {rmse:.4f}, R2: {r2:.4f}")


    plot_confusion_matrix(
        xgb_cm,
        labels=["非漲(0)", "漲(1)"],
        title=f"XGB Confusion Matrix - {ticker}",
        save_path=os.path.join(out_dir, "xgb_confusion_matrix.png")
    )
    pd.DataFrame(xgb_cm).to_csv(os.path.join(out_dir, "xgb_confusion_matrix.csv"), index=False)
    with open(os.path.join(out_dir, "xgb_f1_score.txt"), "w") as f:
        f.write(str(xgb_f1))

    print(f"XGB 分類 F1-score: {xgb_f1:.4f}")

    xgb_reg, xgb_rmse, xgb_r2, xgb_pred_reg = run_xgb_regression(train, test, FEATURES)

    with open(os.path.join(out_dir, "xgb_regression_scores.txt"), "w") as f:
        f.write(f"RMSE: {xgb_rmse:.4f}\n")
        f.write(f"R2: {xgb_r2:.4f}\n")

    print(f"XGB 迴歸 RMSE: {xgb_rmse:.4f}, R2: {xgb_r2:.4f}")
    rf_f1_scores[ticker] = f1
    xgb_f1_scores[ticker] = xgb_f1

    print(f"{ticker} 完成。結果在 {out_dir}")
    time.sleep(random.randint(3, 6))
    
print("\n 計算股票之間的齊漲共現率...")
df_binary = pd.DataFrame(index=ALL_DATES)

for ticker, series in ALL_BINARY.items():
    df_binary[ticker] = series.reindex(ALL_DATES).fillna(0)

co_rise_matrix = df_binary.T.dot(df_binary)

co_dir = os.path.join(RESULT_FOLDER, "_co_movement")
os.makedirs(co_dir, exist_ok=True)

co_rise_matrix.to_csv(os.path.join(co_dir, "co_rise_matrix.csv"))

print("已輸出共現矩陣 co_rise_matrix.csv")

co_rise_matrix = co_rise_matrix.fillna(0).astype(int)
plt.figure(figsize=(10, 8))
sns.heatmap(co_rise_matrix, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Co-Rise Co-movement (同時上漲共現率)")
plt.tight_layout()
plt.savefig(os.path.join(co_dir, "co_rise_heatmap.png"))
plt.close()

print("已輸出共現熱力圖 co_rise_heatmap.png")


ALL_TRAIN_FEATURES = pd.concat(ALL_TRAIN_FEATURES).dropna()
print("AE 訓練資料量：", ALL_TRAIN_FEATURES.shape)

def build_autoencoder(input_dim=10, latent_dim=4):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(8, activation='relu')(inputs)
    latent = layers.Dense(latent_dim, activation='relu')(x)
    x = layers.Dense(8, activation='relu')(latent)
    outputs = layers.Dense(input_dim, activation='linear')(x)

    autoencoder = models.Model(inputs, outputs)
    encoder = models.Model(inputs, latent)

    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

scaler_ae = StandardScaler()
X_train_scaled = scaler_ae.fit_transform(ALL_TRAIN_FEATURES)
autoencoder, encoder = build_autoencoder()
autoencoder.fit(
    X_train_scaled, X_train_scaled,
    epochs=50,
    batch_size=64,
    shuffle=True,
    verbose=1
)

print("Autoencoder 訓練完成！")

print("為所有股票建立 latent code ...")

for ticker, df in ALL_LATENT_RESULTS.items():

    if not all(col in df.columns for col in FEATURES):
        print(f"{ticker} 缺少 FEATURES，跳過 latent 生成")
        continue

    X_scaled = scaler_ae.transform(df[FEATURES])
    latent = encoder.predict(X_scaled)
    df["latent1"] = latent[:,0]
    df["latent2"] = latent[:,1]
    df["latent3"] = latent[:,2]
    df["latent4"] = latent[:,3]

    ALL_LATENT_RESULTS[ticker] = df

from sklearn.decomposition import PCA

all_latent = []
all_labels = []
for ticker, df in ALL_LATENT_RESULTS.items():
    if "latent1" in df:
        l = df[["latent1","latent2","latent3","latent4"]].values
        all_latent.append(l)
        all_labels.append(df["label"].values)

all_latent = np.vstack(all_latent)
all_labels = np.hstack(all_labels)

pca = PCA(n_components=2)
latent_2d = pca.fit_transform(all_latent)

plt.figure(figsize=(8,6))
plt.scatter(latent_2d[:,0], latent_2d[:,1], c=all_labels, cmap="coolwarm", alpha=0.5)
plt.title("Latent Code PCA Visualization")
plt.colorbar(label="label (+1 / 0 / -1)")
plt.savefig(os.path.join(RESULT_FOLDER, "latent_pca.png"), dpi=200)
plt.close()

all_latent = []
all_labels = []

for ticker, df in ALL_LATENT_RESULTS.items():
    if "latent1" in df:
        all_latent.append(df[["latent1","latent2","latent3","latent4"]].values)
        all_labels.append(df["label"].values)

if all_latent:
    all_latent = np.vstack(all_latent)
    all_labels = np.hstack(all_labels)

    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(all_latent)

    plt.figure(figsize=(8,6))
    plt.scatter(latent_2d[:,0], latent_2d[:,1], c=all_labels, cmap="coolwarm", alpha=0.5)
    plt.title("AE Latent Code PCA Visualization")
    plt.colorbar(label="Label (1=上漲, -1=下跌, 0=其他)")
    plt.savefig(os.path.join(RESULT_FOLDER, "latent_pca.png"), dpi=200)
    plt.close()


def recon_mse(x_true, x_pred):
    return np.mean((x_true - x_pred) ** 2, axis=1)

def fit_ae_recon_models(all_latent_results, scaler_ae, features,
                        up_th=UP_THRESHOLD, down_th=DOWN_THRESHOLD,
                        q=0.90, epochs=40, batch_size=64):
    up_X_list = []
    down_X_list = []

    for _, df in all_latent_results.items():
        df2 = df.dropna(subset=features + ["future_return"]).copy()

        up_part = df2[df2["future_return"] > up_th][features].values
        down_part = df2[df2["future_return"] < down_th][features].values

        if len(up_part) > 0:
            up_X_list.append(up_part)
        if len(down_part) > 0:
            down_X_list.append(down_part)

    if (not up_X_list) or (not down_X_list):
        raise ValueError("方法二（recon error）：Up 或 Down 樣本不足，無法訓練 AE_up / AE_down")

    up_X = np.vstack(up_X_list)
    down_X = np.vstack(down_X_list)

    up_Xs = scaler_ae.transform(up_X)
    down_Xs = scaler_ae.transform(down_X)

    ae_up, _ = build_autoencoder(input_dim=len(features), latent_dim=4)
    ae_down, _ = build_autoencoder(input_dim=len(features), latent_dim=4)

    ae_up.fit(up_Xs, up_Xs, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)
    ae_down.fit(down_Xs, down_Xs, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)

    up_pred = ae_up.predict(up_Xs, verbose=0)
    down_pred = ae_down.predict(down_Xs, verbose=0)

    err_up = recon_mse(up_Xs, up_pred)
    err_down = recon_mse(down_Xs, down_pred)

    th_up = np.quantile(err_up, q)
    th_down = np.quantile(err_down, q)

    return ae_up, ae_down, th_up, th_down


def generate_ae_recon_signals(df, ae_up, ae_down, th_up, th_down, scaler_ae, features):
    df2 = df.dropna(subset=features).copy()
    if df2.empty:
        return pd.DataFrame(columns=["signal"])

    X = scaler_ae.transform(df2[features].values)

    err_u = np.full(len(X), np.inf)
    err_d = np.full(len(X), np.inf)

    if ae_up is not None:
        up_rec = ae_up.predict(X, verbose=0)
        err_u = recon_mse(X, up_rec)

    if ae_down is not None:
        down_rec = ae_down.predict(X, verbose=0)
        err_d = recon_mse(X, down_rec)

    signals = []
    for i, dt in enumerate(df2.index):
        if ae_up is not None and th_up is not None:
            if (err_u[i] < err_d[i]) and (err_u[i] <= th_up):
                signals.append((dt, 1))
                continue 

        if ae_down is not None and th_down is not None:
            if (err_d[i] < err_u[i]) and (err_d[i] <= th_down):
                signals.append((dt, -1))

    if not signals:
        return pd.DataFrame(columns=["signal"])

    return pd.DataFrame(signals, columns=["date", "signal"]).set_index("date")

print("建立 One-Class SVM 訓練集...")

up_latents = []
down_latents = []

for ticker, df in ALL_LATENT_RESULTS.items():
    up_latents.append(df[df["future_return"] > UP_THRESHOLD][["latent1","latent2","latent3","latent4"]])
    down_latents.append(df[df["future_return"] < DOWN_THRESHOLD][["latent1","latent2","latent3","latent4"]])

up_latents = pd.concat(up_latents).values
down_latents = pd.concat(down_latents).values

svm_up = OneClassSVM(kernel='rbf', nu=0.3, gamma='scale')
svm_down = OneClassSVM(kernel='rbf', nu=0.3, gamma='scale')

svm_up.fit(up_latents)
svm_down.fit(down_latents)

print("SVM 訓練完成！")

ae_up, ae_down, th_up, th_down = fit_ae_recon_models(
    ALL_LATENT_RESULTS,
    scaler_ae=scaler_ae,
    features=FEATURES,
    up_th=UP_THRESHOLD,
    down_th=DOWN_THRESHOLD,
    q=0.90,          
    epochs=40,
    batch_size=64
)

print("Method2 (AE recon error) 建立完成")
print("   th_up=", th_up, "th_down=", th_down)

def generate_signal(latent_vec):
    up_pred = svm_up.predict([latent_vec])[0]
    down_pred = svm_down.predict([latent_vec])[0]

    if up_pred == 1:   
        return 1

    if down_pred == 1: 
        return -1

    return 0

def generate_svm_signals(df):
    df = df.dropna(subset=["latent1","latent2","latent3","latent4"]).copy()
    signals = []

    for dt, row in df.iterrows():
        z = row[["latent1","latent2","latent3","latent4"]].values.astype(float).reshape(1, -1)

        up_pred = svm_up.predict(z)[0]
        down_pred = svm_down.predict(z)[0]

        if up_pred == 1 and down_pred == -1:
            signals.append((dt, 1))
        elif down_pred == 1 and up_pred == -1:
            signals.append((dt, -1))
        elif up_pred == 1 and down_pred == 1:
            up_score = svm_up.decision_function(z)[0]
            down_score = svm_down.decision_function(z)[0]
            signals.append((dt, 1 if up_score >= down_score else -1))
        else:
            pass

    if not signals:
        return pd.DataFrame(columns=["signal"])

    return pd.DataFrame(signals, columns=["date","signal"]).set_index("date")


MAX_HOLD_DAYS = 3      
STOP_LOSS = -0.03      
TAKE_PROFIT = 0.08    

def backtest(df):
    position = 0            
    entry_price = 0
    hold_days = 0
    trades = []
    signals = []

    closes = df["Close"].values
    opens = df["Open"].values
    dates = df.index.tolist()

    for i in range(len(df) - 1):      
        row = df.iloc[i]
        next_open = opens[i + 1]      

        latent_vec = [
            row["latent1"], row["latent2"],
            row["latent3"], row["latent4"]
        ]

        sig = generate_signal(latent_vec)

        if position == 0:
            if sig == 1:  
                position = 1
                entry_price = next_open
                hold_days = 0
                signals.append((dates[i+1], next_open, "BUY"))
            continue

        else:
            hold_days += 1
            current_r = (closes[i] - entry_price) / entry_price

            if current_r <= STOP_LOSS:
                exit_price = next_open
                trades.append((exit_price - entry_price) / entry_price)
                signals.append((dates[i+1], exit_price, "SELL"))
                position = 0
                continue

            if current_r >= TAKE_PROFIT:
                exit_price = next_open
                trades.append((exit_price - entry_price) / entry_price)
                signals.append((dates[i+1], exit_price, "SELL"))
                position = 0
                continue

            if sig == -1:
                exit_price = next_open
                trades.append((exit_price - entry_price) / entry_price)
                signals.append((dates[i+1], exit_price, "SELL"))
                position = 0
                continue

            if hold_days >= MAX_HOLD_DAYS:
                exit_price = next_open
                trades.append((exit_price - entry_price) / entry_price)
                signals.append((dates[i+1], exit_price, "SELL"))
                position = 0
                continue

    return trades, signals

def select_best_method(q1_csv_path):
    df = pd.read_csv(q1_csv_path)
    best_row = df.sort_values("return", ascending=False).iloc[0]
    return best_row["method"]

for ticker, df in ALL_LATENT_RESULTS.items():
    test_df = df[df.index.year == TEST_YEAR].copy()
    if test_df.empty:
        continue

    out_dir = os.path.join(RESULT_FOLDER, ticker)
    os.makedirs(out_dir, exist_ok=True)
    m2_sig_df = generate_ae_recon_signals(
        test_df,
        ae_up=ae_up,
        ae_down=ae_down,
        th_up=th_up,
        th_down=th_down,
        scaler_ae=scaler_ae,
        features=FEATURES
    )
    m2_trades = simple_backtest(m2_sig_df, test_df)
    m2_ret, m2_wr, m2_dd = evaluate(m2_trades)
    m3_sig_df = generate_svm_signals(test_df)
    m3_trades = simple_backtest(m3_sig_df, test_df)
    m3_ret, m3_wr, m3_dd = evaluate(m3_trades)
    m1_path = os.path.join(out_dir, "method1_pattern_metrics.csv")
    if os.path.exists(m1_path):
        m1 = pd.read_csv(m1_path).iloc[0]
        m1_ret, m1_wr, m1_dd = float(m1["return"]), float(m1["winrate"]), float(m1["maxdd"])
    else:
        m1_ret, m1_wr, m1_dd = 0.0, 0.0, 0.0

    compare_df = pd.DataFrame([
        {"method": "Method1_Pattern", "return": m1_ret, "winrate": m1_wr, "maxdd": m1_dd},
        {"method": "Method2_AE_only", "return": m2_ret, "winrate": m2_wr, "maxdd": m2_dd},
        {"method": "Method3_SVM_only", "return": m3_ret, "winrate": m3_wr, "maxdd": m3_dd},
    ])
    compare_df.to_csv(os.path.join(out_dir, "Q1_three_methods_comparison.csv"), index=False)

    print(f"\n{ticker} 題目一三方法比較表已輸出：Q1_three_methods_comparison.csv")
    print(compare_df)

    train = df[df.index.year < TEST_YEAR].copy()
    test  = df[df.index.year == TEST_YEAR].copy()

    if train.empty or test.empty:
        print(f"{ticker} train/test 為空，跳過 GA")
        continue
    best_ga_pattern, ga_scaler, ga_hist, ga_best_fit, ga_best_gen = run_ga(train)
    ga_hist.to_csv(os.path.join(out_dir, "ga_fitness_history.csv"), index=False)

    plt.figure(figsize=(8,5))
    plt.plot(ga_hist["generation"], ga_hist["best_fitness"], label="Best Fitness")
    plt.plot(ga_hist["generation"], ga_hist["avg_fitness"], label="Avg Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title(f"GA Fitness Curve - {ticker}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ga_fitness_curve.png"))
    plt.close()

    print(
        f"GA 收斂結果：best fitness = {ga_best_fit:.2f}, "
        f"出現在第 {ga_best_gen} 代"
    )

    with open(os.path.join(out_dir, "ga_convergence.txt"), "w") as f:
        f.write(f"best_fitness={ga_best_fit:.4f}\n")
        f.write(f"converged_generation={ga_best_gen}\n")

    ga_pattern_df = pd.DataFrame([best_ga_pattern], columns=FEATURES)
    ga_pattern_df.to_csv(os.path.join(out_dir, "GA_best_pattern.csv"), index=False)

    q1_path = os.path.join(out_dir, "Q1_three_methods_comparison.csv")
    best_method = select_best_method(q1_path)

    print(f"Q1 Best Method for {ticker}: {best_method}")

    ga_pattern_df = pd.DataFrame([best_ga_pattern], columns=FEATURES)

    ga_train_signal_df = generate_pattern_signals_daily(
        ga_pattern_df,
        pd.DataFrame(),     
        ga_scaler,
        FEATURES,
        train,             
        weights,
        threshold=GA_SIM_THRESHOLD
    )

    if ga_train_signal_df is None or len(ga_train_signal_df) == 0:
        print("GA 在訓練年找不到符合型態的日期，無法用最佳方法學習；改用 GA 本身做訊號")
        final_signals = generate_pattern_signals_daily(
            ga_pattern_df,
            pd.DataFrame(),
            ga_scaler,
            FEATURES,
            test,           
            weights,
            threshold=GA_SIM_THRESHOLD
        )
    else:
        ga_train_idx = ga_train_signal_df.index
        ga_train_subset = train.loc[ga_train_idx].copy()

        if best_method == "Method1_Pattern":
            final_signals = generate_pattern_signals_daily(
                ga_pattern_df,
                pd.DataFrame(),
                ga_scaler,
                FEATURES,
                test,            
                weights,
                threshold=GA_SIM_THRESHOLD
            )

        elif best_method == "Method2_AE_only":
            X_ga = scaler_ae.transform(ga_train_subset[FEATURES].values)
            recon = ae_up.predict(X_ga, verbose=0)
            ga_recon_err = np.mean((X_ga - recon) ** 2, axis=1)
            th_ga = float(np.quantile(ga_recon_err, 0.90))
            final_signals = generate_ae_recon_signals(
                test,
                ae_up=ae_up,
                ae_down=None,          
                th_up=th_ga,
                th_down=None,
                scaler_ae=scaler_ae,
                features=FEATURES
            )

        elif best_method == "Method3_SVM_only":
            latent_cols = ["latent1", "latent2", "latent3", "latent4"]

            X_ga = ga_train_subset[latent_cols].values
            svm_ga = OneClassSVM(nu=0.05, kernel="rbf", gamma="scale")
            svm_ga.fit(X_ga)

            X_test = test[latent_cols].values
            pred = svm_ga.predict(X_test)  
            final_signals = pd.DataFrame(index=test.index)
            final_signals["signal"] = np.where(pred == 1, 1, 0)  

        else:
            print(f"未知 best_method={best_method}，改用 Method1_Pattern")
            final_signals = generate_pattern_signals_daily(
                ga_pattern_df,
                pd.DataFrame(),
                ga_scaler,
                FEATURES,
                test,
                weights,
                threshold=GA_SIM_THRESHOLD
            )

    if final_signals is None or len(final_signals) == 0:
        ga_trades = []
    else:
        ga_trades = simple_backtest(final_signals, test)


    ga_ret, ga_wr, ga_dd = evaluate(ga_trades)

    print(f"GA Pattern trades: {len(ga_trades)}")
    if len(ga_trades) == 0:
        print("GA pattern 在 2025 未出現，屬於過往有效但測試年稀有型態")

    q1_path = os.path.join(out_dir, "Q1_three_methods_comparison.csv")
    q1_df = pd.read_csv(q1_path)

    q2_df = q1_df.copy()
    q2_df.loc[len(q2_df)] = {
        "method": "Method4_GA_Pattern",
        "return": ga_ret,
        "winrate": ga_wr,
        "maxdd": ga_dd
    }

    q2_df.to_csv(os.path.join(out_dir, "Q2_GA_vs_Q1_comparison.csv"), index=False)

print("題目二完成：Q2_GA_vs_Q1_comparison.csv 已輸出")

if summaries:
    pd.DataFrame(summaries).to_csv(os.path.join(RESULT_FOLDER, "all_run_summaries.csv"), index=False)
print("\n全部完成！請查看 stock_results 資料夾。")