import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans
from scipy.stats import skew, kurtosis
from scipy.linalg import solve
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def describe(df):
    df = df.select_dtypes(include='number').copy()
    mean = df.mean()
    var = df.var()
    std = df.std()
    q = df.quantile([.25, .5, .75])
    q1, q2, q3 = q.iloc[0, :], q.iloc[1, :], q.iloc[2, :]
    iqr = q3 - q1
    iqr_normal = (iqr / 1.349) / std
    bowley_skew = (q3 + q1 - 2 * q2) / (q3 - q1)
    skewness = pd.Series(skew(df, nan_policy='omit'), index=df.columns)
    kurt = pd.Series(kurtosis(df, nan_policy='omit'), index=df.columns)
    ps = np.arange(.1, .5, .1)
    q_low = df.quantile(ps)
    q_high = df.quantile(1 - ps)
    ratio = pd.DataFrame(
        (q2 - q_low).to_numpy() / (q_high - q2).to_numpy(), 
        columns=df.columns, 
        index=[f'Tail Assymetry {p:.1f}' for p in ps]
    )
    quantiles = df.quantile(np.arange(0, 1.1, .1))
    quantiles.index = [f'Quantile {idx:.1f}' for idx in quantiles.index]
    quantiles = quantiles.drop(index=['Quantile 0.5'])
    descriptive = pd.DataFrame(
        {
            'Mean': mean, 
            'Median': q2, 
            'Variance': var, 
            'Std': std, 
            'Variation Rate': std / mean, 
            'IQR': iqr, 
            'IQR Normal / Std': iqr_normal, 
            'Skewness': skewness, 
            'Skewness Bowley': bowley_skew, 
            'Kurtosis': kurt, 
        }, 
        index=df.columns
    )
    descriptive = pd.concat([descriptive.T, ratio, quantiles], axis=0)
    corr = {
        'Pearson': df.corr(), 
        'Spearman': df.corr(method='spearman')
    }
    return descriptive, corr

def group_binders_aggregates(df, ratios=False):
    y_col_loc = df.columns.get_loc(y_col)
    binder_ori = df[['Cement', 'Fly Ash', 'Blast Furnace Slag']].sum(axis=1)
    aggregates_ori = df[['Coarse Aggregate', 'Fine Aggregate']].sum(axis=1)
    others = df[['Water', 'Superplasticizer']].sum(axis=1)
    water_ratio = df['Water'] / binder_ori
    fine_aggregate_ratio = df['Fine Aggregate'] / aggregates_ori
    if ratios:
        binder = (
            binder_ori / (binder_ori + aggregates_ori + others)
        )
        aggregates = (
            aggregates_ori / (binder_ori + aggregates_ori + others)
        )
    else:
        binder = binder_ori
        aggregates = aggregates_ori
    df.insert(y_col_loc, 'Binder', binder)
    df.insert(y_col_loc + 1, 'Aggregates', aggregates)
    df.insert(y_col_loc + 2, 'Water Ratio', water_ratio)
    df.insert(y_col_loc + 3, 'Fine Aggregate Ratio', fine_aggregate_ratio)
    return df

def basic_model(
        df_train, columns, y, degree=1, df_cv=None, ridge=False, **kwargs
    ):
    alpha = kwargs.get('alpha', 1.0)
    if not ridge:
        pipe = Pipeline(
            [
             ('scaler', StandardScaler()), 
             ('poly', PolynomialFeatures(degree=degree, include_bias=False)), 
             ('reg', LinearRegression()), 
            ]
            )
    else:
        pipe = Pipeline(
            [
             ('scaler', StandardScaler()), 
             ('poly', PolynomialFeatures(degree=degree, include_bias=False)), 
             ('reg', Ridge(alpha=alpha)), 
            ]
            )
    pipe.fit(df_train[columns], df_train[y])
    pred_train = pipe.predict(df_train[columns])
    resids_train = df_train[y] - pred_train
    if df_cv is not None:
        score = pipe.score(df_cv[columns], df_cv[y])
        score = r2_adjusted_score(
            score, 
            df_cv.shape[0], 
            pipe['reg'].coef_.shape[0]
        )
        pred_cv = pipe.predict(df_cv[columns])
        resids_cv = df_cv[y] - pred_cv
        results = {
            'Pipe': pipe, 
            'Train Pred': pred_train, 
            'CV Pred': pred_cv, 
            'Train Resids':resids_train, 
            'CV Resids': resids_cv, 
            'Score':score
        }
    else:
        results = {
            'Pipe': pipe, 'Train Pred': pred_train, 'Train Resids':resids_train
        }
    return results

def r2_adjusted_score(r2_score, n, p):
    return 1 - (1 - r2_score) * (n - 1) / (n - p - 1)

def remove_linear_effects(df, y, cols, return_reg=False):
    if isinstance(cols, str):
        cols = [cols]
    model = LinearRegression().fit(df[cols], df[y])
    residuals = df[y] - model.predict(df[cols])
    if return_reg:
        return residuals + df[y].mean(), (model, df[y].mean())
    return residuals + df[y].mean()

def group_quantile(df):
    cols = df.columns[df.columns != 'Quantile Group']
    df_grouped = (
        df
        .groupby('Quantile Group', observed=False)
        [cols]
        .agg(['mean', 'std'])
    )
    return df_grouped

def interval_cumulative_distribution(
        df, 
        col1, 
        col2, 
        col3=None, 
        n=10, 
        dist='conditional', 
        return_bounds=True
    ):
    
    def bounds_and_mask(col, i):
        bounds = np.linspace(df[col].min(), df[col].max(), n[i])
        mask = df[col].to_numpy()[:, None] <= bounds
        return bounds, mask.astype(np.int64)
    
    if type(n) == int:
        n = [n] * 3
    col1_bounds, col1_mask = bounds_and_mask(col1, 0)
    col2_bounds, col2_mask = bounds_and_mask(col2, 1)
    bounds = [col1_bounds, col2_bounds]
    if col3 is not None:
        col3_bounds, col3_mask = bounds_and_mask(col3, 2)
        bounds.append(col3_bounds)
        joint = np.einsum(
            'ni,nj,nk->ijk', col1_mask, col2_mask, col3_mask, optimize=True
        )
        condition = col2_mask.T @ col3_mask
    else:
        joint = col1_mask.T @ col2_mask
        condition = col2_mask.sum(axis=0)
    if dist == 'conditional':
        div = condition
    elif dist == 'joint':
        div = df.shape[0]
    probs = np.divide(
        joint, 
        div, 
        out=np.zeros(joint.shape, dtype=float), 
        where=condition != 0
    )
    if return_bounds:
        return probs, bounds
    return probs

def evaluate_candidates(
        candidates, train, cv, columns, col, cumulative=False
    ):
    if cumulative:
        df_train = train.copy()
        df_cv = cv.copy()
        cols = columns.copy()
    for k in candidates:
        if not cumulative:
            df_train = train.copy()
            df_cv = cv.copy()
        train_fs = candidates[k](df_train, col)
        cv_fs = candidates[k](df_cv, col)
        for i, (train_f, cv_f) in enumerate(zip(train_fs, cv_fs)):
            if not cumulative:
                cols = columns.copy()
            name = f'{k} - {i}'
            cols.insert(0, name)
            df_train.insert(0, name, train_f)
            df_cv.insert(0, name, cv_f)
            r = basic_model(
                df_train, 
                cols[:-1], 
                y_col, 
                df_cv=df_cv
            )
            print(
                f"{name}: {r['Score']:.4f}, "
                f"{r['Pipe'].score(df_cv[cols[:-1]], df_cv[y_col]):.4f}"
            )
            for age in ['Age Log', 'Age Log Squared']:
                name_age = name + f' - {age}'
                cols.insert(0, name_age)
                df_train.insert(0, name_age, df_train[age] * train_f)
                df_cv.insert(0, name_age, df_cv[age] * cv_f)
                r = basic_model(
                    df_train, 
                    cols[:-1], 
                    y_col, 
                    df_cv=df_cv
                )
                print(
                    f"{name_age}: {r['Score']:.4f}, "
                    f"{r['Pipe'].score(df_cv[cols[:-1]], df_cv[y_col]):.4f}", 
                )
            print() 

def filter_cement_pure(df):
    cement_pure = (
        df[
            (df[['Blast Furnace Slag', 'Fly Ash']].sum(axis=1) == 0)
            & (df['Superplasticizer'] == 0)
        ]
        .drop(columns=['Blast Furnace Slag', 'Fly Ash', 'Superplasticizer'])
    )
    return cement_pure

def filter_superplasticizer_out(df):
    df = df[df['Superplasticizer'] == 0].drop(columns=['Superplasticizer'])
    return df

def engineer_cement_pure_features(df):
    df = df.copy()
    cement_water_ratio = df['Cement'] / df['Water']
    log_age = np.log(df['Age'])
    log_age_squared = log_age ** 2
    log_age_water_ratio = log_age * cement_water_ratio
    df.insert(
        df.shape[1] - 1, 
        'Cement to Water Ratio', 
        cement_water_ratio
    )
    df.insert(
        df.shape[1] - 1, 
        'Log Age', 
        log_age
    )
    df.insert(
        df.shape[1] - 1, 
        'Log Age Squared', 
        log_age_squared
    )
    df.insert(
        df.shape[1] - 1, 
        'Cement to Water Ratio x Log Age', 
        log_age_water_ratio
    )
    return df

def engineer_without_superplasticizer(df):
    df = df.copy()
    binder_cols = ['Cement', 'Blast Furnace Slag']
    binder_ratios = df[binder_cols].to_numpy()/ df[['Binder']].to_numpy()
    aggregate_to_binder_ratio = df['Aggregates'] / df['Binder']
    age_log = np.log(df['Age'])
    age_log_squared = age_log ** 2
    for col, i in zip(binder_cols, range(len(binder_cols))):
        df.insert(df.shape[1] - 1, f'{col} Ratio', binder_ratios[:, i])
    df.insert(
        df.shape[1] - 1, 'Aggregate to Binder Ratio', aggregate_to_binder_ratio
    )
    df.insert(df.shape[1] - 1, 'Age Log', age_log)
    df.insert(df.shape[1] - 1, 'Age Log Squared', age_log_squared)
    return df

def engineer_without_superplasticizer_2(df):
    df = df.copy()
    columns = [
        'Fine Aggregate x Water', 
        'Water Squared', 
        'Cement Squared', 
        'Cement Ratio', 
        'Aggregate to Binder Ratio', 
        'Water Ratio Age Log', 
        'Water Ratio Age Log Squared', 
        'Cement Ratio Age Log', 
        'Cement Ratio Age Log Squared', 
        'Age Log', 
        'Age Log Squared'
    ]
    columns_it = iter(columns)
    fine_agg_water = df['Fine Aggregate'] * df['Water']
    water_squared = df['Water'] ** 2
    cement_square = df['Cement'] ** 2
    cement_ratio = df['Cement'] / df['Binder']
    aggregate_to_binder_ratio = df['Aggregates'] / df['Binder']
    age_log = np.log(df['Age'])
    age_log_squared = age_log ** 2
    water_ratio_age_log = df['Water Ratio'] * age_log
    water_ratio_age_log_squared = df['Water Ratio'] * age_log_squared
    cement_ratio_age_log = cement_ratio * age_log
    cement_ratio_age_log_squared = cement_ratio * age_log_squared
    df.insert(df.shape[1] - 1, next(columns_it), fine_agg_water)
    df.insert(df.shape[1] - 1, next(columns_it), water_squared)
    df.insert(df.shape[1] - 1, next(columns_it), cement_square)
    df.insert(df.shape[1] - 1, next(columns_it), cement_ratio)
    df.insert(df.shape[1] - 1, next(columns_it), aggregate_to_binder_ratio)
    df.insert(df.shape[1] - 1, next(columns_it), water_ratio_age_log)
    df.insert(df.shape[1] - 1, next(columns_it), water_ratio_age_log_squared)
    df.insert(df.shape[1] - 1, next(columns_it), cement_ratio_age_log)
    df.insert(df.shape[1] - 1, next(columns_it), cement_ratio_age_log_squared)
    df.insert(df.shape[1] - 1, next(columns_it), age_log)
    df.insert(df.shape[1] - 1, next(columns_it), age_log_squared)
    return df, columns

def engineer_all_mixtures(df):
    df = df.copy()
    columns = [
        'Binder to Ratio Ratio', 
        'Superplasticizer Present', 
        'Superplasticizer x Water', 
        'Superplasticizer x Water Squared', 
        'Fine Aggregate x Water', 
        'Blast Furnace Slag x Water', 
        'Water Squared', 
        'Cement Squared', 
        'Cement Ratio', 
        'Fly Ash Ratio', 
        'Aggregate to Binder Ratio', 
        'Water Ratio Age Log', 
        'Water Ratio Age Log Squared', 
        'Cement Ratio Age Log', 
        'Cement Ratio Age Log Squared', 
        'Fly Ash Ratio Age Log', 
        'Fly Ash Ratio Age Log Squared', 
        'Age Log', 
        'Age Log Squared'
    ]
    columns_it = iter(columns)
    binder_water_ratio = df['Binder'] / df['Water']
    sp = (df['Superplasticizer'] > 0).astype(int)
    sp_water = df['Superplasticizer'] * df['Water Ratio']
    sp_water_sqr = sp_water ** 2
    fine_agg_water = df['Fine Aggregate'] * df['Water']
    bfs_water = df['Blast Furnace Slag'] * df['Water']
    water_squared = df['Water'] ** 2
    cement_square = df['Cement'] ** 2
    cement_ratio = df['Cement'] / df['Binder']
    fly_ash_ratio = df['Fly Ash'] / df['Binder']
    aggregate_to_binder_ratio = df['Aggregates'] / df['Binder']
    age_log = np.log(df['Age'])
    age_log_squared = age_log ** 2
    water_ratio_age_log = df['Water Ratio'] * age_log
    water_ratio_age_log_squared = df['Water Ratio'] * age_log_squared
    cement_ratio_age_log = cement_ratio * age_log
    cement_ratio_age_log_squared = cement_ratio * age_log_squared
    fly_ash_age_log = fly_ash_ratio * age_log
    fly_ash_age_log_squared = fly_ash_ratio * age_log_squared
    df.insert(df.shape[1] - 1, next(columns_it), binder_water_ratio)
    df.insert(df.shape[1] - 1, next(columns_it), sp)
    df.insert(df.shape[1] - 1, next(columns_it), sp_water)
    df.insert(df.shape[1] - 1, next(columns_it), sp_water_sqr)
    df.insert(df.shape[1] - 1, next(columns_it), fine_agg_water)
    df.insert(df.shape[1] - 1, next(columns_it), bfs_water)
    df.insert(df.shape[1] - 1, next(columns_it), water_squared)
    df.insert(df.shape[1] - 1, next(columns_it), cement_square)
    df.insert(df.shape[1] - 1, next(columns_it), cement_ratio)
    df.insert(df.shape[1] - 1, next(columns_it), fly_ash_ratio)
    df.insert(df.shape[1] - 1, next(columns_it), aggregate_to_binder_ratio)
    df.insert(df.shape[1] - 1, next(columns_it), water_ratio_age_log)
    df.insert(df.shape[1] - 1, next(columns_it), water_ratio_age_log_squared)
    df.insert(df.shape[1] - 1, next(columns_it), cement_ratio_age_log)
    df.insert(df.shape[1] - 1, next(columns_it), cement_ratio_age_log_squared)
    df.insert(df.shape[1] - 1, next(columns_it), fly_ash_age_log)
    df.insert(df.shape[1] - 1, next(columns_it), fly_ash_age_log_squared)
    df.insert(df.shape[1] - 1, next(columns_it), age_log)
    df.insert(df.shape[1] - 1, next(columns_it), age_log_squared)
    return df, columns

def engineer_all_mixtures_2(df):
    df = df.copy()
    columns = [
        'Fine Aggregate x Cement x Water', 
        'Fine Aggregate Log', 
        'Superplasticizer Squared', 
        'Aggregate to Binder Ratio', 
        'Superplasticizer x Water', 
        'Cement Squared', 
        'Fly Ash Squared', 
        'Blast Furnace Slag Squared', 
        'Blast Furnace Slag Ratio Squared', 
        'Water Squared', 
        'Binder to Water Ratio', 
        'Binder Age Log', 
        'Binder Age Log Squared', 
        'Slag Age Log', 
        'Slga Age Log Squared', 
        'Ash Ratio Age Log', 
        'Ash Ratio Age Log Squared', 
        'Aggregate to Binder Ratio Age Log', 
        'Aggregate to Binder Ratio Age Log Squared', 
        'Age Log', 
        'Age Log Squared'
    ]
    columns_it = iter(columns)
    fa_cem_w = df[['Fine Aggregate', 'Cement', 'Water']].prod(axis=1)
    fa_log = np.log(df['Fine Aggregate'])
    sp_sqr = df['Superplasticizer'] ** 2
    agg_to_bind_ratio = df['Aggregates'] / df['Binder']
    sp_water = df['Superplasticizer'] * df['Water']
    cement_sqr = df['Cement'] ** 2
    fly_ash_sqr = df['Fly Ash'] ** 2
    slag_sqr = df['Blast Furnace Slag'] ** 2
    slag_ratio_sqr = (df['Blast Furnace Slag'] / df['Binder']) ** 2
    water_sqr = df['Water'] ** 2
    binder_water = df['Binder'] / df['Water']
    age_log = np.log(df['Age'])
    age_log_sqr = age_log ** 2
    binder_age_log = df['Binder'] * age_log
    binder_age_log_sqr = df['Binder'] * age_log_sqr
    slag_age_log = df['Blast Furnace Slag'] * age_log
    slag_age_log_sqr = df['Blast Furnace Slag'] * age_log_sqr
    ash_ratio_age_log = df['Fly Ash'] / df['Binder'] * age_log
    ash_ratio_age_log_sqr = df['Fly Ash'] / df['Binder'] * age_log_sqr
    agg_bind_ratio_age_log = agg_to_bind_ratio * age_log
    agg_bind_ratio_age_log_sqr = agg_to_bind_ratio * age_log_sqr
    df.insert(df.shape[1] - 1, next(columns_it), fa_cem_w)
    df.insert(df.shape[1] - 1, next(columns_it), fa_log)
    df.insert(df.shape[1] - 1, next(columns_it), sp_sqr)
    df.insert(df.shape[1] - 1, next(columns_it), agg_to_bind_ratio)
    df.insert(df.shape[1] - 1, next(columns_it), sp_water)
    df.insert(df.shape[1] - 1, next(columns_it), cement_sqr)
    df.insert(df.shape[1] - 1, next(columns_it), fly_ash_sqr)
    df.insert(df.shape[1] - 1, next(columns_it), slag_sqr)
    df.insert(df.shape[1] - 1, next(columns_it), slag_ratio_sqr)
    df.insert(df.shape[1] - 1, next(columns_it), water_sqr)
    df.insert(df.shape[1] - 1, next(columns_it), binder_water)
    df.insert(df.shape[1] - 1, next(columns_it), binder_age_log)
    df.insert(df.shape[1] - 1, next(columns_it), binder_age_log_sqr)
    df.insert(df.shape[1] - 1, next(columns_it), slag_age_log)
    df.insert(df.shape[1] - 1, next(columns_it), slag_age_log_sqr)
    df.insert(df.shape[1] - 1, next(columns_it), ash_ratio_age_log)
    df.insert(df.shape[1] - 1, next(columns_it), ash_ratio_age_log_sqr)
    df.insert(df.shape[1] - 1, next(columns_it), agg_bind_ratio_age_log)
    df.insert(df.shape[1] - 1, next(columns_it), agg_bind_ratio_age_log_sqr)
    df.insert(df.shape[1] - 1, next(columns_it), age_log)
    df.insert(df.shape[1] - 1, next(columns_it), age_log_sqr)
    return df, columns

def plot_pred_and_resids(df, pred, y=None, resids=None, x=None, title=None):
    if y is None:
        y = y_col
    if x is None:
        x = range(df.shape[0])
    if resids is None:
        resids = df[y] - pred
    fig, axs = plt.subplots(2, 1, figsize=(20, 25), constrained_layout=True)
    if title is not None:
        plt.suptitle(title, fontsize=14, weight='bold')
    for i in range(2):
        axs[i].grid(True, color='k', linestyle='--', alpha=.2)
    axs[0].plot(x, df[y], label='Original')
    axs[0].plot(
        x, 
        pred, 
        color='r', 
        linestyle='-.', 
        alpha=.5, 
        label='Predicted'
    )
    axs[1].scatter(x, resids, label='Resids')
    axs[1].axhline(0, color='r', linestyle='-.', alpha=.5)
    axs[0].legend()
    axs[1].legend()
    plt.show()

def plot_features_and_resids(df, resids, title=None):
    if y_col in df.columns:
        df = df.drop(columns=[y_col])
    else:
        df = df.copy()
    cols = int(round(df.shape[1] ** .5, 0))
    rows = df.shape[1] // cols
    if rows * cols < df.shape[1]:
        rows += 1
    fig, axs = plt.subplots(
        rows, cols, figsize=(cols * 10, rows * 10), constrained_layout=True
    )
    if title is not None:
        plt.suptitle(title, fontsize=14, weight='bold')
    for i in range(rows * cols):
        m = i // cols
        n = i % cols
        try:
            col = df.columns[i]
        except:
            axs[m, n].set_visible(False)
            continue
        axs[m, n].grid(True, color='k', linestyle='--', alpha=.2)
        axs[m, n].scatter(df[col], resids)
        axs[m, n].set_title(col)
    plt.show()

def scatter_y_col(df, y=None, title=None):
    if y is None:    
        y = df[y_col]
    else:
        y = df[y]
    df = df.drop(columns=[y.name])
    cols = int(round(df.shape[1] ** .5, 0))
    rows = df.shape[1] // cols
    if rows * cols < df.shape[1]:
        rows += 1
    fig, axs = plt.subplots(
        rows, cols, figsize=(cols * 10, rows * 10), constrained_layout=True
    )
    if title is not None:
        plt.suptitle(title, fontsize=14, weight='bold')
    for i in range(rows * cols):
        m = i // cols
        n = i % cols
        try:
            col = df.columns[i]
        except:
            axs[m, n].set_visible(False)
            continue
        axs[m, n].grid(True, color='k', linestyle='--', alpha=.2)
        axs[m, n].scatter(df[col], y)
        axs[m, n].set_title(col)
    plt.show()

def combinatorial_scatter_plot(df, columns, title=None):
    size = np.arange(len(columns), len(columns) - 2, -1).prod() // 2
    if size >= 4:
        cols = int(round(size ** .5, 0))
        rows = size // cols
        if rows * cols < size:
            rows += 1
    else:
        rows, cols = 1, size
    completed = []
    fig, axs = plt.subplots(
        rows, 
        cols, 
        figsize=(cols * 10, rows * 10), 
        constrained_layout=True, 
        squeeze=False
    )
    if title is not None:
        plt.suptitle(title, fontsize=14, weight='bold')
    for col1 in columns:
        for col2 in columns:
            grouped = sorted([col1, col2])
            if (grouped in completed) or (col1 == col2):
                continue
            m = len(completed) // cols
            n = len(completed) % cols
            axs[m, n].grid(True, color='k', linestyle='--', alpha=.2)
            axs[m, n].scatter(df[col1], df[col2])
            axs[m, n].set_xlabel(col1)
            axs[m, n].set_ylabel(col2)
            axs[m, n].set_title(f'{col1} x {col2}')
            completed.append(grouped)
    plt.show()

def plot_conditional_ecdf(df, col_groups, title_header=''):
    if len(title_header) > 0:
        title_header += ' '
    groups = np.sort(df[col_groups].unique())
    size = groups.shape[0]
    columns = df.select_dtypes('number').columns
    features = columns[columns != col_groups]
    cols = int(round(size ** .5, 0))
    rows = size // cols
    yticks = np.linspace(0, 1, num=11)
    if rows * cols < size:
        rows += 1
    for col in features:
        xticks = np.linspace(df[col].min(), df[col].max(), num=11)
        fig, axs = plt.subplots(
            rows, 
            cols, 
            figsize=(cols * 10, rows * 10), 
            constrained_layout=True, 
            squeeze=False
        )
        plt.suptitle(
            f'{title_header}Conditional ECDF - {col}', 
            fontsize=14, 
            weight='bold'
        )
        for i in range(rows * cols):
            m, n = divmod(i, cols)
            try:
                n_group = groups[i]
            except:
                axs[m, n].set_visible(False)
                continue
            axs[m, n].grid(True, color='k', linestyle='--', alpha=.2)
            axs[m, n].ecdf(
                df.loc[df[col_groups] == n_group, col], label='ECDF'
            )
            axs[m, n].ecdf(
                df[col], 
                linestyle='--', 
                color='r', 
                alpha=.7, 
                label='Baseline'
            )
            axs[m, n].set_xticks(xticks)
            axs[m, n].set_yticks(yticks)
            axs[m, n].set_title(f'{col} - Group {n_group}')
            axs[m, n].legend()
        plt.show()

def plot_resids_by_features_quantiles(df, col_resids):
    features = df.columns[df.columns != col_resids]
    cols = int(round(features.shape[0] ** .5, 0))
    rows = features.shape[0] // cols
    if cols * rows < features.shape[0]:
        rows += 1
    fig, axs = plt.subplots(
        rows, cols, figsize=(cols * 10, rows * 10), sharey=True
    )
    for i in range(rows * cols):
        m, n = divmod(i, cols)
        try:
            col = features[i]
        except:
            axs[m, n].set_visible(False)
            continue
        axs[m, n].grid(True, color='k', linestyle='--', alpha=.2)
        group = (
            df
            .groupby(
                pd.qcut(df[col], 10, duplicates='drop'), observed=False
            )
            [col_resids]
            .mean()
        )
        axs[m, n].plot(range(group.shape[0]), group)
        axs[m, n].axhline(0, color='r', linestyle='-.', alpha=.7)
        axs[m, n].set_xticks(range(group.shape[0]))
        axs[m, n].set_xticklabels(group.index, rotation=45)
        axs[m, n].set_title(col)
    plt.show()

class GaussianRBF(RegressorMixin, BaseEstimator):
    
    def __init__(self, pipe, columns, alpha=1., gamma=.3):
        self.pipe = pipe
        self.columns = columns
        self.alpha = alpha
        self.gamma = gamma
    
    def raw_features(self, df):
        cols = train_original.columns[:-1].to_list()
        x = df[cols].astype(float).copy()
        x['Age'] = np.log(x['Age'])
        return x
    
    def fit(self, df, y):
        self.linear = clone(self.pipe)
        self.linear.fit(df[self.columns], y)
        lin_feat_norm = self.linear[:-1].transform(df[self.columns])
        T = np.column_stack([np.ones(df.shape[0]), lin_feat_norm])
        self.scaler_raw = StandardScaler()
        self.raw_feat_norm = self.scaler_raw.fit_transform(
            self.raw_features(df)
        )
        kernel = rbf_kernel(self.raw_feat_norm, gamma=self.gamma)
        A = kernel + self.alpha * np.eye(df.shape[0])
        solved = solve(
            A, 
            np.column_stack([T, np.asarray(y)]), 
            assume_a='pos'
        )
        AiT = solved[:, :-1]
        Aiy = solved[:, -1]
        beta = (
            np.linalg.lstsq(
                T.T @ AiT, 
                T.T @ Aiy, 
                rcond=None
            )
            [0]
        )
        self.dual_ = Aiy - AiT @ beta
        self.linear['reg'].intercept_ = beta[0]
        self.linear['reg'].coef_ = beta[1:]
        return self
    
    def predict(self, df):
        raw_feat_norm = self.scaler_raw.transform(self.raw_features(df))
        correction = (
            rbf_kernel(raw_feat_norm, self.raw_feat_norm, gamma=self.gamma)
            @ self.dual_
        )
        return self.linear.predict(df[self.columns]) + correction

colors = list(mcolors.CSS4_COLORS.keys())
colors = [color for color in colors if 'dark' in color]

# Dataset loading and columns adjustments
base_directory = 'D:\\Alexandre\\Python\\Datasets\\Concrete'
concrete_directory = os.path.join(base_directory, 'Concrete_Data.xls')
concrete = pd.read_excel(concrete_directory)
concrete.columns = concrete.columns.str.extract(r'^(.*)\s\(')[0].str.strip()
y_col = 'Concrete compressive strength'

# Train, CV, Test split based on groups that vary only age
cols = concrete.columns.drop(['Age', y_col]).tolist()
groups = (
    concrete.groupby(cols, dropna=False)
    .ngroup()
    .to_numpy()
)

split_test = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)
rest_idx, test_idx = next(
    split_test.split(concrete, groups=groups)
)

split_cv = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=43
)
train_idx, cv_idx = next(
    split_cv.split(
        concrete.iloc[rest_idx],
        groups=groups[rest_idx]
    )
)
train = concrete.iloc[rest_idx[train_idx]].copy()
cv = concrete.iloc[rest_idx[cv_idx]].copy()
test = concrete.iloc[test_idx].copy()
train_original = train.copy()
train_ratios = train.copy()
train_ratios.iloc[:, :-2] = train_ratios.iloc[:, :-2].div(
    train_ratios.iloc[:, :-2].sum(axis=1),axis=0
)
cv_ratios = cv.copy()
cv_ratios.iloc[:, :-2] = cv_ratios.iloc[:, :-2].div(
    cv_ratios.iloc[:, :-2].sum(axis=1),axis=0
)

# Basic preliminary analysis
train = group_binders_aggregates(train)
cv = group_binders_aggregates(cv)
descriptive, corr = describe(train)
cov = train.cov()

# Raw variables linear baseline and plot
linear_baseline = basic_model(
    train_original, train_original.columns[:-1], y_col, df_cv=cv
)
linear_ratio_baseline = basic_model(
    train_ratios, train_ratios.columns[:-1], y_col, df_cv=cv_ratios
)
print(f"Raw Data Linear Baseline Score: {linear_baseline['Score']:.4f}")
print(
      'Raw Data Linear Ratio Baseline Score: '
      f"{linear_ratio_baseline['Score']:.4f}"
  )

order = np.argsort(cv[y_col].to_numpy())
plot_pred_and_resids(
    cv.iloc[order], 
    linear_baseline['CV Pred'][order], 
    resids=linear_baseline['CV Resids'].iloc[order], 
    title='Raw Data Linear Baseline - CV Pred and Resids'
)
plot_features_and_resids(
    cv, 
    linear_baseline['CV Resids'], 
    title='Raw Data Linear Baseline CV Resids by Features'
)
plot_pred_and_resids(
    cv_ratios.iloc[order], 
    linear_ratio_baseline['CV Pred'][order], 
    title='Raw Data Linear Ratio Baseline - CV Pred and Resids'
)
plot_features_and_resids(
    cv_ratios, 
    linear_ratio_baseline['CV Resids'], 
    title='Raw Data Linear Ratio Baseline CV Resids by Features'
)

# Theory
# Cement and Water produces C-S-H, which influences binding strength
# Water allows for mixing, placing, consolidation and curing
# Excess water to binder ratio reduces strength
# Fly Ash slows early strength development while improving later development
# Blast Furnace Slag has latent hydraulic activity, which promotes cement 
# reaction with water and forms binding products, with a slower reaction.
# Aggregates ratio must allow for coating by the paste.
# Superplasticizer improves the mixture's flow, reducing water needs while
# maintaining similar workability.
# Binders = Cement + Fly Ash + Blast Furnace Slag
# Water: excess reduces strength.
# Water Proportion: Water / Binders
# Fly Ash: Greater strength with higher age requirement.
# Blast Furnace Slag: Lower water need, increased age requirement.
# Fine Aggregate: Greater paste requirement.
# Coarse Aggregate: Reduces paste requirement.
# Fine Aggregate Proportion: Fine Aggregate / Aggregates
# Superplasticizer: lower water requirement.
   
cement_pure = filter_cement_pure(train)
water_ratio_squared = (
    cement_pure['Water Ratio'] ** 2
)
cement_pure.insert(
    cement_pure.shape[1] - 1, 
    'Water Ratio Squared', 
    water_ratio_squared
)
cement_pure_age_controlled = cement_pure.drop(columns=['Binder', 'Age'])
regressors = {}
for col in cement_pure.columns:
    if col in ['Binder', 'Age']:
        continue
    col_controlled, reg = remove_linear_effects(
        cement_pure, col, 'Age', return_reg=True
    )
    cement_pure_age_controlled[col] = col_controlled
    regressors[col] = reg
    
scatter_y_col(
    cement_pure, y=y_col, title='Pure Cement Mixtures x Strength'
)
scatter_y_col(
    cement_pure_age_controlled, 
    y=y_col, 
    title='Pure Cement Mixtures x Strength Controlled by Age'
)

corr_cement_pure_age_controlled = cement_pure_age_controlled.corr()

# There's a clear relationship between the amount of Cement and its 
# corresponding Strength, with the inverse association observed with
# Water, Water Ratio and Fine Aggregate. However, Fine Aggregate displays
# a high absolute correlation with Cement and Water Ratio.

combinatorial_scatter_plot(
    cement_pure_age_controlled, 
    ['Fine Aggregate', 'Cement', 'Water Ratio'], 
    title='Pure Cement Mixtures Combinatorial Scatter Plots'
)

controls = ['Cement', 'Water Ratio', 'Water Ratio Squared']

fine_aggregate_controlled = remove_linear_effects(
    cement_pure_age_controlled, 'Fine Aggregate', controls)
cement_pure_age_controlled.insert(
    cement_pure_age_controlled.shape[1] - 1, 
    'Fine Aggregate Controlled', 
    fine_aggregate_controlled
)
cement_pure_age_controlled['Strength Controlled'] = remove_linear_effects(
    cement_pure_age_controlled, y_col, controls)

fig, ax = plt.subplots(1, 1, figsize=(10, 10), constrained_layout=True)
plt.suptitle(
    'Fine Aggregate and Strength Controlled by Age, Cement and Water Ratio', 
    fontsize=14, 
    weight='bold'
)
ax.grid(True, color='k', linestyle='--', alpha=.2)
ax.scatter(
    cement_pure_age_controlled['Fine Aggregate Controlled'], 
    cement_pure_age_controlled['Strength Controlled']
)
plt.show()

corr_cement_pure_age_controlled = cement_pure_age_controlled.corr()

# As expected, there was a change in the relationship between
# Fine Aggregate and Strength, although it still exhibits very low correlation.

combinatorial_scatter_plot(
    cement_pure_age_controlled, 
    ['Coarse Aggregate', 'Fine Aggregate', 'Cement', 'Water Ratio'], 
    title='Pure Cement Mixtures Combinatorial Scatter Plots'
)

coarse_aggregate_controlled = remove_linear_effects(
    cement_pure_age_controlled, 'Coarse Aggregate', ['Cement', 'Water Ratio'])
cement_pure_age_controlled.insert(
    cement_pure_age_controlled.shape[1] - 2, 
    'Coarse Aggregate Controlled', 
    coarse_aggregate_controlled
)
cement_pure_age_controlled['Strength Lienarly Controlled'] = (
    remove_linear_effects(cement_pure_age_controlled, y_col, controls[:-1])
)
corr_cement_pure_age_controlled = cement_pure_age_controlled.corr()

# Cement Pure Mixtures Linear Regression
columns = [
    'Water Ratio', 
    'Aggregates', 
    y_col
]

cv_cement_pure_age_controlled = (
    cv[
       (cv[['Blast Furnace Slag', 'Fly Ash']].sum(axis=1) == 0)
       & (cv['Superplasticizer'] == 0)
   ]
   .copy()
)
cv_cement_pure_age_controlled = cv_cement_pure_age_controlled[
    columns + ['Age']
]
for col in columns:
    reg, mean = regressors[col]
    cv_cement_pure_age_controlled[col] = (
        cv_cement_pure_age_controlled[col] 
        - reg.predict(cv_cement_pure_age_controlled[['Age']]) 
        + mean
    )

linear_cement_pure = basic_model(
    cement_pure_age_controlled, 
    columns[:-1], 
    y_col, 
    df_cv=cv_cement_pure_age_controlled
)
print(f"Cement Pure Score: {linear_cement_pure['Score']:.4f}")
order = np.argsort(cv_cement_pure_age_controlled[y_col].to_numpy())
plot_pred_and_resids(
    cv_cement_pure_age_controlled.iloc[order, :-1], 
    linear_cement_pure['CV Pred'][order], 
    title='Cement Pure Linear Model - CV Pred and Resids'
)
plot_features_and_resids(
    cv_cement_pure_age_controlled, 
    linear_cement_pure['CV Resids'], 
    title='Cement Pure Linear Model CV Resids by Features'
)

# Cement Pure Non Linear Regression
columns = [
    'Cement to Water Ratio', 
    'Log Age', 
    'Log Age Squared', 
    'Cement to Water Ratio x Log Age', 
    y_col
]
cement_pure = engineer_cement_pure_features(filter_cement_pure(train))
cv_cement_pure = engineer_cement_pure_features(filter_cement_pure(cv))
non_linear_cement_pure = basic_model(
    cement_pure, 
    columns[:-1], 
    y_col, 
    df_cv=cv_cement_pure
)
print(
      "Non Linear Cement Pure Score: "
      f"{non_linear_cement_pure['Score']:.4f}"
)
order = np.argsort(cv_cement_pure[y_col].to_numpy())
plot_pred_and_resids(
    cv_cement_pure.iloc[order], 
    non_linear_cement_pure['CV Pred'][order], 
    title='Cement Pure Non Linear Model - CV Pred and Resids'
)
plot_features_and_resids(
    cv_cement_pure, 
    non_linear_cement_pure['CV Resids'], 
    title='Cement Pure Non Linear Model CV Resids by Features'
)
non_linear_cement_pure_coefs = pd.Series(
    non_linear_cement_pure['Pipe']['reg'].coef_, 
    index=columns[:-1]
)
non_linear_cement_pure_coefs = pd.concat(
    [
     pd.Series(
         non_linear_cement_pure['Pipe']['reg'].intercept_, index=['Bias']
     ), 
     non_linear_cement_pure_coefs
    ]
)

# Binder Mixtures without Superplasticizer
train_wo_sp = filter_superplasticizer_out(train)
cv_wo_sp = filter_superplasticizer_out(cv)

train_wo_sp = engineer_without_superplasticizer(train_wo_sp)
cv_wo_sp = engineer_without_superplasticizer(cv_wo_sp)

columns = [
    'Cement Ratio',
    'Blast Furnace Slag Ratio', 
    'Water Ratio', 
    'Fine Aggregate Ratio', 
    'Aggregate to Binder Ratio', 
    'Age Log', 
    'Age Log Squared', 
    'Concrete compressive strength'
]

linear_wo_sp = basic_model(
    train_wo_sp, 
    columns[:-1], 
    y_col, 
    df_cv=cv_wo_sp
)
print(f"No Superplasticizer Mixture Score: {linear_wo_sp['Score']:.4f}")
order = np.argsort(cv_wo_sp[y_col].to_numpy())
plot_pred_and_resids(
    cv_wo_sp.iloc[order], 
    linear_wo_sp['CV Pred'][order], 
    title='No Superplasticizer Non Linear Model - CV Pred and Resids'
)
plot_features_and_resids(
    cv_wo_sp, 
    linear_wo_sp['CV Resids'], 
    title='No Superplasticier Non Linear Model CV Resids by Features'
)
cv_wo_sp['CV Resids'] = linear_wo_sp['CV Resids']
cv_wo_sp['CV Resids Squared'] = cv_wo_sp['CV Resids'] ** 2
cv_wo_sp['Label'] = pd.qcut(cv_wo_sp['CV Resids'], 10)
cv_wo_sp['Label Squared'] = pd.qcut(
    cv_wo_sp['CV Resids Squared'], 10, labels=False
)

plot_conditional_ecdf(
    cv_wo_sp.loc[:, ~cv_wo_sp.columns.str.contains('Resids')], 
    'Label Squared', 
    title_header='Model Without Superplasticizer CV Squared Residuals'
)
plot_conditional_ecdf(
    cv_wo_sp.loc[
        :, ~cv_wo_sp.columns.str.contains(
            r'(?:Resids|^Label Squared$)', regex=True
        )
    ], 
    'Label', 
    title_header='Model Without Superplasticizer CV Residuals'
)

# Squared Residuals Grouping
# Water: Groups with larger squared residuals concentrated on higher values.
# Fine Aggregate: Groups with larger squared residuals concentrated on 
# lower values.
# Age: Groups with smaller squared residuals concentrated on lower values.
# Binder: Groups with larger squared residuals with slower accumulation.
# Aggregates: Groups with larger squared residuals with faster accumulation.
# Water Ratio: Groups with larger squared residuals with repeated
# accumulations around 0.5884.
# Fine Aggregate Ratio: Groups with larger squared residuals with repeated
# accumulations around 0.4105.
# Cement Ratio: Groups with larger squared residuals with repeated
# accumulations around 0.82.
# Blast Furnace Slag Ratio: Groups with larger squared residuals with repeated
# accumulations around 0.2.
# Aggregate to Binder Ratio: Groups with larger squared residuals with
# repeated accumulations around 4.
# Age Log: Groups with smaller squared residuals with faster accumulation.
# Age Log Squared: Groups with smaller squared residuals with faster 
# accumulation.
# Concrete Compressive Strength: Groups with larger squared residuals with
# slower accumulation.

# Residuals Grouping
# Water: Over- and underprediction accumulation slower in both extremes.
# Coarse Aggregate: Overpredictions are mainly associated with slower 
# accumulations, while underpredictions with faster ones.
# Fine Aggregate: Extremes residuals are mainly associated with slower 
# accumulations.
# Age: Extremes residuals, specially at the higher end, are mainly
# associated with slower accumulations.
# Binder: Exreme residuals are mainly associated with slower accumulations.
# Aggregates: Extreme residuals are mainly associated with faster accumlations.
# Water Ratio: Overpredictions are usually associated with faster 
# accumulations, while underpredictions accumulate rapidly around 0.6.
# Fine Aggregate Ratio: Extreme overprediction associated with faster
# accumulation, while extreme underprediction accumulates rapidly around 0.41.
# Cement Ratio: Largest overprediction group with values varying only 
# between 0.4 and 1.0, while mid overprediction groups display slower 
# accumulations and mid underprediction groups faster ones.
# Blast Furnace Slag Ratio: Largest overprediction group with values varying 
# only between 0. and 0.6, while mid overprediction groups display faster 
# accumulations and mid underprediction groups slower ones.
# Aggregate to Binder Ratio: Extreme preditions accumulate faster.
# Age Log: Extremes residuals, specially at the higher end, are mainly
# associated with slower accumulations.
# Age Log Squared: Groups near 0 residuals tend to accumulate faster, without
# accounting for the higher feature values.
# Concrete Compressive Strength: Extreme underpredictions tend to accumulate
# slower, while the other groups accumulate faster.

train_wo_sp = filter_superplasticizer_out(train)
cv_wo_sp = filter_superplasticizer_out(cv)

train_wo_sp, columns = engineer_without_superplasticizer_2(train_wo_sp)
cv_wo_sp, _ = engineer_without_superplasticizer_2(cv_wo_sp)

columns.insert(0, 'Water')
columns.insert(0, 'Water Ratio')
columns.insert(2, 'Fine Aggregate')
columns.insert(2, 'Fine Aggregate Ratio')
columns += [y_col]

model_wo_sp = basic_model(
    train_wo_sp, 
    columns[:-1], 
    y_col, 
    df_cv=cv_wo_sp
)
print(
      'No Superplasticizer Mixture Enhanced Score: '
      f"{model_wo_sp['Score']:.4f}"
)

train_wo_sp = train_wo_sp.drop(columns=['Fly Ash'])
train_wo_sp['Residuals'] = model_wo_sp['Train Resids']
plot_resids_by_features_quantiles(train_wo_sp, 'Residuals')

# All Mixtures
train_all, columns = engineer_all_mixtures(train)
cv_all, _ = engineer_all_mixtures(cv)

columns.insert(0, 'Blast Furnace Slag')
columns.insert(0, 'Water')
columns.insert(0, 'Water Ratio')
columns.insert(2, 'Fine Aggregate')
columns.insert(2, 'Fine Aggregate Ratio')
columns.insert(4, 'Superplasticizer')
columns += [y_col]

model_all = basic_model(
    train_all, 
    columns[:-1], 
    y_col, 
    df_cv=cv_all
)
print(f"All Mixtures Score: {model_all['Score']:.4f}")

order = np.argsort(cv_all[y_col].to_numpy())
plot_pred_and_resids(
    cv_all.iloc[order], 
    model_all['CV Pred'][order], 
    title='All Mixtures Non Linear Model CV Pred and Resids'
)
plot_features_and_resids(
    cv_all, 
    model_all['CV Resids'], 
    title='All Mixtures Non Linear Model CV Resids by Features'
)

train_all['Residuals'] = model_all['Train Resids']
plot_resids_by_features_quantiles(train_all, 'Residuals')

train_all, columns = engineer_all_mixtures_2(train)
cv_all, _ = engineer_all_mixtures_2(cv)

columns.insert(0, 'Cement')
columns.insert(0, 'Blast Furnace Slag')
columns.insert(0, 'Fly Ash')
columns.insert(3, 'Water')
columns.insert(4, 'Fine Aggregate')
columns.insert(5, 'Fine Aggregate Ratio')
columns.insert(7, 'Superplasticizer')
columns += [y_col]

results = basic_model(
    train_all, 
    columns[:-1], 
    y_col, 
    df_cv=cv_all
)


print(f"All Mixtures Enhanced Score: {results['Score']:.4f}")
print(f"Score: {results['Pipe'].score(cv_all[columns[:-1]], cv_all[y_col])}")
print(f"MSE: {(results['CV Resids'] ** 2).mean():.4f}")

train_all['Residuals'] = results['Train Resids']

order = np.argsort(cv_all[y_col].to_numpy())
plot_pred_and_resids(
    cv_all.iloc[order], 
    results['CV Pred'][order], 
    title='All Mixtures Enhanced Non Linear Model CV Pred and Resids'
)
plot_features_and_resids(
    cv_all, 
    results['CV Resids'], 
    title='All Mixtures Enhanced Non Linear Model CV Resids by Features'
)
columns_order = (
    train_all.columns[:-2].sort_values().to_list() + [y_col, 'Residuals']
)
plot_resids_by_features_quantiles(train_all[columns_order], 'Residuals')

# Model assessment given recipes
cols = train.columns[~train.columns.isin([y_col, 'Age'])].to_list()
duplicated = train_all[cols].duplicated(keep=False)
recipes = train_all[duplicated].copy()
recipes['Recipe ID'] = recipes.groupby(cols, sort=False).ngroup()
recipes_resids = (
    recipes.groupby('Recipe ID')['Residuals'].agg(['mean', 'size'])
)

recipes_norm = recipes[cols].copy()
recipes_scaler = StandardScaler()
recipes_norm = recipes_scaler.fit_transform(recipes_norm)
kmeans = KMeans(n_clusters=25, n_init=20)
kmeans.fit(recipes_norm)

centers = pd.DataFrame(
    recipes_scaler.inverse_transform(kmeans.cluster_centers_), columns=cols
)
labels = pd.Series(kmeans.labels_).value_counts().sort_index()
centers = pd.concat([centers, labels], axis=1)
centers = pd.concat(
    [centers, recipes.groupby(kmeans.labels_)['Residuals'].median()], axis=1
)
recipes['Label'] = kmeans.labels_

# Observed and predicted target comparison
check = (
    pd.concat(
        [
            train_all[y_col].rename('Y').reset_index(), 
            pd.Series(results['Train Pred'], name='Pred')
        ], 
        axis=1
    )
    .sort_values('Y')
    .reset_index(drop=True)
)
check['Residuals'] = check[['Pred', 'Y']].diff(axis=1).iloc[:, 1]

for i in np.arange(5, 21, 5):
    idxs = check.loc[check['Residuals'].abs() > i, 'index'].to_list()
    n = recipes.index[recipes.index.isin(idxs)].shape[0]
    per = n / len(idxs) * 100
    print(f'For absolute error > {i}: {n} = {per:.2f}% with known recipes')

# Large residuals analysis
idxs = check.loc[check['Residuals'].abs() > 20, 'index'].to_list()
resids_high = train_all.loc[idxs]

scaler = StandardScaler()
train_all_scaled = scaler.fit_transform(train_all[cols[:7]])
loc = np.where(train_all.index == resids_high.index[3])[0][0]
dist = pd.Series(
    np.sqrt(((train_all_scaled - train_all_scaled[loc]) ** 2).sum(axis=1)), 
    index=train_all.index, 
    name='Distance'
)

resids_high = train_all.loc[dist <= 2]

resids_high_coef = pd.DataFrame(
    results['Pipe']['scaler'].transform(resids_high[columns[:-1]]) 
    * results['Pipe']['reg'].coef_, 
    columns=columns[:-1], 
    index=resids_high.index
)

diff = resids_high_coef.loc[381].sub(resids_high_coef.loc[354])

# Model displays greater residual variance when Binder to Water Ratio > 3, 
# with the first residual quantile representing -5.3696 x -3.7123 and the third
# 6.4065 x 3.7179. 50% of the absolute residuals greater than 20 are in this
# category, or 66.67% of the residuals above said margin, or even 100%
# when considered the quarter. The tendency for overprediction is largely 
# corrected when only Cement accounts for the Binder Mixture, although 
# underprediction remains strong.

# Model also suffers when Blast Furnace Slag >= 180, although not as much as
# with high Binder to Water Ratio.

# Model also displays greater residual variance when Aggregate to Binder
# Ratio <= 3, with the third residual quantile representing 7.8478 x 3.7123.

# Assessment of unusual concentration of high residuals at high 
# Binder to Water Ratios
binder_wat_high = train_all[train_all['Binder to Water Ratio'] >= 2.7]

fig, ax = plt.subplots(1, 1, figsize=(15, 7))
ax.grid(True, color='k', linestyle='--', alpha=.2)
binder_wat_high['Residuals'].hist(ax=ax, bins=20)
plt.show()

cov = binder_wat_high.cov()
cols = [
        'Residuals', 'Cement', 'Blast Furnace Slag', 'Fly Ash', 'Age', 'Binder'
]
cols_ratio = [col + ' Ratio' for col in cols[1:-2]]

binder_wat_high = binder_wat_high[cols].copy()
binder_wat_high[cols_ratio] = (
    binder_wat_high[cols[1:-2]].to_numpy() 
    / binder_wat_high[cols[-1:]].to_numpy()
)
binder_wat_high['Group'] = pd.qcut(binder_wat_high['Residuals'], 10)
des = (
       binder_wat_high
       .groupby('Group', observed=True)
       .agg(['mean', 'median', 'var'])
)
des = des[des.columns[des.columns.get_level_values(0) != 'Residuals']]
for col in cols[1:-2]:
    binder_wat_high[f'{col} Group'] = pd.cut(
        binder_wat_high[col], 10, duplicates='drop'
    )
binder_wat_high_pivoted = binder_wat_high.pivot_table(
    index='Cement Group', 
    columns=['Blast Furnace Slag Group', 'Fly Ash Group'], 
    values='Residuals', 
    aggfunc='size', 
    observed=True
)

residuals_by_group = train_all[cols].copy()
residuals_by_group[cols_ratio] = (
    residuals_by_group[cols[1:-2]].to_numpy() 
    / residuals_by_group[cols[-1:]].to_numpy()
)
for col in cols[1:-1]:
    if col != 'Age':
        bins = 10
    else:
        bins = [0, 10, 30, 101, 180, 370]
    residuals_by_group[f'{col} Group'] = pd.cut(
        residuals_by_group[col], bins, duplicates='drop'
    )

residuals_by_group_pivoted = residuals_by_group.pivot_table(
    index='Cement Group', 
    columns=['Blast Furnace Slag Group', 'Age Group'], 
    values='Residuals', 
    aggfunc='size', 
    observed=True
)

coefs = (
    pd.Series(results['Pipe']['reg'].coef_, index=columns[:-1]).sort_index()
)

residuals_by_group_age = residuals_by_group[residuals_by_group['Age'] == 28]
filt = residuals_by_group_age['Residuals'] < 0

col1 = 'Cement'
col2 = 'Blast Furnace Slag'
xticks = np.linspace(
    residuals_by_group_age[col1].min(), residuals_by_group_age[col1].max(), 10
)
yticks = np.linspace(
    residuals_by_group_age[col2].min(), residuals_by_group_age[col2].max(), 10
)
fig, ax = plt.subplots(1, 1, figsize=(15, 7), constrained_layout=True)
plt.suptitle(
    f'{col1} x {col2} Residuals given Age == 28', fontsize=14, weight='bold'
)
ax.grid(True, color='k', linestyle='--', alpha=.2)
ax.scatter(
    residuals_by_group_age.loc[~filt, col1], 
    residuals_by_group_age.loc[~filt, col2], 
    s=residuals_by_group_age.loc[~filt, 'Residuals'] ** 2, 
    label='Residuals >= 0'
)
ax.scatter(
    residuals_by_group_age.loc[filt, col1], 
    residuals_by_group_age.loc[filt, col2], 
    s=residuals_by_group_age.loc[filt, 'Residuals'] ** 2, 
    color='r', 
    label='Residuals < 0'
)
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_xlabel(col1)
ax.set_ylabel(col2)
ax.legend()
plt.show()

residuals_by_group_age['Sqr Residuals'] = (
    residuals_by_group_age['Residuals'] ** 2
)

probs, bounds = interval_cumulative_distribution(
    residuals_by_group_age, 
    'Cement', 
    'Blast Furnace Slag', 
    'Sqr Residuals', 
    dist='joint', 
    return_bounds=True
)
bounds = pd.DataFrame(bounds)

# Assessment of candidate features
col = 'Blast Furnace Slag'
candidates = {
    'Binder': lambda df, col: [
        df[col] * df['Cement'], 
        df[col] * df['Cement'] / df['Binder'], 
        df[col] / df['Binder'] * df['Cement'] / df['Binder'], 
        df[col] * df['Water Ratio'], 
        df[col] / df['Water'], 
        df[col] * df['Fly Ash'], 
        df[col] / df['Binder'] * df['Fly Ash'] / df['Binder'], 
    ], 
    'Spline': lambda df, col: [
        np.where(df[col].between(132, 190), df[col] - 132, 0) * df['Cement'], 
        np.where(df[col].between(132, 190), df[col] - 132, 0) / df['Binder'], 
        np.where(df[col].between(132, 190), df[col] - 132, 0) * (
            df['Cement'] / df['Binder']
        ), 
        np.where(df[col].between(132, 190), df[col] - 132, 0) * (
            df['Water Ratio']
        ), 
        np.where(df[col].between(132, 190), df[col] - 132, 0) / df['Water'], 
        np.where(df[col].between(132, 190), df[col] - 132, 0) * df['Fly Ash'], 
        np.where(df[col].between(132, 190), df[col] - 132, 0) / df['Binder'] * (
            df['Fly Ash'] / df['Binder']
        )
    ]
}

evaluate_candidates(candidates, train_all, cv_all, columns, col)

# Extensive testing showed the need of a different model capable of learning 
# said relationships, while also maintaining the ones already found.

results = basic_model(
    train_all,
    columns[:-1],
    y_col,
    df_cv=cv_all
)
model = GaussianRBF(
    pipe=results['Pipe'],
    columns=columns[:-1],
    alpha=1.0,
    gamma=0.3
)
model.fit(train_all, train_all[y_col])

pred_train = model.predict(train_all)
pred_cv = model.predict(cv_all)

results.update({
    'Pipe': model,
    'Train Pred': pred_train,
    'CV Pred': pred_cv,
    'Train Resids': train_all[y_col] - pred_train,
    'CV Resids': cv_all[y_col] - pred_cv,
    'Score': model.score(cv_all, cv_all[y_col])
})

print(f"All Mixtures Final Score: {results['Score']:.4f}")
print(f"MSE: {(results['CV Resids'] ** 2).mean():.4f}")

cv_all['Residuals'] = results['CV Resids']
order = np.argsort(cv_all[y_col].to_numpy())
plot_pred_and_resids(
    cv_all.iloc[order], 
    results['CV Pred'][order], 
    title='All Mixtures Final Model CV Pred and Resids'
)
plot_features_and_resids(
    cv_all, 
    results['CV Resids'], 
    title='All Mixtures Final Model CV Resids by Features'
)
plot_resids_by_features_quantiles(cv_all[columns_order], 'Residuals')

# Observed and predicted target comparison
check = (
    pd.concat(
        [
            cv_all[y_col].reset_index(drop=True).rename('Y'), 
            pd.Series(results['CV Pred']).rename('Pred')
        ], 
        axis=1
    )
    .sort_values('Y')
)

cols = columns.copy()
cols.insert(0, 'Pred')
train_all['Residuals'] = results['Train Resids']
cv_all['Residuals'] = results['CV Resids']

# Evaluation of the final model on the test set
test_all = test.copy()
test_all = group_binders_aggregates(test_all)
test_all, _ = engineer_all_mixtures_2(test_all)
print(f'Final model test set score: {model.score(test_all, test_all[y_col])}')


























