import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    #local level model
    import pandas as pd
    import numpy as np
    from scipy import stats
    from scipy.optimize import minimize

    import statsmodels.api as sm
    import statsmodels.tsa.api as tsa

    from matplotlib import pyplot as plt
    import seaborn as sns
    sns.set()
    return minimize, np, pd, plt, stats, tsa


@app.cell
def _(np, stats):
    #local levelは水準成分のみ
    #状態はランダムウォークをしつつ、ノイズが加わる。
    #make data
    np.random.seed(1)
    sim_size=100
    mu = np.cumsum(stats.norm.rvs(loc=0, scale=1, size=sim_size).round(1)) + 30
    return mu, sim_size


@app.cell
def _(mu, sim_size, stats):
    y = mu + stats.norm.rvs(loc=0, scale=5, size=sim_size).round(1)
    return (y,)


@app.cell
def _(mu, pd, y):
    local_level_df = pd.DataFrame({'mu':mu, 'y':y})
    return (local_level_df,)


@app.cell
def _(local_level_df):
    local_level_df.plot()
    return


@app.cell
def _(pd, sim_size, y):
    y_ts = pd.Series(
        y, index=pd.date_range(start='2020-01-01', periods=sim_size, freq='D')
    )
    return (y_ts,)


@app.cell
def _(minimize, np, pd, stats):
    class LocalLevel:
        # データを格納(pd.Seriesで、日付インデックスがついている想定)
        def __init__(self, ts_data):
            self.ts_data = ts_data
            self.a = pd.Series(np.zeros(len(ts_data)), index=ts_data.index)
            self.P = pd.Series(np.zeros(len(ts_data)), index=ts_data.index)
            self.v = pd.Series(np.zeros(len(ts_data)), index=ts_data.index)
            self.F = pd.Series(np.zeros(len(ts_data)), index=ts_data.index)
            self.K = pd.Series(np.zeros(len(ts_data)), index=ts_data.index)
            self.s_level = None     # 過程誤差の分散
            self.s_irregular = None # 観測誤差の分散

        # 状態の初期値を設定する
        def initialize(self, initial_a, initial_P):
            self.initial_a = initial_a
            self.initial_P = initial_P

        # 1時点先の予測値を計算する
        def _forecast_step(self, a_pre, P_pre, s_irregular, s_level, first=False):
            if first:
                a_forecast = self.initial_a    # 初回に限り、初期値を代入
                P_forecast = self.initial_P    # 初回に限り、初期値を代入
            else:
                a_forecast = a_pre             # 状態の予測値
                P_forecast = P_pre + s_level   # 状態の予測値の分散

            y_forecast = a_forecast            # 観測値の予測値
            F = P_forecast + s_irregular       # 観測値の予測値の残差の分散

            return(pd.Series([a_forecast, P_forecast, y_forecast, F], 
                             index=['a', 'P', 'y', 'F']))

        # 1時点のフィルタリングをする
        def _filter_step(self, forecasted, y, s_irregular):
            v = y - forecasted.y                # 観測値の1時点先予測値の残差
            K = forecasted.P / forecasted.F     # カルマンゲイン
            a_filter = forecasted.a + K * v     # フィルタ化推定量
            P_filter = (1 - K) * forecasted.P   # フィルタ化推定量の分散

            return(pd.Series([a_filter, P_filter, v, K], 
                             index=['a', 'P', 'v', 'K']))

        # フィルタリングを行う
        def filter(self, s_irregular, s_level):
            for i in range(0, len(self.ts_data)):
                if(i == 0):
                    # 初回のみ、初期値の値を利用して予測する
                    forecast_loop = self._forecast_step(
                        a_pre=None, P_pre=None, 
                        s_irregular=s_irregular, s_level=s_level, first=True)
                else:
                    # 2時点目以降は、1時点前の値を参照して予測する
                    forecast_loop = self._forecast_step(
                        a_pre=self.a.iloc[i - 1], P_pre=self.P.iloc[i - 1], 
                        s_irregular=s_irregular, s_level=s_level)

                # フィルタリングの実行
                filter_loop = self._filter_step(
                    forecasted=forecast_loop, y=self.ts_data.iloc[i],
                    s_irregular=s_irregular
                )

                # 結果の保存
                self.a.iloc[i] = filter_loop.a
                self.P.iloc[i] = filter_loop.P
                self.F.iloc[i] = forecast_loop.F
                self.K.iloc[i] = filter_loop.K
                self.v.iloc[i] = filter_loop.v

        # 対数尤度の計算
        def llf(self):
            return np.sum(np.log(stats.norm.pdf(
                x=self.v, loc=0, scale=np.sqrt(self.F)
            )))

        # パラメータの推定と状態の再当てはめ
        def fit(self, start_params):
            # パラメータを指定して対数尤度の-1倍を出力する内部関数
            def calc_llf(params):
                self.filter(np.exp(params[0]), np.exp(params[1]))
                return self.llf() * -1

            # 最適化の実行
            opt_res = minimize(calc_llf, start_params, 
                               method='Nelder-Mead', tol=1e-6, 
                               options={'maxiter':2000})

            # パラメータの保存
            self.s_irregular = np.exp(opt_res.x[0])
            self.s_level = np.exp(opt_res.x[1])

            # 最適なパラメータでもう一度フィルタリングを行う
            self.filter(self.s_irregular, self.s_level)

        # 推定された状態の可視化
        def plot_level(self):
            plot_df = pd.concat([self.a, self.ts_data], axis=1)
            plot_df.columns = column=['filtered', 'y']
            plot_df.plot()
    return (LocalLevel,)


@app.cell
def _(LocalLevel, y_ts):
    local_level = LocalLevel(y_ts)
    local_level.initialize(initial_a=0.001, initial_P=1000000)
    local_level.filter(s_irregular=10.0, s_level=1.0)
    return (local_level,)


@app.cell
def _(local_level, plt):
    local_level.plot_level()
    plt.show()
    return


@app.cell
def _(local_level):
    # パラメータの推定
    local_level.fit(start_params=[1, 1])

    # 対数尤度
    local_level.llf()
    return


@app.cell
def _(local_level, np):
    # 推定されたパラメータ
    print('観測誤差の分散', np.round(local_level.s_irregular, 5))
    print('過程誤差の分散', np.round(local_level.s_level, 5))
    return


@app.cell
def _(tsa, y_ts):
    #make various models by statsmodel
    mod_local_level_fix = tsa.UnobservedComponents(y_ts, level='local level', loglikelihood_burn=0)
    return (mod_local_level_fix,)


@app.cell
def _(mod_local_level_fix, np, pd):
    mod_local_level_fix.initialize_approximate_diffuse(1000000)
    res_local_level_fix = mod_local_level_fix.filter(pd.Series(np.array([10, 1])))
    return (res_local_level_fix,)


@app.cell
def _(res_local_level_fix):
    print(res_local_level_fix.level['filtered'])
    return


@app.cell
def _(tsa, y_ts):
    mod_local_level = tsa.UnobservedComponents(y_ts, level='local level', loglikelihood_burn=0)
    mod_local_level.initialize_approximate_diffuse(1000000)
    res_local_level = mod_local_level.fit(
        start_params=[1, 1], method='nm', maxiter=2000
    )

    print(res_local_level.summary())
    return (res_local_level,)


@app.cell
def _(plt, res_local_level):
    _ = res_local_level.plot_components(which='filtered', observed=False)
    plt.show()
    return


@app.cell
def _(res_local_level):
    res_local_level.forecast(5)
    return


@app.cell
def _(plt, res_local_level):
    _ = res_local_level.plot_diagnostics(lags=48, figsize=(15,8))
    plt.show()
    return


@app.cell
def _(pd):
    sales_day = pd.read_csv("C:/Users/rally/python3_12/5-6-1-daily-sales-data.csv",
                            index_col=0, parse_dates=True,
                           dtype='float')
    return (sales_day,)


@app.cell
def _(plt, sales_day):
    sales_day.plot(subplots=True)
    plt.show()
    return


@app.cell
def _(pd):
    #make holiday flg
    # 祝日を内閣府のWebサイトから読み込む
    holiday = pd.read_csv(
        'https://www8.cao.go.jp/chosei/shukujitsu/syukujitsu.csv', 
        encoding='CP932', parse_dates=True, index_col=0
    )

    # 結果の確認
    print(holiday.head(3))
    return (holiday,)


@app.cell
def _(holiday, sales_day):
    is_holiday = sales_day.index.isin(holiday.index).astype(int)
    y_st = (sales_day.index.month == 1) & sales_day.index.day.isin([2, 3])
    y_en = (sales_day.index.month == 12) & sales_day.index.day.isin([30, 31])
    is_holiday = is_holiday + y_st + y_en
    sales_day['holiday'] = is_holiday
    return (is_holiday,)


@app.cell
def _(is_holiday, sales_day):
    # 日曜日かつ祝日の日は、通常の祝日と区別する
    sales_day['sun_holiday'] = is_holiday & (sales_day.index.dayofweek == 6)

    # 結果の確認
    print(sales_day.head(3))
    return


@app.cell
def _(sales_day):
    #make flyer lag datas
    sales_day['flyer_lag'] = sales_day.flyer.shift(1).fillna(0)
    sales_day['flyer_lag2'] = sales_day.flyer.shift(2).fillna(0)
    return


@app.cell
def _(sales_day):
    sales_day.head(60)
    return


@app.cell
def _(sales_day, tsa):
    #周期性と季節性を加味したモデル、７日周期３６５日周期
    mod_bsts = tsa.UnobservedComponents(sales_day['sales'],
                                       level='smooth trend',
                                       seasonal=7,
                                       exog=sales_day[['flyer', 'flyer_lag', 'flyer_lag2', 'holiday', 'sun_holiday']],
                                       autoregressive=2,
                                       freq_seasonal=[{'period':365.25, 'harmonics':1}])
    return (mod_bsts,)


@app.cell
def _(mod_bsts):
    #fit
    res_bsts = mod_bsts.fit(method='nm',
                maxiter=5000)
    return (res_bsts,)


@app.cell
def _(res_bsts):
    res_bsts.summary()
    return


@app.cell
def _(plt, res_bsts):
    res_bsts.plot_diagnostics(lags=30, fig=plt.figure(tight_layout=True, figsize=(15, 8)))
    return


@app.cell
def _(pd, plt, res_bsts, sales_day):
    # 2つのモデルの比較(水準成分)

    # DataFrameにまとめる
    plot_df = pd.DataFrame({
        'sales': sales_day['sales'],
        'mod_level': res_bsts.level['smoothed']
    })

    # 可視化
    # グラフサイズの指定
    fig, ax = plt.subplots(figsize=(8, 4), tight_layout=True)

    # 折れ線グラフを描く
    ax.plot(plot_df['sales'], color='black', label='original')
    ax.plot(plot_df['mod_level'], linewidth=3, color='orange',
            label='bsts')

    # 軸ラベルとタイトル・凡例
    ax.set_xlabel('yyyymm', size=14)
    ax.set_ylabel('sales', size=14)
    ax.legend()

    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
