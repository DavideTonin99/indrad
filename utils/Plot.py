import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm
import seaborn as sns


class Plot:
    DEFAULT_FIGSIZE = (30, 20)
    DEFAULT_COLORS = {'ts_true': '#025378', 'ts_ok': '#00824a', 'ts_pred': '#69b0d1', 'anomaly': '#cc2929',
                      'score_threshold': '#ff8400', 'score': '#4992bf'}
    DEFAULT_MARKERS = {'anomaly': 'o', 'score': 'o'}

    def __init__(self, save_path: str = None, **kwargs):
        """
        @param save_path:
        """
        self.save_path = save_path
        self.figsize = Plot.DEFAULT_FIGSIZE
        self.colors = {}
        self.markers = {}

        if 'figsize' in kwargs:
            self.figsize = kwargs['figsize']
        if 'colors' in kwargs:
            self.colors = kwargs['colors']
        if 'markers' in kwargs:
            self.markers = kwargs['markers']

        for key, value in Plot.DEFAULT_COLORS.items():
            if key not in self.colors:
                self.colors[key] = value

        for key, value in Plot.DEFAULT_MARKERS.items():
            if key not in self.markers:
                self.markers[key] = value

    def ts(self, title: str, time_series: dict, features: list, n_rows: int, n_cols: int, show: bool = True,
           save: bool = False) -> None:
        """
        @param title:
        @param ts:
        @param features:
        @param n_rows:
        @param n_cols:
        @param show:
        @param save:
        @return:
        """
        fig, axs = plt.subplots(n_rows, n_cols, figsize=self.figsize)
        fig.suptitle(title)
        col = 0
        feature_index = 0
        for feature in features:
            for i in range(n_rows):
                for key, value in time_series.items():
                    marker = self.markers[key] if key in self.markers else None
                    color = self.colors[key] if key in self.colors else None

                    axs[i, col].plot(value[:, feature_index], color=color, marker=marker)
                    axs[i, col].set_title(f'{feature} {i}')
                feature_index += 1
            col += 1
        if show:
            plt.show()
        if save:
            pass

    def anomaly_score(self, ts_name: str, anomaly_score: np.array, anomaly_mask: np.array, threshold: float,
                      show: bool = True, save: bool = False) -> None:
        """
        @param ts_name:
        @param anomaly_score:
        @param anomaly_mask:
        @param threshold:
        @param show:
        @param save:
        @return:
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        plt.title(f"Anomaly score: {ts_name}")
        ax.axhline(y=threshold, color=self.colors['score_threshold'], linestyle='--')
        ax.plot(anomaly_score, color=self.colors['score'], marker=self.markers['score'])

        if anomaly_mask is not None:
            anomalies = anomaly_score
            anomalies[~anomaly_mask] = np.nan
            ax.scatter(x=np.arange(len(anomaly_score)), y=anomalies, color=self.colors['anomaly'])
        if show:
            plt.show()
        if save:
            pass

    def stats(self, title: str, ts_data: np.array, features: list, n_rows: int, n_cols: int, plot_type: str,
              show: bool = True, save: bool = False) -> None:
        """
        Plot stats
        @param title:
        @param ts_data:
        @param features:
        @param n_rows:
        @param n_cols:
        @param plot_type:
        @param show:
        @param save:
        @return:
        """
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=self.figsize)
        fig.suptitle(title)
        col = 0
        feature_index = 0

        for _ in features:
            for i in range(7):
                if plot_type == 'quantile':
                    sm.qqplot(ts_data[:, feature_index], ax=axs[i, col], line='45')
                elif plot_type == 'distribution':
                    sns.kdeplot(data=ts_data[:, feature_index], ax=axs[i, col], fill=True, alpha=0.5)
                feature_index += 1
            col += 1

        if show:
            plt.show()
        if save:
            pass
