import numpy as np
from matplotlib import pyplot as plt


class Plot:
    DEFAULT_FIGSIZE = (15, 5)
    DEFAULT_COLORS = {'ts': 'black', 'pca_inverse': 'orange', 'anomaly': 'red', 'score_threshold': 'orange',
                      'score': '#dddddd'}
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

    def ts(self, title: str, ts: np.array, features: list, n_rows: int, n_cols: int, show: bool = True,
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
                for key, value in ts.items():
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
            ax.plot(anomaly_score[anomaly_mask], color=self.colors['anomaly'], marker=self.markers['score'])
        if show:
            plt.show()
        if save:
            pass
