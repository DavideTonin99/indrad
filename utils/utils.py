from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import sklearn

from models.TimeSeries import TimeSeries


def fit_pca(dataset, n_components=21, show_plot_variance=False) -> PCA:
    """
    Fit a PCA model to the dataset
    :param dataset: dataset to fit the PCA model
    :param n_components: number of components to keep
    :param show_plot_variance: show the plot of the variance
    """
    pca = PCA(n_components=n_components)
    pca.fit(dataset)
    if show_plot_variance:
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.show()
    return pca


def test_sklearn(y_true, y_pred):
    """
    Test the model with sklearn metrics
    :param y_true: true labels
    :param y_pred: predicted labels
    """
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    recall = sklearn.metrics.recall_score(y_true, y_pred, pos_label=-1)
    precision = sklearn.metrics.precision_score(y_true, y_pred, pos_label=-1)
    f1 = sklearn.metrics.f1_score(y_true, y_pred, pos_label=-1)
    roc_auc = sklearn.metrics.roc_auc_score(y_true, y_pred)

    print('Precision: %f' % precision)
    print('Recall: %f' % recall)
    print('F1 score: %f' % f1)
    print('Accuracy: %f' % accuracy)
    print('Roc auc score: %f' % roc_auc)
    print('%f\t%f\t%f\t%f' % (precision, recall, f1, accuracy))
    print(sklearn.metrics.precision_recall_fscore_support(y_true, y_pred))

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_pred)
    plt.figure(1)
    plt.plot(fpr, tpr, label=('ROC curve (area = %0.2f)' % roc_auc))
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.show()

    precision_rt, recall_rt, threshold_rt = sklearn.metrics.precision_recall_curve(
        y_true, y_pred)
    plt.figure(1)
    plt.plot(threshold_rt, precision_rt[1:], label="Precision", linewidth=1)
    plt.plot(threshold_rt, recall_rt[1:], label="Recall", linewidth=1)
    plt.title('Precision and recall for different threshold values')
    plt.xlabel('Threshold')
    plt.ylabel('Precision/Recall')
    plt.legend()
    plt.show()


def is_pos_def(A):
    """
    Check if a matrix is positive definite
    :param A: matrix
    :return: True if the matrix is positive definite, False otherwise
    """
    if np.allclose(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


def cov_matrix(data: np.array) -> (np.array, np.array):
    """
    Compute the covariance matrix and its inverse
    :param data: data
    :return: covariance matrix and its inverse
    """
    covariance_matrix = np.cov(data, rowvar=False)
    if is_pos_def(covariance_matrix):
        inv_covariance_matrix = np.linalg.inv(covariance_matrix)

        if is_pos_def(inv_covariance_matrix):
            return [covariance_matrix, inv_covariance_matrix]
        else:
            print("Error: Inverse of Covariance Matrix is not positive definite!")
    else:
        print("Error: Covariance Matrix is not positive definite!")


def mahalanobis_dist(inv_cov_matrix, mean_distr, data):
    """
    Compute the mahalanobis distance
    :param inv_cov_matrix: inverse of the covariance matrix
    :param mean_distr: mean distribution
    :param data: data
    :return: mahalanobis distance
    """
    inv_covariance_matrix = inv_cov_matrix
    vars_mean = mean_distr
    diff = data - vars_mean

    left = np.dot(diff, inv_covariance_matrix)
    mahal = np.dot(left, diff.T)
    return mahal.diagonal()


def compute_errors(ts_true: TimeSeries, ts_pred: TimeSeries, abs: bool = True) -> np.array:
    """
    Compute the errors between the true dataset and the predicted dataset
    :param dataset_true: true dataset
    :param dataset_pred: predicted dataset
    :return: errors
    """
    errors = ts_pred.data - ts_true.data
    if abs:
        errors = np.abs(errors)
    return errors
