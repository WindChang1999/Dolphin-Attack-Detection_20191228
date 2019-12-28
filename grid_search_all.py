from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from Load_data import Load_data
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


# Utility function to move the midpoint of a colormap to be around
# the values of interest.

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

if __name__ == '__main__':
    scaler = StandardScaler()
    data, label = Load_data('all')
    data = scaler.fit_transform(data)
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    kernel_range = ('linear', 'rbf')
    param_grid = dict(gamma=gamma_range, C=C_range, kernel=kernel_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=206)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(data, label)

    print("the best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

    scores = grid.cv_results_['mean_test_score']

    linear_scores = scores[::2].reshape(len(C_range), len(gamma_range))
    rbf_scores = scores[1::2].reshape(len(C_range), len(gamma_range))

    for i, score in enumerate([linear_scores, rbf_scores]):
        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
        plt.imshow(score, interpolation='nearest', cmap=plt.cm.hot,
                   norm=MidpointNormalize(vmin=0.3, midpoint=0.95))
        plt.xlabel('gamma')
        plt.ylabel('C')
        plt.colorbar()
        plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
        plt.yticks(np.arange(len(C_range)), C_range)
        if i == 0:
            plt.title('Linear kernel validation accuracy')
        else:
            plt.title('RBF kernel validation accuracy')
    plt.show()