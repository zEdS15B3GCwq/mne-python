# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator

from .._fiff.pick import _picks_to_idx, pick_info, pick_types
from ..parallel import parallel_func
from ..utils import logger, verbose
from .base import _set_cv
from .transformer import MNETransformerMixin


class EMS(MNETransformerMixin, BaseEstimator):
    """Transformer to compute event-matched spatial filters.

    This version of EMS :footcite:`SchurgerEtAl2013` operates on the entire
    time course. No time
    window needs to be specified. The result is a spatial filter at each
    time point and a corresponding time course. Intuitively, the result
    gives the similarity between the filter at each time point and the
    data vector (sensors) at that time point.

    .. note:: EMS only works for binary classification.

    Attributes
    ----------
    filters_ : ndarray, shape (n_channels, n_times)
        The set of spatial filters.
    classes_ : ndarray, shape (n_classes,)
        The target classes.

    References
    ----------
    .. footbibliography::
    """

    def __sklearn_tags__(self):
        """Return sklearn tags."""
        from sklearn.utils import ClassifierTags

        tags = super().__sklearn_tags__()
        if tags.classifier_tags is None:
            tags.classifier_tags = ClassifierTags()
        tags.classifier_tags.multi_class = False
        return tags

    def __repr__(self):  # noqa: D105
        if hasattr(self, "filters_"):
            return (
                f"<EMS: fitted with {len(self.filters_)} filters "
                f"on {len(self.classes_)} classes.>"
            )
        else:
            return "<EMS: not fitted.>"

    def fit(self, X, y):
        """Fit the spatial filters.

        .. note : EMS is fitted on data normalized by channel type before the
                  fitting of the spatial filters.

        Parameters
        ----------
        X : array, shape (n_epochs, n_channels, n_times)
            The training data.
        y : array of int, shape (n_epochs)
            The target classes.

        Returns
        -------
        self : instance of EMS
            Returns self.
        """
        X, y = self._check_data(X, y=y, fit=True, return_y=True)
        classes, y = np.unique(y, return_inverse=True)
        if len(classes) > 2:
            raise ValueError("EMS only works for binary classification.")
        self.classes_ = classes
        filters = X[y == 0].mean(0) - X[y == 1].mean(0)
        filters /= np.linalg.norm(filters, axis=0)[None, :]
        self.filters_ = filters
        return self

    def transform(self, X):
        """Transform the data by the spatial filters.

        Parameters
        ----------
        X : array, shape (n_epochs, n_channels, n_times)
            The input data.

        Returns
        -------
        X : array, shape (n_epochs, n_times)
            The input data transformed by the spatial filters.
        """
        X = self._check_data(X)
        Xt = np.sum(X * self.filters_, axis=1)
        return Xt


@verbose
def compute_ems(
    epochs, conditions=None, picks=None, n_jobs=None, cv=None, *, verbose=None
):
    """Compute event-matched spatial filter on epochs.

    This version of EMS :footcite:`SchurgerEtAl2013` operates on the entire
    time course. No time
    window needs to be specified. The result is a spatial filter at each
    time point and a corresponding time course. Intuitively, the result
    gives the similarity between the filter at each time point and the
    data vector (sensors) at that time point.

    .. note : EMS only works for binary classification.

    .. note : The present function applies a leave-one-out cross-validation,
              following Schurger et al's paper. However, we recommend using
              a stratified k-fold cross-validation. Indeed, leave-one-out tends
              to overfit and cannot be used to estimate the variance of the
              prediction within a given fold.

    .. note : Because of the leave-one-out, this function needs an equal
              number of epochs in each of the two conditions.

    Parameters
    ----------
    epochs : instance of mne.Epochs
        The epochs.
    conditions : list of str | None, default None
        If a list of strings, strings must match the epochs.event_id's key as
        well as the number of conditions supported by the objective_function.
        If None keys in epochs.event_id are used.
    %(picks_good_data)s
    %(n_jobs)s
    cv : cross-validation object | str | None, default LeaveOneOut
        The cross-validation scheme.
    %(verbose)s

    Returns
    -------
    surrogate_trials : ndarray, shape (n_trials // 2, n_times)
        The trial surrogates.
    mean_spatial_filter : ndarray, shape (n_channels, n_times)
        The set of spatial filters.
    conditions : ndarray, shape (n_classes,)
        The conditions used. Values correspond to original event ids.

    References
    ----------
    .. footbibliography::
    """
    logger.info("...computing surrogate time series. This can take some time")

    # Default to leave-one-out cv
    cv = "LeaveOneOut" if cv is None else cv
    picks = _picks_to_idx(epochs.info, picks)

    if not len(set(Counter(epochs.events[:, 2]).values())) == 1:
        raise ValueError(
            "The same number of epochs is required by "
            "this function. Please consider "
            "`epochs.equalize_event_counts`"
        )

    if conditions is None:
        conditions = epochs.event_id.keys()
        epochs = epochs.copy()
    else:
        epochs = epochs[conditions]

    epochs.drop_bad()

    if len(conditions) != 2:
        raise ValueError(
            "Currently this function expects exactly 2 "
            f"conditions but you gave me {len(conditions)}"
        )

    ev = epochs.events[:, 2]
    # Special care to avoid path dependent mappings and orders
    conditions = list(sorted(conditions))
    cond_idx = [np.where(ev == epochs.event_id[k])[0] for k in conditions]

    info = pick_info(epochs.info, picks)
    data = epochs.get_data(picks=picks)

    # Scale (z-score) the data by channel type
    # XXX the z-scoring is applied outside the CV, which is not standard.
    for ch_type in ["mag", "grad", "eeg"]:
        if ch_type in epochs:
            # FIXME should be applied to all sort of data channels
            if ch_type == "eeg":
                this_picks = pick_types(info, meg=False, eeg=True)
            else:
                this_picks = pick_types(info, meg=ch_type, eeg=False)
            data[:, this_picks] /= np.std(data[:, this_picks])

    # Setup cross-validation. Need to use _set_cv to deal with sklearn
    # changes in cv object handling.
    y = epochs.events[:, 2]
    _, cv_splits = _set_cv(cv, "classifier", X=y, y=y)

    parallel, p_func, n_jobs = parallel_func(_run_ems, n_jobs=n_jobs)
    # FIXME this parallelization should be removed.
    #   1) it's numpy computation so it's already efficient,
    #   2) it duplicates the data in RAM,
    #   3) the computation is already super fast.
    out = parallel(
        p_func(_ems_diff, data, cond_idx, train, test) for train, test in cv_splits
    )

    surrogate_trials, spatial_filter = zip(*out)
    surrogate_trials = np.array(surrogate_trials)
    spatial_filter = np.mean(spatial_filter, axis=0)

    return surrogate_trials, spatial_filter, epochs.events[:, 2]


def _ems_diff(data0, data1):
    """Compute the default diff objective function."""
    return np.mean(data0, axis=0) - np.mean(data1, axis=0)


def _run_ems(objective_function, data, cond_idx, train, test):
    """Run EMS."""
    d = objective_function(*(data[np.intersect1d(c, train)] for c in cond_idx))
    d /= np.sqrt(np.sum(d**2, axis=0))[None, :]
    # compute surrogates
    return np.sum(data[test[0]] * d, axis=0), d
