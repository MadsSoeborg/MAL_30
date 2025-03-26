import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split # For validation split
from sklearn.metrics import r2_score
import warnings # To warn about empty validation sets etc.

class MyLinReg:
    """
    Linear Regression using Gradient Descent with optional Early Stopping.

    Parameters
    ----------
    eta0 : float, default=0.1
        Initial learning rate.
    max_iter : int, default=1000
        Maximum number of iterations (epochs for SGD/Mini-Batch).
    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when validation score
        is not improving.
    validation_split : float, default=0.1
        Fraction of training data to set aside as validation set for early stopping.
        Only used if early_stopping is True. Must be between 0 and 1.
    n_iter_no_change : int, default=10
        Number of iterations with no improvement on the validation loss to wait
        before stopping. Only used if early_stopping is True.
    tol : float, default=1e-4
        The tolerance for the optimization. If validation loss does not improve by
        at least tol for n_iter_no_change iterations, training stops.
        Only used if early_stopping is True.

    Attributes
    ----------
    coef_ : array of shape (n_features,)
        Estimated coefficients for the linear regression problem.
    intercept_ : float
        Estimated intercept (bias term) for the linear regression problem.
    history_ : list
        History of training loss values during training.
    val_history_ : list
        History of validation loss values during training (if early_stopping=True).
    """
    def __init__(self, eta0=0.1, max_iter=1000,
                 early_stopping=False, validation_split=0.1,
                 n_iter_no_change=10, tol=1e-4):
        self.eta0 = eta0
        self.max_iter = max_iter

        self.early_stopping = early_stopping
        self.validation_split = validation_split
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol

        self.coef_ = None
        self.intercept_ = None
        self.history_ = []
        self.val_history_ = []
        self._best_weights = None

    def __str__(self):
        params = ", ".join(f"{k}={v}" for k, v in self.get_params(deep=False).items())
        return f"{self.__class__.__name__}({params})"

    def _add_intercept(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X]

    def _compute_loss(self, X_with_intercept, y, weights):
        m = X_with_intercept.shape[0]
        if m == 0: return np.inf
        y_pred = np.dot(X_with_intercept, weights)
        mse = np.mean((y.ravel() - y_pred.ravel())**2)
        return mse

    def _compute_gradient(self, X_with_intercept, y, weights):
        m = X_with_intercept.shape[0]
        if m == 0: return np.zeros_like(weights)
        y_pred = np.dot(X_with_intercept, weights)
        # Ensure y is 1D for consistent subtraction
        error = y_pred.ravel() - y.ravel()
        gradient = (2 / m) * X_with_intercept.T.dot(error)
        return gradient # Gradient should have shape (n_features + 1,)

    def LossHistory(self):
        return self.history_

    def ValLossHistory(self):
        return self.val_history_

    def _get_adaptive_learning_rate(self, iteration, initial_lr=None):
        # Simple decay, kept from original
        if initial_lr is None: initial_lr = self.eta0
        decay_rate = 0.01
        return initial_lr / (1 + decay_rate * iteration)

    # --- Gradient Descent Methods with Integrated Early Stopping Logic ---

    def _batch_gradient_descent(self, X_train, y_train, adaptive_lr=False, X_val=None, y_val=None):
        n_samples, n_features = X_train.shape
        X_train_int = self._add_intercept(X_train)
        weights = np.zeros(n_features + 1)

        # Early stopping setup
        perform_early_stopping = X_val is not None and y_val is not None
        X_val_int = self._add_intercept(X_val) if perform_early_stopping else None
        best_val_loss = np.inf
        no_improvement_count = 0
        self._best_weights = None # Reset for this fit

        for i in range(self.max_iter):
            # --- Training Step ---
            current_train_loss = self._compute_loss(X_train_int, y_train, weights)
            self.history_.append(current_train_loss)
            gradient = self._compute_gradient(X_train_int, y_train, weights)
            lr = self._get_adaptive_learning_rate(i) if adaptive_lr else self.eta0
            weights -= lr * gradient

            # Early stopping check
            if perform_early_stopping:
                current_val_loss = self._compute_loss(X_val_int, y_val, weights)
                self.val_history_.append(current_val_loss)

                if current_val_loss < best_val_loss - self.tol:
                    best_val_loss = current_val_loss
                    self._best_weights = weights.copy()
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if no_improvement_count >= self.n_iter_no_change:
                    print(f"Early stopping triggered at iteration {i+1} (Batch GD).")
                    break

        # Finalize weights
        final_weights = self._best_weights if self._best_weights is not None else weights
        self.intercept_ = final_weights[0]
        self.coef_ = final_weights[1:]
        return self

    def _stochastic_gradient_descent(self, X_train, y_train, adaptive_lr=False, X_val=None, y_val=None):
        n_samples, n_features = X_train.shape
        X_train_int = self._add_intercept(X_train)
        weights = np.zeros(n_features + 1)

        # Early Stopping Setup
        perform_early_stopping = X_val is not None and y_val is not None
        X_val_int = self._add_intercept(X_val) if perform_early_stopping else None
        best_val_loss = np.inf
        no_improvement_count = 0
        self._best_weights = None

        for i in range(self.max_iter): # Loop over epochs
            X_shuffled, y_shuffled = shuffle(X_train_int, y_train, random_state=i)

            # --- Training Step (Samples within epoch) ---
            for j in range(n_samples):
                x_j = X_shuffled[j:j+1]
                y_j = y_shuffled[j:j+1]
                gradient = self._compute_gradient(x_j, y_j, weights)
                lr = self._get_adaptive_learning_rate(i + j / n_samples) if adaptive_lr else self.eta0
                weights -= lr * gradient

            # --- Calculate Losses and Check Early Stopping (End of Epoch) ---
            current_train_loss = self._compute_loss(X_train_int, y_train, weights) # Loss on full train set
            self.history_.append(current_train_loss)

            if perform_early_stopping:
                current_val_loss = self._compute_loss(X_val_int, y_val, weights)
                self.val_history_.append(current_val_loss)

                if current_val_loss < best_val_loss - self.tol:
                    best_val_loss = current_val_loss
                    self._best_weights = weights.copy()
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if no_improvement_count >= self.n_iter_no_change:
                    print(f"Early stopping triggered at epoch {i+1} (SGD).")
                    break # Exit epoch loop

        # --- Finalize Weights ---
        final_weights = self._best_weights if self._best_weights is not None else weights
        self.intercept_ = final_weights[0]
        self.coef_ = final_weights[1:]
        return self


    def _mini_batch_gradient_descent(self, X_train, y_train, batch_size=32, adaptive_lr=False, X_val=None, y_val=None):
        n_samples, n_features = X_train.shape
        X_train_int = self._add_intercept(X_train)
        weights = np.zeros(n_features + 1)

        if batch_size <= 0 or batch_size > n_samples:
             batch_size = min(max(1, batch_size), n_samples) # Basic validation

        # Early Stopping Setup
        perform_early_stopping = X_val is not None and y_val is not None
        X_val_int = self._add_intercept(X_val) if perform_early_stopping else None
        best_val_loss = np.inf
        no_improvement_count = 0
        self._best_weights = None

        for i in range(self.max_iter):
            X_shuffled, y_shuffled = shuffle(X_train_int, y_train, random_state=i)

            # --- Training Step (Batches within epoch) ---
            for j in range(0, n_samples, batch_size):
                end_idx = min(j + batch_size, n_samples)
                X_batch = X_shuffled[j:end_idx]
                y_batch = y_shuffled[j:end_idx]
                gradient = self._compute_gradient(X_batch, y_batch, weights)
                lr = self._get_adaptive_learning_rate(i + j/n_samples) if adaptive_lr else self.eta0
                weights -= lr * gradient

            # --- Calculate Losses and Check Early Stopping (End of Epoch) ---
            current_train_loss = self._compute_loss(X_train_int, y_train, weights) # Loss on full train set
            self.history_.append(current_train_loss)

            if perform_early_stopping:
                current_val_loss = self._compute_loss(X_val_int, y_val, weights)
                self.val_history_.append(current_val_loss)

                if current_val_loss < best_val_loss - self.tol:
                    best_val_loss = current_val_loss
                    self._best_weights = weights.copy()
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if no_improvement_count >= self.n_iter_no_change:
                    print(f"Early stopping triggered at epoch {i+1} (Mini-Batch GD).")
                    break # Exit epoch loop

        # --- Finalize Weights ---
        final_weights = self._best_weights if self._best_weights is not None else weights
        self.intercept_ = final_weights[0]
        self.coef_ = final_weights[1:]
        return self

    def fit(self, X, y, method='batch', batch_size=32, adaptive_lr=False):
        X = np.asarray(X)
        y = np.asarray(y).ravel() # Ensure y is 1D

        # Reset state for a new fit
        self.history_ = []
        self.val_history_ = []
        self._best_weights = None
        self.intercept_ = None
        self.coef_ = None

        X_train, y_train = X, y
        X_val, y_val = None, None

        # Prepare validation set if early stopping is enabled
        if self.early_stopping:
            if 0.0 < self.validation_split < 1.0:
                try:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=self.validation_split, random_state=42
                    )
                    if len(X_val) == 0: # Handle case of empty validation set
                        warnings.warn("Validation split resulted in an empty set. Disabling early stopping for this fit.")
                        X_train, y_train = X, y # Use all data for training
                        X_val, y_val = None, None
                except ValueError as e:
                     warnings.warn(f"Could not create validation split (Error: {e}). Disabling early stopping for this fit.")
                     X_train, y_train = X, y
                     X_val, y_val = None, None
            else:
                warnings.warn("validation_split must be > 0 and < 1 for early stopping. Disabling early stopping for this fit.")

        # Call the selected gradient descent method
        common_args = (X_train, y_train)
        common_kwargs = {'adaptive_lr': adaptive_lr, 'X_val': X_val, 'y_val': y_val}

        if method == 'batch':
            self._batch_gradient_descent(*common_args, **common_kwargs)
        elif method == 'sgd':
            self._stochastic_gradient_descent(*common_args, **common_kwargs)
        elif method == 'mini_batch':
            self._mini_batch_gradient_descent(*common_args, batch_size=batch_size, **common_kwargs)
        else:
            raise ValueError(f"Unknown method: {method}. Choose from 'batch', 'sgd', or 'mini_batch'")

        # Check if fitting actually produced coefficients
        if self.intercept_ is None or self.coef_ is None:
             warnings.warn("Model fitting did not complete successfully.")

        return self

    def predict(self, X):
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Model not trained yet or training failed. Call fit() first.")

        X = np.asarray(X)
        X_with_intercept = self._add_intercept(X)
        # Reconstruct full weights vector for prediction
        weights = np.concatenate(([self.intercept_], self.coef_))
        return np.dot(X_with_intercept, weights)

    def score(self, X, y):
        y = np.asarray(y).ravel() # Ensure y is 1D
        try:
            y_pred = self.predict(X)
        except RuntimeError:
            return np.nan
        return r2_score(y, y_pred)

    def get_params(self, deep=True):
        return {
            'eta0': self.eta0,
            'max_iter': self.max_iter,
            'early_stopping': self.early_stopping,
            'validation_split': self.validation_split,
            'n_iter_no_change': self.n_iter_no_change,
            'tol': self.tol,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            if parameter in self.get_params():
                setattr(self, parameter, value)
            else:
                raise ValueError(f"Invalid parameter {parameter} for estimator {self}.")
        return self