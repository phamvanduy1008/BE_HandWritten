import numpy as np

def knn_predict(X_train, y_train, X_test, k=10):
    distances = np.sqrt(((X_train - X_test) ** 2).sum(axis=1))
    indices = np.argpartition(distances, k)[:k]
    nearest_labels = y_train[indices]
    return np.bincount(nearest_labels).argmax()

class SVM:
    def __init__(self, num_classes=10, num_features=784, lr=0.0001, reg_lambda=0.001, epochs=10):
        self.num_classes = num_classes
        self.num_features = num_features
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.epochs = epochs
        
        self.num_uncertain = 0
        self.W = np.random.randn(self.num_classes, self.num_features) * 0.01
        self.b = np.zeros(self.num_classes)

    def compute_loss(self, X, y):
        n_samples = X.shape[0]
        total_loss = 0
        for i in range(n_samples):
            x_i = X[i]
            y_true = y[i]
            y_ovr = np.where(np.arange(self.num_classes) == y_true, 1, -1)
            margins = 1 - y_ovr * (np.dot(self.W, x_i) + self.b)
            margins = np.maximum(0, margins)
            total_loss += np.sum(margins)
        total_loss /= n_samples
        total_loss += (self.reg_lambda / 2) * np.sum(self.W ** 2)
        return total_loss

    def train(self, X, y):
        n_samples = X.shape[0]
        for epoch in range(self.epochs):
            for i in range(n_samples):
                x_i = X[i]
                y_true = y[i]
                y_ovr = np.where(np.arange(self.num_classes) == y_true, 1, -1)
                margins = y_ovr * (np.dot(self.W, x_i) + self.b)
                for j in range(self.num_classes):
                    if margins[j] < 1:
                        self.W[j] -= self.lr * (self.reg_lambda * self.W[j] - y_ovr[j] * x_i)
                        self.b[j] -= self.lr * (-y_ovr[j])
                    else:
                        self.W[j] -= self.lr * (self.reg_lambda * self.W[j])
            loss = self.compute_loss(X, y)
            y_pred = self.predict(X, 0, 0, None, None)  # Gọi predict với X_train, y_train là None trong train
            acc = np.mean(y_pred == y)
            print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {loss:.4f} - Train Accuracy: {acc:.4f}")

    def predict(self, X, lower_threshold, upper_threshold, X_train=None, y_train=None):
        if X.ndim == 1:
            X = X.reshape(1, -1)

        scores = np.dot(self.W, X.T) + self.b[:, np.newaxis]
        predictions = np.argmax(scores, axis=0)

        margins = np.max(scores, axis=0) - np.sort(scores, axis=0)[-2]
        uncertain_indices = np.where((margins < upper_threshold) & (margins > lower_threshold))[0]

        self.num_uncertain = len(uncertain_indices)
        if self.num_uncertain > 0:
            print(f"Số ảnh không chắc chắn: {self.num_uncertain}")
            if X_train is not None and y_train is not None:
                for idx in uncertain_indices:
                    predictions[idx] = knn_predict(X_train, y_train, X[idx])
            else:
                print("Cảnh báo: X_train hoặc y_train không được cung cấp, không thể dùng KNN.")

        return predictions