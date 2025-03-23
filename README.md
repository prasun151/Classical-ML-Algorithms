# Machine Learning Algorithm Implementations

A collection of machine learning algorithms implemented from scratch for the Iris dataset classification problem.

## Algorithms Implemented

### 1. Linear Regression
- Basic implementation using normal equation method
- Accuracy: ~82%
- Features: Simple matrix operations, no iterative optimization

### 2. Logistic Regression
- Gradient descent optimization
- Accuracy: ~87%
- Features: Sigmoid activation, iterative training

### 3. Newton's Method Logistic Regression
- Second-order optimization
- Accuracy: ~89%
- Features: Faster convergence using Hessian matrix

### 4. K-Nearest Neighbors (KNN)
- Non-parametric learning
- Accuracy: ~93%
- Features: Distance-based classification, no training phase

### 5. Random Forest
- Ensemble learning method
- Accuracy: ~95%
- Features: Bootstrap sampling, feature randomization

### 6. Decision Tree
- Tree-based learning
- Accuracy: ~91%
- Features: Gini impurity, recursive splitting

### 7. Gaussian Naive Bayes
- Probabilistic classifier
- Accuracy: ~94%
- Features: Assumes feature independence

### 8. Linear Discriminant Analysis (LDA)
- Dimensionality reduction and classification
- Accuracy: ~96%
- Features: Class-conditional densities

### 9. Gradient Boosting
- Sequential ensemble method
- Accuracy: ~97%
- Features: Additive modeling, gradient optimization

## Performance Comparison

| Algorithm                    | Accuracy | Training Speed | Prediction Speed | Memory Usage |
|-----------------------------|---------:|----------------|------------------|--------------|
| Linear Regression           |    82%   | Very Fast      | Very Fast        | Low          |
| Logistic Regression         |    87%   | Medium         | Fast            | Low          |
| Newton's Method            |    89%   | Fast           | Fast            | Medium       |
| KNN                        |    93%   | None           | Slow            | High         |
| Random Forest              |    95%   | Medium         | Medium          | High         |
| Decision Tree              |    91%   | Fast           | Fast            | Medium       |
| Naive Bayes                |    94%   | Very Fast      | Fast            | Low          |
| LDA                        |    96%   | Fast           | Fast            | Low          |
| Gradient Boosting          |    97%   | Slow           | Medium          | High         |

## Implementation Details

Each algorithm is implemented with:
- Pure NumPy operations
- Standardized input/output interface
- Visualization capabilities
- Probability estimation where applicable
- Cross-validation support

## Usage

Basic example:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Prepare data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Train model (example with Gradient Boosting)
model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1)
model.fit(X_train, y_train)

# Evaluate
accuracy = np.mean(model.predict(X_test) == y_test)
```

## Visualizations

Each implementation includes:
- Prediction visualization
- Probability distribution plots
- Uncertainty estimation
- Decision boundaries where applicable

## Dependencies

- NumPy
- Pandas
- Matplotlib
- Scikit-learn (for data splitting and preprocessing only)