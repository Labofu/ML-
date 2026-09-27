# XGBoost

XGBoost (Extreme Gradient Boosting) is an optimized and regularized implementation of gradient boosting. It builds decision trees sequentially, where each new tree tries to correct the errors made by the previous trees.

## Gradient Boosting vs XGBoost

### Gradient Boosting
- Builds trees sequentially.
- Each new tree focuses on the errors/residuals of the previous model.
- Usually slower because tree construction is less optimized.
- Can overfit if the model is too complex.
- Provides fewer built-in optimization and regularization techniques.

### XGBoost
XGBoost improves upon traditional Gradient Boosting by adding several optimizations:

- **Regularization (L1 & L2)** → helps reduce overfitting.
- **Parallel processing** → makes tree construction faster.
- **Subsampling** → can use only a fraction of rows for each tree.
- **Column sampling** → can use a fraction of features for each tree.
- **Missing-value handling** → can automatically learn how to handle missing values.
- **Efficient tree-building algorithms** → generally faster and more memory-efficient.
- **Early stopping** → can stop training when additional trees no longer improve performance.

## Important Hyperparameters

- `n_estimators` → number of trees.
- `learning_rate` → contribution of each tree.
- `max_depth` → maximum depth of each tree.
- `subsample` → fraction of training rows used for each tree.
- `colsample_bytree` → fraction of features used for each tree.
- `gamma` → minimum loss reduction required for a split.
- `reg_alpha` → L1 regularization.
- `reg_lambda` → L2 regularization.

## Advantages

- Usually gives strong performance on tabular data.
- Faster and more optimized than traditional Gradient Boosting.
- Has several mechanisms to control overfitting.
- Supports parallel computation.
- Handles missing values automatically.
- Provides extensive hyperparameter control.

## Disadvantages

- More hyperparameters to understand and tune.
- Can still overfit if poorly configured.
- Training can become computationally expensive with many trees or complex trees.
- Less interpretable than a single decision tree.
- Requires careful tuning to get the best performance.

## Key Idea

### Gradient Boosting

Trees are added sequentially:

Tree 1 → Tree 2 → Tree 3 → ... → Final Model

Each new tree tries to improve the current model by fitting the
negative gradient of the loss function.

> Note: "Gradient Boosting uses Information Gain" is not generally
> correct. The exact tree-splitting criterion depends on the
> implementation. For example, sklearn's GradientBoostingClassifier
> uses regression trees with `friedman_mse` by default.

### XGBoost

XGBoost follows the same sequential boosting idea, but the tree
construction is more mathematically optimized.

For each boosting iteration:

Current Model
      ↓
Calculate Gradients + Hessians
      ↓
Evaluate candidate splits using a regularized gain
      ↓
Build the next tree
      ↓
Add the tree to the model

Important:

- Tree building is part of the training process.
- Boosting iterations are sequential.
- XGBoost uses parallel computation inside the tree-building process.
- XGBoost uses both first-order gradients and second-order Hessians.
- XGBoost's split/gain calculation is different from the
  traditional implementation of Gradient Boosting.
- Regularization is included in the objective to help control
  overfitting.

### Subsampling

If:

`subsample = 0.8`

then each boosting iteration uses a random 80% of the training
rows to build the new tree.

### Important Distinction

Do NOT think:

Tree 1, Tree 2, Tree 3 → built independently in parallel ❌

Think:

Tree 1 → Tree 2 → Tree 3 → sequential boosting
          +
Parallel computation during the construction of each tree ✅
