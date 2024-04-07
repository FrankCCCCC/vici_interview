# %%
import pickle

import numpy as np

from sklearn import datasets, manifold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# unused but required import for doing 3d projections with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F401
from matplotlib import ticker

# %%
def data_range_shape(data: np.ndarray):
    print(data.shape)
    print(data)
    print(f"Value Range: [{np.min(data)}:{np.max(data)}]")
    
def label_range_shape(label: np.ndarray):
    print(label.shape)
    print(label)
    uniques, counts = np.unique(label, return_counts=True)
    total = sum(counts)
    lab = []
    nums = []
    ratios = []
    infos = []
    for unique, count in zip(uniques, counts):
        lab.append(unique)
        nums.append(count)
        ratios.append(count / total)
        infos.append(f"{count} ({str(round(count / total * 100, 2))}%)")
    print(f"Class Num: {np.unique(label)}, {dict(zip(uniques, infos))}")
    
def data_amplify(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, amplify: None):
    duplication = 1
    uniques = np.unique(y_train)
    if amplify is None:
        duplication = [1] * uniques
    elif isinstance(amplify, int):
        duplication = [amplify] * uniques
    elif isinstance(amplify, list):
        if len(amplify) == 1:
            duplication = [amplify] * uniques
        elif len(amplify) != uniques:
            raise ValueError(f'the length of amplify list should be the same as the number of unique values')
        else:
            duplication = amplify
    else:
        raise TypeError(f'amplify should be None, int, or list, not {type(amplify)}')
    
    

# %%
train_data = np.load('Test/train_data.npy')
data_range_shape(data=train_data)

train_labels = np.load('Test/train_labels.npy')
label_range_shape(label=train_labels)

# %%
eval_data = np.load('Test/eval_data.npy')
data_range_shape(data=eval_data)

eval_labels = np.load('Test/eval_labels.npy')
label_range_shape(label=eval_labels)

# %%
sc = StandardScaler().fit(train_data)
train_data_sc = sc.transform(train_data)
eval_data_sc = sc.transform(eval_data)
data_range_shape(train_data_sc)

# %%
# pca = PCA(n_components=50).fit(train_data_sc)
# train_data_sc = pca.transform(train_data_sc)
# eval_data_sc = pca.transform(eval_data_sc)
# data_range_shape(train_data_sc)

# %%
# iso = IsolationForest(n_estimators=100, random_state=0, n_jobs=-1, warm_start=True).fit(train_data_sc)
# outliner_idxs = iso.predict(train_data_sc)
# train_data_sc = train_data_sc[outliner_idxs == 1]
# train_labels = train_labels[outliner_idxs == 1]

# data_range_shape(train_data_sc)
# print((outliner_idxs[outliner_idxs == 1]).shape)
# print((train_labels[outliner_idxs == 1]).shape)
# print(tmp.shape)

# %%
def correlation_map(data):
    d = pd.DataFrame(data=data, columns=list(range(data.shape[1])))

    # Compute the correlation matrix
    corr = d.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
correlation_map(train_data)

# %% [markdown]
# # Manifold Compression

# %%
def plot_3d(points, points_color, title):
    x, y, z = points.T

    fig, ax = plt.subplots(
        figsize=(24, 24),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    fig.suptitle(title, size=16)
    col = ax.scatter(x, y, z, c=points_color, s=50, alpha=0.8)
    ax.view_init(azim=-60, elev=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    plt.show()


def plot_2d(points, points_color, title):
    fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, points_color)
    plt.show()


def add_2d_scatter(ax, points, points_color, title=None):
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=50, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())


# %%
n_components = 2
params = {
    "n_neighbors": 12,
    "n_components": n_components,
    "eigen_solver": "auto",
    "random_state": 0,
}

manifold_X_train, _, manifold_y_train, _ = train_test_split(train_data_sc, train_labels, train_size=0.001, shuffle=True, random_state=42)
color_map = ['tab:blue', 'tab:orange', 'tab:green']
S_color = [color_map[y] for y in manifold_y_train]

# %%
# lle_standard = manifold.LocallyLinearEmbedding(method="standard", **params)
# S_standard = lle_standard.fit_transform(train_data_sc)

# plot_3d(S_standard, S_color, "Locally Linear Embedding")

# lle_ltsa = manifold.LocallyLinearEmbedding(method="ltsa", **params)
# S_ltsa = lle_ltsa.fit_transform(train_data_sc)

# lle_hessian = manifold.LocallyLinearEmbedding(method="hessian", **params)
# S_hessian = lle_hessian.fit_transform(train_data_sc)

# lle_mod = manifold.LocallyLinearEmbedding(method="modified", **params)
# S_mod = lle_mod.fit_transform(train_data_sc)

# %%
tsne = manifold.TSNE(n_components=n_components,
                    init="random",
                    perplexity=1000,
                    learning_rate="auto",
                    n_iter=300
                    ).fit_transform(manifold_X_train)

# plot_3d(tsne, S_color, "TSNE")
plot_2d(tsne, S_color, "TSNE")

# %%
def cofusion_matrix(clf, X_test: np.array, y_test: np.array):
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Normalized confusion matrix", "true"),
    ]
    for title, normalize in titles_options:
        disp = ConfusionMatrixDisplay.from_estimator(
            clf,
            X_test,
            y_test,
            display_labels=[0, 1, 2],
            cmap=plt.cm.Blues,
            normalize=normalize,
        )
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.show()
    
def plot_importance(importances: np.array, clf=None):
    if clf != None:
        std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    forest_importances = pd.Series(importances)

    fig, ax = plt.subplots()
    if clf != None:
        forest_importances.plot.bar(yerr=std, ax=ax)
    else:
        forest_importances.plot.bar(ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

# %%
def evaluate_model(clf, eval_data: np.array, eval_labels: np.array, train_data: np.array, train_labels: np.array):
    # Testing Dataset
    eval_pred_label = clf.predict(eval_data)
    eval_acc: float = accuracy_score(eval_labels, eval_pred_label)
    eval_f1: float = f1_score(eval_labels, eval_pred_label, average='weighted')
    print(f"Test Acc: {eval_acc}, Test F1: {eval_f1}")
    cofusion_matrix(clf, eval_data, eval_labels)

    # Training Dataset
    train_pred_label = clf.predict(train_data)
    train_acc: float = accuracy_score(train_labels, train_pred_label)
    train_f1: float = f1_score(train_labels, train_pred_label, average='weighted')
    print(f"Train Acc: {train_acc}, Train F1: {train_f1}")
    cofusion_matrix(clf, train_data, train_labels)
    return eval_acc, eval_f1, train_acc, train_f1

# %% [markdown]
# # Random Forest

# %%
# clf = RandomForestClassifier(max_depth=None, n_estimators=100, random_state=0, n_jobs=-1)
# clf.fit(train_data_sc, train_labels)

# %%
# eval_acc, eval_f1, train_acc, train_f1 = evaluate_model(clf=clf, eval_data=eval_data_sc, eval_labels=eval_labels, train_data=train_data_sc, train_labels=train_labels)

# %%
# importances = clf.feature_importances_
# plot_importance(importances=importances, clf=clf)

# %% [markdown]
# # Ridge Classifier

# %%
# ridge_clf = RidgeClassifier(alpha=0, tol=1e-4, max_iter=10000, class_weight='balanced').fit(train_data_sc, train_labels)
# eval_acc, eval_f1, train_acc, train_f1 = evaluate_model(clf=ridge_clf, eval_data=eval_data_sc, eval_labels=eval_labels, train_data=train_data_sc, train_labels=train_labels)

# %% [markdown]
# # Support Vector Classifier

# %%
# svc_clf = SVC(C=1.0, 
#               kernel='rbf', 
#               gamma='scale',
#               tol=1e-3,
#               ).fit(train_data_sc, train_labels)
# eval_acc, eval_f1, train_acc, train_f1 = evaluate_model(clf=svc_clf, eval_data=eval_data_sc, eval_labels=eval_labels, train_data=train_data_sc, train_labels=train_labels)

# %% [markdown]
# # XGBoost

# %%
best_param = {'colsample_bytree': 1, 'gamma': 0.2, 'learning_rate': 0.5, 'max_depth': 10, 'min_child_weight': 1000, 'reg_lambda': 1, 'scale_pos_weight': 0.5, 'subsample': 1}

xgboostModel = XGBClassifier(
                            # learning_rate=0.01,
                            n_estimators=1000,
                            # max_depth=5,
                            # min_child_weight=5000,
                            # gamma=1,
                            # subsample=0.5,
                            # lambda=1,
                            # colsample_bytree=0.8,
                            # reg_alpha=5,
                            # scale_pos_weight=1
                            *best_param
                            )
xgboostModel.fit(train_data_sc, train_labels)
# pred_label = xgboostModel.predict(eval_data_sc)

# %%
importances = xgboostModel.feature_importances_
plot_importance(importances=importances)

# %%
eval_acc, eval_f1, train_acc, train_f1 = evaluate_model(clf=xgboostModel, eval_data=eval_data_sc, eval_labels=eval_labels, train_data=train_data_sc, train_labels=train_labels)

# %% [markdown]
# # Check Training-Testing Distribution Shift

# %%
# data_sc: np.array = np.concatenate((train_data_sc, eval_data_sc), axis=0)
# labels: np.array = np.concatenate((train_labels, eval_labels), axis=None)
# X_test, X_train, y_test, y_train = train_test_split(data_sc, labels, test_size=0.5, random_state=42, shuffle=True)


# %%
# clf1 = RandomForestClassifier(max_depth=None, n_estimators=100, random_state=0, n_jobs=-1)
# clf1.fit(X_train, y_train)
# eval_acc, eval_f1, train_acc, train_f1 = evaluate_model(clf=clf1, eval_data=X_test, eval_labels=y_test, train_data=X_train, train_labels=y_train)

# xgboostModel = XGBClassifier(n_estimators=100,max_depth=20,)
# xgboostModel.fit(X_train, y_train)
# eval_acc, eval_f1, train_acc, train_f1 = evaluate_model(clf=xgboostModel, eval_data=X_test, eval_labels=y_test, train_data=X_train, train_labels=y_train)

# %% [markdown]
# # Grid Search

# %%
param_test3 = {
 'learning_rate': [0.01, 0.05, 0.1, 0.5, 1],
 'gamma':[i/10.0 for i in range(0,5)],
 'max_depth': [3, 5, 7, 10, 15, 20],
 'min_child_weight': [10, 100, 1000], 
 'subsample': [0.8],
 'colsample_bytree': [0.8],
 'scale_pos_weight': [1],
}

param_test3 = {
 'learning_rate': [0.5],
 'gamma':[0.2],
 'max_depth': [10],
 'min_child_weight': [10, 100, 1000, 5000, 10000], 
 'subsample': [0.5, 0.8, 1],
 'colsample_bytree': [0.5, 0.8, 1],
 'reg_lambda': [0.01, 0.1, 0.5, 1, 10],
#  'reg_alpha': [],
 'scale_pos_weight': [0.5, 1, 2],
}

from sklearn.model_selection import GridSearchCV

gsearch = GridSearchCV(estimator = XGBClassifier(n_estimators=100,objective= 'multi:softmax',seed=27), 
 param_grid = param_test3, scoring='f1_weighted',n_jobs=-1, cv=5)
gsearch.fit(train_data_sc, train_labels)

with open('gsearch_stage2.pkl', 'wb') as f:
    pickle.dump(gsearch, f, pickle.HIGHEST_PROTOCOL)

# %%
with open('gsearch_stage2.pkl', 'rb') as f:
    grid_search = pickle.load(f)
# print(grid_search.cv_results_)
print(grid_search.best_params_)
print(grid_search.best_score_)

# %%



