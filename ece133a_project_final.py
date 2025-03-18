import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, \
    confusion_matrix
import joblib
import os
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier


def part1():
    df = pd.read_csv("./OnlineNewsPopularity.csv")

    # preprocessing & Clean-up
    df = df.iloc[:, 2:]  # remove first two columns (URL, timedelta)

    missing_vals = df.isnull().sum().sum()
    print(f"Total missing values: {missing_vals}")

    df_mod = df.dropna().reset_index(drop=True)  # remove rows with missing values
    m = df_mod.to_numpy()  # Convert to NumPy array
    ft_names = df_mod.columns.tolist()

    # selecting key features based on SVD results
    key_features = [' n_tokens_title', ' n_tokens_content', ' n_unique_tokens',
                    ' num_hrefs', ' num_self_hrefs', ' num_imgs', ' num_videos']
    exist_keys = list(set(df_mod.columns) & set(key_features))
    m_keys = df_mod[exist_keys].to_numpy()

    '''
    print(m.shape)
    print(m_keys.shape)
    '''

    return m, m_keys, ft_names


def part2(std_m, m_keys, ft_names):
    # feature mean and std
    ft_means = np.mean(std_m, axis=0)
    ft_stdvs = np.std(std_m, axis=0)
    ft_stats = pd.DataFrame({'ft': ft_names, 'Mean': ft_means, 'Std Dev': ft_stdvs})
    print("\nft stats:\n", ft_stats)

    # k-Means clustering for all features (elbow)
    sq_euc_dist = []
    k_values = range(1, 21)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(std_m)
        sq_euc_dist.append(kmeans.inertia_)

    best_k = np.argmin(np.diff(sq_euc_dist)) + 1  # stable elbow selection
    print(f"\nbest # of clusters from elbow method: {best_k}")

    '''
    # plot elbow curve
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, sq_euc_dist, marker='o', linestyle='-', color='b')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (Sum of Squared Distances)")
    plt.title("Elbow Method for Best k")
    plt.xticks(k_values)  # Ensure k values are labeled on the x-axis
    plt.grid(True)
    plt.show()
    '''

    # silhouette analysis for all features
    sil_avgs = {}
    sil_scores = []
    for k in range(2, 21):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=20)
        labels = kmeans.fit_predict(std_m)
        sil_avgs[k] = silhouette_score(std_m, labels)
        sil_scores.append(silhouette_score(std_m, labels))

    best_k = max(sil_avgs, key=sil_avgs.get)
    print(f"best # of clusters from silhouette method: {best_k}")

    '''
    # plot silhouette score
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, 21), sil_scores, marker='o', linestyle='-', color='g')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Analysis for Best k")
    plt.xticks(range(2, 21))
    plt.grid(True)
    plt.show()
    '''

    # k-Means clustering for selected features (elbow)
    sq_euc_dist_keys = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(std_m)
        sq_euc_dist_keys.append(kmeans.inertia_)

    best_k = np.argmin(np.diff(sq_euc_dist_keys)) + 1
    print(f"\nbest # of clusters from elbow method (key clusters): {best_k}")

    '''
    # plot elbow curve
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, sq_euc_dist, marker='o', linestyle='-', color='b')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (Sum of Squared Distances)")
    plt.title("(Selected Features) Elbow Method for Best k")
    plt.xticks(k_values)  # Ensure k values are labeled on the x-axis
    plt.grid(True)
    plt.show()
    '''

    # silhouette analysis for selected features
    sil_avgs_keys = {}
    sil_scores_keys = []
    for k in range(2, 21):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=20)
        labels = kmeans.fit_predict(std_m)
        sil_avgs_keys[k] = silhouette_score(std_m, labels)
        sil_scores_keys.append(silhouette_score(std_m, labels))

    best_k = max(sil_avgs_keys, key=sil_avgs_keys.get)
    print(f"best # of clusters from silhouette method (key clusters): {best_k}")

    '''
    # plot Silhouette Score
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, 21), sil_scores_keys, marker='o', linestyle='-', color='g')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("(Selected Features) Silhouette Analysis for Best k")
    plt.xticks(range(2, 21))
    plt.grid(True)
    plt.show()
    '''

    # SVD
    U, S, Vt = np.linalg.svd(std_m, full_matrices=False)
    sg_vals = S[:10]
    sg_fts = [ft_names[i] for i in np.argsort(-Vt[0])[:10]]  # using Vt[0] for ranking features

    svd_result = pd.DataFrame({'singular val': sg_vals, 'ft': sg_fts})
    print("\ntop SVD fts:\n", svd_result)

    # correlation matrix
    corr_m = np.corrcoef(std_m, rowvar=False)
    threshold = 0.8
    sig_corr_fts = [
        (ft_names[i], ft_names[j], corr_m[i, j])
        for i in range(len(ft_names))
        for j in range(i + 1, len(ft_names))
        if abs(corr_m[i, j]) > threshold
    ]

    corr_result = pd.DataFrame(sig_corr_fts, columns=['ft 1', 'ft 2', 'correlation'])
    print("\nsignificant correlated fts:\n", corr_result)


def part3a(std_m, ft_names):
    # target var (shares) and input features
    X = std_m  # complete feature space
    y = (std_m[:, -1] > np.median(std_m[:, -1])).astype(int)  # binary classify as "popular vs. not popular"

    # k-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold = 1
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    auc_scores = []
    rms_errors = []
    model_params = []
    os.makedirs("models", exist_ok=True)

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # linear regression classifier
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_prob = model.predict(X_test)  # raw probabilities
        y_prob = np.clip(y_prob, 0, 1)  # ensure probabilities are within valid range
        y_pred = (y_prob >= 0.5).astype(int)  # threshold to get binary output

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        rms_error = np.sqrt(mean_squared_error(y_test, y_prob))

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        auc_scores.append(auc)
        rms_errors.append(rms_error)

        # save model parameters (coefficients)
        model_params.append(model.coef_)
        joblib.dump(model, f"models/linear_regression_fold{fold}.pkl")

        print(
            f"fold {fold} - accuracy: {accuracy:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, "
            f"f1: {f1:.4f}, auc: {auc:.4f}, rms error: {rms_error:.4f}"
        )

        fold += 1

    joblib.dump(model_params, "models/linear_regression_params.pkl")  # save model parameters

    print(f"\navg scores across folds (least squares classifier):")
    print(f"avg accuracy: {np.mean(accuracy_scores):.4f}")
    print(f"avg precision: {np.mean(precision_scores):.4f}")
    print(f"avg recall: {np.mean(recall_scores):.4f}")
    print(f"avg f1 score: {np.mean(f1_scores):.4f}")
    print(f"avg auc: {np.mean(auc_scores):.4f}")
    print(f"avg rms error: {np.mean(rms_errors):.4f}")

    # compute numerical stability summary
    model_params = np.array(model_params)  # convert list to numpy array
    # print(model_params)
    param_means = np.mean(model_params, axis=0)
    param_stds = np.std(model_params, axis=0)

    print("\nnumerical stability summary (linear regression coefficients):")
    print(f"mean parameter values: {np.mean(param_means):.6f}")
    print(f"std deviation of parameters: {np.mean(param_stds):.6f}")

    return model_params


def part3b(std_m, ft_names):
    # convert to dataframe
    df = pd.DataFrame(std_m, columns=ft_names)

    # apply transformations
    df_sqrt = np.sqrt(np.abs(df))  # square-root transformation for numerical stability
    df_sqrt.columns = [col + '_sqrt' for col in df.columns]
    df_square = np.square(df)  # squared transformation for feature interaction
    df_square.columns = [col + '_square' for col in df.columns]

    # concatenate transformed features
    df = pd.concat([df, df_sqrt, df_square], axis=1)

    # final feature selection
    X_ready = df.to_numpy()
    y_ready = (std_m[:, -1] > np.median(std_m[:, -1])).astype(int)

    # split data into train and test sets (80:20)
    X_train, X_test, y_train, y_test = train_test_split(X_ready, y_ready, test_size=0.2, random_state=42)

    # k-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold = 1
    accuracy_scores, precision_scores, recall_scores, f1_scores, auc_scores, rms_errors = [], [], [], [], [], []
    model_params = []
    os.makedirs("models", exist_ok=True)

    for train_idx, test_idx in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[test_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[test_idx]

        # train model
        model = LinearRegression()
        model.fit(X_train_fold, y_train_fold)

        # predictions
        y_prob = np.clip(model.predict(X_val_fold), 0, 1)
        y_pred = (y_prob >= 0.5).astype(int)

        # performance metrics
        accuracy_scores.append(accuracy_score(y_val_fold, y_pred))
        precision_scores.append(precision_score(y_val_fold, y_pred, zero_division=0))
        recall_scores.append(recall_score(y_val_fold, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_val_fold, y_pred))
        auc_scores.append(roc_auc_score(y_val_fold, y_prob))
        rms_errors.append(np.sqrt(mean_squared_error(y_val_fold, y_prob)))

        # save model parameters (coefficients)
        model_params.append(model.coef_)
        joblib.dump(model, f"models/linear_regression_fold{fold}_feature_eng.pkl")

        print(f"fold {fold} - accuracy: {accuracy_scores[-1]:.4f}, precision: {precision_scores[-1]:.4f}, "
              f"recall: {recall_scores[-1]:.4f}, f1: {f1_scores[-1]:.4f}, "
              f"auc: {auc_scores[-1]:.4f}, rms error: {rms_errors[-1]:.4f}")
        fold += 1

    joblib.dump(model_params, "models/linear_regression_params_feature_eng.pkl")

    # display average performance across folds
    print(f"\navg scores across folds (least squares classifier, post feature engr):")
    print(f"avg accuracy: {np.mean(accuracy_scores):.4f}")
    print(f"avg precision: {np.mean(precision_scores):.4f}")
    print(f"avg recall: {np.mean(recall_scores):.4f}")
    print(f"avg f1 score: {np.mean(f1_scores):.4f}")
    print(f"avg auc: {np.mean(auc_scores):.4f}")
    print(f"avg rms error: {np.mean(rms_errors):.4f}")

    # compute numerical stability summary
    model_params = np.array(model_params)  # convert list to numpy array
    param_means = np.mean(model_params, axis=0)
    param_stds = np.std(model_params, axis=0)

    print("\nnumerical stability summary (linear regression coefficients):")
    print(f"mean parameter values: {np.mean(param_means):.6f}")
    print(f"std deviation of parameters: {np.mean(param_stds):.6f}")

    # evaluate best model on test set
    best_model = LinearRegression()
    best_model.fit(X_train, y_train)
    y_pred_best = (best_model.predict(X_test) >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred_best)
    print("\nconfusion matrix:\n", cm)
    print(f"parameter norm: {np.linalg.norm(best_model.coef_):.4f}")

    return best_model


def part3c(std_m, ft_names, prev_best_model):
    # reconvert to dataframe
    df = pd.DataFrame(std_m, columns=ft_names)

    df_sqrt = np.sqrt(np.abs(df))
    df_sqrt.columns = [col + '_sqrt' for col in df.columns]
    df_square = np.square(df)
    df_square.columns = [col + '_square' for col in df.columns]
    df = pd.concat([df, df_sqrt, df_square], axis=1)
    df = pd.concat([df, df_sqrt], axis=1)

    X_ready = df.to_numpy()
    Y_ready = (std_m[:, -1] > np.median(std_m[:, -1])).astype(int)

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_ready, Y_ready, test_size=0.2, random_state=42)

    # extract best model parameters from part3b and remove unsupported parameters
    best_params = prev_best_model.get_params()
    valid_params = {k: v for k, v in best_params.items() if k in Ridge().get_params()}

    # define regularization parameters
    reg_params = [0.01, 0.1, 1, 10, 100]
    best_model = None
    best_param = None
    best_score = float('inf')

    for reg in reg_params:
        model = Ridge(alpha=reg, **valid_params)
        model.fit(X_train, y_train)
        y_prob = model.predict(X_test)
        y_prob = np.clip(y_prob, 0, 1)
        y_pred = (y_prob >= 0.5).astype(int)
        rms_error = np.sqrt(mean_squared_error(y_test, y_prob))

        if rms_error < best_score:
            best_score = rms_error
            best_model = model
            best_param = reg

    # final predictions using best model
    y_pred_best = best_model.predict(X_test)
    y_pred_best = (y_pred_best >= 0.5).astype(int)
    y_prob_best = np.clip(best_model.predict(X_test), 0, 1)

    # compute performance metrics
    accuracy = accuracy_score(y_test, y_pred_best)
    precision = precision_score(y_test, y_pred_best, zero_division=0)
    recall = recall_score(y_test, y_pred_best, zero_division=0)
    f1 = f1_score(y_test, y_pred_best)
    auc = roc_auc_score(y_test, y_prob_best)

    # save best model norm
    param_norm = np.linalg.norm(best_model.coef_)

    # report results
    print(f"least squares classifier, post regularization.")
    print(f"best regularization parameter: {best_param}")
    print(f"rms error: {best_score:.4f}")
    print(f"accuracy: {accuracy:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"recall: {recall:.4f}")
    print(f"f1 score: {f1:.4f}")
    print(f"auc: {auc:.4f}")
    print(f"parameter norm: {param_norm:.4f}")

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred_best)
    print("\nconfusion matrix:\n", cm)

    return best_model, param_norm


def part3d(std_m, ft_names):
    # target variable (shares) and input features
    X = std_m
    y = (std_m[:, -1] > np.median(std_m[:, -1])).astype(int)  # binary classification: "popular vs. not popular"

    # explicit train-test split (80% train, 20% test)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # k-fold cross-validation (only on the 80% training set)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold = 1
    accuracy_scores, precision_scores, recall_scores, f1_scores, auc_scores, rms_errors = [], [], [], [], [], []
    model_params = []
    os.makedirs("models", exist_ok=True)

    best_model = None
    best_score = float('inf')
    best_fold = None

    for train_idx, val_idx in kf.split(X_train_full):
        X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
        y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]

        # non-linear least squares classifier using a simple neural network
        model = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='lbfgs', max_iter=500, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        # performance metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_prob)
        rms_error = np.sqrt(mean_squared_error(y_val, y_prob))

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        auc_scores.append(auc)
        rms_errors.append(rms_error)

        # save model parameters
        model_params.append(model.coefs_)  # store model weight matrices
        joblib.dump(model, f"models/nlls_fold{fold}.pkl")

        print(
            f"fold {fold} - accuracy: {accuracy:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, "
            f"f1: {f1:.4f}, auc: {auc:.4f}, rms err: {rms_error:.4f}"
        )

        # track best model based on lowest rms error
        if rms_error < best_score:
            best_score = rms_error
            best_model = model
            best_fold = fold

        fold += 1

    joblib.dump(model_params, "models/nlls_params.pkl")

    print("\navg scores across folds (nonlinear least squares classifier):")
    print(f"avg accuracy: {np.mean(accuracy_scores):.4f}")
    print(f"avg precision: {np.mean(precision_scores):.4f}")
    print(f"avg recall: {np.mean(recall_scores):.4f}")
    print(f"avg f1 score: {np.mean(f1_scores):.4f}")
    print(f"avg auc: {np.mean(auc_scores):.4f}")
    print(f"avg rms err: {np.mean(rms_errors):.4f}")

    # compute numerical stability summary for model parameters
    param_magnitudes = [np.linalg.norm(np.concatenate([p.flatten() for p in params])) for params in model_params]
    print("\nnumerical stability summary (weight matrices):")
    print(f"mean parameter norm: {np.mean(param_magnitudes):.6f}")
    print(f"std deviation of parameter norms: {np.std(param_magnitudes):.6f}")

    print(f"\nfold {best_fold} produced the best model with rms error: {best_score:.4f}")

    # confusion matrix for best model from cross-validation
    y_train_full_pred = best_model.predict(X_train_full)
    cm_train = confusion_matrix(y_train_full, y_train_full_pred)
    print("\nconfusion matrix for best model on training data:\n", cm_train)

    # evaluate the best model on the held-out 20% test set
    y_test_pred = best_model.predict(X_test)
    y_test_prob = best_model.predict_proba(X_test)[:, 1]

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_prob)
    test_rms_error = np.sqrt(mean_squared_error(y_test, y_test_prob))

    print("\nfinal evaluation on 20% test set:")
    print(f"test accuracy: {test_accuracy:.4f}")
    print(f"test precision: {test_precision:.4f}")
    print(f"test recall: {test_recall:.4f}")
    print(f"test f1 score: {test_f1:.4f}")
    print(f"test auc: {test_auc:.4f}")
    print(f"test rms err: {test_rms_error:.4f}")

    # confusion matrix for test set
    cm_test = confusion_matrix(y_test, y_test_pred)
    print("\nconfusion matrix on test set:\n", cm_test)

    return best_model


def main():
    print("part_1")
    m_p1, m_keys_p1, ft_names = part1()
    print("=================================================================")

    # standardization
    scale = StandardScaler()
    std_m = scale.fit_transform(m_p1)
    std_m_keys = scale.fit_transform(m_keys_p1)

    # print("part_2")
    # part2(std_m, std_m_keys, ft_names)
    # print("=================================================================")

    print("part_3a")
    part3a(std_m, ft_names)
    print("=================================================================")

    print("part_3b")
    best_model = part3b(std_m, ft_names)
    print("=================================================================")

    print("part_3c")
    part3c(std_m, ft_names, best_model)
    print("=================================================================")

    print("part_3d")
    part3d(std_m, ft_names)
    print("=================================================================")


if __name__ == "__main__":
    main()