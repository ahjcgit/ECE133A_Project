import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os
def part1():
    df = pd.read_csv("./OnlineNewsPopularity.csv")

    # Preprocessing & Clean-up
    df = df.iloc[:, 2:]  # Remove first two columns (URL, timedelta)

    missing_vals = df.isnull().sum().sum()
    print(f"Total missing values: {missing_vals}")

    df_mod = df.dropna().reset_index(drop=True)  # Remove rows with missing values
    m = df_mod.to_numpy()  # Convert to NumPy array
    ft_names = df_mod.columns.tolist()

    # Selecting key features based on SVD results
    key_features = [' n_tokens_title', ' n_tokens_content', ' n_unique_tokens',
                    ' num_hrefs', ' num_self_hrefs', ' num_imgs', ' num_videos']
    exist_keys = list(set(df_mod.columns) & set(key_features))
    m_keys = df_mod[exist_keys].to_numpy()

    '''
    print(m.shape)
    print(m_keys.shape)
    '''

    return m, m_keys, ft_names


def part2(m, m_keys, ft_names):
    # Standardization
    scale = StandardScaler()
    std_m = scale.fit_transform(m)
    std_m_keys = scale.fit_transform(m_keys)

    # Feature Mean and Standard Deviation Report
    ft_means = np.mean(std_m, axis=0)
    ft_stdvs = np.std(std_m, axis=0)
    ft_stats = pd.DataFrame({'ft': ft_names, 'Mean': ft_means, 'Std Dev': ft_stdvs})
    print("\nft stats:\n", ft_stats)

    # K-Means Clustering for All Features - Elbow Method
    sq_euc_dist = []
    k_values = range(1, 21)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(std_m)
        sq_euc_dist.append(kmeans.inertia_)

    best_k = np.argmin(np.diff(sq_euc_dist)) + 1  # More stable elbow selection
    print(f"\nbest # of clusters from elbow method: {best_k}")

    # Plot Elbow Curve
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, sq_euc_dist, marker='o', linestyle='-', color='b')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (Sum of Squared Distances)")
    plt.title("Elbow Method for Best k")
    plt.xticks(k_values)  # Ensure k values are labeled on the x-axis
    plt.grid(True)
    plt.show()

    # Silhouette Analysis for All Features
    sil_avgs = {}
    sil_scores = []
    for k in range(2, 21):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=20)
        labels = kmeans.fit_predict(std_m)
        sil_avgs[k] = silhouette_score(std_m, labels)
        sil_scores.append(silhouette_score(std_m, labels))

    best_k = max(sil_avgs, key=sil_avgs.get)
    print(f"best # of clusters from silhouette method: {best_k}")

    # Plot Silhouette Score
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, 21), sil_scores, marker='o', linestyle='-', color='g')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Analysis for Best k")
    plt.xticks(range(2, 21))
    plt.grid(True)
    plt.show()

    # K-Means Clustering for Selected Features - Elbow Method
    sq_euc_dist_keys = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(std_m_keys)
        sq_euc_dist_keys.append(kmeans.inertia_)

    best_k = np.argmin(np.diff(sq_euc_dist_keys)) + 1
    print(f"\nbest # of clusters from elbow method (key clusters): {best_k}")

    # Plot Elbow Curve
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, sq_euc_dist, marker='o', linestyle='-', color='b')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (Sum of Squared Distances)")
    plt.title("(Selected Features) Elbow Method for Best k")
    plt.xticks(k_values)  # Ensure k values are labeled on the x-axis
    plt.grid(True)
    plt.show()

    # Silhouette Analysis for Selected Features
    sil_avgs_keys = {}
    sil_scores_keys = []
    for k in range(2, 21):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=20)
        labels = kmeans.fit_predict(std_m_keys)
        sil_avgs_keys[k] = silhouette_score(std_m_keys, labels)
        sil_scores_keys.append(silhouette_score(std_m_keys, labels))

    best_k = max(sil_avgs_keys, key=sil_avgs_keys.get)
    print(f"best # of clusters from silhouette method (key clusters): {best_k}")

    # Plot Silhouette Score
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, 21), sil_scores_keys, marker='o', linestyle='-', color='g')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("(Selected Features) Silhouette Analysis for Best k")
    plt.xticks(range(2, 21))
    plt.grid(True)
    plt.show()

    # Singular Value Decomposition (SVD) - Identify Top Features
    U, S, Vt = np.linalg.svd(std_m, full_matrices=False)
    sg_vals = S[:10]
    sg_fts = [ft_names[i] for i in np.argsort(-Vt[0])[:10]]  # Using Vt[0] for ranking features

    svd_result = pd.DataFrame({'singular val': sg_vals, 'ft': sg_fts})
    print("\ntop SVD fts:\n", svd_result)

    # Correlation Matrix - Identify Highly Correlated Features
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


def part3(m, m_keys, ft_names):
    # target var (shares) and input features
    X = m_keys  # previously using selected key features
    y = (m[:, -1] > np.median(m[:, -1])).astype(int)  # binary classify as "popular vs. not popular"

    # k-Fold Cross-Validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold = 1
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    auc_scores = []
    model_params = []
    os.makedirs("models", exist_ok=True)

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # RF Classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  # Get probability scores for AUC
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        auc_scores.append(auc)

        # model parameters saving
        model_params.append(model.get_params())
        joblib.dump(model, f"models/random_forest_fold{fold}.pkl")
        print(
            f"Fold {fold} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        fold += 1

    joblib.dump(model_params, "models/random_forest_params.pkl")  # Saved model parameters

    print(f"\nAverage Scores across folds:")
    print(f"Accuracy: {np.mean(accuracy_scores):.4f}")
    print(f"Precision: {np.mean(precision_scores):.4f}")
    print(f"Recall: {np.mean(recall_scores):.4f}")
    print(f"F1 Score: {np.mean(f1_scores):.4f}")
    print(f"AUC: {np.mean(auc_scores):.4f}")

    # Plot AUC Score distribution
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), auc_scores, marker='o', linestyle='-', color='b')
    plt.xlabel("Fold")
    plt.ylabel("AUC Score")
    plt.title("AUC Score Distribution Across Folds")
    plt.xticks(range(1, 11))
    plt.grid(True)
    plt.show()


def main():
    m_p1, m_keys_p1, ft_names = part1()
    # part2(m_p1, m_keys_p1, ft_names)
    part3(m_p1, m_keys_p1, ft_names)


if __name__ == "__main__":
    main()
