import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def part1():
    df = pd.read_csv("./OnlineNewsPopularity.csv")

    # Preprocessing & Clean-up
    url_array = df.iloc[:, 0].values  # extract news urls mapped by index
    df = df.iloc[:, 1:]  # remove first column urls
    df = df.iloc[:, 1:]  # remove second column timedelta

    missing_vals = df.isnull().sum().sum()
    # print(f"total missing values: {missing_vals}")
    df_mod = df.dropna().reset_index(drop=True)  # remove rows with missing or null values

    m = df_mod.to_numpy()  # phase 1 matrix
    ft_names = df_mod.columns.tolist()

    '''
    print("Dim: ", m_p1.shape)
    print(url_array)
    print(m_p1)
    '''
    return m, ft_names


def part2(m, ft_names):
    # Standardize raw features
    scale = StandardScaler()
    std_m = scale.fit_transform(m)

    # Report every features' mean and standard deviations across samples
    ft_means = np.mean(std_m, axis=0)
    ft_stdvs = np.std(std_m, axis=0)
    ft_stats = pd.DataFrame({'ft': ft_names, 'Mean': ft_means, 'Std Dev': ft_stdvs})
    print("\nft stats:\n", ft_stats)

    # K-Means clustering for various k values
    sq_euc_dist = []
    k_values = range(1, 11)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # KMeans from sklearn
        kmeans.fit(std_m)
        sq_euc_dist.append(kmeans.inertia_)
    best_k = np.argmin(np.gradient(sq_euc_dist)) + 1  # elbow method
    print(f"\nbest # of clusters: {best_k}")

    # Identify top features with largest SVs
    U, S, Vt = np.linalg.svd(std_m, full_matrices=False)  # SVD
    sg_vals = S[:10]  # prepare extract 10 SVs
    sg_fts = [ft_names[i] for i in np.argsort(-S)[:10]]  # fts corresponding to highest singular values
    svd_result = pd.DataFrame({'singular val': sg_vals, 'ft': sg_fts})
    print("\ntop SVD fts:\n", svd_result)

    # Create correlation matrix to identify sig corr features
    corr_m = np.corrcoef(std_m, rowvar=False)
    threshold = 0.8  # self picked threshold
    sig_corr_fts = []
    for i, col1 in enumerate(ft_names):
        for j, col2 in enumerate(ft_names):
            if i < j and abs(corr_m[i, j]) > threshold:
                sig_corr_fts.append((col1, col2, corr_m[i, j]))

    corr_result = pd.DataFrame(sig_corr_fts, columns=['ft 1', 'ft 2', 'correlation'])
    print("\nsignificant correlated fts:\n", corr_result)


def main():
    m_p1, ft_names = part1()
    part2(m_p1, ft_names)


if __name__ == "__main__":
    main()
