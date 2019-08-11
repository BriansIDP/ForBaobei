'''
This is the main script for clustering
'''
import argparse
import generate_data
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn import metrics, decomposition
from scipy import ndimage, sparse

import generate_data

parser = argparse.ArgumentParser(description='Pseudo User Generator')
parser.add_argument('--number_of_users', metavar='N', type=int,
                   help='an integer for the number of users')
parser.add_argument('--jobs', type=str, default='jobs.txt',
                   help='input job list')
parser.add_argument('--countries', type=str, default='countries.txt',
                   help='input country list')
parser.add_argument('--embed', type=str, default='used_word_embeddings.txt',
                   help='a list of word embeddings')
parser.add_argument('--stddev', type=float, default=1.0,
                   help='a float for spectral clustering Gaussian blur')
parser.add_argument('--threshold', type=float, default=0.8,
                   help='a float for spectral clustering affinity threshold')

args = parser.parse_args()

users = generate_data.generate(args.number_of_users, args.jobs, args.countries) 
user_attributes = ['age', 'gender', 'education', 'nationality', 'occupation']

scales = {'age':0, 'gender':1, 'education':1, 'nationality':1, 'occupation':1}

embeddings = {}
fin = open(args.embed)
for i, line in enumerate(fin):
    elems = line.split()
    word = elems[0]
    vec = elems[1:]
    embeddings[word] = vec
    if i % 10000 == 0:
        print(i)
fin.close()

def get_user_embeddings(user):
    user_embeds = []
    emb_size = len(embeddings['male'])
    for key in user_attributes:
        # get user embeddings from word embedding list
        scale_factor = scales[key]
        if key == 'age':
            user_embeds.append(scale_factor * np.array([user[key]])) 
        else:
            word_list = user[key].split()
            word_emb = np.zeros(emb_size)
            # If a job or an education level has more than one words
            # use the average of the two
            for word in word_list:
                word_emb += np.array(embeddings[word], dtype=np.float64)
            user_embeds.append(word_emb * scale_factor)
    return np.concatenate(user_embeds)

def pre_processing(X, std_dev=1, threshold=0.80):
    m, n = X.shape
    # Set diagonal elements
    for i in range(m):
            X[i][i] = 0
            v = X[i]
            maxv = np.amax(v)
            X[i][i] = maxv

    # Gaussian blur
    X = ndimage.gaussian_filter(X, std_dev)

    # Thresholding
    for i in range(m):
            v = X[i]
            maxv = np.amax(v)
            v[v < maxv*threshold] = maxv * 0.01
            X[i] = v

    # Symmetrization
    Y = np.maximum(X, X.transpose())

    # Diffusion
    X = np.dot(Y, Y.T)

    # Row-wise Max Normalization
    for i in range(m):
        v = X[i]
        maxv = np.amax(v)
        v = v/maxv
        X[i] = v

    # Symmetrization
    Y = np.zeros((m, n))
    for i in range(m):
          for j in range(n):
                  Y[i][j] = max(X[i][j], X[j][i])
    return Y

def calc_centroids(X):
    return np.mean(X, axis=0)

def KmeansClustering(X, num_of_clusters):
    labels = KMeans(n_clusters=num_of_clusters, random_state=0).fit_predict(X)

    # Add some visualization
    pca = PCA(n_components=2)
    pca.fit(user_matrix)
    manifold = pca.transform(user_matrix)
    x = manifold[:, 0]
    y = manifold[:, 1]
    clusters = defaultdict(list)
    centroids = {}
    for i, vec in enumerate(X):
        cluster_id = labels[i]
        clusters[cluster_id].append(vec)
    for i, cluster in clusters.items():
        centroids[i] = calc_centroids(cluster)
    centroids_X = np.stack(list(centroids.values()))
    transformed_centroids = pca.transform(centroids_X)
    cent_x = transformed_centroids[:, 0]
    cent_y = transformed_centroids[:, 1]

    plt.figure(figsize=(9, 3))
    title = 'K-means Clustering'
    plt.title(title, size=9)
    plt.scatter(x, y, c=labels, s=10)
    plt.scatter(cent_x, cent_y, c='red', s=16)
    plt.show()
    return labels, clusters, centroids

def KMeansOnlineAddOne(x, clusters, centroids):
    min_dist = 100000
    for i, cent in centroids.items():
        dist = np.linalg.norm(cent-x)
        if dist < min_dist:
            min_dist = dist
            min_label = i
    clusters[min_label].append(x)
    centroids[min_label] = calc_centroids(clusters[min_label])
    return min_label, clusters, centroids

def SpecClustering(X):
    # Compute affinity matrices
    print("Computing the affinity matrices")
    cos_metric_orig = metrics.pairwise.cosine_similarity(X)
    metric = (cos_metric_orig+1)/2
    metric = pre_processing(metric, std_dev=args.stddev, threshold=args.threshold)

    # Decide number of clusters
    print("Get eigenvalues to decide cluster number")
    eigVal, eigVec = sparse.linalg.eigs(metric, 20, return_eigenvectors = True)
    eigVal_abs = np.absolute(eigVal)
    print("First 20 eigenvalues: ")
    print(eigVal[0:20])
    Valptr = eigVal[0]
    ValRatio = []
    count = 0
    while Valptr > 0 and count < 18:
            count += 1
            Valptr = eigVal[count]
            ValRatio.append(eigVal_abs[count] / eigVal_abs[count+1])
    cluster_number = np.argmax(ValRatio) + 2
    print('Cluster number is {}'.format(cluster_number))

    # Calculate cluster labels
    spectral = SpectralClustering(n_clusters=cluster_number, eigen_solver='arpack', affinity='precomputed')
    spectral.fit(metric)
    y_pred = spectral.labels_.astype(np.int)
    print("Finished clustering")

    # Add some viualization
    pca = PCA(n_components=2)
    X_vis = pca.fit_transform(X)
    plt.figure(figsize=(9, 3))
    title = 'Spectral Clustering'
    plt.title(title, size=9)

    # plot the result 
    plt.subplot(1, 2, 1)
    plt.title('Combined Similarity', size=12)
    plt.imshow(1-metric, cmap='gray', interpolation='nearest')
    plt.subplot(1, 2, 2)
    plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_pred)
    plt.show()

#---------------------------------------------------------
# The main script
#--------------------------------------------------------
user_vecs = []
for user in users:
   user_vecs.append(get_user_embeddings(user))

# data visualization
user_matrix = np.array(user_vecs)
# Use k-means clustering, input number of clusters
labels, clusters, centroids = KmeansClustering(user_matrix, 3)
new_user = generate_data.generate(1, args.jobs, args.countries)[0]
new_user_vec = get_user_embeddings(new_user)
label, clusters, centroids = KMeansOnlineAddOne(new_user_vec, clusters, centroids)
print(label)
# Use spectral clustering
# SpecClustering(user_matrix)
# Now use PCA for visualization
# PCA extract the 2 dimensions with the highest variance from
# a high dimensional space using linear transform
# Future Laogong will add t-SNE which uses a non-linear manifold
