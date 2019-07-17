'''
This is the main script for clustering
'''
import argparse
import generate_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser(description='Pseudo User Generator')
parser.add_argument('--number_of_users', metavar='N', type=int,
                   help='an integer for the number of users')
parser.add_argument('--jobs', type=str, default='jobs.txt',
                   help='input job list')
parser.add_argument('--countries', type=str, default='countries.txt',
                   help='input country list')
parser.add_argument('--embed', type=str, default='used_word_embeddings.txt',
                   help='a list of word embeddings')

args = parser.parse_args()

users = generate_data.generate(args.number_of_users, args.jobs, args.countries) 
user_attributes = ['age', 'gender', 'education', 'nationality', 'occupation']
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
    for key in user_attributes:
        if key == 'age':
            user_embeds.append(np.array([user[key]])) 
        else:
            word_emb = np.array(embeddings[user[key]])
            user_embeds.append(word_emb)
    # not considering age for now
    return np.concatenate(user_embeds[1:])

user_vecs = []
for user in users:
   user_vecs.append(get_user_embeddings(user))

user_matrix = np.array(user_vecs)
print(user_matrix)
pca = PCA(n_components=2)
manifold = pca.fit_transform(user_matrix)
x = manifold[:, 0]
y = manifold[:, 1]
plt.scatter(x, y)
plt.show()
