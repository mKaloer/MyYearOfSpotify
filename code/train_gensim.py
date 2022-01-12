import sys
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS, MeanShift
import plotly.express as px
import numpy as np
import pandas as pd
import random
import pickle


import dataloading

RANDOM_WALK_LEN = 10


class GeneratorIterator():
    def __init__(self, generator_function):
        self.generator_function = generator_function
        self.generator = self.generator_function()

    def __iter__(self):
        # reset the generator
        self.generator = self.generator_function()
        return self

    def __next__(self):
        result = next(self.generator)
        if result is None:
            raise StopIteration
        else:
            return result


class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        print('Loss after epoch {}: {}'.format(self.epoch, loss))
        self.epoch += 1


def create_iterator(datadir, key):
    def track_generator():
        for playlist in dataloading.get_playlists(datadir):
            unique_items = set()
            tracks = []
            for t in playlist['tracks']:
                if t[key] in unique_items:
                    continue
                else:
                    unique_items.add(t[key])
                    tracks.append(t)

            if len(tracks) <= 1:
                continue
            for _ in range(0, len(tracks)*5):
                # In average 5 random "paths" per track
                # Generate random path
                walk = np.random.choice(
                    tracks, min(RANDOM_WALK_LEN, len(tracks)), replace=False)
                yield list(map(lambda x: x[key], walk))
    return track_generator


def create_iterator_rand(datadir, identifier, unique_keys):
    unique_mapping = dict()

    def track_generator():
        for playlist in dataloading.get_playlists(datadir):
            unique_items = set()
            tracks = []
            for t in playlist['tracks']:
                key = tuple([t[k] for k in unique_keys])
                if key in unique_items:
                    continue
                else:
                    unique_items.add(key)
                    tracks.append(t)
                    if not key in unique_mapping:
                        unique_mapping[key] = t[identifier]
            # Random walk approximation
            for i in range(0, 5):
                random.shuffle(tracks)
                yield list(map(lambda x: unique_mapping[tuple([x[k] for k in unique_keys])], tracks))
    return track_generator


def train_tracks(datadir):
    model = Word2Vec(sentences=GeneratorIterator(create_iterator_rand(datadir, 'track_uri', ['track_name', 'artist_uri'])), vector_size=128,
                     window=5, min_count=5, workers=8, callbacks=[callback()], compute_loss=True, epochs=10)

    model.save("track2vec.model")


def train_artists(datadir):
    model = Word2Vec(sentences=GeneratorIterator(create_iterator_rand(datadir, 'artist_uri', ['artist_uri'])), vector_size=128,
                     window=5, min_count=5, workers=8, callbacks=[callback()], compute_loss=True, epochs=10)

    model.save("artist2vec.model")


def eval(model_path):
    model = Word2Vec.load(model_path)
    ms = model.wv.most_similar('spotify:track:5iGleL7HpEThuuYQ3us2jh', 10)
    for x in ms:
        print(x[0], x[1])


def shrink_track_map(model_path):
    model = Word2Vec.load(model_path)
    track_dict = pickle.load(open('track_map.pkl', 'rb'))
    for k in list(track_dict.keys()):
        if not k in model.wv.key_to_index:
            del track_dict[k]
    pickle.dump(track_dict, open('track_map_clean.pkl', 'wb'))


def plot_weights(model_path):
    print("Loading labels")
    track_dict = pickle.load(open('track_map.pkl', 'rb'))
    print("Loading model")
    model = Word2Vec.load(model_path)
    X = model.wv.get_normed_vectors()
    X = X[:100000]
    print("Running PCA")
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X)

    print("Running tsne")
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X_pca)
    labels = []
    print("Generating labels")
    for v in model.wv.index_to_key[:100000]:
        t = track_dict[v]
        labels.append(t['track_name'] + " - " + t['artist_name'])

    print("Clustering")
    #clustering = OPTICS(eps=3, min_samples=10)
    clustering = MeanShift(n_jobs=-1)
    cluster_idx = clustering.fit_predict(X_tsne)

    print("Plotting...")
    df = pd.DataFrame(X_tsne, columns=['x', 'y'])
    df['Title'] = labels
    df['Cluster'] = cluster_idx
    fig = px.scatter(df, x='x', y='y', hover_data=[
                     'Title'], color="Cluster")
    fig.update_layout(template='seaborn')
    fig.update_traces(marker=dict(size=4),
                      selector=dict(mode='markers'))
    fig.show()
    fig.write_html('plot2.html')


if sys.argv[1] == 'tracks':
    train_tracks(sys.argv[2])
elif sys.argv[1] == 'artists':
    train_artists(sys.argv[2])
elif sys.argv[1] == 'eval':
    eval(sys.argv[2])
elif sys.argv[1] == 'plot':
    plot_weights(sys.argv[2])
else:
    print("Unknown type: " + sys.argv[1])
