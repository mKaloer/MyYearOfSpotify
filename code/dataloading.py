import glob
import json


def get_playlists(datadir):
    for f in glob.glob('{}/*.json'.format(datadir)):
        with open(f) as json_f:
            for playlist in json.load(json_f)['playlists']:
                yield playlist
