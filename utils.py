import pandas as pd
import os
import json

PATH = 'data/preprocessed/'


def load_data(dataset='kiba', mode='train'):
    print('Loading ' + mode + 'ing dataset', dataset)
    if mode is 'train':
        path = PATH + dataset + '_train.csv'
        print(path)
    elif mode is 'test':
        path = PATH + dataset + '_test.csv'
        print(path)
    else:
        print('Valid modes are "test" and "train".')
        os._exit(0)
    assert os.path.exists(path), print('Invalid data path:', path)
    data = pd.read_csv(path)
    return data


def save_track(path, dict_new_epoch):
    last=(*dict_new_epoch,)[-1]
    if last == 0:
        with open(path, "w") as f:
            print(dict_new_epoch)
            json.dump(dict_new_epoch, f)
    else:
        file = open(path)
        dict_track = json.load(file)
        dict_track.update(dict_new_epoch)
        print(dict_track)
        with open(path, "w") as f:
            json.dump(dict_track, f)
    return
