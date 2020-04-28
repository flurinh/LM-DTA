import os
import numpy as np
import pickle
import ast
import collections
from collections import OrderedDict
import json
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

import torch
from architectures.loss import *
import matplotlib.pyplot as plt

# ======================================================================================================================


VALPATH = 'evaluation/validation_'
TRAINPATH = 'evaluation/training_'
"""
model_dict_davis = {'cnn': ['01002', '01006', '01010', '01014'],
                    'lstm': ['01022', '01025', '01008', '01011'],
                    'bidir': ['01003', '01026', '01029', '01012']}

model_dict_kiba = {'cnn': ['02001', '02004', '02007', '02010'],
                   'lstm': ['02002', '02005', '02008', '02011'],
                   'bidir': ['02003', '02006', '02009', '02012']}
"""
model_dict_davis = {'cnn': ['03001'],
                    'lstm': ['03002']}


model_dict_kiba = {'cnn': ['04001'],
                   'lstm': ['04002']}

# ======================================================================================================================

def ensemble_eval(ids, loss_type=None):
    if isinstance(ids, int):
        ids = [ids]
    model_dict = get_model_dict(ids)
    idstrs = [str(x).zfill(5) for x in ids]
    summary = {}
    for idx, i in enumerate(idstrs):
        # data = load_predictions(id=i, model_dict=model_dict)
        if int(i) > 2000:
            fac = 10
            add = 8
        else:
            fac = 6
            add = 5
        data, _ = get_data_from_pkl(id=i)
        data = np.array(data)
        if idx == 0:
            pred = data[:, 1]
        else:
            pred += data[:, 1]
        data[:, 1] = pred / len(ids)
    dict_id = 'Ensemble-'+'-'.join(idstrs)
    if loss_type is None:
        summary.update({dict_id: get_losses(data, fac=fac, add=add)})
    else:
        summary.update({dict_id: get_losses(data, fac=fac, add=add)[loss_type]})
    return summary


def evaluate(ids=1002, loss_type=None):
    if isinstance(ids, int):
        ids = [ids]
    ids = list(ids)
    model_dict = get_model_dict(ids)
    idstrs = [str(x).zfill(5) for x in ids]
    summary = {}
    for i in idstrs:
        # data = load_predictions(id=i, model_dict=model_dict)
        if int(i) > 2000:
            fac = 10
            add = 8
        else:
            fac = 6
            add = 5
        data, _ = get_data_from_pkl(id=i)
        data = np.array(data)
        # print(data)
        key_not_found = True
        for key, value in model_dict.items():
            if i in value:
                key_not_found = False
                dict_id = key + '_' + i
        if key_not_found:
            print("Did not find model type for specified run {}.".format(i))
            dict_id = i
        if loss_type is None:
            summary.update({dict_id: get_losses(data, fac=fac, add=add)})
        else:
            summary.update({dict_id: get_losses(data, fac=fac, add=add)[loss_type]})
    return summary


def get_model_dict(ids):
    if isinstance(ids, int):
        ids = [ids]
    ids = list(ids)
    if ids[0] < 2000:
        return model_dict_davis
    else:
        return model_dict_kiba


def load_predictions(id='01002', setting='val', epoch=29, model_dict=model_dict_davis):
    data = None
    if id in model_dict['cnn']:
        epoch = 99
    val_path = VALPATH + id + '.txt'
    train_path = TRAINPATH + id + '.txt'
    if setting == 'val':
        path = val_path
    else:
        path = train_path
    with open(path, 'r') as f:
        step = 0
        for line in f:
            line_ = np.asarray([float(x) for x in line.strip(
                '\n').replace(' ', '').strip('[').strip(']').split(',')])
            if int(line_[0]) == epoch:
                if step == 0:
                    data = line_[1:3]
                    step += 1
                else:
                    data = np.vstack((data, line_[1:3]))
    return data


def get_losses(data, fac=5, add=6):
    losses = []
    rmse_ = 0
    mse_ = 0
    mae_ = 0
    hub_ = 0
    logc_ = 0
    N = data.shape[0] - 1
    # print(data.shape)
    for i in range(data.shape[0]):
        # print(data[i])
        p = data[i][1] * fac + add
        t = data[i][2] * fac + add
        rmse_ += rmse(p, t)
        mse_ += mse(p, t)
        mae_ += mae(p, t)
        hub_ += huber(p, t, delta=10)
        # logc_ += logcosh(p, t)
    output = {'mse': mse_ / N,
            'mae': mae_ / N,
            'huber': hub_ / N}
    print(output)
    return output


# ======================================================================================================================


def select_models(data, model_type=None, loss_types='mse', runs=None):
    if model_type is not None:
        for key in data.keys():
            if not model_type in key:
                del data[key]
    if not loss_types is None:
        if isinstance(loss_types, str):
            loss_types = [loss_types]
        for key in data.keys():
            for loss_key in data[key].keys():
                print()
                print(loss_key)
                print(data[key][loss_key])
                if loss_key in loss_types:
                    pass
                else:
                    del data[key][loss_key]


# ======================================================================================================================

# Load evalutations...

def get_data_from_pkl(id='01001'):
    if not os.path.isfile('evaluation/validation_' + id + '.txt'):
        # Todo: run parser with setting to generate the validation set
        pass
    val = [np.asarray(ast.literal_eval(line.rstrip('\n'))) for line in open('evaluation/validation_' + id + '.txt')]
    val_rep = pickle.load(open('evaluation/validation_repr_' + id + '.txt', 'rb'))
    rep = collections.OrderedDict(sorted(val_rep.items()))
    return val, rep


def repr_analysis(rep, mode='tsne', scale=True, plot=True, top=10, id='00000'):
    if int(id) > 2000:
        dataset = 'kiba'
    else:
        dataset = 'davis'
    ids = np.vstack([prot[0] for prot in rep.items()])
    Occ = np.vstack([prot[1][2] for prot in rep.items()]).flatten()
    Acc = np.vstack([prot[1][1] for prot in rep.items()]).flatten()
    y = Acc / Occ
    X = np.vstack([prot[1][0] for prot in rep.items()])
    if scale:
        X = StandardScaler().fit_transform(X)
    if not mode is 'tsne':
        op = PCA(n_components=2)
    else:
        op = TSNE(n_components=2)
    x = op.fit_transform(X)
    if not mode is 'tsne':
        # print("Total explained variance: ", np.sum(op.explained_variance_ratio_))
        pass
    if plot:
        plot_representation(x, y, Occ, ids, top=top, name='plots/'+mode+'/'+id+'_representation.png', dataset=dataset)
    return x, Occ, Acc, ids


def plot_representation(x, y, z, ids, top, name, dataset):
    fig, ax = plt.subplots()
    fig.set_size_inches(13, 10)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('Protein representation map', fontsize=20)
    top_prots = idx_smallest_err(y, top)
    top_x = x[top_prots]
    top_err = y[top_prots]
    top_ids = ids[top_prots][:, 0]
    top_id_names = [protein_name(idx, dataset) for idx in top_ids]
    # print("Top {} proteins ids: {}".format(top, top_prots))

    # Todo: add ids in plot to the best performing proteins
    plt.scatter(x[:, 0],  # pc 1
                x[:, 1],  # pc 2
                c=y,  # color -> average accuracy
                cmap='viridis',
                s=(z + 15) / 2)  # size -> occurances

    for t in range(top):
        plt.annotate(top_id_names[t], top_x[t])

    cbar = plt.colorbar()
    cbar.set_label('Mean absolute error score')
    plt.savefig(name)
    plt.close()
    return


def idx_smallest_err(y, k):
    idx = np.argpartition(y.ravel(), k)
    return np.array(np.unravel_index(idx, y.shape))[0, range(k)].tolist()


def protein_name(id, dataset='davis'):
    prots = {}
    proteins = json.load(open('data/' + dataset + '/proteins.txt'), object_pairs_hook=OrderedDict)
    for _, t in enumerate(proteins.keys()):
        prots.update({t: str(_)})
    for name, id_ in prots.items():
        if int(id) is int(id_):
            # print(name)
            return name
    return ''

# ======================================================================================================================


def plot_losses(data, plot_type):
    plt.title(plot_type)
    plt.bar(range(len(data)), list(data.values()), align='center')
    plt.xticks(range(len(data)), list(data.keys()))
    name = ''  # str(np.random.randint(0, 10000))
    plt.savefig('plots/' + name + '_' + plot_type + '.png')
    plt.close()
    return


# ======================================================================================================================


def load_trained_model(id=1001):
    path = 'models/run_{}/model_{}'.format(int(id) // 100, int(id)) + '.pt'
    print("Loading model from path", path)
    model = torch.load(path)
    return model


# ======================================================================================================================

plot_type = 'mse'
summary_ = evaluate(ids=[3001, 4001], loss_type=plot_type)
plot_losses(summary_, 'cnn_cv' + plot_type)

"""
summary_ = evaluate(ids=[31, 32], loss_type=plot_type)
plot_losses(summary_, 'cnn_lstm_bs128_24_epochs30_' + plot_type)

summary_ = evaluate(ids=[33, 36], loss_type=plot_type)
plot_losses(summary_, 'lstm_kiba_bs8_trained_untrained_' + plot_type)

summary_ = evaluate(ids=[33, 32, 34, 35], loss_type=plot_type)
plot_losses(summary_, 'lstm_kiba_bs_8_24_36_64_' + plot_type)


summary_ = evaluate(ids=[1002, 1006, 1010, 1014], loss_type=plot_type)
plot_losses(summary_, 'cnn_davis_' + plot_type)
#print('summary_')

summary_ = evaluate(ids=[1022, 1025, 1008, 1011], loss_type=plot_type)
plot_losses(summary_, 'lstm_davis_' + plot_type)
#print('summary_')

summary_ = evaluate(ids=[1003, 1026, 1029, 1012], loss_type=plot_type)
plot_losses(summary_, 'bidir_davis_' + plot_type)
#print('summary_')

summary_ = evaluate(ids=[2001, 2004, 2007, 2010], loss_type=plot_type)
plot_losses(summary_, 'cnn_kiba_' + plot_type)
#print('summary_')

summary_ = evaluate(ids=[2002, 2005, 2008, 2011], loss_type=plot_type)
plot_losses(summary_, 'lstm_kiba_' + plot_type)
#print('summary_')

summary_ = evaluate(ids=[2003, 2006], loss_type=plot_type)
plot_losses(summary_, 'bidir_kiba_' + plot_type)
#print('summary_')

summary_ = evaluate(ids=[1014, 1011, 1012], loss_type=plot_type)
plot_losses(summary_, 'davis_crosscomp_' + plot_type)

summary_ = evaluate(ids=[2010, 2011], loss_type=plot_type)
plot_losses(summary_, 'kiba_crosscomp_' + plot_type)


dicts = [model_dict_davis]
all_runs = []
for filename in os.listdir('evaluation'):
    if 'validation' in filename:
        if not 'repr' in filename:
            all_runs.append(filename[-9:-4])

pbar = tqdm(all_runs)
for run in tqdm(pbar):
    pbar.set_description("Processing %s" % run)
    val, rep = get_data_from_pkl(run)
    x, occ, acc, ids = repr_analysis(rep, top=15, mode='pca', id=run)
    x, occ, acc, ids = repr_analysis(rep, top=15, mode='tsne', id=run)

dicts = [model_dict_kiba]
all_runs = []
for filename in os.listdir('evaluation'):
    if 'validation' in filename:
        if not 'repr' in filename:
            all_runs.append(filename[-9:-4])

pbar = tqdm(all_runs)
for run in tqdm(pbar):
    pbar.set_description("Processing %s" % run)
    val, rep = get_data_from_pkl(run)
    x, occ, acc, ids = repr_analysis(rep, top=15, mode='pca', id=run)
    x, occ, acc, ids = repr_analysis(rep, top=15, mode='tsne', id=run)
"""
