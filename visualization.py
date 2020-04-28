# TODO: visualization of smiles? visualization of proteins?
# TODO: Visualization of interesting sequence regions based on network weights?
# TODO: json loss representation
# https://github.com/lanpa/tensorboardX
# https://tensorboardx.readthedocs.io/en/latest/tutorial.html
# setting up a port http://cerfacs.fr/helios/research_blog/tensorboard/

from Processing.preprocessing import *
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import numpy as np


def plot_smiles(smiles, one_hot, ID=-1, size=(500, 500), show=True, save=False, path='data/visuals'):
    dense = smiles_decoder(smiles, one_hot=one_hot, _ignore_special=True)
    m = Chem.MolFromSmiles(dense)
    fig = Draw.MolToImage(m, size=size)
    if show:
        fig.show()
    if save:
        fig.save(path + '/' + str(ID) + '.png')
    return


def plot_char_dist(seqs, datasetname, seq_type='SMI', mode='norm', ignore_filler=True, show=True, save=False,
                   path='data/visuals'):
    if seq_type is 'SMI':
        bins = SMILES_CHARS
    if seq_type is 'SEQ':
        bins = SEQ_CHARS
    # print(seqs)
    all_chars = ''.join(seqs)
    n_all_chars = len(all_chars)
    counts = []
    for char in bins:
        counts.append(all_chars.count(char))
    sum_counts = sum(counts)
    # print(bins)
    # print(counts)
    try:
        assert n_all_chars == sum_counts, print("Number of characters and total number of counts dont add up! "
                                                "Total missing characters:", np.abs(sum_counts - n_all_chars))
    except:  # Not implemented
        pass
    del_chars = [i for i, x in enumerate(counts) if x == 0]
    counts = [x for i, x in enumerate(counts) if del_chars.count(i) == 0]
    bins = [x for i, x in enumerate(bins) if del_chars.count(i) == 0]
    if ignore_filler:
        counts = counts[:-1]
        bins = bins[:-1]
    log = False
    sum_counts = sum(counts)
    if mode is 'norm':
        log = True
        counts[:] = [x / sum_counts for x in counts]
    else:
        counts[:] = [x / 1000 for x in counts]
    plot_xy_hist(bins, counts, log, datasetname, show, save, path)
    return bins, counts


def plot_xy_hist(bins, counts, log, datasetname, show, save, path):
    plt.figure(figsize=(15, 15))
    plt.bar(bins, counts, log=log)
    plt.title('character distribution ' + datasetname)
    plt.xlabel('bins')
    if log:
        plt.ylabel('counts (log-scale) in %')
    else:
        plt.ylabel('absolute counts (in 10^3)')
    if show:
        plt.show()
    if save:
        plt.savefig(path + '.png')
    return


def plot_length_dist(seqs, cut_off=.8, seq_type='SMI', bin_size=20, ignore_filler=True, show=True, save=False,
                     path='data/visuals/'):
    path = path + seq_type + '_length_hist'
    lengths = []
    for seq in seqs:
        if ignore_filler:
            l = len((seq.replace(FILLER, '').replace(UNKNOWN, '')))
        else:
            l = len(seq)
        lengths.append(l - (l % bin_size))
    max_len = max(lengths)
    x_axis = np.linspace(0, max_len, dtype=int, num=int(max_len / bin_size) + 1)
    y_axis = np.zeros(x_axis.shape)
    for l in lengths:
        bin = int(l / bin_size)
        y_axis[bin] += 1
    percent_line = cut_off
    start = 0.01
    tot_counts = np.sum(y_axis)
    red = True
    i = 0
    while red and i < y_axis.shape[0]:
        percent = np.sum(y_axis[:i]) / tot_counts
        i += 1
        if percent < start:
            not_included = i
        if percent > percent_line:
            red = False
            # red_line = i
            cut_off_val = x_axis[i]
    x_axis = x_axis[not_included:]
    y_axis = y_axis[not_included:]
    plt.bar(x_axis, y_axis, log=False)
    plt.title(seq_type + ': length distribution in dataset')
    plt.ylabel(seq_type + ' absolute counts')
    plt.xlabel('length')
    plt.figure(figsize=(15, 15))
    if show:
        plt.show()
    if save:
        plt.savefig(path + '.png')
    if show:
        print("Suggested cut-off for " + seq_type + " (α = " + str(cut_off) + ') at ' + str(cut_off_val) + '.')
    return cut_off_val
