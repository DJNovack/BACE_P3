import argparse
import functools
import multiprocessing
import numpy as np
import os
import deeptime
import scipy.io
import scipy.sparse
import sys
import logging

# Global variables and logger configuration
version = 1.0
logger = logging.getLogger('BACE')
logger.setLevel(logging.INFO)
sh = logging.StreamHandler(stream=sys.stdout)
formatter = logging.Formatter(fmt='%(asctime)s - %(message)s', datefmt="%H:%M:%S")
sh.setFormatter(formatter)
logger.addHandler(sh)
logger.propagate = False

# License information
LicenseString = f"""--------------------------------------------------------------------------------
BACE version {version}

Written by Gregory R. Bowman, UC Berkeley

--------------------------------------------------------------------------------
Copyright 2012 University of California, Berkeley.

BACE comes with ABSOLUTELY NO WARRANTY.

BACE is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

--------------------------------------------------------------------------------
Please cite:
GR Bowman. Improved coarse-graining of Markov state models via explicit consideration of statistical uncertainty. J Chem Phys 2012;137;134111.

Currently available as arXiv:1201.3867 2012.

--------------------------------------------------------------------------------
"""

# Argument parser configuration
def add_argument(group, *args, **kwargs):
    if 'default' in kwargs:
        d = f"Default: {kwargs['default']}"
        kwargs['help'] = f"{kwargs.get('help', '')} {d}"
    group.add_argument(*args, **kwargs)

parser = argparse.ArgumentParser(
    description='''Bayesian agglomerative clustering engine (BACE) for coarse-graining MSMs. For example, building macrostate models from microstate models.

The algorithm works by iteratively merging states until the final desired number of states (the nMacro parameter) is reached.

Results are often obtained most quickly by forcing the program to use dense matrices (with the -f option) and using a single processor. Sparse matrices (and possibly multiple processors) are useful when insufficient memory is available to use dense matrices.

A macrostate model may be attractive for further analysis if further reducing the number of macrostates (M) causes a large increase in the Bayes factor (cost), as reported in the bayesFactors.dat output file described below. For example, if the Bayes factor increases steadily as one goes from models with M-5, M-4, ..., M states but increases much more dramatically when going from M to M-1 states, then a model with M states may be of interest because the sudden increase in the Bayes factor for going to M-1 states suggests two very distinct free energy basins are being merged. To make these judgments, it is often useful to plot the Bayes factor as a function of the number of macrostates.

Once you have chosen the number of macrostates (M) you wish to analyze further, you can calculate the appropriate transition matrices using the BuildMSM.py script. For example, to build a model with 5 macrostates you might run something like
BuildMSM.py -l 1 -a Data/Assignments.Fixed.h5 -m Output_BACE/map5.dat -o BACE_5state
The -m option is the crucial addition for directing the script to apply the specified mapping from the microstates in the h5 file to the macrostates specified by the -m option.

Outputs (stored in the directory specified with outDir):
bayesFactors.dat = the Bayes factors (cost) for each merging of two states. The first column is the number of macrostates (M) and the second column is the Bayes factor (cost) for coarse-graining from M+1 states to M states.
mapX.dat = the mapping from the original state numbering to X coarse-grained states.''',
    formatter_class=argparse.RawDescriptionHelpFormatter
)

add_argument(parser, '-c', dest='tCountFn', help='Path to transition count matrix file (sparse and dense formats accepted) or a NumPy array.', required=True)
add_argument(parser, '-n', dest='nMacro', help='Minimum number of macrostates to make.', default=2, type=int)
add_argument(parser, '-p', dest='nProc', help='Number of processors to use.', default=1, type=int, required=False)
add_argument(parser, '-f', dest='forceDense', help='If true, the program will force the transition matrix into a dense format. Using the dense format is faster if you have enough memory.', default=False, type=bool, required=False, nargs='?', const=True)
add_argument(parser, '-o', dest='outDir', help='Path to save the output to.', default="Output_BACE", required=False)

def get_indices(c, state_inds, chunk_size, is_sparse, update_single_state=None):
    """Get indices for states with sufficient statistics."""
    indices = []
    for s in state_inds:
        if is_sparse:
            dest = np.where(c[s, :].toarray()[0] > 1)[0]
        else:
            dest = np.where(c[s, :] > 1)[0]
        if update_single_state is not None:
            dest = dest[np.where(dest != update_single_state)[0]]
        else:
            dest = dest[np.where(dest > s)[0]]
        if dest.shape[0] == 0:
            continue
        elif dest.shape[0] < chunk_size:
            indices.append((s, dest))
        else:
            i = 0
            while dest.shape[0] > i:
                if i + chunk_size > dest.shape[0]:
                    indices.append((s, dest[i:]))
                else:
                    indices.append((s, dest[i:i + chunk_size]))
                i += chunk_size
    return indices

def run(c, n_macro, n_proc, multi_dist, out_dir, filter_func, chunk_size=100):
    """Run the main coarse-graining algorithm."""
    logger.info("Checking for states with insufficient statistics")
    c, map_states, states_keep = filter_func(c, n_proc)

    w = np.array(c.sum(axis=1)).flatten()
    w[states_keep] += 1

    unmerged = np.zeros(w.shape[0], dtype=np.int8)
    unmerged[states_keep] = 1

    ind_recalc = get_indices(c, states_keep, chunk_size, scipy.sparse.issparse(c))
    d_mat = scipy.sparse.lil_matrix(c.shape) if scipy.sparse.issparse(c) else np.zeros(c.shape, dtype=np.float32)

    if scipy.sparse.issparse(c):
        c = c.tocsr()

    i = 0
    n_current_states = states_keep.shape[0]
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/bayesFactors.dat", 'w') as f_bayes_fact:
        d_mat, min_x, min_y = calc_dmat(c, w, f_bayes_fact, ind_recalc, d_mat, n_proc, states_keep, multi_dist, unmerged, chunk_size)
        logger.info("Coarse-graining...")

        while n_current_states > n_macro:
            logger.info(f"Iteration {i}, merging {n_current_states} states")
            c, w, ind_recalc, d_mat, map_states, states_keep, unmerged, min_x, min_y = merge_two_closest_states(
                c, w, f_bayes_fact, ind_recalc, d_mat, n_proc, map_states, states_keep, min_x, min_y, multi_dist, unmerged, chunk_size)
            n_current_states -= 1
            np.savetxt(f"{out_dir}/map{n_current_states}.dat", map_states, fmt="%d")
            i += 1

def merge_two_closest_states(c, w, f_bayes_fact, ind_recalc, d_mat, n_proc, map_states, states_keep, min_x, min_y, multi_dist, unmerged, chunk_size):
    """Merge the two closest states."""
    c_is_sparse = scipy.sparse.issparse(c)
    if c_is_sparse:
        c = c.tolil()

    if unmerged[min_x]:
        c[min_x, states_keep] += unmerged[states_keep] * 1.0 / c.shape[0]
        unmerged[min_x] = 0
        if c_is_sparse:
            c[states_keep, min_x] += np.matrix(unmerged[states_keep]).transpose() * 1.0 / c.shape[0]
        else:
            c[states_keep, min_x] += unmerged[states_keep] * 1.0 / c.shape[0]

    if unmerged[min_y]:
        c[min_y, states_keep] += unmerged[states_keep] * 1.0 / c.shape[0]
        unmerged[min_y] = 0
        if c_is_sparse:
            c[states_keep, min_y] += np.matrix(unmerged[states_keep]).transpose() * 1.0 / c.shape[0]
        else:
            c[states_keep, min_y] += unmerged[states_keep] * 1.0 / c.shape[0]

    c[min_x, states_keep] += c[min_y, states_keep]
    c[states_keep, min_x] += c[states_keep, min_y]
    c[min_y, states_keep] = 0
    c[states_keep, min_y] = 0
    d_mat[min_x, :] = 0
    d_mat[:, min_x] = 0
    d_mat[min_y, :] = 0
    d_mat[:, min_y] = 0

    if c_is_sparse:
        c = c.tocsr()

    w[min_x] += w[min_y]
    w[min_y] = 0
    states_keep = states_keep[np.where(states_keep != min_y)[0]]
    ind_change = np.where(map_states == map_states[min_y])[0]
    map_states = renumber_map(map_states, map_states[min_y])
    map_states[ind_change] = map_states[min_x]
    ind_recalc = get_indices(c, [min_x], chunk_size, c_is_sparse, update_single_state=min_x)
    d_mat, min_x, min_y = calc_dmat(c, w, f_bayes_fact, ind_recalc, d_mat, n_proc, states_keep, multi_dist, unmerged, chunk_size)
    return c, w, ind_recalc, d_mat, map_states, states_keep, unmerged, min_x, min_y

def renumber_map(map_states, state_drop):
    """Renumber the map after merging states."""
    for i in range(map_states.shape[0]):
        if map_states[i] >= state_drop:
            map_states[i] -= 1
    return map_states

def calc_dmat(c, w, f_bayes_fact, ind_recalc, d_mat, n_proc, states_keep, multi_dist, unmerged, chunk_size):
    """Calculate the distance matrix."""
    n_recalc = len(ind_recalc)
    if n_recalc > 1 and n_proc > 1:
        if n_recalc < n_proc:
            n_proc = n_recalc
        pool = multiprocessing.Pool(processes=n_proc)
        n = len(ind_recalc)
        step_size = n // n_proc
        dlims = [(i, min(i + step_size, n)) for i in range(0, n, step_size)]
        args = [ind_recalc[start:stop] for start, stop in dlims]
        result = pool.map_async(functools.partial(multi_dist, c=c, w=w, states_keep=states_keep, unmerged=unmerged, chunk_size=chunk_size), args)
        result.wait()
        d = np.vstack(result.get())
        pool.close()
    else:
        d = multi_dist(ind_recalc, c, w, states_keep, unmerged, chunk_size)

    for i in range(len(ind_recalc)):
        d_mat[ind_recalc[i][0], ind_recalc[i][1]] = d[i][:len(ind_recalc[i][1])]

    if scipy.sparse.issparse(d_mat):
        min_x, min_y = -1, -1
        max_d = 0
        for x in states_keep:
            if len(d_mat.data[x]) == 0:
                continue
            pos = np.argmax(d_mat.data[x])
            if d_mat.data[x][pos] > max_d:
                max_d = d_mat.data[x][pos]
                min_x = x
                min_y = d_mat.rows[x][pos]
    else:
        ind_min = d_mat.argmax()
        min_x = ind_min // d_mat.shape[1]
        min_y = ind_min % d_mat.shape[1]

    f_bayes_fact.write(f"{states_keep.shape[0] - 1} {1. / d_mat[min_x, min_y]}\n")
    return d_mat, min_x, min_y

def multi_dist_dense(indices_list, c, w, states_keep, unmerged, chunk_size):
    """Calculate distances for dense matrices."""
    d = np.zeros((len(indices_list), chunk_size), dtype=np.float32)
    for j in range(len(indices_list)):
        indices = indices_list[j]
        ind1 = indices[0]
        c1 = c[ind1, states_keep] + unmerged[ind1] * unmerged[states_keep] * 1.0 / c.shape[0]
        d[j, :indices[1].shape[0]] = 1. / multi_dist_dense_helper(indices[1], c1, w[ind1], c, w, states_keep, unmerged)
    return d

def multi_dist_dense_helper(indices, c1, w1, c, w, states_keep, unmerged):
    """Helper function for dense distance calculation."""
    d = np.zeros(indices.shape[0], dtype=np.float32)
    p1 = c1 / w1
    for i in range(indices.shape[0]):
        ind2 = indices[i]
        c2 = c[ind2, states_keep] + unmerged[ind2] * unmerged[states_keep] * 1.0 / c.shape[0]
        p2 = c2 / w[ind2]
        cp = c1 + c2
        cp /= (w1 + w[ind2])
        d[i] = c1.dot(np.log(p1 / cp)) + c2.dot(np.log(p2 / cp))
    return d

def multi_dist_sparse(indices_list, c, w, states_keep, unmerged, chunk_size):
    """Calculate distances for sparse matrices."""
    d = np.zeros((len(indices_list), chunk_size), dtype=np.float32)
    for j in range(len(indices_list)):
        indices = indices_list[j]
        ind1 = indices[0]
        c1 = c[ind1, states_keep].toarray()[0] + unmerged[ind1] * unmerged[states_keep] * 1.0 / c.shape[0]
        d[j, :indices[1].shape[0]] = 1. / multi_dist_sparse_helper(indices[1], c1, w[ind1], c, w, states_keep, unmerged)
    return d

def multi_dist_sparse_helper(indices, c1, w1, c, w, states_keep, unmerged):
    """Helper function for sparse distance calculation."""
    d = np.zeros(indices.shape[0], dtype=np.float32)
    p1 = c1 / w1
    for i in range(indices.shape[0]):
        ind2 = indices[i]
        c2 = c[ind2, states_keep].toarray()[0] + unmerged[ind2] * unmerged[states_keep] * 1.0 / c.shape[0]
        p2 = c2 / w[ind2]
        cp = c1 + c2
        cp /= (w1 + w[ind2])
        d[i] = c1.dot(np.log(p1 / cp)) + c2.dot(np.log(p2 / cp))
    return d

def filter_func_dense(c, n_proc):
    """Filter function for dense matrices."""
    w = np.array(c.sum(axis=1)).flatten()
    w += 1
    map_states = np.arange(c.shape[0], dtype=np.int32)
    pseud = np.ones(c.shape[0], dtype=np.float32) / c.shape[0]
    indices = np.arange(c.shape[0], dtype=np.int32)
    states_keep = np.arange(c.shape[0], dtype=np.int32)
    unmerged = np.ones(c.shape[0], dtype=np.float32)

    n_ind = len(indices)
    if n_ind > 1 and n_proc > 1:
        if n_ind < n_proc:
            n_proc = n_ind
        pool = multiprocessing.Pool(processes=n_proc)
        step_size = n_ind // n_proc
        dlims = [(i, min(i + step_size, n_ind)) for i in range(0, n_ind, step_size)]
        args = [indices[start:stop] for start, stop in dlims]
        result = pool.map_async(functools.partial(multi_dist_dense_helper, c1=pseud, w1=1, c=c, w=w, states_keep=states_keep, unmerged=unmerged), args)
        result.wait()
        d = np.concatenate(result.get())
        pool.close()
    else:
        d = multi_dist_dense_helper(indices, pseud, 1, c, w, states_keep, unmerged)

    states_prune = np.where(d < 1.1)[0]
    states_keep = np.where(d >= 1.1)[0]
    logger.info(f"Merging {states_prune.shape[0]} states with insufficient statistics into their kinetically-nearest neighbor")

    for s in states_prune:
        row = c[s, :]
        row[s] = 0
        dest = row.argmax()
        c[dest, :] += c[s, :]
        c[:, dest] += c[:, s]
        c[s, :] = 0
        c[:, s] = 0
        map_states = renumber_map(map_states, map_states[s])
        map_states[s] = map_states[dest]

    return c, map_states, states_keep

def filter_func_sparse(c, n_proc):
    """Filter function for sparse matrices."""
    w = np.array(c.sum(axis=1)).flatten()
    w += 1
    map_states = np.arange(c.shape[0], dtype=np.int32)
    pseud = np.ones(c.shape[0], dtype=np.float32) / c.shape[0]
    indices = np.arange(c.shape[0], dtype=np.int32)
    states_keep = np.arange(c.shape[0], dtype=np.int32)
    unmerged = np.ones(c.shape[0], dtype=np.int8)

    n_ind = len(indices)
    if n_ind > 1 and n_proc > 1:
        if n_ind < n_proc:
            n_proc = n_ind
        pool = multiprocessing.Pool(processes=n_proc)
        step_size = n_ind // n_proc
        dlims = [(i, min(i + step_size, n_ind)) for i in range(0, n_ind, step_size)]
        args = [indices[start:stop] for start, stop in dlims]
        result = pool.map_async(functools.partial(multi_dist_sparse_helper, c1=pseud, w1=1, c=c, w=w, states_keep=states_keep, unmerged=unmerged), args)
        result.wait()
        d = np.concatenate(result.get())
        pool.close()
    else:
        d = multi_dist_sparse_helper(indices, pseud, 1, c, w, states_keep, unmerged)

    states_prune = np.where(d < 1.1)[0]
    states_keep = np.where(d >= 1.1)[0]
    logger.info(f"Merging {states_prune.shape[0]} states with insufficient statistics into their kinetically-nearest neighbor")

    for s in states_prune:
        row = c[s, :].toarray()[0]
        row[s] = 0
        dest = row.argmax()
        c[dest, :] += c[s, :]
        c[:, dest] += c[:, s]
        c[s, :] = 0
        c[:, s] = 0
        map_states = renumber_map(map_states, map_states[s])
        map_states[s] = map_states[dest]

    return c, map_states, states_keep

def bace_main(args=None):
    """Main function to run BACE."""
    if args is None:
        args = parser.parse_args()
    
    print(LicenseString)

    c = None
    multi_dist = None
    filter_func = None
    
    if isinstance(args.tCountFn, str):
        if args.tCountFn.endswith(".mtx"):
            c = scipy.sparse.lil_matrix(scipy.io.mmread(args.tCountFn), dtype=np.float32)
            multi_dist = multi_dist_sparse
            filter_func = filter_func_sparse
            if args.forceDense:
                logger.info("Forcing dense")
                c = c.toarray()
                multi_dist = multi_dist_dense
                filter_func = filter_func_dense
        else:
            c = np.loadtxt(args.tCountFn, dtype=np.float32)
            multi_dist = multi_dist_dense
            filter_func = filter_func_dense
    elif isinstance(args.tCountFn, np.ndarray):
        c = args.tCountFn
        if scipy.sparse.issparse(c):
            multi_dist = multi_dist_sparse
            filter_func = filter_func_sparse
            if args.forceDense:
                logger.info("Forcing dense")
                c = c.toarray()
                multi_dist = multi_dist_dense
                filter_func = filter_func_dense
        else:
            multi_dist = multi_dist_dense
            filter_func = filter_func_dense
    else:
        raise ValueError("tCountFn must be a file path or a NumPy array")

    if args.nProc is None:
        args.nProc = multiprocessing.cpu_count()
    logger.info(f"Set number of processors to {args.nProc}")

    run(c, args.nMacro, args.nProc, multi_dist, args.outDir, filter_func, chunk_size=100)

def coarse_grain_msm(fine_grained_msm, map_file_path):
    """
    Coarse grains a fine-grained MSM using a map file.

    Parameters:
    fine_grained_msm (deeptime.markov.msm.MSM): The fine-grained MSM.
    map_file_path (str): Path to the map file that defines the coarse graining.

    Returns:
    deeptime.markov.msm.MSM: The coarse-grained MSM.
    """
    # Read the mapping from the map file
    with open(map_file_path, 'r') as f:
        mapping = np.loadtxt(f, dtype=int)

    # Determine the number of coarse states
    n_coarse_states = np.max(mapping) + 1

    # Initialize the coarse transition count matrix
    coarse_counts = np.zeros((n_coarse_states, n_coarse_states))

    # Aggregate the fine-grained transition counts into the coarse counts
    for i in range(fine_grained_msm.transition_matrix.shape[0]):
        for j in range(fine_grained_msm.transition_matrix.shape[1]):
            coarse_counts[mapping[i], mapping[j]] += fine_grained_msm.count_model.count_matrix[i, j]

    # Convert coarse counts to a transition matrix
    coarse_transitions = coarse_counts / coarse_counts.sum(axis=1, keepdims=True)

    # Handle any rows that sum to zero (avoid division by zero)
    coarse_transitions[np.isnan(coarse_transitions)] = 0

    # Create the coarse-grained MSM
    coarse_grained_msm = deeptime.markov.msm.MarkovStateModel(coarse_transitions)

    return coarse_grained_msm

def calculate_coarse_grained_centers(cluster_centers, populations, mapping):
    """
    Calculate the population centers of coarse-grained cluster centers.

    Parameters:
    cluster_centers (np.ndarray): Positions of the fine-grained cluster centers (n_clusters, n_dimensions).
    populations (np.ndarray): Populations of the fine-grained states (n_clusters,).
    mapping (np.ndarray): Mapping from fine-grained states to coarse-grained states (n_clusters,).

    Returns:
    np.ndarray: Positions of the coarse-grained cluster centers (n_coarse_clusters, n_dimensions).
    np.ndarray: Populations of the coarse-grained states (n_coarse_clusters,).
    """
    # Determine the number of coarse-grained states
    n_coarse_clusters = np.max(mapping) + 1
    n_dimensions = cluster_centers.shape[1]

    # Initialize arrays for coarse-grained centers and populations
    coarse_centers = np.zeros((n_coarse_clusters, n_dimensions))
    coarse_populations = np.zeros(n_coarse_clusters)

    # Accumulate the populations and weighted positions for each coarse-grained state
    for i, coarse_state in enumerate(mapping):
        coarse_populations[coarse_state] += populations[i]
        coarse_centers[coarse_state] += cluster_centers[i] * populations[i]

    # Normalize the positions by the populations to get the population centers
    for i in range(n_coarse_clusters):
        if coarse_populations[i] > 0:
            coarse_centers[i] /= coarse_populations[i]

    return coarse_centers, coarse_populations

if __name__ == '__main__':
    bace_main()

