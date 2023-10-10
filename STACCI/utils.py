import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation
from sklearn import metrics


def compare_labels(save_path, gt_labels, cluster_labels): # Tid: helper
    # re-order cluster labels for constructing diagonal-like matrix
    if max(gt_labels)==max(cluster_labels):
        matrix = np.zeros([max(gt_labels)+1, max(cluster_labels)+1], dtype=int)
        n_samples = len(cluster_labels)
        for i in range(n_samples):
            matrix[gt_labels[i], cluster_labels[i]] += 1
        matrix_size = max(gt_labels)+1
        order_seq = np.arange(matrix_size)
        matrix = np.array(matrix)
        #print(matrix)
        norm_matrix = matrix/matrix.sum(1).reshape(-1,1)
        #print(norm_matrix)
        norm_matrix_2_arr = norm_matrix.flatten()
        sort_index = np.argsort(-norm_matrix_2_arr)
        #print(sort_index)
        sort_row, sort_col = [], []
        for tmp in sort_index:
            sort_row.append(int(tmp/matrix_size))
            sort_col.append(int(tmp%matrix_size))
        sort_row = np.array(sort_row)
        sort_col = np.array(sort_col)
        #print(sort_row)
        #print(sort_col)
        done_list = []
        for j in range(len(sort_index)):
            if len(done_list) == matrix_size:
                break
            if (sort_row[j] in done_list) or (sort_col[j] in done_list):
                continue
            done_list.append(sort_row[j])
            tmp = sort_col[j]
            sort_col[sort_col == tmp] = -1
            sort_col[sort_col == sort_row[j]] = tmp
            sort_col[sort_col == -1] = sort_row[j]
            order_seq[sort_row[j]], order_seq[tmp] = order_seq[tmp], order_seq[sort_row[j]]

        reorder_cluster_labels = []
        for k in cluster_labels:
            reorder_cluster_labels.append(order_seq.tolist().index(k))
        matrix = matrix[:, order_seq]
        norm_matrix = norm_matrix[:, order_seq]
        plt.imshow(norm_matrix)
        plt.savefig(save_path + '/compare_labels_Matrix.png')
        plt.close()
        np.savetxt(save_path+ '/compare_labels_Matrix.txt', matrix, fmt='%3d', delimiter='\t')
        reorder_cluster_labels = np.array(reorder_cluster_labels, dtype=int)

    else:
        print('not square matrix!!')
        reorder_cluster_labels = cluster_labels
    return reorder_cluster_labels

from mpl_toolkits.axes_grid1 import make_axes_locatable
def draw_map(args, adj_0, barplot=False, title='DSTC', bi=False): # Tid: XXX
    data_folder = args.data_path + args.data_name+'/'
    save_path = args.result_path

    print(f"Saving drew maps in {save_path}")

    f = open(save_path + '/types.txt')            
    line = f.readline() # drop the first line  
    cell_cluster_type_list = []

    while line: 
        tmp = line.split('\t')
        cell_id = int(tmp[0]) # index start is start from 0 here
        #cell_type_index = int(tmp[1])
        cell_cluster_type = int(tmp[1].replace('\n', ''))
        cell_cluster_type_list.append(cell_cluster_type)
        line = f.readline() 
    f.close() 
    n_clusters = max(cell_cluster_type_list) + 1 # start from 0
    print('n clusters in drwaing:', n_clusters)
    coordinates = np.load(data_folder + 'coordinates.npy')

    plt.figure()
    sc_cluster = plt.scatter(x=coordinates[:,0], y=-coordinates[:,1], s=5, c=cell_cluster_type_list, cmap='viridis')
    plt.legend(*sc_cluster.legend_elements(), bbox_to_anchor=(1, 0.5), loc='center left', prop={'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.axis('scaled')
    plt.title(title)
    plt.savefig(save_path + '/spacial.png', dpi=400, bbox_inches='tight') 
    plt.clf()

    if bi:
        f = open(save_path + '/bi_logits.txt')            
        line = f.readline() # drop the first line  
        cell_cluster_bi_logit_list = []

        while line: 
            tmp = line.split('\t')
            cell_id = int(tmp[0]) # index start is start from 0 here
            #cell_type_index = int(tmp[1])
            cell_cluster_bi_logit = float(tmp[1].replace('\n', ''))
            cell_cluster_bi_logit_list.append(cell_cluster_bi_logit)
            line = f.readline() 
        f.close()

        plt.figure()
        sc_heatmap = plt.scatter(x=coordinates[:,0], y=-coordinates[:,1], s=5, c=cell_cluster_bi_logit_list, cmap='viridis')
        # cbar = plt.colorbar(sc_heatmap)
        # plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.axis('scaled')
        plt.title(title)
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(sc_heatmap, cax=cax)
        plt.savefig(save_path + '/heatmap.png', dpi=400, bbox_inches='tight') 
        plt.clf()

    # draw barplot
    if barplot:
        total_cell_num = len(cell_cluster_type_list)
        barplot = np.zeros([n_clusters, n_clusters], dtype=int)
        source_cluster_type_count = np.zeros(n_clusters, dtype=int)
        p1, p2 = adj_0.nonzero()
        def get_all_index(lst=None, item=''):
            return [i for i in range(len(lst)) if lst[i] == item]

        for i in range(total_cell_num):
            source_cluster_type_index = cell_cluster_type_list[i]
            edge_indeces = get_all_index(p1, item=i)
            paired_vertices = p2[edge_indeces]
            for j in paired_vertices:
                neighbor_type_index = cell_cluster_type_list[j]
                barplot[source_cluster_type_index, neighbor_type_index] += 1
                source_cluster_type_count[source_cluster_type_index] += 1

        np.savetxt(save_path + '/cluster_' + str(n_clusters) + '_barplot.txt', barplot, fmt='%3d', delimiter='\t')
        norm_barplot = barplot/(source_cluster_type_count.reshape(-1, 1))
        np.savetxt(save_path + '/cluster_' + str(n_clusters) + '_barplot_normalize.txt', norm_barplot, fmt='%3f', delimiter='\t')

        for clusters_i in range(n_clusters):
            plt.bar(range(n_clusters), norm_barplot[clusters_i], label='graph '+str(clusters_i))
            plt.xlabel('cell type index')
            plt.ylabel('value')
            plt.title('barplot_'+str(clusters_i))
            plt.savefig(save_path + '/barplot_sub' + str(clusters_i)+ '.jpg')
            plt.clf()

    return

import os.path as osp
import anndata as ad
import os
def draw_sub_type_map(
        bi_type, 
        data_name,
        types_dic,
        node_embed,
        method,
        time_stamp,
        seed,
):
    # outdir=osp.join('results', data_name, time_stamp, f'SubType_{method}')
    outdir=osp.join('results', data_name, time_stamp, method)
    if not osp.exists(outdir):
        os.makedirs(outdir)
    generated_data_fold = 'generated/' + data_name +'/'

    cell_types = pd.read_csv(generated_data_fold + 'cell_types.csv').iloc[:, 0]
    increment=0.02
    fixed_clus_count = 3
    for t in types_dic:
        if t == bi_type:
            type_id_list = cell_types[cell_types == t].index.to_list()

            # Fixed subtype number
            adata = ad.AnnData(node_embed[type_id_list])
            sc.pp.neighbors(adata, n_neighbors=20, n_pcs=10, key_added='SPACE', random_state=seed)
            for res in sorted(list(np.arange(0.02, 5, increment)), reverse=True):
                sc.tl.leiden(adata, resolution=res, neighbors_key='SPACE', random_state=seed)
                count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
                if count_unique_leiden == fixed_clus_count:
                    print('Resolution:', res)
                    break
            eval_resolution = res
            sc.tl.leiden(adata, key_added="leiden", neighbors_key='SPACE', resolution=eval_resolution, random_state=seed)
            sc.tl.umap(adata, neighbors_key='SPACE', random_state=seed)
            cluster_labels = np.array(adata.obs['leiden'])
            cluster_labels = [ [idx, int(x)] for idx, x in zip(type_id_list, cluster_labels) ]
            np.savetxt(outdir + f"/fixed_n_{t.replace('/', 'or')}_types.txt", np.array(cluster_labels), fmt='%3d', delimiter='\t')
            draw_default_outdir(generated_data_fold, outdir, f"SPACE_{t}", f"fixed_n_spatial_{t.replace('/', 'or')}", f"/fixed_n_{t.replace('/', 'or')}_types.txt")

            # Fixed leiden resolution
            adata = ad.AnnData(node_embed[type_id_list])
            sc.pp.neighbors(adata, n_neighbors=20, n_pcs=10, key_added='SPACE', random_state=seed)
            eval_resolution = 1
            sc.tl.leiden(adata, key_added="leiden", neighbors_key='SPACE', resolution=eval_resolution, random_state=seed)
            sc.tl.umap(adata, neighbors_key='SPACE', random_state=seed)
            cluster_labels = np.array(adata.obs['leiden'])
            cluster_labels = [ [idx, int(x)] for idx, x in zip(type_id_list, cluster_labels) ]
            np.savetxt(outdir + f"/fixed_reso_{t.replace('/', 'or')}_types.txt", np.array(cluster_labels), fmt='%3d', delimiter='\t')
            draw_default_outdir(generated_data_fold, outdir, f"SPACE_{t}", f"fixed_reso_spatial_{t.replace('/', 'or')}", f"/fixed_reso_{t.replace('/', 'or')}_types.txt")


def draw_default_outdir(data_folder, save_path, title, fig_name, type_txt_file_name):
    # print(f"Saving drew maps in {save_path}")

    f = open(save_path + f'/types.txt') if type_txt_file_name is None else open(save_path + type_txt_file_name)           
    line = f.readline() # drop the first line  
    cell_cluster_type_list = []
    id_list = []

    while line: 
        tmp = line.split('\t')
        cell_id = int(tmp[0]) # index start is start from 0 here
        id_list.append(cell_id)
        #cell_type_index = int(tmp[1])
        cell_cluster_type = int(tmp[1].replace('\n', ''))
        cell_cluster_type_list.append(cell_cluster_type)
        line = f.readline() 
    f.close() 
    n_clusters = max(cell_cluster_type_list) + 1 # start from 0
    print('n clusters in drwaing:', n_clusters)
    coordinates = np.load(data_folder + 'coordinates.npy')[id_list]

    sc_cluster = plt.scatter(x=coordinates[:,0], y=-coordinates[:,1], s=5, c=cell_cluster_type_list, cmap='rainbow')
    if n_clusters <= 3:
        plt.legend(*sc_cluster.legend_elements(), bbox_to_anchor=(1, 0.5), loc='center left', prop={'size': 9})
    else:
        plt.legend(*sc_cluster.legend_elements(num=n_clusters), bbox_to_anchor=(1, 0.5), loc='center left', prop={'size': 9})
    #cb_cluster = plt.colorbar(sc_cluster, boundaries=np.arange(n_types+1)-0.5).set_ticks(np.arange(n_types))    
    plt.xticks([])
    plt.yticks([])
    plt.axis('scaled')
    #plt.xlabel('X')
    #plt.ylabel('Y')
    plt.title(f'{title}')
    plt.savefig(save_path + f'/{fig_name}.png', dpi=400, bbox_inches='tight') 
    plt.clf()

import scipy.sparse as sp
def get_cell_type_aware_adj(X, adj_0, reso=0.2):
    adata = ad.AnnData(X)
    if not sp.isspmatrix_coo(adj_0):
        adj = adj_0.tocoo()
    else:
        adj = adj_0.copy()
    adj = adj.astype(np.float32)
    Graph_df = pd.DataFrame({'St': adj.row, 'Ed': adj.col, 'Data': adj.data})

    print('------Pre-clustering using louvain with resolution=%.2f' %reso)
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata)
    pre_labels = 'expression_louvain_label'
    sc.tl.louvain(adata, resolution=reso, key_added='expression_louvain_label')
    label = adata.obs[pre_labels]

    print('------Pruning the graph...')
    print('%d edges before pruning.' %Graph_df.shape[0])
    pro_labels_dict = dict(zip(list(map(int, label.index)), label))
    Graph_df['St_label'] = Graph_df['St'].map(pro_labels_dict)
    Graph_df['Ed_label'] = Graph_df['Ed'].map(pro_labels_dict)
    Graph_df = Graph_df.loc[Graph_df['St_label']==Graph_df['Ed_label'],]
    print('%d edges after pruning.' %Graph_df.shape[0])

    prune_G = sp.coo_matrix((np.ones(Graph_df.shape[0]), (Graph_df['St'], Graph_df['Ed'])))
    return prune_G

def sparse_mx_to_torch_edge_list(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    edge_list = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    return edge_list

def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index

# 将字典保存到文件
def save_dict_to_file(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file)

# 从文件中加载字典
def load_dict_from_file(filename):
    with open(filename, 'rb') as file:
        dictionary = pickle.load(file)
    return dictionary


import os
import ot
import torch
import random
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from torch.backends import cudnn
#from scipy.sparse import issparse
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from sklearn.neighbors import NearestNeighbors 

def filter_with_overlap_gene(adata, adata_sc):
    # remove all-zero-valued genes
    #sc.pp.filter_genes(adata, min_cells=1)
    #sc.pp.filter_genes(adata_sc, min_cells=1)
    
    if 'highly_variable' not in adata.var.keys():
       raise ValueError("'highly_variable' are not existed in adata!")
    else:    
       adata = adata[:, adata.var['highly_variable']]
       
    if 'highly_variable' not in adata_sc.var.keys():
       raise ValueError("'highly_variable' are not existed in adata_sc!")
    else:    
       adata_sc = adata_sc[:, adata_sc.var['highly_variable']]   

    # Refine `marker_genes` so that they are shared by both adatas
    genes = list(set(adata.var.index) & set(adata_sc.var.index))
    genes.sort()
    print('Number of overlap genes:', len(genes))

    adata.uns["overlap_genes"] = genes
    adata_sc.uns["overlap_genes"] = genes
    
    adata = adata[:, genes]
    adata_sc = adata_sc[:, genes]
    
    return adata, adata_sc

def permutation(feature):
    # fix_seed(FLAGS.random_seed) 
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutated = feature[ids]
    
    return feature_permutated 

def construct_interaction(adata, n_neighbors=3):
    """Constructing spot-to-spot interactive graph"""
    position = adata.obsm['spatial']
    
    # calculate distance matrix
    distance_matrix = ot.dist(position, position, metric='euclidean')
    n_spot = distance_matrix.shape[0]
    
    adata.obsm['distance_matrix'] = distance_matrix
    
    # find k-nearest neighbors
    interaction = np.zeros([n_spot, n_spot])  
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1
         
    adata.obsm['graph_neigh'] = interaction
    
    #transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj>1, 1, adj)
    
    adata.obsm['adj'] = adj
    
def construct_interaction_KNN(adata, n_neighbors=3):
    position = adata.obsm['spatial']
    n_spot = position.shape[0]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(position)  
    _ , indices = nbrs.kneighbors(position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    interaction = np.zeros([n_spot, n_spot])
    interaction[x, y] = 1
    
    adata.obsm['graph_neigh'] = interaction
    
    #transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj>1, 1, adj)
    
    adata.obsm['adj'] = adj
    print('Graph constructed!')   

def preprocess(adata, n_top_genes=3000, get_highly_vars=True):
    if get_highly_vars:
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)

import time
import os.path as osp
import pickle
from scipy import sparse
def get_adj(generated_data_fold, thresholds):
    coordinates = np.load(generated_data_fold + 'coordinates.npy')
    if not os.path.exists(generated_data_fold):
        os.makedirs(generated_data_fold) 
    ############# get batch adjacent matrix
    cell_num = len(coordinates)

    ############ the distribution of distance 
    if not osp.exists(generated_data_fold + 'distance_array.npy'):
        distance_list = []
        print ('calculating distance matrix, it takes a while')
        st = time.time()

        distance_list = []
        for j in range(cell_num):
            for i in range (cell_num):
                if i!=j:
                    distance_list.append(np.linalg.norm(coordinates[j]-coordinates[i]))

        distance_array = np.array(distance_list)
        np.save(generated_data_fold + 'distance_array.npy', distance_array)
        print(f'Total time: {time.time() - st}s.')
    else:
        distance_array = np.load(generated_data_fold + 'distance_array.npy')

    ###try different distance thrdescribeold, so that on average, each cell has x neighbor cells, see Tab. S1 for results

    for threshold in thresholds:
        num_big = np.where(distance_array<threshold)[0].shape[0]
        print (threshold,num_big,str(num_big/(cell_num*2))) #300 22064 2.9046866771985256
        from sklearn.metrics.pairwise import euclidean_distances

        distance_matrix = euclidean_distances(coordinates, coordinates)
        distance_matrix_threshold_I = np.zeros(distance_matrix.shape)
        distance_matrix_threshold_W = np.zeros(distance_matrix.shape)
        for i in range(distance_matrix_threshold_I.shape[0]):
            for j in range(distance_matrix_threshold_I.shape[1]):
                if distance_matrix[i,j] <= threshold and distance_matrix[i,j] > 0:
                    distance_matrix_threshold_I[i,j] = 1
                    distance_matrix_threshold_W[i,j] = distance_matrix[i,j]

        ############### get normalized sparse adjacent matrix
        distance_matrix_threshold_W_N = np.float32(distance_matrix_threshold_W) ## do not normalize adjcent matrix
        # # normalize the distance matrix using min-max normalization
        # max_val = np.max(distance_matrix_threshold_W_N)
        # print(max_val)
        # distance_matrix_threshold_W_N = distance_matrix_threshold_W_N / max_val
        distance_matrix_threshold_W_N_crs = sparse.csr_matrix(distance_matrix_threshold_W_N)
        with open(generated_data_fold + f'Adjacent', 'wb') as fp:
            pickle.dump(distance_matrix_threshold_W_N_crs, fp)
    
def get_feature(adata, deconvolution=False):
    if deconvolution:
       adata_Vars = adata
    else:   
       adata_Vars =  adata[:, adata.var['highly_variable']]
       
    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
       feat = adata_Vars.X.toarray()[:, ]
    else:
       feat = adata_Vars.X[:, ] 
    
    # data augmentation
    feat_a = permutation(feat)
    
    adata.obsm['feat'] = feat
    adata.obsm['feat_a'] = feat_a    
    
def add_contrastive_label(adata):
    # contrastive label
    n_spot = adata.n_obs
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    label_CSL = np.concatenate([one_matrix, zero_matrix], axis=1)
    adata.obsm['label_CSL'] = label_CSL
    
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)+np.eye(adj.shape[0])
    return adj_normalized 

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_adj_sparse(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)    

def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'