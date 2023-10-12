import os
import os.path as osp
import pandas as pd
import scanpy as sc
import numpy as np
import pickle
import time
from scipy import sparse
import matplotlib.pyplot as plt
from scipy.spatial import distance
from .utils import preprocess, get_feature, save_dict_to_file


def get_bi_type(cell_types, generated_data_fold):
    types_dic = np.loadtxt(generated_data_fold + 'types_dic.txt', delimiter='\t', dtype=str)
    for type in types_dic:
        bi_types_idx = (cell_types == type).astype(int).to_numpy()
        np.save(generated_data_fold + f"cell_type_indeces_{type.replace('/', 'or')}.npy", np.array(bi_types_idx))

def get_type(cell_types, generated_data_fold):
    types_dic = []
    types_idx = []
    for t in cell_types:
        if not t in types_dic:
            types_dic.append(t) 
        id = types_dic.index(t)
        types_idx.append(id)
    
    np.save(generated_data_fold + 'cell_type_indeces.npy', np.array(types_idx))
    np.savetxt(generated_data_fold + 'types_dic.txt', np.array(types_dic), fmt='%s', delimiter='\t')

def get_adj(generated_data_fold, boundary, k):
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

    for threshold in [boundary]:
        num_big = np.where(distance_array<threshold)[0].shape[0]
        print (threshold, num_big, str(num_big / (cell_num * 2))) #300 22064 2.9046866771985256
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
        distance_matrix_threshold_I_N = np.float32(distance_matrix_threshold_I)
        # # normalize the distance matrix using min-max normalization
        # max_val = np.max(distance_matrix_threshold_W_N)
        # print(max_val)
        # distance_matrix_threshold_W_N = distance_matrix_threshold_W_N / max_val
        distance_matrix_threshold_W_N_crs = sparse.csr_matrix(distance_matrix_threshold_W_N)
        distance_matrix_thredhold_I_N_crs = sparse.csr_matrix(distance_matrix_threshold_I_N)
        with open(generated_data_fold + f'adj_{k}_w', 'wb') as fp:
            pickle.dump(distance_matrix_threshold_W_N_crs, fp)
        with open(generated_data_fold + f'adj_{k}_i', 'wb') as fp:
            pickle.dump(distance_matrix_thredhold_I_N_crs, fp)

def draw_map(generated_data_fold):
    coordinates = np.load(generated_data_fold + 'coordinates.npy')
    cell_type_indeces = np.load(generated_data_fold + 'cell_type_indeces.npy')
    n_cells = len(cell_type_indeces)
    n_types = max(cell_type_indeces) + 1 # start from 0
    
    types_dic = np.loadtxt(generated_data_fold+'types_dic.txt', dtype='|S15',   delimiter='\t').tolist()
    for i,tmp in enumerate(types_dic):
        types_dic[i] = tmp.decode()
    print(types_dic)

    sc_cluster = plt.scatter(x=coordinates[:,0], y=-coordinates[:,1], s=5, c=cell_type_indeces, cmap='rainbow')  
    plt.legend(*sc_cluster.legend_elements(num=n_types), bbox_to_anchor=(1,0.5), loc='center left', prop={'size': 9}) 
    
    plt.xticks([])
    plt.yticks([])
    plt.axis('scaled')
    #plt.xlabel('X')
    #plt.ylabel('Y')
    plt.title('Annotation')
    plt.savefig(generated_data_fold+'/spacial.png', dpi=400, bbox_inches='tight') 
    plt.clf()

def draw_bi_map(generated_data_fold):
    types_dic = np.loadtxt(generated_data_fold + 'types_dic.txt', delimiter='\t', dtype=str)
    for type in types_dic:
        coordinates = np.load(generated_data_fold + 'coordinates.npy')
        cell_type_indeces = np.load(generated_data_fold + f"cell_type_indeces_{type.replace('/', 'or')}.npy")
        n_cells = len(cell_type_indeces)
        n_types = max(cell_type_indeces) + 1 # start from 0

        sc_cluster = plt.scatter(x=coordinates[:,0], y=-coordinates[:,1], s=5, c=cell_type_indeces, cmap='viridis')  
        plt.legend(*sc_cluster.legend_elements(), bbox_to_anchor=(1,0.5), loc='center left', prop={'size': 9}) 
        
        plt.xticks([])
        plt.yticks([])
        plt.axis('scaled')
        #plt.xlabel('X')
        #plt.ylabel('Y')
        plt.title(f'Annotation_{type}')
        plt.savefig(generated_data_fold+f"/spacial_{type.replace('/', 'or')}.png", dpi=400, bbox_inches='tight') 
        plt.clf()

def generate_data(args):
    data_fold = args.raw_path + args.data_name + '/'
    generated_data_fold = args.data_path + args.data_name + '/'
    if not os.path.exists(generated_data_fold):
        os.makedirs(generated_data_fold)
    adata_h5ad = sc.read_h5ad(osp.join(data_fold, f'{args.h5_name}.h5ad'))

    adata = adata_h5ad.copy()
    adata.var = adata.var.drop('highly_variable', axis=1) if 'highly_variable' in adata.var.keys() else adata.var

    get_highly_vars = True if 'highly_variable' not in adata.var.keys() else False
    preprocess(adata, n_top_genes=args.n_input, get_highly_vars=get_highly_vars)

    if 'feat' not in adata.obsm.keys():
        get_feature(adata, args.use_whole_gene)

    # gene_ids = adata.var['gene_ids']
    coordinates = adata.obsm['spatial']
    features = adata.obsm['feat']

    adata.obs['SubClass'].to_csv(generated_data_fold + 'cell_types.csv', index=False)
    np.save(generated_data_fold + 'features.npy', features)
    np.save(generated_data_fold + 'coordinates.npy', np.array(coordinates))

    cell_num = len(coordinates)
    adj_coo = coordinates
    edgeList = []
    k = args.k
    for sigma_num in range(100):
        print(sigma_num)
        boundary = 1e8
        for node_idx in range(cell_num):
            tmp = adj_coo[node_idx, : ].reshape(1, -1)
            distMat = distance.cdist(tmp, adj_coo, 'euclidean')
            res = distMat.argsort()[: k + 1]
            tmpdist = distMat[0, res[0][1: k + 1]]
            boundary = min(np.mean(tmpdist) + sigma_num * np.std(tmpdist), boundary)

        cell_cluster_type_list = {}
        edge_num = 0
        for node_idx in range(cell_num):
            cell_cluster_type_list[node_idx] = [0] * cell_num
            tmp = adj_coo[node_idx, : ].reshape(1, -1)
            distMat = distance.cdist(tmp, adj_coo, 'euclidean')
            res = distMat.argsort()[: k + 1]
            tmpdist = distMat[0, res[0][1: k + 1]]
            for j in np.arange(1, k+1):
                if distMat[0, res[0][j]] <= boundary:
                    weight = 1.0
                    edge_num += 1
                else:
                    weight = 0.0
                cell_cluster_type_list[node_idx][res[0][j]] = weight

        if edge_num / cell_num > (k - 0.2) and edge_num / cell_num < k:
            print('Boundary: ', boundary)
            print('Average number of neighbor: ', edge_num / cell_num)
            break

    cell_types = adata.obs['SubClass']

    get_adj(generated_data_fold, boundary=boundary, k=k)
    get_type(cell_types, generated_data_fold)
    get_bi_type(cell_types, generated_data_fold)
    draw_map(generated_data_fold)
    draw_bi_map(generated_data_fold)
    
    save_dict_to_file(adata.uns['LR_cell_weight'], generated_data_fold + 'lr_cell_weight')
    np.save(generated_data_fold + 'highly_var_gene_symbols.npy', adata.var[adata.var['highly_variable']].index.to_numpy())

    LR_dict = {}
    for inter_act in adata.uns['LR_cell_weight'].keys():
        LR_dict[inter_act] = {}
    with open(generated_data_fold + f'Interaction2symbol', 'wb') as fp:
        pickle.dump(LR_dict, fp)