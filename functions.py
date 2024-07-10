# Use this as a source folder for all custom functions we create throughout the different sections. 
# The idea is that we use this as a source from which we can pull functions we introduced in an earlier section

import matplotlib.pyplot as plt
import matplotlib.patches as Patch
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import astropy.stats
from palettable import wesanderson
import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.optim as optim
import torch.nn as nn
import sklearn as sk
import seaborn as sns
from sklearn.metrics import precision_recall_curve , average_precision_score , recall_score ,  PrecisionRecallDisplay
from tqdm.notebook import tqdm
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from itertools import chain

def get_edge_attributes(G):
    """
    Extracts edge attributes from a graph.

    Args:
        G (networkx.Graph): The graph from which to extract edge attributes.

    Returns:
        list: A list of edge attributes.

    Raises:
        ValueError: If the graph G is empty or not defined.
    """
    if not G:
        raise ValueError("The graph is empty or not defined.")

    # Extract edge attributes
    edge_attributes = list(set(chain.from_iterable(d.keys() for *_, d in G.edges(data=True)))
    )
    return edge_attributes

def draw_network_with_node_attrs(G, node_attributes, communities=None, title='Network Visualization', color_attr=None, shape_attr=None, figsize=(20,10), layout='spring', cmap_name='tab20', with_labels=False, savefig=False, save_path_prefix=''):
    """
    Draws a network graph with nodes colored and shaped based on their attributes, and optionally colored by community membership.

    Args:
        G (networkx.Graph): The graph to be drawn.
        node_attributes (dict): A dictionary where keys are node names and values are dictionaries of attributes.
        communities (List[List[Any]], optional): A list where each sublist contains the nodes belonging to a community. Default is None.
        title (str, optional): The title of the plot. Default is 'Network Visualization'.
        color_attr (str, optional): Node attribute to color nodes by. Default is None.
        shape_attr (str, optional): Node attribute to shape nodes by. Default is None.
        figsize (tuple, optional): The size of the figure. Default is (20, 10).
        layout (str, optional): The layout algorithm for positioning nodes ('spring', 'circular', etc.). Default is 'spring'.
        cmap_name (str, optional): The name of the colormap to use for coloring. Default is 'tab20'.
        with_labels (bool, optional): Whether to draw labels for the nodes. Default is False.

    Raises:
        ValueError: If the graph G is empty or not defined.
        ValueError: If node_attributes is empty or not defined.

    The function draws the graph with nodes colored and/or shaped based on their attributes. If communities are provided, nodes are colored by their community memberships. A legend is added to indicate the mapping of attributes to colors and shapes.
    """
    
    if not G:
        raise ValueError("The graph is empty or not defined.")
    if not node_attributes:
        raise ValueError("Node attributes are empty or not defined.")

    if communities:
        community_dict = {node: idx for idx, community in enumerate(communities) for node in community}
        node_attributes['community']=community_dict
    
    shapes = ['o', '^', 's', 'p', 'h', 'H', '8', 'd', 'D', 'v', '<', '>', 'P', '*', 'X']
    cmap = plt.get_cmap(cmap_name)
    
    unique_attr_vals_color = list(set(node_attributes[color_attr].values())) if color_attr else []
    color_map = {val: cmap(i / len(unique_attr_vals_color)) for i, val in enumerate(unique_attr_vals_color)}
    node_colors_from_attribute = [color_map[node_attributes[color_attr][node]] for node in G.nodes()] if color_attr else ['blue']*len(list(G.nodes()))
    
    unique_attr_vals_shape = list(set(node_attributes[shape_attr].values())) if shape_attr else []
    shape_map = {val: shapes[i] for i, val in enumerate(unique_attr_vals_shape)}
    node_shapes_from_attribute = [shape_map[node_attributes[shape_attr][node]] if shape_attr else 'o' for node in G.nodes()]

    plt.figure(figsize=figsize)
    pos = getattr(nx, f'{layout}_layout')(G) if hasattr(nx, f'{layout}_layout') else nx.spring_layout(G)
    
    node_list_by_shape = {}
    node_idx_list_by_shape = {}
    for shape_marker in shapes[:len(unique_attr_vals_shape)]:
        node_list_by_shape[shape_marker] = [node for node in list(G.nodes()) if shape_map[node_attributes[shape_attr][node]] == shape_marker]
        node_idx_list_by_shape[shape_marker] = [node_idx for node_idx, node in enumerate(list(G.nodes())) if shape_map[node_attributes[shape_attr][node]] == shape_marker]
    
    for shape_marker, node_list in node_list_by_shape.items():
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=node_list_by_shape[shape_marker],
            node_color=[node_colors_from_attribute[i] for i in node_idx_list_by_shape[shape_marker]],
            node_shape=shape_marker,
            node_size=400,
            edgecolors='yellow'
        )

    nx.draw_networkx_edges(G, pos, width=0.5)
    if with_labels:
        nx.draw_networkx_labels(G, pos, font_size=12)

    legend_fontsize = 14
    if shape_attr:
        shape_legend_handles = [Line2D([0], [0], marker=shape_map[val], color='w', label=f'{val}', markerfacecolor='k', markersize=10) for val in unique_attr_vals_shape]
        leg1 = plt.legend(handles=shape_legend_handles, title=f"{shape_attr.capitalize()} (Shape)", loc='upper left', bbox_to_anchor=(1, 0.5), fontsize=legend_fontsize, title_fontsize=legend_fontsize)
    if color_attr:
        color_legend_handles = [mpatches.Patch(facecolor=color_map[val], label=f'{color_attr} {val}') for val in unique_attr_vals_color]
        leg2 = plt.legend(handles=color_legend_handles, title=f"{color_attr.capitalize()} (Color)", loc='upper left', bbox_to_anchor=(1, 1), fontsize=legend_fontsize, title_fontsize=legend_fontsize)
    if shape_attr:
        plt.gca().add_artist(leg1)
    plt.title(title, fontsize=20)
    
    save_path = None
    if savefig:
        time_string = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = save_path_prefix + f'_{time_string}.png' if save_path_prefix else f'net_image_{time_string}.png' 
        plt.savefig(save_path)
        
    plt.show()
    
    return save_path

def message_passing(node, G):
    """
    Perform message passing for a given node in a graph.

    Args:
        node (int): The node for which message passing is performed.
        G (networkx.Graph): The graph containing the node and its neighbors.

    Returns:
        numpy.ndarray: The aggregated message from the neighboring nodes.

    Notes:
        This function gathers the messages for a single node and will be used in the message_passing_iteration.
        The function performs propagation and aggregation.
        Propagation: Gather the node features of all neighboring nodes.
        Aggregation: Aggregate the gathered messages using median aggregation.
    """
    neighbors = list(G.neighbors(node))
    if not neighbors:
        return G.nodes[node]['feature']
    else:
        neighbor_features = [G.nodes[neighbor]['feature'] for neighbor in neighbors]
        aggregated_message = np.median(neighbor_features, axis=0)
        
        return aggregated_message

def message_passing_iteration(G):
    """
    Perform message passing iteration on a graph.

    Args:
        G (networkx.Graph): The input graph.

    Returns:
        None

    Description:
        This function performs message passing iteration on a graph. It iterates through all nodes in the graph and updates their features based on the aggregated message from neighboring nodes.

    Note:
        The function `message_passing` is defined above.

    """
    updated_features = {}

    for node in G.nodes:
        updated_features[node] = message_passing(node, G)
    
    for node, feature in updated_features.items():
        G.nodes[node]['feature'] = feature

def pearson_corr(data):
    """
    Calculate the Pearson correlation coefficient matrix for a given DataFrame.

    Args:
        data (pandas.DataFrame): The input DataFrame containing numeric data.

    Returns:
        pandas.DataFrame: The correlation coefficient matrix.

    """
    data = data._get_numeric_data()
    cols = data.columns
    idx = cols.copy()
    mat = data.to_numpy(dtype=float, na_value=np.nan, copy=False)
    mat = mat.T

    K = len(cols)
    correl = np.empty((K, K), dtype=np.float32)
    mask = np.isfinite(mat)

    cov = np.cov(mat)

    for i in range(K):
        correl[i, :] = cov[i, :] / np.sqrt(cov[i, i] * np.diag(cov))

    return pd.DataFrame(data=correl, index=idx, columns=cols, dtype=np.float32)

def abs_bicorr(data):
    """
    Calculate the absolute biweight midcorrelation matrix for the given data.

    Args:
        data (pandas.DataFrame): The input DataFrame containing numeric data.

    Returns:
        pandas.DataFrame: The absolute biweight midcorrelation matrix.

    """
    data = data._get_numeric_data()
    cols = data.columns
    idx = cols.copy()
    mat = data.to_numpy(dtype=float, na_value=np.nan, copy=False)
    mat = mat.T

    K = len(cols)
    correl = np.empty((K, K), dtype=np.float32)
    mask = np.isfinite(mat)

    bicorr = astropy.stats.biweight_midcovariance(mat)

    for i in range(K):
        correl[i, :] = bicorr[i, :] / np.sqrt(bicorr[i, i] * np.diag(bicorr))

    return pd.DataFrame(data=correl, index=idx, columns=cols, dtype=np.float32)

def get_k_neighbours(df, k):
    """
    Returns a dictionary of k-nearest neighbors for each node in the dataframe.

    Args:
        df (DataFrame): The similarity matrix dataframe.
        k (int): The number of neighbors to retrieve.

    Returns:
        dict: A dictionary where the keys are the nodes and the values are lists of k-nearest neighbors.
    """
    k_neighbours = {}
    if abs(df.max().max()) > 1:
        print('Dataframe should be a similarity matrix of max value 1')
    else:
        np.fill_diagonal(df.values, 1)
        for node in df.index:
            neighbours = df.loc[node].nlargest(k+1).index.to_list()[1:]  # Exclude the node itself
            k_neighbours[node] = neighbours

    return k_neighbours

def plot_knn_network(df , K , labels , node_colours = ['skyblue'] , node_size = 300) : 
    """
    Plots a k-nearest neighbors network based on the given dataframe and parameters.

    Args:
        df (pandas.DataFrame): The input dataframe.
        K (int): The number of nearest neighbors to consider.
        labels (pandas.Series): The labels for each node in the dataframe.
        node_colours (list, optional): The colors for the nodes. Defaults to ['skyblue'].
        node_size (int, optional): The size of the nodes. Defaults to 300.

    Returns:
        nx.Graph: The NetworkX graph representing the k-nearest neighbors network.
    """
    
    # Get K-nearest neighbours for each node
    k_neighbours = get_k_neighbours(df , k = K)
    
    # Create a NetworkX graph
    G = nx.Graph()
    
    # Add nodes to the graph
    G.add_nodes_from(df.index)
    
    nx.set_node_attributes(G , labels.astype('category').cat.codes , 'label')
    nx.set_node_attributes(G , pd.Series(np.arange(len(df.index)) , index=df.index) , 'idx')

    # Add edges based on the k-nearest neighbours
    for node, neighbours in k_neighbours.items():
        for neighbor in neighbours:
            G.add_edge(neighbor, node)

    plt.figure(figsize=(10, 8))
    nx.draw(G, with_labels=False, font_weight='bold', node_size=node_size, node_color=node_colours, font_size=8)
    
    return G

def gen_graph_legend(node_colours, G, attr):
    """
    Generate a legend for a graph based on node colors and attributes.

    Args:
        node_colours (pd.Series): A series of node colors.
        G (networkx.Graph): The graph object.
        attr (str): The attribute to use for labeling.

    Returns:
        patches (list): A list of matplotlib patches representing the legend.

    """
    patches = []
    for col, lab in zip(node_colours.drop_duplicates(), pd.Series(nx.get_node_attributes(G, attr)).drop_duplicates()):
        patches.append(mpatches.Patch(color=col, label=lab))

    return patches

def train(g, h, train_split , val_split , device ,  model , labels , epochs , lr):
    """
    Trains a model using the specified graph, node features, 
    train/validation splits, device, model, labels, epochs, and learning rate.

    Args:
        g (Graph): The graph object.
        h (Tensor): The node features tensor.
        train_split (Tensor): The train split tensor.
        val_split (Tensor): The validation split tensor.
        device (str): The device to train the model on.
        model (nn.Module): The model to train.
        labels (Tensor): The labels tensor.
        epochs (int): The number of training epochs.
        lr (float): The learning rate.

    Returns:
        tuple: A tuple containing the figures for training and validation loss.
    """

    # loss function, optimizer and scheduler
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr , weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

    train_loss = []
    val_loss   = []
    train_acc  = []
    val_acc    = []
    
    # training loop
    epoch_progress = tqdm(total=epochs, desc='Loss : ', unit='epoch')
    for epoch in range(epochs):
        model.train()

        logits  = model(g, h)

        loss = loss_fcn(logits[train_split], labels[train_split].float())
        train_loss.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step()
        
        if (epoch % 5) == 0 :
            
            _, predicted = torch.max(logits[train_split], 1)
            _, true = torch.max(labels[train_split] , 1)
            train_acc.append((predicted == true).float().mean().item())

            valid_loss , valid_acc , *_ = evaluate(val_split, device, g , h, model , labels)
            val_loss.append(valid_loss.item())
            val_acc.append(valid_acc)
            
            epoch_desc = (
                "Epoch {:05d} | Loss {:.4f} | Train Acc. {:.4f} | Validation Acc. {:.4f} ".format(
                    epoch, np.mean(train_loss[-5:]) , np.mean(train_acc[-5:]), np.mean(val_acc[-5:])
                )
            )
            
            epoch_progress.set_description(epoch_desc)
            epoch_progress.update(5)

    fig1 , ax1 = plt.subplots(figsize=(6,4))
    ax1.plot(train_loss , label = 'Train Loss')
    ax1.plot(range(5 , len(train_loss)+1 , 5) , val_loss  , label = 'Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    fig2 , ax2 = plt.subplots(figsize=(6,4))
    ax2.plot(train_acc  , label = 'Train Accuracy')
    ax2.plot(val_acc  , label = 'Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_ylim(0,1)
    ax2.legend()
    
    # Close tqdm for epochs
    epoch_progress.close()

    return fig1 , fig2

def evaluate(split, device, g , h, model , labels):
    """
    Evaluate the performance of a model on a given dataset split.

    Args:
        split (Tensor): The index of the dataset split to evaluate.
        device: The device to perform the evaluation on.
        g: The graph input to the model.
        h: The node features input to the model.
        model: The model to evaluate.
        labels: The ground truth labels for the dataset.

    Returns:
        tuple: A tuple containing the evaluation metrics:
            - loss (float): The loss value.
            - acc (float): The accuracy value.
            - F1 (float): The F1 score.
            - PRC (float): The precision-recall curve value.
            - SNS (float): The sensitivity value.
    """
    model.eval()
    loss_fcn = nn.CrossEntropyLoss()
    acc = 0
    
    with torch.no_grad() : 
        logits = model(g, h)
        
        loss = loss_fcn(logits[split], labels[split].float())

        _, predicted = torch.max(logits[split], 1)
        _, true = torch.max(labels[split] , 1)
        acc = (predicted == true).float().mean().item()
        
        logits_out = logits[split].cpu().detach().numpy()
        binary_out = (logits_out == logits_out.max(1).reshape(-1,1))*1
        
        labels_out = labels[split].cpu().detach().numpy()
        
        PRC =  average_precision_score(labels_out , binary_out , average="weighted")
        SNS = recall_score(labels_out , binary_out , average="weighted")
        F1 = 2*((PRC*SNS)/(PRC+SNS))
        
    
    return loss , acc , F1 , PRC , SNS
    
class GCN(nn.Module):
    """
    Graph Convolutional Network (GCN) class.
    
    This class represents a Graph Convolutional Network (GCN) model.
    It inherits from the `nn.Module` class of PyTorch.
    
    Args:
        input_dim (int): The dimensionality of the input features.
        hidden_feats (list): A list of integers representing the number of hidden units in each layer.
        num_classes (int): The number of output classes.
    """
    
    def __init__(self, input_dim,  hidden_feats, num_classes):
        
        super().__init__()
        
        self.gcnlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.num_layers = len(hidden_feats) + 1
        
        for layers in range(self.num_layers) :
            if layers < self.num_layers -1 :
                if layers == 0 : 
                    self.gcnlayers.append(
                        GraphConv(input_dim , hidden_feats[layers])
                    )
                else :
                    self.gcnlayers.append(
                        GraphConv(hidden_feats[layers-1] , hidden_feats[layers])
                    )
                self.batch_norms.append(nn.BatchNorm1d(hidden_feats[layers]))
            else : 
                self.gcnlayers.append(
                    GraphConv(hidden_feats[layers-1] , num_classes)
                )
                
        self.drop = nn.Dropout(0.05)

    def forward(self, g, h):
        """
        Forward pass of the GCN model.
        
        This method performs the forward pass of the GCN model for an arbitrary number of layers.
        
        Args:
            g (Graph): The input graph.
            h (Tensor): The input node features.
        
        Returns:
            Tensor: The output scores of the model.
        """
        
        for layers in range(self.num_layers) : 
            if layers == self.num_layers - 1 : 
                h = self.gcnlayers[layers](g , h)
            else : 
                h = self.gcnlayers[layers](g, h)
                h = self.drop(F.relu(h))
            
        score = self.drop(h)
            
        return score
    

def filter_low_expression_genes(data, threshold=1.0):
    """
    Filter out low-expressed genes from the dataset.

    Calculates the mean expression level for each gene and filters out
    genes whose mean expression level is below the specified threshold.

    Parameters:
    data (DataFrame): Expression data with genes as columns.
    threshold (float): Minimum mean expression level to retain a gene.
                       Default is 1.0.

    Returns:
    DataFrame: Filtered data with genes above the threshold.
    """
    # Calculate the mean expression for each gene
    gene_means = data.mean(axis=0)
    # Filter out genes with mean expression below the threshold
    mask = gene_means >= threshold
    filtered_data = data.loc[:, mask]
    return filtered_data


def filter_high_variance_genes(data, threshold):
    """
    Filter out genes with variance below the specified threshold.

    Calculates the variance for each gene and filters out genes whose 
    variance is below the specified threshold.

    Parameters:
    data (DataFrame): Gene expression data with genes as columns and samples as rows.
    threshold (float): Minimum variance level to retain a gene.

    Returns:
    DataFrame: Filtered data with genes having variance above the threshold.
    """

    # Calculate the variance for each gene (column)
    gene_variances = data.var(axis=0)
    # Create a boolean mask to filter out genes with variance below the threshold
    mask = gene_variances >= threshold
    # Apply the mask to filter the DataFrame
    filtered_data = data.loc[:, mask]
    return filtered_data

def calc_abs_bicorr(data):
    """
    Calculate the absolute biweight midcorrelation matrix for numeric data.

    Parameters:
    data (pd.DataFrame): Input DataFrame with numeric data.

    Returns:
    pd.DataFrame: DataFrame containing the absolute biweight midcorrelation matrix.
    """

    # Select only numeric data
    data = data._get_numeric_data()
    cols = data.columns
    idx = cols.copy()
    mat = data.to_numpy(dtype=float, na_value=np.nan, copy=False)
    mat = mat.T

    K = len(cols)
    correl = np.empty((K, K), dtype=np.float32)

    # Calculate biweight midcovariance
    bicorr = astropy.stats.biweight_midcovariance(mat, modify_sample_size=True)

    for i in range(K):
        for j in range(K):
            if i == j:
                correl[i, j] = 1.0
            else:
                denominator = np.sqrt(bicorr[i, i] * bicorr[j, j])
                if denominator != 0:
                    correl[i, j] = bicorr[i, j] / denominator
                else:
                    correl[i, j] = 0  # Or handle it in another appropriate way

    return pd.DataFrame(data=np.abs(correl), index=idx, columns=cols, dtype=np.float32)


def create_graph_from_correlation(correlation_matrix, threshold=0.8):
    """
    Creates a graph from a correlation matrix using a specified threshold.

    Parameters:
    correlation_matrix (pd.DataFrame): DataFrame containing the correlation matrix.
    threshold (float): Threshold for including edges based on correlation value.

    Returns:
    G (nx.Graph): Graph created from the correlation matrix.
    """
    G = nx.Graph()

    # Add nodes
    for node in correlation_matrix.columns:
        G.add_node(node)

    # Add edges with weights above the threshold
    for i in range(correlation_matrix.shape[0]):
        for j in range(i + 1, correlation_matrix.shape[1]):
            if i != j:  # Ignore the diagonal elements
                weight = correlation_matrix.iloc[i, j]
                if abs(weight) >= threshold:
                    G.add_edge(correlation_matrix.index[i], correlation_matrix.columns[j], weight=weight)

    return G


def print_graph_info(G):
    """
    Print basic information about a NetworkX graph.

    
    Parameters:
    G (nx.Graph): The NetworkX graph.
    """
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print("Sample nodes:", list(G.nodes)[:10])  # Print first 10 nodes as a sample
    print("Sample edges:", list(G.edges(data=True))[:10])  # Print first 10 edges as a sample
    
    info_str = "Graph type: "
    is_directed = G.is_directed()
    if is_directed:
        info_str += "directed"
    else:
        info_str += "undirected"
    print(info_str)

    # Check for self-loops
    self_loops = list(nx.selfloop_edges(G))
    if self_loops:
        print(f"Number of self-loops: {len(self_loops)}")
        print("Self-loops:", self_loops)
    else:
        print("No self-loops in the graph.")

    # density of the graph
    density = nx.density(G)
    print(f"Graph density: {density}")

    # Find and print the number of connected components
    num_connected_components = nx.number_connected_components(G)
    print(f"Number of connected components: {num_connected_components}")

    # Calculate and print the clustering coefficient of the graph
    clustering_coeff = nx.average_clustering(G)
    print(f"Average clustering coefficient: {clustering_coeff}")


# Function to visualize the graph
def visualise_graph(G, title='Gene Co-expression Network'):
    """
    Visualizes the graph using Matplotlib and NetworkX.

    Parameters:
    G (nx.Graph): Graph to visualize.
    title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, k=0.1)  # k controls the distance between nodes
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='blue', alpha=0.7)
    nx.draw_networkx_edges(G, pos, width=0.2, alpha=0.5)
    plt.title(title)
    plt.show()


def clean_graph(G, degree_threshold=1, keep_largest_component=True):
    """
    Cleans the graph by performing several cleaning steps:
    - Removes unconnected nodes (isolates)
    - Removes self-loops
    - Removes nodes with a degree below a specified threshold
    - Keeps only the largest connected component (optional)

    Parameters:
    G (nx.Graph): The NetworkX graph to clean.
    degree_threshold (int): Minimum degree for nodes to keep.
    keep_largest_component (bool): Whether to keep only the largest connected component.

    Returns:
    G (nx.Graph): Cleaned graph.
    """
    G = G.copy()  # Work on a copy of the graph to avoid modifying the original graph

    # Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    # Remove nodes with no edges (isolates)
    G.remove_nodes_from(list(nx.isolates(G)))

    # Remove nodes with degree below the threshold
    low_degree_nodes = [node for node, degree in dict(G.degree()).items() if degree < degree_threshold]
    G.remove_nodes_from(low_degree_nodes)

    # Keep only the largest connected component
    if keep_largest_component:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    return G


def plot_degree_distribution(G):
    """
    Plots the degree distribution of the graph.
    
    Parameters:
    G (nx.Graph): The NetworkX graph.
    """
    degrees = [d for n, d in G.degree()]
    plt.figure(figsize=(10, 6))
    sns.histplot(degrees, bins=30, kde=False, edgecolor='black')
    plt.title('Degree Distribution')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.show()

def visualise_edge_weight_distribution(G):
    """
    Visualizes the distribution of edge weights.

    Parameters:
    edge_weights (list): List of edge weights.
    """
    plt.figure(figsize=(10, 6))
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    # Histogram
    sns.histplot(edge_weights, bins=30, kde=False)
    
    plt.title('Distribution of Edge Weights')
    plt.xlabel('Edge Weight')
    plt.ylabel('Frequency')
    plt.show()

def threshold_sparsification(graph, threshold):
    """
    Sparsifies the graph by removing edges below the specified weight threshold.

    Parameters:
    graph (nx.Graph): The original NetworkX graph.
    threshold (float): The weight threshold.

    Returns:
    nx.Graph: The sparsified graph.
    """
    graph_copy = graph.copy()
    sparsified_graph = nx.Graph()
    sparsified_graph.add_nodes_from(graph_copy.nodes(data=True))
    sparsified_graph.add_edges_from((u, v, d) for u, v, d in graph_copy.edges(data=True) if d.get('weight', 0) >= threshold)
    return sparsified_graph

def top_percentage_sparsification(graph, top_percentage):
    """
    Sparsifies the graph by keeping the top percentage of edges by weight.

    Parameters:
    graph (nx.Graph): The original NetworkX graph.
    top_percentage (float): The percentage of top-weight edges to keep.

    Returns:
    nx.Graph: The sparsified graph.
    """
    graph_copy = graph.copy()
    sorted_edges = sorted(graph_copy.edges(data=True), key=lambda x: x[2].get('weight', 0), reverse=True)
    top_edges_count = max(1, int(len(sorted_edges) * (top_percentage / 100)))
    sparsified_graph = nx.Graph()
    sparsified_graph.add_nodes_from(graph_copy.nodes(data=True))
    sparsified_graph.add_edges_from(sorted_edges[:top_edges_count])
    return sparsified_graph


def remove_by_degree(graph, min_degree):
    """
    Sparsifies the graph by removing nodes with degree below the specified threshold.

    Parameters:
    graph (nx.Graph): The original NetworkX graph.
    min_degree (int): The minimum degree threshold.

    Returns:
    nx.Graph: The sparsified graph.
    """
    graph_copy = graph.copy()
    nodes_to_remove = [node for node, degree in dict(graph_copy.degree()).items() if degree < min_degree]
    
    graph_copy.remove_nodes_from(nodes_to_remove)
    return graph_copy

def knn_sparsification(graph, k):
    """
    Sparsifies the graph by keeping only the top-k edges with the highest weights for each node.

    Parameters:
    graph (nx.Graph): The original NetworkX graph.
    k (int): The number of nearest neighbors to keep for each node.

    Returns:
    nx.Graph: The sparsified graph.
    """
    graph_copy = graph.copy()
    sparsified_graph = nx.Graph()
    sparsified_graph.add_nodes_from(graph_copy.nodes(data=True))
    
    for node in graph_copy.nodes():
        edges = sorted(graph_copy.edges(node, data=True), key=lambda x: x[2].get('weight', 0), reverse=True)
        sparsified_graph.add_edges_from(edges[:k])
    
    return sparsified_graph


def spanning_tree_sparsification(graph):
    """
    Sparsifies the graph by creating a minimum spanning tree.

    Parameters:
    graph (nx.Graph): The original NetworkX graph.

    Returns:
    nx.Graph: The sparsified graph.
    """
    graph_copy = graph.copy()
    return nx.minimum_spanning_tree(graph_copy, weight='weight')

def analyse_and_plot_density(graph):
    """
    Calculates and plots the density of the graph for a predefined series of thresholds.

    Parameters:
    graph (nx.Graph): The original NetworkX graph.

    Returns:
    densities (list of float): Densities of the graph at each threshold.
    """
    thresholds = [0.7 + i * 0.01 for i in range(31)]
    densities = []

    for threshold in thresholds:
        filtered_edges = [(u, v) for u, v, d in graph.edges(data=True) if d['weight'] > threshold]
        temp_graph = nx.Graph()
        temp_graph.add_edges_from(filtered_edges)
        densities.append(nx.density(temp_graph))

    # Plot the densities
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, densities, marker='o')
    plt.xlabel('Threshold')
    plt.ylabel('Density')
    plt.title('Density vs. Threshold')
    plt.grid(True)
    plt.show()

    return densities

def get_highest_degree_nodes(graph, top_n=10):
    """
    Returns the nodes with the highest degree in the graph.

    Parameters:
    graph (nx.Graph): The NetworkX graph.
    top_n (int): The number of top nodes to return.

    Returns:
    List of tuples: Each tuple contains a node and its degree.
    """
    degrees = dict(graph.degree())
    sorted_degrees = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    return sorted_degrees[:top_n]

def fetch_gene_info(gene_list):
    """
    Fetches gene information from MyGene.info.

    Parameters:
    gene_list (list): List of gene symbols or Ensembl IDs.

    Returns:
    list: List of dictionaries containing gene information.
    """
    mg = mygene.MyGeneInfo()
    gene_info = mg.querymany(gene_list, scopes='symbol,ensembl.gene', 
                             fields='name,symbol,entrezgene,summary,disease,pathway', 
                             species='human')
    return gene_info

def print_gene_info_with_degree(top_genes_with_degrees, gene_info):
    """
    Prints gene information including the degree.

    Parameters:
    top_genes_with_degrees (list): List of tuples containing gene symbols and their degrees.
    gene_info (list): List of dictionaries containing gene information.
    """
    for gene, degree in top_genes_with_degrees:
        info = next((item for item in gene_info if item['query'] == gene), None)
        if info:
            print(f"Gene Symbol: {info.get('symbol', 'N/A')}")
            print(f"Degree: {degree}")
            print(f"Gene Name: {info.get('name', 'N/A')}")
            print(f"Entrez ID: {info.get('entrezgene', 'N/A')}")
            print(f"Summary: {info.get('summary', 'N/A')}")
            if 'disease' in info:
                diseases = ', '.join([d['term'] for d in info['disease']])
                print(f"Diseases: {diseases}")
            else:
                print("Diseases: N/A")
            if 'pathway' in info:
                pathways = []
                if isinstance(info['pathway'], dict):
                    for key in info['pathway']:
                        pathway_data = info['pathway'][key]
                        if isinstance(pathway_data, list):
                            pathways.extend([p['name'] for p in pathway_data if 'name' in p])
                        elif isinstance(pathway_data, dict) and 'name' in pathway_data:
                            pathways.append(pathway_data['name'])
                        elif isinstance(pathway_data, str):
                            pathways.append(pathway_data)
                print(f"Pathways: {', '.join(pathways) if pathways else 'N/A'}")
            else:
                print("Pathways: N/A")
            print("-" * 40)
        else:
            print(f"Gene not found: {gene}")
            print(f"Degree: {degree}")
            print("-" * 40)

