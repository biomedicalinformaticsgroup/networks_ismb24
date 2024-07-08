# Section 1 - Network Basics

## Introduction to Network Basics

Networks are powerful tools that allow us to model and understand complex biological systems by representing entities such as genes, proteins, or patients as nodes, and their interactions or relationships as edges.

## Graphs

A graph $G = (V, E)$, is a tuple of a node set $V$ and an edge set $E$. Nodes encapsulate entities of interest, while edges capture the relationships or interactions between these entities.
Graphs can be: 
- directed,
- undirected, or 
- weighted. 

The adjacency matrix, symbolized as $A$, represents the connections. For an undirected, unweighted graph with $n$ nodes, $A$ is an $n \times n$ matrix where $A_{ij}$ is 1 if nodes $i$ and $j$ are connected, and 0 otherwise. For a directed graph, $A_{ij} = 1$ if and only if there is a directed edge from $i$ to $j$. For weighted graphs, the value of $A_{ij} \in \mathbb{R}^+$  indicates the strength of the connections. 

## TCGA-LUAD Project

![fishy](./tcga.png)

- **Title**: The Cancer Genome Atlas Lung Adenocarcinoma (TCGA-LUAD)
- **Main Focus**: Study of lung adenocarcinoma (a common type of lung cancer)
- **Data Collected**: Comprehensive genomic, epigenomic, transcriptomic, and proteomic data from lung adenocarcinoma samples
- **Disease Types**:
  - Acinar Cell Neoplasms
  - Adenomas and Adenocarcinomas
  - Cystic, Mucinous, and Serous Neoplasms
- **Number of Cases**: 585 (498 with transcriptomic data)
- **Data Accessibility**: Available on the NIH-GDC Data Portal

- **Link**: [TCGA-LUAD Project Page](https://portal.gdc.cancer.gov/projects/TCGA-LUAD)

## Part 1: Gene Expression Network

### Tabular data to graph data
Our main goal is to learn how to convert tabular data into network data in a way that is robust and reproducible. We will discuss different ways how to define the edges from the tabular data. We will also look at common techniques how to clean the resulting network all the while we are learning functionality of a popular Python package for working with network data, called [NetworkX](https://networkx.org/documentation/stable/reference/index.html).

By the end of this session you will be comfortable to inspect, analyse and create you own network data for tasks such as protein-protein interaction (PPI), gene regulatory network (GRN) and patient-similarity network (PSN) analysis.

### Correlation Matrices
We will calculate correlation matrices from gene expression data, which will serve as the basis for constructing gene expression networks. The correlation matrix captures the relationships between genes by calculating the correlation coefficient between their expression profiles.

There are a few correlation metrics one could consider:
- [Pearson](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)  
  - O(n^2) complexity, fast for large datasets
- [Spearman](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)
  -  O(n^2 log n) complexity, relatively fast but can be slower than Pearson
- [Absolute biweight midcorrelation](https://en.wikipedia.org/wiki/Biweight_midcorrelation)
  - Robust but slower than Pearson and Spearman, suitable for datasets with outliers


### Graph Construction
From the correlation matrix, we will construct a gene expression network where the nodes are genes and the edges represent the strength of the correlation between them. The network will allow us to identify highly connected genes, which are likely to be functionally related.

![graph_features](./vertex_types.png)

[(source)](https://medium.com/@athrav.kale20/graph-theoretic-algorithms-81fc5291460e)

### Network Cleaning
Networks generated from real-world data often contain noisy or irrelevant connections, which can impact downstream analysis. We will discuss various strategies for cleaning networks, such as removing low-confidence edges, filtering nodes based on their connectivity, and identifying and removing outlier nodes.

### Sparsification Methods
Sparsification is a technique used to reduce the density of a network by removing edges while preserving the overall network structure. We will explore different sparsification methods, such as thresholding based on edge weights or network density, to simplify the network and improve interpretability.

![network sparsification](./sparsification.png)

[(source)](https://dl.acm.org/doi/10.1145/1989323.1989399)

### Describing Highly Connected Nodes
Highly connected nodes, also known as hubs, play a crucial role in network structure and function. We will discuss how to identify and characterise hubs in a network, including their biological relevance.

## Part 2: Patient Similarity Network
In the second half of this exercise we are going to generate two patient similarity networks. One of them is based on the gene expression data we used in Part 1. The other one is based on DNA methylation as a preliminary step for Section 2.



<!-- Embed an HTML file using an iframe
<iframe src="../../ismb_data/genecoexp_plotly_network.html" style="width:100%; height:600px; border:none;"></iframe> -->



<!-- 1. **Biomedical Networks**
   - 1.1. Libraries
   - 1.2. TCGA-LUAD project
   - 1.3. Metadata Analysis
   - 1.4. Expression Data Analysis

2. **Data Filtering and Preprocessing**
   - 2.1. Thresholding Based on Gene Expression Levels
   - 2.2. Gene Retention at Various Thresholds
   - 2.3. Thresholding Based on Gene Expression Variance
   - 2.4. Gene Retention at Different Variance Levels

3. **Gene Symbol Conversion**
   - 3.1. Checking for Missing Data and Outliers
   - 3.2. Addressing Data Heterogeneity

4. **Correlation Matrix Calculation**

5. **Graph Construction from Correlation Matrices**

6. **Network Cleaning**

7. **Sparsification Methods for Networks**

8. **Describing Highly Connected Nodes**

9. **Storing and Managing Generated Networks**

10. **Patient Similarity Network Construction**

11. **DNA Methylation Network Analysis** -->




<!-- # EXAMPLE md 

Section 3 - SNF + GNN's

## Introduction to Graph Neural Networks 
Graph Neural Networks (GNN) are a powerful architecture for the learning of graph structure and information in a supervised setting. In this workshop we will implement a Graph Convolutional Network (GCN) model from the [Deep Graph Library](https://www.dgl.ai/) in Python. 

The goal of GNN's is to learn an embedding space for nodes which captures both node feature representation and graph structure. Intuitively, if two nodes are connected and belong to the same class they should be close together in the embedding space. Conversely, if two nodes are connected but do not belong to the same class we want them to be separated in the embedding space. Thus, we cannot rely on graph structure alone and necessitates the requirement to include node feature representation also. The method in which we capture this similarity is through the message passing algorithm discussed below. 

```{image} ./GNN_Learning.png
:alt: fishy
:width: 600px
:align: center
```

The differentiation between GCN and neural network architectures is their ability to learn from the local neighbourhood as opposed to handcrafted network features. The performance of GCN and other GNN architectures has been demonstrated on a variety of benchmark tasks, hence extending their application to a biomedical setting is an exciting avenue.  -->
