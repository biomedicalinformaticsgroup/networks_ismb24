# Concept Glossary

## Absolute Biweight Midcorrelation

Statistical measure used to quantify the similarity between two sets of variables or features. Useful when dealing with noisy or heterogenous data.

## Adjacency Matrix

Matrix where each row and column corresponds to a node, and entries indicate whether there's a direct connection (edge) between those nodes (denoted by 1, otherwise 0).

## Augmentation

Process commonly done in the propagation phase of message passing, where node features are transformed or enhanced before being sent to their neighbors.

## Classifier

Algorithm that takes a set of input data and assigns each input to one of several predefined categories or classes. The goal of a classifier is to predict the class/category of an instance based on its features.

## Clustering

Machine Learning technique used to group similar data points together based on their features or attributes.

## CpG Sites

Specific regions in the DNA sequence (Cytosine followed by Guanine from 5′ to 3′ direction) that are often targets for DNA methylation.

## DNA Methylation

Biochemical process where methyl groups (-CH3) are added to the DNA molecule, specifically at cytosine bases within the sequence. It acts as a molecular switch for turning genes on or off.

## Edge

Represents the relationship, link, or connection between two nodes.

## Edge Prediction

Task in GNNs where the goal is to predict whether a missing edge should exist between two nodes, based on their features and the overall structure of the graph.

## Element-Wise Mean Pooling

Data processing technique where the mean value of each feature across a set of instances is calculated.

## Embedding Space

Vector space, typically of lower dimensionality, in which you map your data. 

## EWAS

Epigenome-Wide Association Study; research method used in genetics and epigenetics to identify associations between epigenetic modifications (such as DNA methylation patterns) and specific traits or diseases.

## Feature Representation

Process of transforming raw data into format that is suitable for machine learning model.

## Graph

Mathematical structure to model pairwise relations between objects. It is made up of nodes and edges that connect these nodes.

## Graph Classification

Task in GNNs where the goal is to predict the label/class of an entire graph based on its structure and the features of its nodes.

## Graph Neural Networks

Type of neural network designed to perform machine learning tasks on graph data.

## Log-Transformation

Mathematical operation applied to data to reduce skewness and compress a wide range of values into a smaller range.

## Message Passing

Iterative method of information exchange among nodes to update their own features. It consists of three steps: propagation, aggregation, and update.

## Neighborhood of a Node

Set of nodes directly connected that node by an edge.

## Network Sparsification

Process of reducing network complexity by selectively removing certain edges that are less informative or influential while preserving its essential structure and functionality.

## Node

Fundamental unit of the graph, representing various entities depending on the problem e.g. person, gene, protein, etc.

## Node Classification

Task in GNNs where the label/class of individual nodes in a graph is predicted based on their features as well as the structure of the graph.

## Non-Euclidean Graph

A data structure that does not adhere to traditional Euclidean principles of space. For example, in a social network graph, the "distance" between two people doesn't refer to a physical distance, but might instead refer to the number of connections between them.

## Normalisation

Data pre-processing step to transform data into a common scale (with a mean of zero and a standard deviation of one). It ensures that different features or variables have comparables ranges or distributions.

## One-Hot Encoding

Data pre-processing step to convert categorical data into a format that can be provided for machine learning algorithms to improve prediction. For each unique category in the dataset, it create a binary (0 or 1) feature representing whether the instance belongs to that category.

## Patient Similarity Network

Type of network where each node represents a patient, and edges between nodes represent a measure of similarity between the patients. This similarity could be based on various factors such as genetic data, clinical features, etc.

## Phenotype

Observable traits or characteristics of an organism, influences by its genetic makeup (genotype) and environmental factors. It encompasses physical features, behaviors, and physiological properties that can be seen or measured.

## Pearson Correlation

Statistical measure used to quantify the strength and direction of the linear relationship between two continuous variables.

## ReLU

Rectified Linear Unit (ReLU) is a popular activation function in deep learning models. It introduces non-linearity into the model to allow for the learning and representation of more complex data patterns. It returns 0 if the input is negative, and the input itself if the input is positive or 0.

## Semi-Supervised Learning

Machine learning approach where the model is trained using a combination of a small amount of labeled data and a large amount of unlabeled data.

## Similarity Network Fusion

Algorithm used to integrate multiple networks into a single one, by iteratively exhanging information between them until they converge towards a common representation.

## Spearman's Rank Correlation

Nonparametric measure of rank correlation. It assesses how well the relationship between two variables can be described using a monotonic function.

## Supervised Learning

Machine learning approach where the model is trained on labeled data and makes predictions on unlabeled/unseen data.

## Transcriptomics

The study of all RNA transcripts produced by the cells of an organism. It involves analyzing the types and quantities of RNA molecules present in a cell or tissue at a specific time or under certain conditions.
