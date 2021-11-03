# GNNs_rec_public

Graph Neural Networks in Spektral for Energy Reconstruction 

The code was built in Google Colab. Install library dependencies accordingly. 
To see versions of each library installed in Google Colab, see the 'dependencies' file. Tensorflow and spektral versions can be checked there.

The data used in the code can be found in the following links:

Link to Google Colab notebook:
https://colab.research.google.com/drive/1rMO5ggPIG-fWrfRPndWJx3JXuCEVFmes?usp=sharing

Saved GNN model:
https://drive.google.com/drive/folders/1ugTVAPtCrUjyRu4xwt2dDG8MUdi-VXN6?usp=sharing

PMT coordinates:
https://drive.google.com/file/d/1TI1frpA8AKWPDADj7BpaqvKphK6fMi6C/view?usp=sharing

In spektral, we need to create graph datasets, containing a number of graphs. We have an output Energy corresponding to each graph.
To create a graph dataset, we need to subclass the Dataset class defined in spektral. To understand how to create spektral datasets, use the following:
https://graphneural.network/creating-dataset/

There are very few examples of graph regression in spektral. I referred to the following 2 examples to build the custom training loops and custom datasets:
https://github.com/danielegrattarola/spektral/blob/master/examples/graph_prediction/custom_dataset.py
https://github.com/danielegrattarola/spektral/blob/master/examples/graph_prediction/qm9_ecc.py

These subclasses are named as:
MyDataset (training dataset)
MyTestDataset (test dataset)
MyMonoEnergeticDataset1
MyMonoEnergeticDataset5
MyMonoEnergeticDataset8
MyMonoEnergeticDataset10
MyMonoEnergeticDataset20
MyMonoEnergeticDataset30
MyMonoEnergeticDataset40


We then pass the number of graphs to generate to these classes to create a dataset object. For example: 
data = MyDataset(max_no_of_graphs_training)

Making predictions in the current version of the code is inefficient but well defined as follows:
First we save the model weights after training the model.
The predictions are made by passing the test data to the training loop, 
The predicted value is extracted as the output in the current iteration of the training loop. Then it is made sure that the model forget the newly trained weights at the end of each iteration, so as to use only the training model weights for prediction of further iterations. 

Also the test input graphs are passed using the DisjointLoader in batches. So in each iteration, the predictions are saved in a chunks with size = batch size. These chunks are then combined to create a continuous array of predictions. 

This can be seen in the 'Making predictions' section of the python code. The rest of the code is self explanatory once basics of spektral are studied along with an understanding of the two spektral examples mentioned above.

Reach out to: 'yashrajsmotwani@gmail.com' for further queries
