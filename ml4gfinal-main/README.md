# ml4gfinal

```python3 train.py``` will do training for the two gene diff input models.

The available model classes are ConvTransNet (CNN-Transformer), BetterConvNet (6-layer CNN), SimpleConvNet (Simple CNN) and TransformerNoConv (model with Transformer-encoder layers only). Currently the model architectures are configured to use their best performing versions.

To interchange between models, change line 189 in ```train.py``` to use the model of your choice. Be sure to change the ```TRIAL_NAME``` variable on line 19 to use your desired name so that models get saved to the correct folder. 

```python3 train_single.py``` will do training for SingleNet. Be sure to change the ```TRIAL_NAME``` variable on line 18 to use your desired name so that the model gets saved to the correct folder. 

```python3 graph.py``` will display our results that were obtained for the paper. 

The folder ProcessedData contains all of the data we used, which was originally from https://github.com/QData/DeepDiffChrome/tree/master/data/ProcessedData. X.expr.csv files contain the gene expression values for cell X, while the X.train.csv and X.valid.csv files contain cell X's histone modification counts for the training and validation gene sequences, respectively.  
