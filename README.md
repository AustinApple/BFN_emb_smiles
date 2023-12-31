# Using bayesian flow networks to generate molecular SMILES embedding


## Training logP predictor
Training a fully connected network to predict logP from 256D embeddings of molecular smiles.
```
python train_predictor.py
```
## Training a BFN model 
Training a BFN generative model to learn the distribution of original embedding of smiles.
```
python train_emb_smiles.py
```
## Generating SMILES embedding
Using the trained BFN to generate embedding smiles which are following the original distribution.
```
python generate_common_emb_smiles.py
```
<p align="middle">
  <img src="figure/emb_smiles_generated.png" width="400" />
  <img src="figure/logP_generated.png" width="400" /> 
</p>

## Training a conditional BFN model
Training a conditonal BFN model by simply adding logP label into the timestep.
```
python train_emb_smiles_conditioned.py
```
## Generating SMILES embedding with desired logP
The conditioned BFN model generates embedding smiles with desired logP. 
```
python generate_common_emb_smiles_conditioned.py
```
<p align="middle">
  <img src="figure/emb_smiles_generated_conditioned.png" width="400" />
  <img src="figure/logP_generated_conditioned.png" width="400" /> 
</p>
