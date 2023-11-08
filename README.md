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

![emb_smiles_generated](image/emb_smiles_generated.png) ![logP_generated](image/logP_generated.png)

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
![emb_smiles_generated_conditioned](image/emb_smiles_generated_conditioned.png) ![logP_generated_conditioned](image/logP_generated_conditioned.png)