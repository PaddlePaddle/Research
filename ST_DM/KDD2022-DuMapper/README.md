
## DuMapper: Towards Automatic Verification of Large-Scale Points of Interest with Street Views at Baidu Maps

### How to run

- Deep multimodal embedding (DME) of a POI: This neural module takes the signboard image and the coordinates of a POI as input to generate its multimodal vector representation. 

- Approximate nearest neighbor (ANN) search through the large-scale POI database: This module takes advantage of ANN algorithms to conduct a more accurate search through billions of archived POI embeddings in the database for verification within milliseconds.

- train.py: train and build DME model.

- infer.py: generate multimodal embedding of a POI for ANN search.


