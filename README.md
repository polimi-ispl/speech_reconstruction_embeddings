# Reconstructing speech from CNN embeddings
In this work we study one aspect of this problem by reconstructing speech from the intermediate embeddings computed by a CNN.
Specifically, we consider a pre-trained network that acts as a feature extractor from speech audio.
We investigate the possibility of inverting these features, reconstructing the input  signals in a black-box scenario, and quantitatively measure the reconstruction quality by measuring the word-error-rate of an off-the-shelf ASR model. 
