

### Dataset preparation

> **Using juliet dataset for C/C++** it is compilable, it has safe and unsafe code constructions, it is synthetic (so prob easier to learn/understand).

> **Targetting BF Input/Output Bugs Model** [see list of CWEs here](https://usnistgov.github.io/BF/info/bf-classes/_inp/model/)

1.  Build the datasets with patched/vulnerable versions of code.
2.  Double check vulnerabilities using the esbmc module (assuming they compile). 
    Exclude: (1) patch version is buggy, (2) patch/vuln has syntax errors



### Model stuff
> Using llama 3.1 8B (we can run locally; the quantizied version is (in theory) very equivalent to the original one; there are pretrained SAEs)
1. Extract embeddings. 
    1.1 Explore quantized vs original version (if there is no big different, use quantized)
2. Finetuned a pretrained autoencoder to learn a general-purpose latent representation of the embeddings 
3. Contrastive learning to align the latent space to distinguish between vulnerable/safe samples.
    3.1 Create paired samples (vulnerable/safe)
        We want to teach the model the changes that make code vulnerable or safe.
    3.2 Add contrastive loss term to autoencoder training to encourage positive pairs to have similar representations; negative pairs must differ
    3.3 Train autoencoder w reconstruction + contrastive losses

4. Post hoc analysis to interpret what the autoencoder's latent space has learned. 
    4.1 Plot these latent representations. 
    4.2 Correlation between each latent feature and program properties? Heatmaps



**Tasks:**
1. Get approved by meta. -- using  llama 3.1 8B 
2. Find the best layer in llama 3.1 8B
    2.1 Get dataset -- claudia defects4j
    2.2 Test dataset pairwise -- rui
    2.3. Get layer number
3. If available SAE for corresponding layer number then 🥳 else 😭
    3.1 If 😭: can we train it ?
    3.2 If 🥳: go to 4
