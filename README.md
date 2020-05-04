Please cite:
```
  @inproceedings{vikram2019approach,
  title={An Approach for Multimodal Medical Image Retrieval using Latent Dirichlet Allocation},
  author={Vikram, Mandikal and Anantharaman, Aditya and BS, Suhas},
  booktitle={Proceedings of the ACM India Joint International Conference on Data Science and Management of Data},
  pages={44--51},
  year={2019}
  }
```
  
  [Link to paper](https://vikram-mm.github.io/cods_comad_camera_ready.pdf)

# Multimodal-Image-Retrieval
In this work, a multi-modal medical image retrieval approach that incorporates both visual and textual features for improved image retrieval performance is presented. In the discussed model, SIFT features are used for capturing the important visual features of the medical images and Latent Dirichlet Allocation (LDA) is used to effectively represent the topics of the clustered SIFT features. To derive the composite feature set, two different fusion techniques were experimented with - early and late fusion. In early fusion, features obtained from an autoencoder and a modified VGG-16 model were used. The late fusion approach was implemented as an ensemble of both visual and textual features, aided by a SVM based classification for improving retrieval performance. Experiments showed that the drop in performance when the textual features are incorporated indicates that the co-occurrence matrix was not a effective way of fusing the textual and visual features in this case. Further attempts to decrease sparsity using autoencoder and using VGG features did not improve the performance. Separating out the textual and visual components using the late fusion approach gave better results. The performance with visual-features-only model was improved by re-ranking the result list using an independently trained text classifier. This outperformed the early fusion approaches proposed in this work as well as those described in other contemporary works.


## File descriptions
| File | Description |
|------|--------------|
|step1_image.py| Used for generating the visual bag of words|
|tf_kmeans.py| Tensorflow K-Means used in step1_image.py|
|lda_image.py| Computes the latent visual topics|
|step_1_text.py| Textual feature extraction|
|step_2_text.py| SVM on textual features|
|co_occurence.py | The get_vistex()is used to compute the co-occurence matrix described in the early fusion approach|
|autoencoder.py | The autoencoder used for reducing sparsity in the early fusion approach|
|vgg_net.py | The VGG network used for reducing sparsity in the early fusion approach|
|metrics.py| The various metrics which are used for evaluation
|evaluate_just_visual.py | Used to compute the performance by using just the visual features|
|evaluate_vistex.py	| Used to compute the performance by using just the co-occurence matrix|
|evaluate_autoencoder.py	| Used to compute the performance by using just the co-occurence matrix compressed using autoencoder|
|evaluate_vgg.py	| Used to compute the performance by using just the co-occurence matrix compressed using VGG|
|evaluate_late_fusion.py	| The late fusion approach and its evaluation|
|other util code | gen_path_list.py, irma_reader.py|
