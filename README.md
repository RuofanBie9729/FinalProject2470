# FinalProject2470
### Introduction
In this project, we intend to perform image segmentation with prostate Magnetic Resonance Imaging (MRI) data. 

Prostate cancer is the second most frequent cancer diagnosis made in men and the fifth leading cause of death worldwide. [1] A few techniques are used for early detection of prostate cancer, including blood tests, biopsy and imaging tests. The Magnetic Resonance Imaging (MRI) scans create detailed images of soft tissues in the body using radio waves and strong magnets. MRI scans can give doctors a very clear picture of the prostate and nearby areas. [2] 

MRI of prostate cancer usually consists of two non-overlapping adjacent regions: the peripheral zone (PZ) and the transition zone (TZ). An example of prostate MRI with labelled zones is shown in Figure 1. Identifying prostate zones is important for diagnostic and therapies. However, the identification work requires substantial expertise in reading MRI scans. Therefore, automatic segmentation of prostate zones is instrumental for prostate lesion detection.

![](https://challengepost-s3-challengepost.netdna-ssl.com/photos/production/software_photos/001/734/680/datas/small.jpg "Figure 1: T2-weighted MRI at the maximal axial plane in 48-year-old with history acute prostatitis 3 years previously and currently with Mondor’s disease.") 

The problem of prostate zone segmentation is challenging because of the lack of a clear prostate boundary, prostate tissue heterogeneity, and the wide inter-individual variety of prostate shapes. [3] In this project, we will be implementing some existing CNN and RNN models for image segmentation using prostate MRI data. We will use a survey for image segmentation using deep learning [4] as a guide, implement selected models and compare their performance.

### Related Work
#### Prior Work
There are some works done for segmenting prostate MRI images using deep learning.

[5] focus on the work done using fully convolutional neural networks (FCNN). They suggested eight different FCNNs-based deep 2D network structures for automatic MRI prostate segmentation by analysing various structures of shortcut connections together with the size of a deep network using the PROMISE12 dataset [6]. 

[7] mentions that 3D neural networks have strong potential for prostate MRI segmentation. However, substantial computational resources are required due to the large number of trainable parameters. They proposed a network architecture called V-net Light (VnL), which is based on an efficient 3D Module called 3D Light,  that minimises the number of network parameters while maintaining state-of-art segmentation results. The proposed architecture replaces regular 3D convolutions of the V-net architecture [8] with novel 3D Light modules.  Figure 2 shows the architecture of VnL. The original V-net model consists of encoder and decoder paths with convolutional layers. To reduce the number of parameters, [7] inserts pooling layers between the encoder stages. The novel 3D-Light Module is used in each stage of the encoder and decoder. The 3D-Light Module is a parameter-efficient 3D convolutional block consisting of parallel convolution blocks, blocks composed of regular convolutions, followed by a group convolution. It reduces the number of parameters by 88%−92% in comparison to V-net. The VnL achieves comparable results to V-Net on the PROMISE12 dataset [6] while requiring 90% fewer learning parameters, 90% less hard-disk storage and just 3.3% of the FLOPs. 

![](https://challengepost-s3-challengepost.netdna-ssl.com/photos/production/software_photos/001/734/678/datas/small.jpg "Figure 2: Macro structure network of VnL.")

[9] proposed a transfer learning method based on deep neural networks for prostate MRI segmentation. They also designed a multi-level edge attention module using wavelet decomposition to overcome the difficulty of ambiguous boundaries in the task.

#### Public Implementation
Some of the model architectures we would like to implement have publicly available implementation.
* DeconvNet: https://github.com/HyeonwooNoh/DeconvNet/blob/master/model/DeconvNet/DeconvNet_inference_deploy.prototxt (Caffe)
* U-Net: https://github.com/milesial/Pytorch-UNet/tree/master/unet (Pytorch).

### Data
We use a set of prostate MRI data from The Medical Segmentation Decathlon -- a biomedical image analysis challenge. The Decathlon challenge made ten data sets available online. All data sets have been released with a permissive copyright license (CC-BY-SA 4.0), thus allowing for data sharing, redistribution, and commercial usage. [10]

According to [10], all images were de-identified and reformatted to the Neuroimaging Informatics Technology Initiative (NIfTI) format (https://nifti.nimh.nih.gov). All images were transposed (without resampling) to the most approximate right-anterior-superior coordinate frame, ensuring the data matrix x−y−z direction was consistent. Lastly, non-quantitative modalities (e.g., MRI) were robust min-max scaled to the same range. For each segmentation task, a pixel-level label annotation was provided. 

The Decathlon challenge provides users with training sets (images and labels) and test sets (images without labels). To evaluate the performance with true labels, we only use the training set provided and randomly select a third of the data to be our own test set.

The prostate data set was acquired at Radboud University Medical Center, Nijmegen Medical Centre, Nijmegen, The Netherlands. It consists of 48 prostate multiparametric MRI (mpMRI) studies, 32 of them have corresponding region-of-interest (ROI) targets (background= 0, TZ= 1 and PZ= 2). Each study contains approximately 15 to 20 slices of MRI images, resulting in 602 images in total. We will use 10 studies as the test set and the remaining 22 studies as the training set. Figure 3 shows the 20 slices from one study.

![](https://challengepost-s3-challengepost.netdna-ssl.com/photos/production/software_photos/001/734/681/datas/small.png "Figure 3: MRI slices (first and thrid rows) from one study with labels (second and fourth rows).")

### Methodology
#### Fully Convolutional Networks (FCNs)
As the name suggests, the fully convolutional networks only use convolutional layers in the architecture. As Figure 4 suggested, the FCN is constructed by very deep convolutional layers with a final pixel-wise prediction layer. [11] proposed using FCNs for image segmentation. By combining a deep coarse layer for appearance information with a shallow fine layer for fine-tuning, the architecture proposed can implement a significant improvement in segmentation accuracy.  Based on the architecture proposed in [11], we consider using three convolutional layers in our FCN model and add a deconvolution layer in the final step to implement pixel-wise prediction. To mimic the coarse layer combined with finer layer architecture in [11], we set the first convolutional layer with 15 filters with stride 4 and kernel size 3x3. The second convolutional layer is set to have 8 filters with stride 3 and kernel size 3x3.  The first two layers mimic the deep coarse layer, which grasps the appearance information. The third layer is set to have 4 filters with stride 1 and kernel size 2. We hope the third layer can mimic the fine-tuning process as described in [11]. 

![](https://challengepost-s3-challengepost.netdna-ssl.com/photos/production/software_photos/001/734/684/datas/small.png "Figure 4: Architecture example of Fully Convolutional Networks.")

#### Encoder-Decoder Based Models
Most of the popular DL-based segmentation models use some kind of encoder-decoder architecture. A basic encoder-decoder model to implement image segmentation is to use convolutional layers as encoders and then use deconvolutional or convolution-transpose layers for decoders. We will implement two of them: DeConvNet for general image segmentation and U-Net for medical image segmentation.

##### DeConvNet
The DeConvNet [13] is designed on top of the convolutional layers adopted from the VGG 16-layer net [12].  As shown in Figure 5, DeConvNet is composed of convolution and deconvolution networks, where the convolution network acts as the feature extractor and the deconvolution network is a shape generator. The proposed architecture aims to overcome two limitations of FCNs. First, using FCN models, label prediction is done with only local information for large objects. Also, FCNs often ignore small objects and classify them as background. Second, in FCN, the input to the deconvolutional layer is too coarse and the deconvolution procedure is overly simple.

![](https://challengepost-s3-challengepost.netdna-ssl.com/photos/production/software_photos/001/757/698/datas/small.png "Figure 5: Architecture example of DeConvNet.")

There are no unpooling layers defined in TensorFlow. We will make use of the implementation from https://github.com/aizawan/segnet/blob/master/ops.py.

##### U-Net
The U-Net model [14] is built upon [11]. It is designed specifically for biomedical data, where there is very little training data available. Different from [11],  a large number of feature channels is used in the upsampling part. The modification allows the network to propagate context information to higher resolution layers. The network does not have any fully connected layers and only uses the valid part of each convolution. We will implement the U-Net model shown in Figure 6 with a three-channel output corresponding to the three segmentation areas.

![](https://challengepost-s3-challengepost.netdna-ssl.com/photos/production/software_photos/001/734/679/datas/small.png "Figure 6: Architecture example of U-Net.")

#### Dilated Convolutional Models
Due to the translation-invariant property of the convolutional layer, the FCN model is reliable in predicting the presence and roughly the position of objects in an image. However, as a trade-off between classification accuracy and localization accuracy, the FCN model might not be able to sketch the exact outline of the object.  [15]  proposed to add a fully connected CRF layer after the convolutional layers as presented in Figure 7. In this project, we intend to add a fully connected CRF layer to the FCN model described above. By comparing the performance of the FCN model with this dilated convolutional model, we intend to explore how much improvement the fully connected CRF can bring to the FCN model. Since CRF is not a regular TensorFlow layer, we intend to refer to code from http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/18/image-segmentation-with-tensorflow-using-cnns-and-conditional-random-fields/  to implement this model.

![](https://challengepost-s3-challengepost.netdna-ssl.com/photos/production/software_photos/001/734/682/datas/small.png "Figure 7: Architecture example of Deep Convolutional with CRF Networks.")

Since we only have about 600 images, we firstly use data augmentation to add rotated or flipped images in the original training set to increase the training sample size and also increase the robustness of our model.  All four models are trained with 3 epochs through the whole training set with batch size 100. The trained models are then applied to the test set and compute the accuracy metrics describes below.

### Metrics
The model performance for image segmentation is measured differently from for classification. We will evaluate the model using a few new metrics. [4]

#### Pixel Accuracy
Pixel accuracy (PA) measures the proportion of correctly classified pixels. For K+1 classes, the pixel accuracy is defined as

PA = \frac{\sum_{i=0}^Kp_{ii}}{\sum_{i=0}^K\sum_{j=0}^Kp_{ij}},

where ![](https://latex.codecogs.com/gif.download?p_%7Bij%7D) is the number of pixels of class predicted as belonging to class j.

#### Mean Pixel Accuracy
Mean pixel accuracy (MPA) extends PA to the proportion of correctly predicted pixels in a per-class manner, and then average over the total number of classes:

MPA = \frac{1}{K+1}\sum_{i=0}^K\frac{p_{ii}}{\sum_{j=0}^Kp_{ij}}.

#### Intersection over Union
Pixel accuracy has limitations such that it has a bias in the presence of very imbalanced classes, while mean pixel accuracy is not suitable for data with a strong background class. Another segmentation evaluation metric is the intersection over union (IoU). It is defined as the area of intersection between the predicted segmentation map A and the ground truth map B, divided by the area of the union between the two maps:

IoU=\frac{|A\cap B|}{|A\cup B|}.

The mean intersection over union (Mean-IoU) is defined as the average IoU over all classes.

In this project, we would expect an accuracy of 50% for all models using the intersection over union metric as a baseline. Our goal is to implement 70-75% of accuracy for the four models. If these accuracies are easily achieved, we would consider adjusting the model to achieve around 90% accuracy.

### Results
| Model | Pixel Accuracy (PA) | Mean Pixel Accyracy (MPA) | Intersection over Union (IoU) |
| --- | --- | --- | --- |
| FCN | | | |
| DeConvNet | | | |
|U-Net | | | |

### Ethics
Magnetic resonance imaging (MRI) is a medical imaging technique that uses a magnetic field and computer-generated radio waves to create detailed images of the organs and tissues in a patients' body. It's also an important tool for doctors to detect any abnormalities of the tissue or organ. Developing an image segmentation neural network that can reach high accuracy of detecting prostate can help to relieve doctors' burden in manually checking the MRI and can increase efficiency in the medical system. However, developing such neural networks doesn't necessarily mean physicians never have to look at MRI. The neural network results would justify the physician's diagnosis to secure the diagnosis process.

### Division of labour
Ruofan is responsible for running the FCN and dilated convolutional models. Ruya is responsible for running the encoder-decoder models.

### References

[1] Prashanth Rawla. Epidemiology of prostate cancer. World journal of oncology, 10(2):63, 2019. pages 3

[2] American Cancer Society. Cancer Statistics Center. Tests to diagnose and stage prostate cancer. URL http://cancerstatisticscenter.cancer.org [Accessed8thNovember2021]. pages 3

[3] Nader Aldoj, Federico Biavati, Florian Michallek, Sebastian Stober, and Marc Dewey. Automatic prostate and prostate zones segmentation of magnetic resonance images using densenet-like u-net. Scientific reports, 10(1):1–17, 2020. pages 3

[4] Shervin Minaee, Yuri Y Boykov, Fatih Porikli, Antonio J Plaza, Nasser Kehtarnavaz, and Demetri Terzopoulos. Image segmentation using deep learning: A survey. IEEE Transactions on pattern analysis and Machine Intelligence, 2021. pages 3, 8

[5] Tahereh Hassanzadeh, Leonard GC Hamey, and Kevin Ho-Shon. Convolutional neural networks for prostate magnetic resonance image segmentation. IEEE  Access, 7:36748–36760, 2019. pages 3

[6] Geert Litjens, Robert Toth, Wendy van de Ven, Caroline Hoeks, Sjoerd Kerkstra, Bram van Gin-neken, Graham Vincent, Gwenael Guillard, Neil Birbeck, Jindang Zhang, Robin Strand, FilipMalmberg, Yangming Ou, Christos Davatzikos, Matthias Kirschner, Florian Jung, Jing Yuan, Wu Qiu, Qinquan Gao, Philip aEddiea Edwards, Bianca Maan, Ferdinand van der Heijden, Soumya Ghose, Jhimli Mitra, Jason Dowling, Dean Barratt, Henkjan Huisman, and Anant Madabhushi. Evaluation of prostate segmentation algorithms for mri: The promise12 challenge. Medical Image Analysis, 18(2):359–373, 2014. ISSN 1361-8415. doi: https://doi.org/10.1016/j.media.2013.12.002. URL https://www.sciencedirect.com/science/article/pii/S1361841513001734. pages 3, 4

[7] Ophir Yaniv, Orith Portnoy, Amit Talmon, Nahum Kiryati, Eli Konen, and Arnaldo Mayer. V-netlight-parameter-efficient 3-d convolutional neural network for prostate mri segmentation. In 2020 IEEE 17th International Symposium on Biomedical Imaging (ISBI), pages 442–445. IEEE, 2020. pages 3

[8] Fausto Milletari, Nassir Navab, and Seyed-Ahmad Ahmadi. V-net: Fully convolutional neural networks for volumetric medical image segmentation. In 2016 fourth international conference on 3D vision (3DV), pages 565–571. IEEE, 2016. pages 3

[9] Xiangxiang Qin. Transfer learning with edge attention for prostate mri segmentation. arXivpreprint arXiv:1912.09847, 2019. pages 4

[10] Michela Antonelli, Annika Reinke, Spyridon Bakas, Keyvan Farahani, Bennett A Landman, GeertLitjens, Bjoern Menze, Olaf Ronneberger, Ronald M Summers, Bram van Ginneken, et al. The medical segmentation decathlon. arXiv preprint arXiv:2106.05735, 2021. pages 4

[11] Jonathan Long, Evan Shelhamer, and Trevor Darrell. Fully convolutional networks for semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3431–3440, 2015. pages 5

[12] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014. pages 6

[13] Hyeonwoo Noh, Seunghoon Hong, and Bohyung Han. Learning deconvolution network for semantic segmentation. In Proceedings of the IEEE international conference on computer vision, pages 1520–1528, 2015. pages 6

[14] Olaf  Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234–241. Springer, 2015. pages 6

[15] Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, and Alan L Yuille. Semantic image segmentation with deep convolutional nets and fully connected crfs. arXivpreprint arXiv:1412.7062, 2014. pages 7

