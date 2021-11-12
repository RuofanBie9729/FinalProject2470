# FinalProject2470
### Introduction
n this project, we intend to perform image segmentation with prostate Magnetic Resonance Imaging(MRI) data.Prostate cancer is the second most frequent cancer diagnosis made in men and the fifth leading causeof death worldwide.[1] A few techniques are used for early detection of prostate cancer, includingblood test, biopsy and imaging tests.  The Magnetic Resonance Imaging (MRI) scans create detailedimages of soft tissues in the body using radio waves and strong magnets. MRI scans can give doctorsa very clear picture of the prostate and nearby areas. [2] MRI of prostate cancer usually consists of twonon-overlapping  adjacent  regions:   peripheralzone (PZ) and the transition zone (TZ). An ex-ample  of  prostate  MRI  with  labeled  zones  is shown  in  Fig. 1.   Identifying  prostate  zones  isimportant  for  diagnostic  and  therapies.   How-ever, the identification work requires substantialexpertise in reading MRI scans. Therefore, auto-matic segmentation of prostate zones is instru-mental for prostate lesion detection.The  problem  of  prostate  zone  segmentationis  challenging  because  of  the  lack  of  a  clearprostate boundary, prostate tissue heterogeneity,and the wide inter-individual variety of prostateshapes.[3] In this project, we will be implementing some existing CNN and RNN models for imagesegmentation using prostate MRI data.   We will use a survey for image segmentation using deeplearning [4] as a guide, implement selected models and compare their performance.

### Related Work
#### Prior Work
There are some work done for segmenting prostate MRI images using deep learning.[5] focus on the work done using fully convolutional neural networks (FCNN). They suggested eightdifferent  FCNNs-based  deep  2D  network  structures  for  automatic  MRI  prostate  segmentation  byanalysing various structures of shortcut connections together with the size of a deep network us-ing the PROMISE12 dataset [6].[7] mentions that 3D neural networks have strong potential for prostate MRI segmentation.  How-ever, substantial computational resources are required due to the large number of parameters. Theyproposed a network architecture calledV-net Light (VnL), which is based on an efficient 3D Modulecalled3D Light,  that minimises the number of network parameters while maintaining state-of-artsegmentation results.  The proposed architecture replaces regular 3D convolutions of the V-net ar-chitecture [8] with novel3D Light modules.  Fig. 2 shows the architecture ofVnL. The V-net modelconsists of encoder and decoder paths with convolutional layers.  To reduce the number of paramers, pooling layers are inserted between the encoder stages.  The novel3D-Light Moduleis used ineach stage of the encoder and decoder. The3D-Light Moduleis a parameter-efficient 3D convolutionalblock consists of parallel convolution blocks, blocks composed of regular convolutions, followed by agroup convolution. It reduces the number of parameters by88%−92%in comparison to V-net. TheVnLachieves comparable results to V-Net on the PROMISE12 dataset [6] while requiring90%lesslearning parameters,90%less hard-disk storage and just3.3%of the FLOPs.[9] proposed a transfer learning method based on deep neural network for prostate MRI segmenta-tion.  They also designed a multi-level edge attention module using wavelet decomposition to over-come the difficulty of ambiguous boundary in the task.

#### Public Implementation
Some of the model architectures we would liketo implement have publicly available implemen-tation.

### Data
We use a set of prostate MRI data from The Medical Segmentation Decathlon – a biomedical image analysis challenge.  The Decathlon challenge made ten data sets available online.  All data setshave been released with a permissive copyright-license (CC-BY-SA 4.0), thus allowing for data shar-ing, redistribution, and commercial usage. [10]
According to [10], all images were de-identified and reformatted to the Neuroimaging InformaticsTechnology Initiative (NIfTI) format https://nifti.nimh.nih.gov. All images were transposed (withoutresampling)  to  the  most  approximate  right-anterior-superior  coordinate  frame,  ensuring  the  datamatrixx−y−zdirection was consistent.  Lastly, non-quantitative modalities (e.g., MRI) were robustmin-max scaled to the same range.  For each segmentation task, a pixel-level label annotation wasprovided.The Decathlon challenge provides users with training set (images and labels) and test set (imageswithout labels).  In order to evaluate the performance with true labels, we only use the training setprovided and randomly select a third of the data to be our own test set.The prostate data set was acquired at Radboud University Medical Center, Nijmegen Medical Centre,Nijmegen, The Netherlands.  It consists of 48 prostate multiparametric MRI (mpMRI) studies, 32 ofthem have corresponding region-of-interest (ROI) targets (background= 0,T Z= 1andP Z= 2).Each study contains approximately 15 to 20 slices of MRI images, resulting in 602 images in total.We will use 10 studies as test set and the remaining 22 studies as training set.  Fig.3 shows the 20slices from one study.

### Methodology
#### Fully Convolutional Networks (FCNs)
As the name suggests, the fully convolutional networks only uses convolutional layers in the archi-tecture. As figure 4 suggested, the FCN is constructed by very deep convolutional layers with a finalpixelwise prediction layer. [11] proposed using FCNs for image segmentation. By combining a deepcoarse layer for appearance information with a shallow fine layer for fine tuning, the architeture pro-posed can implement significant improvement in segmentation accuracy.  Based on the architectureproposed in [11], we consider using three convolutional layer in our FCN model and add a deconvo-lution layer in the final step to implement pixelwise prediction. To mimic the coarse layer combinedwith finer layer architecture in [11], we set the first convolutional layer with 15 filters with stride 4and kernel size3. The second convolutional layer is set to have 8 filters with stride 3 and kernel size3×3.  The first two layers mimics the deep coarse layer, which grasps the appearance information.The third layer is set to have 4 filters with stride 1 and kernel size 2.  We hope the third layer canmimic the fine-tuning process as described in [11].

#### Encoder-Decoder Based Models
A basic encoder-decoder model to implement image segmentation is to use convolutional layers asencoders and then use deconvolutional or convolution-transpose layers for decoders.  As the archi-tecture of the U-Net ([12]) presented in figure 5 shows, we intend to implement a simplified U-Netmodel as the encoder-decoder model.  For the encoder, we consider using three convolutional lay-ers.   All the three convolutional layers would have 10 filters with kernel size 3 and stride size 2.Then the three convolutional layers are followed by three convolution-transpose layers.  All of theconvolution-transpose layers would have 10 filters with kernel size 3 and stride size 2.

#### Dilated Convolutional Models
Due to the translation invariant property of convolutional layer, the FCN model is reliable in pre-diction the presence and roughly the position of objects in an image.  However, as a trade-off be-tween classification accuracy and localization accuracy, the FCN model might not be able to sketchthe  exact  the  outline  of  the  object.   [13]  proposed  to  add  a  fully  connected  CRF  layer  after  theconvolutional layers as presented in figure 6.   In this project,  we intend to add a fully connectedCRF layer to the FCN model described above.  By comparing the performance of FCN model withthis dilated convolutional model, we intend to explore how much improvement the fully connected CRF  can  bring  to  the  FCN  model.   Since  CRF  is  not  a  regular  tensorflow  layer,  we  intend  to  re-fer  to  code  from 
http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/18/image-segmentation-with-tensorflow-using-cnns-and-conditional-random-fields/ to implement thismodelFigure 6:Architecture example of Deep Convolutional with CRF Networks

#### RNN Based Models

The RNN models are usually used in dealing with sequential data.  [14] proposed a Graph-LSTMmodel, which can apply LSTM layer in image segmentation problem.  The LSTM structure usuallyshare forget gates with neighboring nodes to remember information from neighboring nodes.  How-ever, in images, non-neighboring pixel within the same segment might be informative than neighbor-ing pixels.  Consequently, [14] proposed to pass deep convolutional layer results to a1×1convolu-tional layer to generate a confidence map and then use the confidence map results as initial states forthe LSTM layers. In this project, we intend to pass the FCN output, as described in the FCN model toa1×1convolution layer, and use the results of the confidence map as initial state for 2 LSTM layers, which use the output of FCN layers as input.

Since we only have about 600 images, we firstly use data augmentation to add rotated or flippedimaged in the original training set to increase training sample size and also increase the robustness ofour model.  All the four models are trained with 3 epochs through the whole training set with batchsize 100. The trained models are then applied to test set and compute the accuracy metrics describesas below.

### Metrics
The model performance for image segmentation is measured differently from for classification.  Wewill evaluate the model using a few new metrics. [4]

#### Pixel Accuracy
Pixel accuracy (PA) measures the proportion of correctly classified pixels. For K+ 1 classes, the pixelaccuracy is defined as

\begin{equation}
    PA = \frac{\sum_{i=0}^Kp_{ii}}{\sum_{i=0}^K\sum_{j=0}^Kp_{ij}},
\end{equation}

, where $p_{ij}$ is the number of pixels of classipredicted as belonging to classj.

#### Mean Pixel Accuracy
Mean pixel accuracy (MPA) extends PA to the proportion of correctly predicted pixels in a per-class manner, and then average over the total number of classes:
\begin{equation}
    MPA = \frac{1}{K+1}\sum_{i=0}^K\frac{p_{ii}}{\sum_{j=0}^Kp_{ij}}.
\end{equation}

#### Intersection over Union
Pixel accuracy has the limitation such that it has bias in the presence of very imbalanced classes, while mean pixel accuracy is not suitable for data with strong background class. Another segmentation evaluation metric is the intersection over union (IoU). It is defined as the area of intersection between the predicted segmentation map A and the ground truth map B, divided by the area of the union between the two maps:
\begin{equation}
    IoU=\frac{|A\cap B|}{|A\cup B|}.
\end{equation}
The mean intersection over union (Mean-IoU) is defined as the average IoU over all classes.

In this project, we would expect an accuracy of 50% for all models using the intersection over union metric as a baseline. Our goal is to implement 70-75% of accuracy for the four models. If these accuracy are easily achieved, we would consider to adjust model to achieve around 90% accuracy.

### Ethics
Magnetic resonance imaging (MRI) is a medical imaging technique that uses a magnetic field and computer-generated radio waves to create detailed images of the organs and tissues in patients' body. It's also an important tool for doctors to detect any abnormalities of the tissue or organ. Developing an image segmentation neural network that can reach high accuracy of detecting prostate can help to relief doctors' burden in manually checking the MRI and can increase efficiency in the medical system. However, developing such neural network doesn't necessarily mean physicians never have to give a look to MRI. The neural network results would justify the physician's diagnosis to secure the diagnosis process.

### Division of labor
Ruofan is responsible for running the FCN and dilated convolutional models. Ruya is responsible for running the encoder-decoder model and the RNN based model.

### References

[1]  Prashanth Rawla.   Epidemiology of prostate cancer.World journal of oncology,  10(2):63,2019. pages 3

[2]  American Cancer Society. Cancer Statistics Center.  Tests to diagnose and stage prostate cancer.URLhttp://cancerstatisticscenter.cancer.org[Accessed8thNovember2021]. pages 3

[3]  Nader Aldoj, Federico Biavati, Florian Michallek, Sebastian Stober, and Marc Dewey. Automaticprostate and prostate zones segmentation of magnetic resonance images using densenet-likeu-net.Scientific reports, 10(1):1–17, 2020. pages 3

[4]  Shervin Minaee, Yuri Y Boykov, Fatih Porikli, Antonio J Plaza, Nasser Kehtarnavaz, and DemetriTerzopoulos. Image segmentation using deep learning: A survey.IEEE Transactions on PatternAnalysis and Machine Intelligence, 2021. pages 3, 8

[5]  Tahereh Hassanzadeh, Leonard GC Hamey, and Kevin Ho-Shon. Convolutional neural networksfor  prostate  magnetic  resonance  image  segmentation.IEEE  Access,  7:36748–36760,  2019.pages 3

[6]  Geert Litjens, Robert Toth, Wendy van de Ven, Caroline Hoeks, Sjoerd Kerkstra, Bram van Gin-neken, Graham Vincent, Gwenael Guillard, Neil Birbeck, Jindang Zhang, Robin Strand, FilipMalmberg,  Yangming  Ou,  Christos  Davatzikos,  Matthias  Kirschner,  Florian  Jung,  Jing  Yuan,Wu  Qiu,  Qinquan  Gao,  PhilipˆaEddieˆa  Edwards,  Bianca  Maan,  Ferdinand  van  der  Heijden,Soumya  Ghose,  Jhimli  Mitra,  Jason  Dowling,  Dean  Barratt,  Henkjan  Huisman,  and  AnantMadabhushi.   Evaluation of prostate segmentation algorithms for mri:  The promise12 chal-lenge.Medical Image Analysis, 18(2):359–373, 2014. ISSN 1361-8415. doi: https://doi.org/10.1016/j.media.2013.12.002. URLhttps://www.sciencedirect.com/science/article/pii/S1361841513001734. pages 3, 4

[7]  Ophir Yaniv, Orith Portnoy, Amit Talmon, Nahum Kiryati, Eli Konen, and Arnaldo Mayer.  V-netlight-parameter-efficient 3-d convolutional neural network for prostate mri segmentation.  In2020 IEEE 17th International Symposium on Biomedical Imaging (ISBI), pages 442–445.IEEE, 2020. pages 3

[8]  Fausto Milletari, Nassir Navab, and Seyed-Ahmad Ahmadi.  V-net:  Fully convolutional neuralnetworks for volumetric medical image segmentation.  In2016 fourth international confer-ence on 3D vision (3DV), pages 565–571. IEEE, 2016. pages 3

[9]  Xiangxiang Qin.  Transfer learning with edge attention for prostate mri segmentation.arXivpreprint arXiv:1912.09847, 2019. pages 4

[10]  Michela Antonelli, Annika Reinke, Spyridon Bakas, Keyvan Farahani, Bennett A Landman, GeertLitjens, Bjoern Menze, Olaf Ronneberger, Ronald M Summers, Bram van Ginneken, et al.  Themedical segmentation decathlon.arXiv preprint arXiv:2106.05735, 2021. pages 4

[11]  Jonathan Long, Evan Shelhamer, and Trevor Darrell.  Fully convolutional networks for seman-tic segmentation.  InProceedings of the IEEE conference on computer vision and patternrecognition, pages 3431–3440, 2015. pages 5

[12]  Olaf  Ronneberger,  Philipp  Fischer,  and  Thomas  Brox.    U-net:   Convolutional  networks  forbiomedical image segmentation.  InInternational Conference on Medical image computingand computer-assisted intervention, pages 234–241. Springer, 2015. pages 6

[13]  Liang-Chieh Chen,  George Papandreou,  Iasonas Kokkinos,  Kevin Murphy,  and Alan L Yuille.Semantic image segmentation with deep convolutional nets and fully connected crfs.arXivpreprint arXiv:1412.7062, 2014. pages 7

[14]  Xiaodan  Liang,  Xiaohui  Shen,  Jiashi  Feng,  Liang  Lin,  and  Shuicheng  Yan.   Semantic  objectparsing  with  graph  lstm.   InEuropean  Conference  on  Computer  Vision,  pages  125–143.Springer, 2016. pages 7
