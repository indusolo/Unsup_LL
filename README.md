# Unsup_LL 
Unsupervised Lifelong Learning
#### U-TELL: UNSUPERVISED TASK EXPERT LIFELONG LEARNING - 
- U-TELL model is evaluated in an unsupervised setting for Class-IL and Domain-IL frameworks for six prominent continual learning datasets namely split MNIST (SMNIST), split CIFAR (SCIFAR), split SVHN (SSVHN), split WAFER (SWAFER) defect, rotated MNIST (RMNIST), and permuted MNIST (PMNIST). 
- The Class-IL setting for SMNIST, SCIFAR, SSVHN, SWAFER datasets is with five tasks $(t_1, t_2, t_3, t_4, t_5)$, where each task has two mutually exclusive classes {{0,1}, {2,3}, {4,5}, {6,7}, {8,9}}.
- SMNIST, SCIFAR10, and SSVHN with a training set of 10000 samples per task and a test set of 1000 samples per task and SWAFER dataset with training and testing sets of 1700 and 200 samples per task .
- CIFAR100 datastream has a sequence of 10 tasks ($t_1,\ldots,t_{10}$) with a total of 20 coarse classes and 2 classes per each task with train data of 5000 training images per task and 200 non-overlapping test images per task.
- The subset of TinyImageNet data used in our experiments has 5 tasks ($t_1,\ldots,t_5$) with 2 classes per task with train data of 1000 samples per task and 100 non-overlapping test samples per task.
- The Domain-IL setting with RMNIST and PMNIST datasets is with four task sequences $(t_1, t_2, t_3, t_4$) with a training set of 15000 samples per task and a test set of 1250 samples per task. 
- PMNIST dataset is generated by giving a distribution drift through random permutations to the original MNIST dataset.
- RMNIST dataset is created by random rotations of ($\{[0-30],[31-60], [61-90], [91-120]\}$) degrees to the original MNIST dataset.

#### Processor and GPU setting
* Intel® Xeon® CPU E5-2630 v4 2.20 GHz
* Nvidia Quadro RTX 8000 GPU with memory of 48 GB having 4608 cores

#### Libraries and versions
* python 3.9.12
* sklearn== 1.2.2
* numpy== 1.24.3
* torch== 2.0.1+cu117
* torchvision== 0.15.2+cu117

* Datastream generation and image rotation codes are taken from publicly available codes at https://tinyurl.com/AutonomousDCN

#### Links for baselines' publicly avaiilable code used in our experiments
* SCALE: Online Self-Supervised Lifelong Learning without Prior Knowledge -  https://github.com/Orienfish/SCALE
* CaSSLe: Self-Supervised Models are Continual Learners - CaSSLe experiments were conducted with unknown task boundary settings with the CaSSLe code available at https://github.com/Orienfish/SCALE 
* UPL-STAM: Unsupervised Progressive Learning and the STAM Architecture - https://github.com/CameronTaylorFL/stam
* KIERA: Unsupervised Continual Learning via Self-Adaptive Deep Clustering Approach - https://tinyurl.com/AutonomousDCN

#### Domain-IL case RMNIST images generated by task structure signatures by SDG block

![rmnist_img1](https://github.com/indusolo/Unsup_LL/assets/99342079/40866201-e8dd-410d-ace8-579f25040468 width="100" height="100")

#### 

