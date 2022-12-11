# PeerJ-Computer-Science

Task Offloading Decision Algorithm
for Vehicular Edge Network Based
on Multi-dimensional Information Deep
Learning


Deep reinforcement learning based offloading decision algorithm for vehicular edge computing
https://pubmed.ncbi.nlm.nih.gov/36262145/


[6] Deep reinforcement learning based offloading decision algorithm for vehicular edge computing
作者：Hu X, Huang Y.
出处：PeerJ Computer Science, 2022, 8: e1126.
摘要：任务卸载决策是车载边缘计算的核心技术之一。高效的卸载决策不仅可以满足复杂的车辆任务在时间、能耗和计算性能方面的要求，还可以减少网络资源的竞争和消耗。传统的分布式任务卸载决策是由车辆根据本地状态做出的，无法最大化移动边缘计算（MEC）服务器的资源利用率。此外，为了简化，很少考虑车辆的移动性。本文提出了一种基于深度强化学习的任务卸载决策算法，即面向车辆边缘计算（VEC）的基于深度强化学习的卸载决策（DROD）。在这项工作中，在我们最小化系统开销的最优问题中考虑了车辆的移动性和VEC环境中常见的信号阻塞。为了解决最优问题，DROD采用马尔可夫决策过程对车辆与MEC服务器之间的交互进行建模，并使用改进的深度确定性策略梯度算法NLDDPG迭代训练模型以获得最优决策。NLDDPG将归一化状态空间作为输入，将LSTM结构引入到actor-critic网络中，以提高学习效率。最后，进行了两个系列的实验来探索DROD。首先，讨论了核心超参数对DROD性能的影响，并确定了最优值。其次，将DROD与其他一些基线算法进行比较，结果表明DROD比DQN好25%，比NLDQN好10%，比DDDPG好130%。
链接：
https://peerj.com/articles/cs-1126/
代码：
https://github.com/YYYYYYHuang/PeerJ-Computer-Science.git
