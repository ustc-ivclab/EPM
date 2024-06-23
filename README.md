<div align="center">

<h1>Efficient Partition Map Prediction  via Token Sparsification for Fast VVC Intra Coding</h1>

<div>
    <a href='https://zhexinliang.github.io/' target='_blank'>Xinmin Feng</a>&emsp;
    <a href='https://faculty.ustc.edu.cn/lil1/en/index.htm' target='_blank'>Li Li</a>&emsp;
    <a href='https://faculty.ustc.edu.cn/dongeliu/en/index.htm' target='_blank'>Dong Liu</a>&emsp;
    <a href='https://scholar.google.com/citations?user=5bInRDEAAAAJ&hl=en&oi=ao' target='_blank'>Feng Wu</a>
</div>
<div>
    Intelligent Visual Lab, University of Science and Technology of China &emsp; 
</div>

<div>
   <strong>MMSP2024, Under Peer Review</strong>
</div>
<div>
    <h4 align="center">
    </h4>
</div>

---

</div>

Although the partition map-based fast block partitioning algorithm for VVC intra-coding has achieved advanced encoding time savings and coding efficiency, it still faces high inference overhead. To deploy this algorithm efficiently, we first present a lightweight network based on the hierarchical vision transformer that efficiently predicts the partition map with less computational complexity compared to the previous approach, distributing the reduced inference complexity across each region. Then, we introduce token sparsification, where we select the most informative tokens using a pre-defined pruning ratio, achieving content-adaptive computation reduction and parallel-friendly inference. Experimental results demonstrate that the proposed method reduces the inference complexity significantly with a negligible BDBR increase compared to the original method. 

<div>
   <strong>Performance Evaluation</strong>
</div>

![Performance Evaluation](BDBR_ETS.png)



## :running_woman: Previous work

[Partition Map Prediction for Fast Block Partitioning in VVC Intra-frame Coding](https://github.com/AolinFeng/PMP-VVC-TIP2023)
