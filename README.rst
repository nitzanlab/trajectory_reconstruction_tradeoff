

Trajectory Reconstruction Tradeoff
==================================

.. image:: https://github.com/nitzanlab/trajectory_reconstruction_tradeoff/raw/main/.images/fig1.png
   :width: 500px
   :align: center

Charting cellular trajectories over gene expression is key to understanding dynamic cellular processes and their underlying mechanisms. 
While advances in single-cell RNA-sequencing technologies and  computational methods have pushed forward the recovery of such trajectories, trajectory inference remains a challenge due to the noisy, sparse, and high-dimensional nature of single-cell data. 
This challenge can be alleviated by increasing either the number of cells sampled along the trajectory (breadth) or the sequencing depth, i.e. the number of reads captured per cell (depth). 
Generally, these two factors are coupled due to an inherent breadth-depth tradeoff that arises when the sequencing budget is constrained due to financial or technical limitations. 
Here we study the optimal allocation of a fixed sequencing budget to optimize the recovery of trajectory attributes. 

Data
----
In the manuscript, we focus on five single-cell RNA-sequencing datasets encompassing differentiation of embryonic stem cells, pancreatic $\beta$ cells, hepatoblast and multipotent haematopoietic cells, as well as induced reprogramming of embryonic fibroblasts into neurons. 
All were downloaded from \href{https://zenodo.org/record/1443566\#.YEExrpMzbDI}{https://zenodo.org/record/1443566\#.YEExrpMzbDI} in .rds format.
We load and save separately the component of each dataset <dataset> as 'counts_<dataset>.csv', '<dataset>_cell_info.csv', and '<dataset>_milestone_network.csv' with R script 'scripts/rds_to_csv.R'.
See example datasets of mESC ('hayashi') and fibroblasts in 'datasets' folder.

Reconstruction error for subsampling experiments
------------------------------------------------
We first construe subsampling experiments with just cell(breadth) subsample, read (depth) subsample or with both being sampled under constant budgets. 
For example:
``python run.py hayashi --sample cells  --desc example``
``python run.py hayashi --sample reads  --desc example``
``python run.py hayashi --sample tradeoff  --desc example``

.. image:: https://github.com/nitzanlab/trajectory_reconstruction_tradeoff/raw/main/.images/hayashi_pc.png
   :width: 150px
   :align: left

.. image:: https://github.com/nitzanlab/trajectory_reconstruction_tradeoff/raw/main/.images/hayashi_pt.png
   :width: 150px
   :align: center

.. image:: https://github.com/nitzanlab/trajectory_reconstruction_tradeoff/raw/main/.images/hayashi_tradeoff.png
   :width: 150px
   :align: right

Analyzing results
-----------------
Empirical results reveal that trajectory reconstruction accuracy scales with the logarithm of either the breadth or depth of sequencing. 
We additionally observe a power law relationship between the optimal number of sampled cells and the corresponding sequencing budget.
See 'notebooks/fig23.ipynb' for example.

Expression pattern analysis along the trajectory
------------------------------------------------
We further demonstrate the impact of the breadth-depth tradeoff on downstream analysis of expression patterns along linear trajectories.
To compute the quality of expression pattern under subsampling, we use the following command:

``python run.py hayashi --sample exp --desc example``

We then plot the results in 'notebooks/fig4.ipynb'.



.. Reference
.. ---------
.. Noa Moriel, Edvin Memmet, and Mor Nitzan (2022), Optimal sequencing budget allocation for trajectory reconstruction,
.. `bioRxiv <bioarxiv link>`_.

Support
-------
Please get in touch `email <mailto:noa.moriel@mail.huji.ac.il>`_.