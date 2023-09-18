# Hardware implementation of Dysta scheduler

## Evaluation Methodology

We compare the resource consumption of Dysta scheduler against the classical sparse CNN accelerator Eyeriss, to evaluate the relative area/resource overhead brought by our Dysta scheduler.

### Resource Usage of Eyeriss-V2

We get resource estimation based on the open-source [code](https://github.com/karthisugumar/CSE240D-Hierarchical_Mesh_NoC-Eyeriss_v2). We found some clusters in their top noc design are optimized away during synthesis, so there are some bugs in the way they connect clusters. To get accurate resource consumption, instead of using the top noc module, we just use the design of a single cluster to get synthesis results.  Then we get the total resource consumption, we multiply the resource usage of a single cluster by the total number of clusters used in our simulator.

The generated Vivado synthesis report of one-cluster Eyeriss can be downloaded [here](https://drive.google.com/drive/folders/1OcTIqF1nYl-7CEH0_VI6dFULcXZe2PWm?usp=sharing).

### Resource Consumption

To evaluate the resource consumption of the scheduler, we use a dummy NPU that just receives/outputs data. The dummy NPU has the same input/output interface as the Eyeriss-V2.
The resource consumption is controlled by three design parameters: 
- 1. REQST_DEPTH, means there are maximally 2^REQST_DEPTH number of requests. We define this for scalability evaluation; 
- 2. SCORE_BITWIDTH, the precision used to calculate the score, which will affect the DSP usage; 
- 3. COMPUTE_OPT, where to fuse operations into one unified compute unit to save compute resources.

The generated Vivado synthesis report of the scheduler with different design parameters can be downloaded [here](https://drive.google.com/drive/folders/1OcTIqF1nYl-7CEH0_VI6dFULcXZe2PWm?usp=sharing).

Pls note that, as we didn't find any high-quality open-source hardware code for **Dynamic Sparse** CNN/AttNN accelerators with main functionality evaluated, we put a dummy NPU in the project to interact with our hardware scheduler. When you adopt our scheduler in your own accelerator, more interaction design and verification are needed.
