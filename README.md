# MonoDETR: Depth-guided Transformer for Monocular 3D Object Detection
This is an unofficial fork of Mono-DETR (['MonoDETR: Depth-guided Transformer for Monocular 3D Object Detection'](https://arxiv.org/pdf/2203.13310.pdf)), for the reproducibility challenge of Advanced 3D Vision course, 2023 spring, NYCU.

## Data preparation
Download [KITTI object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset, and layout them in the form of:
```
│<repository-root>/
├──...
├──<dataset-root>/
│   ├──ImageSets/
│   ├──training/
│   ├──testing/
├──...
```
then change the `root_dir` property in configs/\<your-config-file\> to the dataset root.

You will need the following files from the KITTI website:
 - left color images of object data set
 - camera calibration matrices of object data set
 - training labels of object data set

## Environment Setting
Follow the instruction [below](#installation).

## Our reproduce results
Our reproduce results are listed in the following table:

<table>
    <tr>
        <td div align="center">Models</td>
        <td div align="center">Easy</td> 
        <td div align="center">Mod.</td> 
        <td div align="center">Hard</td> 
        <td div align="center">Logs</td>
        <td div align="center">Ckpts</td>
    </tr>
    <tr>
        <td rowspan="4" div align="center">
            MonoDETR (reproduced)
        </td>
        <td colspan="3" div align="center">
            Val, AP<sub>3D</sub>
        </td>
        <td rowspan="4" div align="center">
            <a href="https://drive.google.com/file/d/1jo5MXCMR9DgE_YHuUptPCYcEM3gVimuL/view?usp=sharing">
                log
            </a>
        </td>
        <td rowspan="4" div align="center">
            <a href="https://drive.google.com/file/d/1SGCJ-a4EPkgkIELobpSHOgGiQGAJEkCu/view?usp=sharing">
                ckpt
            </a>
        </td>
    </tr>
    <tr>
        <td div align="center">18.63%</td> 
        <td div align="center">14.55%</td> 
        <td div align="center">12.58%</td> 
    </tr>
    <tr>
        <td colspan="3" div align="center">
            Val, AP<sub>3D|R40</sub>
        </td>
    </tr>
    <tr>
        <td div align="center">12.33%</td> 
        <td div align="center">8.26%</td> 
        <td div align="center">6.70%</td> 
    </tr>
</table>

We followed the official instruction and trained the model on a RTX 2080Ti with batch size = 4. We can see that there is a performance gap between the reported scores and ours. According to the authors, they intentionally tweaked some training code / hyperparameters to protect their work since they are still submitting this work to other conferences apparently.

We have validated the scores they reported using the checkpoint they provided, and it does match. Since the weight name in their checkpoint does not match with the public code, we wrote a script to convert the checkpoints runnable.

To evaluate the checkpoint, follow the instructions:
1. Run `python tools/rename_weight.py -i <input_checkpoint_name> -o <output_checkpoint_name>` **(Skip this step if you are running our reproduced checkpoint)**
2. Place the checkpoint under `runs/monodetr/<checkpoint_name>`
3. Edit L34 in `lib/helpers/tester_helper.py` to load your checkpoint
4. Run the test script following [this section](#test)

More experiment results and visualizations about results we reproduced can be found in our [Google Drive](https://drive.google.com/drive/folders/1KHBAmsGsbIwNr-ygHkTJD8dnhdviS5Uu?usp=sharing).

---

Below are information of the original repository

---

Official implementation of the paper ['MonoDETR: Depth-guided Transformer for Monocular 3D Object Detection'](https://arxiv.org/pdf/2203.13310.pdf).

**For our multi-view version, MonoDETR-MV on nuScenes dataset, please refer to [MonoDETR-MV](https://github.com/ZrrSkywalker/MonoDETR-MV).**

## Introduction
MonoDETR is the **first DETR-based model** for monocular 3D detection **without additional depth supervision, anchors or NMS**, which achieves leading performance on KITTI *val* and *test* set. We enable the vanilla transformer in DETR to be depth-aware and enforce the whole detection process guided by depth. In this way, each object estimates its 3D attributes adaptively from the depth-informative regions on the image, not limited by center-around features.
<div align="center">
  <img src="pipeline.jpg"/>
</div>

## Main Results
This repo contains only an intermediate version of MonoDETR. Our paper is still under review, but has been **intentionally plagiarized** by several times in character level, submitting to NeurIPS, CVPR, and other conferences.
Given this, we plan to release the complete code after our paper been accepted.
Thanks for your understanding.

The randomness of training for monocular detection would cause the variance of ±1 AP<sub>3D</sub>. For reproducibility, we provide four training logs of MonoDETR on KITTI *val* set for the car category: (the stable version is still under tuned)

**We have relased the ckpts of our implementation for reproducibility. The module names might have some mismatch, which will be rectified in a few days.**

<table>
    <tr>
        <td rowspan="2",div align="center">Models</td>
        <td colspan="3",div align="center">Val, AP<sub>3D|R40</sub></td>   
        <td rowspan="2",div align="center">Logs</td>
        <td rowspan="2",div align="center">Ckpts</td>
    </tr>
    <tr>
        <td div align="center">Easy</td> 
        <td div align="center">Mod.</td> 
        <td div align="center">Hard</td> 
    </tr>
    <tr>
        <td rowspan="4",div align="center">MonoDETR</td>
        <td div align="center">28.84%</td> 
        <td div align="center">20.61%</td> 
        <td div align="center">16.38%</td> 
        <td div align="center"><a href="https://drive.google.com/file/d/124u2WW_DqDyKrpUe3lQ8TR6xth8rn9YH/view?usp=sharing">log</a></td>
        <td div align="center"><a href="https://drive.google.com/drive/folders/1eIQtH3RzJqOCHm9hwgmjmQAnG5qJNFCN?usp=sharing">ckpt</a></td>
    </tr>  
    <tr>
        <td div align="center">26.66%</td> 
        <td div align="center">20.14%</td> 
        <td div align="center">16.88%</td> 
        <td div align="center"><a href="https://drive.google.com/file/d/1gSof60oOnno_qAHRViXKQ6CyqRI7O0tr/view?usp=sharing">log</a></td>
        <td div align="center"><a href="https://drive.google.com/drive/folders/1eIQtH3RzJqOCHm9hwgmjmQAnG5qJNFCN?usp=sharing">ckpt</a></td>
    </tr> 
    <tr>
        <td div align="center">29.53%</td> 
        <td div align="center">20.13%</td> 
        <td div align="center">16.57%</td> 
        <td div align="center"><a href="https://drive.google.com/file/d/1rrayzzwHGpddE1f_mfvq0RQb5xpWcPAL/view?usp=sharing">log</a></td>
        <td div align="center"><a href="https://drive.google.com/drive/folders/1eIQtH3RzJqOCHm9hwgmjmQAnG5qJNFCN?usp=sharing">ckpt</a></td>
    </tr> 
    <tr>
        <td div align="center">27.11%</td> 
        <td div align="center">20.08%</td> 
        <td div align="center">16.18%</td> 
        <td div align="center"><a href="https://drive.google.com/file/d/1D6IOkscfypGSEbsXcHZ60-q492zvMLp7/view?usp=sharing">log</a></td>
        <td div align="center"><a href="https://drive.google.com/drive/folders/1eIQtH3RzJqOCHm9hwgmjmQAnG5qJNFCN?usp=sharing">ckpt</a></td>
    </tr> 
</table>

MonoDETR on *test* set from official [KITTI benckmark](http://www.cvlibs.net/datasets/kitti/eval_object_detail.php?&result=22a0e176d4f7794e7c142c93f4f8891749aa738f) for the car category:
<table>
    <tr>
        <td rowspan="2",div align="center">Models</td>
        <td colspan="3",div align="center">Test, AP<sub>3D|R40</sub></td>   
    </tr>
    <tr>
        <td div align="center">Easy</td> 
        <td div align="center">Mod.</td> 
        <td div align="center">Hard</td> 
    </tr>
    <tr>
        <td rowspan="2",div align="center">MonoDETR</td>
        <td div align="center">24.52%</td> 
        <td div align="center">16.26%</td> 
        <td div align="center">13.93%</td> 
    </tr>  
    <tr>
        <td div align="center">25.00%</td> 
        <td div align="center">16.47%</td> 
        <td div align="center">13.58%</td> 
    </tr>  
    
</table>


## Installation
1. Clone this project and create a conda environment:
    ```
    git clone https://github.com/ZrrSkywalker/MonoDETR.git
    cd MonoDETR

    conda create -n monodetr python=3.8
    conda activate monodetr
    ```
    
2. Install pytorch and torchvision matching your CUDA version:
    ```
    conda install pytorch torchvision cudatoolkit
    ```
    
3. Install requirements and compile the deformable attention:
    ```
    pip install -r requirements.txt

    cd lib/models/monodetr/ops/
    bash make.sh
    
    cd ../../../..
    ```
    
4. Make dictionary for saving training losses:
    ```
    mkdir logs
    ```
 
5. (Skip) Download [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) datasets and prepare the directory structure as:
    ```
    │MonoDETR/
    ├──...
    ├──data/KITTIDataset/
    │   ├──ImageSets/
    │   ├──training/
    │   ├──testing/
    ├──...
    ```
    You can also change the data path at "dataset/root_dir" in `configs/monodetr.yaml`.
    
## Get Started

### Train
You can modify the settings of models and training in `configs/monodetr.yaml` and appoint the GPU in `train.sh`:

    bash train.sh configs/monodetr.yaml > logs/monodetr.log
   
### Test
The best checkpoint will be evaluated as default. You can change it at "tester/checkpoint" in `configs/monodetr.yaml`:

    bash test.sh configs/monodetr.yaml


## Acknowlegment
This repo benefits from the excellent [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR) and [MonoDLE](https://github.com/xinzhuma/monodle).

## Citation
```bash
@article{zhang2022monodetr,
  title={MonoDETR: Depth-aware Transformer for Monocular 3D Object Detection},
  author={Zhang, Renrui and Qiu, Han and Wang, Tai and Xu, Xuanzhuo and Guo, Ziyu and Qiao, Yu and Gao, Peng and Li, Hongsheng},
  journal={arXiv preprint arXiv:2203.13310},
  year={2022}
}
```

## Contact
If you have any question about this project, please feel free to contact zhangrenrui@pjlab.org.cn.
