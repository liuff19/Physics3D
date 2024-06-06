# Physics3D
Official implementation of Physics3D: Learning Physical Properties of 3D Gaussians via Video Diffusion

[Fangfu Liu](https://liuff19.github.io/)*, Hanyang Wang*, [Shunyu Yao](https://scholar.google.com/citations?user=i4kyLbwAAAAJ), Shengjun Zhang, [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ), [Yueqi Duan](https://duanyueqi.github.io/)

## [Paper]() | [Project page](https://liuff19.github.io/Physics3D/) | [Data](https://1drv.ms/f/s!At4g_orSPJVNiFqjfZdl2itnNmyb?e=IZjM2w)

<p align="center">
    <img src="assets/teaser.png">
</p>

Physics3D is a unified simulation-rendering pipeline based on 3D Gaussians, which learn physics dynamics from video diffusion model.

## More features 

The repo is still being under construction, thanks for your patience. 
- [x] Training code release.
- [x] Synthetic data release.
- [ ] Detailed tutorial.
- [ ] Detailed local demo.


## Preparation for training

### Linux System Setup.
```angular2html
conda create -n Physics3D python=3.9
conda activate Physics3D

pip install -r requirements.txt
git clone https://github.com/graphdeco-inria/gaussian-splatting
pip install -e gaussian-splatting/submodules/diff-gaussian-rasterization/
pip install -e gaussian-splatting/submodules/simple-knn/
```

### Quick Start.

1. Download the Gaussian models from [OneDrive](https://1drv.ms/f/s!At4g_orSPJVNiFqjfZdl2itnNmyb?e=IZjM2w). You can also load your own 3D Gaussian pre-trained models to this pipeline following [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting). For the setting details of physical configs, you can refer to [PhysGaussian](https://github.com/XPandora/PhysGaussian).
    ```
    Physics3D
        ├──model
            ├── ball/
        ├──config
            ├── ball_config.json
    ```

2. We support using text-to-video ([ModelScope](https://huggingface.co/ali-vilab/text-to-video-ms-1.7b)) diffusion models to guide the optimization of physical parameters. You can use the following command:
    ```bash
    python simulation.py --model_path ./model/ball/ --prompt "a basketball falling down" --output_path ./output --physics_config ./config/ball_config.json
    ```

## Tips to get better results

1. Parameter initialization that aligns with physical facts can significantly accelerate the convergence of Physics3D and improve training effectiveness.

2. For some high-frequency elastic objects, simulation effectiveness can be enhanced by increasing particle density.

## Acknowledgement

We have intensively borrowed code from the following repositories. Many thanks to the authors for sharing their code.
- [DreamPhysics](https://github.com/tyhuang0428/DreamPhysics)
- [threestudio](https://github.com/threestudio-project/threestudio) and its extension for [Animate124](https://github.com/HeliosZhao/Animate124/tree/threestudio).
- [warp-mpm](https://github.com/zeshunzong/warp-mpm)
- [PhysGaussian](https://github.com/XPandora/PhysGaussian)

We have also used open-source datasets from the following repositories.
- [PhysDreamer](https://github.com/a1600012888/PhysDreamer)
- [BlenderKit](https://github.com/BlenderKit/BlenderKit) for free models and [BlenderNeRF](https://github.com/maximeraafat/BlenderNeRF) for synthetic NeRF datasets within Blender


## Citation

If you found Physics3D helpful, please cite our report:
```bibtex
```

## Contact
If you have any question about this project, please feel free to contact liuff23@mails.tsinghua.edu.cn or hanyang-21@mails.tsinghua.edu.cn.
