# CDRNet (ICCV 2023 Workshops)
This is a repository for "Cross-Dimensional Refined Learning for Real-Time 3D Visual Perception from Monocular Video".
### [Project Page](https://hafred.github.io/cdrnet/) | [Paper](https://openaccess.thecvf.com/content/ICCV2023W/JRDB/papers/Hong_Cross-Dimensional_Refined_Learning_for_Real-Time_3D_Visual_Perception_from_Monocular_ICCVW_2023_paper.pdf) | [Poster](https://github.com/stanfordironman/cdrnet.torch.2023ICCV/blob/main/iccv23_poster_cdrnet_final.pdf)
![CDRNet Real-Time Demo](assets/cdrnet_github.gif)

## Run Inference
```bash
python valscene_inference.py
```

## Configuration
The key modes can be configured under `configs/inference.yaml`, where disabling `MODEL.DEPTH_PREDICTION` release the model into the geometric-semantic inference mode. The geometric-semantic information has been learned by MAP optimization with the help of 2D priors.  

## Citation
Please consider citing our paper and give a ⭐ if you find this repository useful.
```
@inproceedings{hong2023cross,
    author    = {Hong, Ziyang and Yue, C. Patrick},
    title     = {Cross-Dimensional Refined Learning for Real-Time 3D Visual Perception from Monocular Video},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2023},
    pages     = {2169-2178}
}
```
