# DASC7600 - Data Science Project: Medical Diagnosis Explanation by Intervening Self-Learning Bottleneck Visual Concepts

## Introduction
Our research focuses on the application of conceptual explanation in the process of deep learning of medical images. We proposed to treat the global concepts obtained through self-supervised learning as proxies for expert knowledge, used in instance explanation analysis and intervention tests. Once a certain level of accuracy is achieved, we believe that the black-box model decision logic provided by concept-based explainable techniques can serve as a proxy for expert interpretations of model predication or discrimination pattern. Therefore, important concepts identified during the learning process can serve as expert interpretive lenses rich in visual information to explain the preferences and logic of bottleneck layer decision-making, just like the score learned from professional information and obtained in the bottleneck model.

## Visualization of the Intervention Process

![image](https://github.com/BarcelonaTong/IBotCL/blob/main/1.png)

## Dataset

We use the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) from Kaggle for our project.

## Usage

The following commands provide a guide on how to use our project for various tasks: 

```shell
# Pre-training of backbone:
python main_contrast.py --num_classes 4 --base_model resnet50 --lr 0.0001 --epoch 30 --lr_drop 2 --pre_train True --dataset COVID-19 --dataset_dir /input0

# Training for BotCL:
python main_contrast.py --num_classes 4 --num_cpt 6 --base_model resnet50 --lr 0.0001 --epoch 30 --lr_drop 2 --dataset COVID-19 --dataset_dir /input0 --weak_supervision_bias 0.1 --quantity_bias 0.1 --distinctiveness_bias 0.05 --consistence_bias 0.01

# First run process.py to extarct the activation for all dataset samples:
python process.py --num_classes 4 --num_cpt 6 --base_model resnet50 --dataset COVID-19 --dataset_dir /input0 --process True --batch_size 64

# Then see the generated concepts by:
python vis_contrast.py --num_classes 4 --num_cpt 6 --base_model resnet50 --top_sample 100 --dataset COVID-19 --dataset_dir /input0

python adjust_score_as_ep.py --num_classes 4 --num_cpt 6 --base_model resnet50 --index 666 --dataset COVID-19 --dataset_dir /input0

python batch_adjust_score_as_ep.py --num_classes 4 --num_cpt 6 --base_model resnet50 --dataset COVID-19 --dataset_dir /input0

python vis_val_masks.py --num_classes 4 --num_cpt 6 --base_model resnet50 --dataset COVID-19 --dataset_dir /input0

python batch_adjust_saliency_map_as_ep.py --num_classes 4 --num_cpt 6 --base_model resnet50 --dataset COVID-19 --dataset_dir /input0

python quantify_model.py --num_classes 4 --num_cpt 6 --base_model resnet50 --dataset COVID-19 --dataset_dir /input0

python result_analysis.py --num_classes 4 --num_cpt 6 --base_model resnet50 --dataset COVID-19 --dataset_dir /input0

python vis.py --num_classes 4 --num_cpt 6 --base_model resnet50 --dataset COVID-19 --dataset_dir /input0 --index 666

python vis_gui.py --num_classes 4 --num_cpt 6 --base_model resnet50 --dataset COVID-19 --dataset_dir /input0
```

## Acknowledgments

This project is based on the [BotCL](https://github.com/wbw520/BotCL) repository. Thanks to the original authors for their work which helped in developing our project.
