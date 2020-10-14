Code for NeurIPS 2020 Paper ["On the Loss Landscape of Adversarial Training: Identifying Challenges and How to Overcome Them"](https://arxiv.org/pdf/2006.08403).
Suitable to run on NVIDIA GPU machines.

## Requirements

```
python = 3.7
numpy >= 1.16
torch >= 1.3
torchvision >= 0.4
```

## Abstract

We analyze the influence of adversarial training on the loss landscape of machine learning models.
To this end, we first provide analytical studies of the properties of adversarial loss functions under different adversarial budgets.
We then demonstrate that the adversarial loss landscape is less favorable to optimization, due to increased curvature and more scattered gradients.
Our conclusions are validated by numerical analyses, which show that training under large adversarial budgets impede the escape from suboptimal random initialization, cause non-vanishing gradients and make the model find sharper minima.
Based on these observations, we show that a periodic adversarial scheduling (PAS) strategy can effectively overcome these challenges, yielding better results than vanilla adversarial training while being much less sensitive to the choice of learning rate. 

## Modules

Folder `util` contains all supporting functions.
Specially, `util/attack.py` has implmentations for different attackers, `util/seq_parser.py` has all sequential functions for learning rate scheduling and adversarial budget scheduling; `util/optim_parser.py` has constructors of all supported optimizers; `util/models.py` has all model architectures, parameterized by a width factor w.

Folder `run` contains all scripts to run experiments.
You can use `python run/XXX.py -h` to get the information about all command line parameters.
We briefly introduce different files below:

```
# Train and Test (Section 4 and 5)
run/train_normal.py: train models by defining different PGD attacks, learning rate scheduling and adversarial budget scheduling.
run/test_ensemble.py: test the performance of models or ensemble of models under adversarial attacks.
# Numerical Analysis (Section 4)
run/perturb_param.py: perturb the model parameter given the original model and the perturbation.
run/scan_param.py: given the model, two directions in the parameter space and adversarial attacks, test the accuracy and loss of parameters spanned by these two directions.
run/generate_adversary.py: generate adversarial examples by PGD given the model and the adversarial budgets.
run/calc_hessian.py: estimate the top Hessian eigenvalues given a trained model.
# Find the flat curves connecting the parameters of two models (Appendix C)
run/train_curve.py: train the Bezier curves to connect two given minima.
run/scan_curve.py: scan the loss and the accuracy along a trained Bezier curve.
```

Folder `analysis` contains some functions to analyze the checkpoints produced by scripts under `run`:

```
analysis/analyze_adversary.py: calculate the average cosine similarity of the perturbations.
analysis/calc_param_distance.py: calculate the distance of two models in the parameter space.
```

## Examples

Below are the configurations for different attackers used in the paper, use the following configurations after flag `--attack` in the command when you need to construct the corresponding attacker. For each attacker, the first config is for MNIST and the second is for CIFAR10.

* PGD10: `name=pgd,step_size=0.01,threshold=0.4,iter_num=100,order=-1` `name=pgd,step_size=2,threshold=8,iter_num=10,order=-1`
* PGD100: `name=pgd,step_size=0.01,threshold=0.4,iter_num=100,order=-1` `name=pgd,step_size=1,threshold=8,iter_num=100,order=-1`
* APGD100 CE: `name=apgd,threshold=0.4,iter_num=100,order=-1,rho=0.75,loss_type=ce` `name=apgd,threshold=8,iter_num=100,order=-1,rho=0.75,loss_type=ce`
* APGD100 DLR: `name=apgd,threshold=0.4,iter_num=100,order=-1,rho=0.75,loss_type=dlr` `name=apgd,threshold=8,iter_num=100,order=-1,rho=0.75,loss_type=dlr`
* Square5K: `name=square,threshold=0.4,iter_num=5000,order=-1,window_size_factor=0` `name=square,threshold=8,iter_num=5000,order=-1,window_size_factor=0`

Below we give some examples to run the experiments we mentioned in the paper. Replace the name in `$$` with the one you prefer.

1. Calculate the Hessian top 20 eigenvalues of a MNIST model under the adversarial budget $\epsilon = 0.1$.

```
python run/calc_hessian.py --model_type lenet --width 16 --model2load $MODEL_TO_LOAD$ --attack name=pgd,step_size=0.01,threshold=0.1,iter_num=20,order=-1 --out_file $OUTPUT_FILE$ --topk 20 --max_iter 50 --gpu $GPU_ID$
```

2. Using cosine scheduler to train LeNet model on MNIST against adversarial attacks under the budget $\epsilon = 0.4$.

```
python run/train_normal.py --dataset mnist --epoch_num 100 --epoch_ckpts 50,75,100 --model_type lenet --width 16 --out_folder $FOLDER$ --model_name $MODEL_NAME$ --optim name=adam,lr=1e-4 --attack name=pgd,step_size=0.01,iter_num=50,order=-1,threshold=0.4 --attack_threshold_schedule name=cycle_cos,eps_min=0,eps_max=0.6,ckpt_list=0:100,max=0.4 --gpu $GPU_ID$ --lr_schedule name=constant,start_v=1e-4
```

3. Test the ensemble of three ResNet18 models on CIFAR10 against APGD DLR attacks under the budget $\epsilon = 8 / 255$

```
python run/test_ensemble.py --batch_size 100 --model2load $MODEL1$,$MODEL2$,$MODEL3$ --out_file $OUT_FILE$ --gpu $GPU_ID$ --attack name=apgd,threshold=8,iter_num=100,order=-1,rho=0.75,loss_type=dlr --model_type resnet --dataset cifar10 --width 8
```

4. Train a Bezier curve to connect two CIFAR10 models: $MODEL1$ and $MODEL2$. The adversarial budget size is $\epsilon = 8/255$

```
python run/train_curve.py --epoch_num 200 --dataset cifar10 --model_type resnet --width 8 --fix_points 1,0,0,0,1 --model2load $MODEL1$,,,,$MODEL2$ --curve_type bezier --out_folder $OUT_FOLDER$ --model_name $MODEL_NAME$ --optim name=sgd,lr=0.1,momentum=0.9,weight_decay=1e-6 --lr_schedule name=jump,min_jump_pt=100,jump_freq=50,power=0.1,start_v=1e-1 --attack name=pgd,step_size=2,threshold=8,iter_num=10,order=-1 --gpu $GPU_ID$
```

## Model Checkpoints

The model checkpoints of Table 1 are provided under the folder `models`. All experiments are run for three times.

## Acknowledgments

The AutoPGD and SquareAttack modules are downloaded from the AutoAttack repo: [https://github.com/fra31/auto-attack](https://github.com/fra31/auto-attack).
They are put in the `external` folder.

## Contact

Please contact Chen Liu (chen.liu@epfl.ch) regarding this repository.

## Citation

```
@inproceedings{liu2020loss,
  title={On the Loss Landscape of Adversarial Training: Identifying Challenges and How to Overcome Them},
  author={Liu, Chen and Salzmann, Mathieu and Lin, Tao and Tomioka, Ryota and S{\"u}sstrunk, Sabine},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```
