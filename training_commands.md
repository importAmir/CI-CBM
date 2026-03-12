## Pretrained CIL (standard backbones)

### CIFAR10 (5 tasks)

```bash
python main.py --dataset cifar10 --strategy backbone_prototype --SAGA_lr 0.1 --n_iters 1000 --backbone resnet18 --clip_name ViT-B/16 --batch_size 100
python evaluate_cbm.py --dataset cifar10 --strategy backbone_prototype --backbone resnet18 --clip_name ViT-B/16

python main.py --dataset cifar10 --strategy backbone_prototype --SAGA_lr 0.1 --n_iters 1000 --backbone ViT-B/16-IN21K --clip_name ViT-B/16 --batch_size 100
python evaluate_cbm.py --dataset cifar10 --strategy backbone_prototype --backbone ViT-B/16-IN21K --clip_name ViT-B/16
```

#### Distillation (LwF) weight ablation (CIFAR10)
Only affects the projection-layer training for `exp_id > 0` in LwF strategies.

```bash
python main.py --dataset cifar10 --strategy backbone_prototype --distill_weight 0.0  --save_dir distill_weight_saved_models/distill_weight_0.0
python evaluate_cbm.py --dataset cifar10 --strategy backbone_prototype --backbone resnet18 --clip_name ViT-B/16 --save_dir distill_weight_saved_models/distill_weight_0.0

python main.py --dataset cifar10 --strategy backbone_prototype --distill_weight 0.25 --save_dir distill_weight_saved_models/distill_weight_0.25
python evaluate_cbm.py --dataset cifar10 --strategy backbone_prototype --backbone resnet18 --clip_name ViT-B/16 --save_dir distill_weight_saved_models/distill_weight_0.25

python main.py --dataset cifar10 --strategy backbone_prototype --distill_weight 0.5  --save_dir distill_weight_saved_models/distill_weight_0.5
python evaluate_cbm.py --dataset cifar10 --strategy backbone_prototype --backbone resnet18 --clip_name ViT-B/16 --save_dir distill_weight_saved_models/distill_weight_0.5

python main.py --dataset cifar10 --strategy backbone_prototype --distill_weight 1.0  --save_dir distill_weight_saved_models/distill_weight_1.0
python evaluate_cbm.py --dataset cifar10 --strategy backbone_prototype --backbone resnet18 --clip_name ViT-B/16 --save_dir distill_weight_saved_models/distill_weight_1.0

python main.py --dataset cifar10 --strategy backbone_prototype --distill_weight 2.0  --save_dir distill_weight_saved_models/distill_weight_2.0
python evaluate_cbm.py --dataset cifar10 --strategy backbone_prototype --backbone resnet18 --clip_name ViT-B/16 --save_dir distill_weight_saved_models/distill_weight_2.0

python main.py --dataset cifar10 --strategy backbone_prototype --distill_weight 5.0 --save_dir distill_weight_saved_models/distill_weight_5.0
python evaluate_cbm.py --dataset cifar10 --strategy backbone_prototype --backbone resnet18 --clip_name ViT-B/16 --save_dir distill_weight_saved_models/distill_weight_5.0
```

### CIFAR100 (5 / 10 / 20 tasks)

```bash
# 5 tasks
python main.py --dataset cifar100 --strategy backbone_prototype --SAGA_lr 0.1 --n_iters 1000 --backbone resnet18           --clip_name ViT-B/16 --batch_size 100 --n_experiences 5
python evaluate_cbm.py --dataset cifar100 --strategy backbone_prototype --backbone resnet18           --clip_name ViT-B/16 --n_experiences 5

python main.py --dataset cifar100 --strategy backbone_prototype --SAGA_lr 0.1 --n_iters 1000 --backbone ViT-B/16-IN21K    --clip_name ViT-B/16 --batch_size 100 --n_experiences 5
python evaluate_cbm.py --dataset cifar100 --strategy backbone_prototype --backbone ViT-B/16-IN21K    --clip_name ViT-B/16 --n_experiences 5

# 10 tasks
python main.py --dataset cifar100 --strategy backbone_prototype --SAGA_lr 0.1 --n_iters 1000 --backbone resnet18           --clip_name ViT-B/16 --batch_size 100 --n_experiences 10
python evaluate_cbm.py --dataset cifar100 --strategy backbone_prototype --backbone resnet18           --clip_name ViT-B/16 --n_experiences 10

python main.py --dataset cifar100 --strategy backbone_prototype --SAGA_lr 0.1 --n_iters 1000 --backbone ViT-B/16-IN21K    --clip_name ViT-B/16 --batch_size 100 --n_experiences 10
python evaluate_cbm.py --dataset cifar100 --strategy backbone_prototype --backbone ViT-B/16-IN21K    --clip_name ViT-B/16 --n_experiences 10

# 20 tasks
python main.py --dataset cifar100 --strategy backbone_prototype --SAGA_lr 0.1 --n_iters 1000 --backbone resnet18           --clip_name ViT-B/16 --batch_size 100 --n_experiences 20
python evaluate_cbm.py --dataset cifar100 --strategy backbone_prototype --backbone resnet18           --clip_name ViT-B/16 --n_experiences 20

python main.py --dataset cifar100 --strategy backbone_prototype --SAGA_lr 0.1 --n_iters 1000 --backbone ViT-B/16-IN21K    --clip_name ViT-B/16 --batch_size 100 --n_experiences 20
python evaluate_cbm.py --dataset cifar100 --strategy backbone_prototype --backbone ViT-B/16-IN21K    --clip_name ViT-B/16 --n_experiences 20
```

### CUB (4 / 10 / 20 tasks)

```bash
# ResNet18 backbone
python main.py --dataset cub --strategy backbone_prototype --SAGA_lr 0.1 --n_iters 500 --lam 0.0002 --backbone resnet18        --clip_name ViT-B/16 --batch_size 100 --n_experiences 4
python evaluate_cbm.py --dataset cub --strategy backbone_prototype --backbone resnet18        --clip_name ViT-B/16 --n_experiences 4

python main.py --dataset cub --strategy backbone_prototype --SAGA_lr 0.1 --n_iters 500 --lam 0.0002 --backbone resnet18        --clip_name ViT-B/16 --batch_size 100 --n_experiences 10
python evaluate_cbm.py --dataset cub --strategy backbone_prototype --backbone resnet18        --clip_name ViT-B/16 --n_experiences 10

python main.py --dataset cub --strategy backbone_prototype --SAGA_lr 0.1 --n_iters 500 --lam 0.0002 --backbone resnet18        --clip_name ViT-B/16 --batch_size 100 --n_experiences 20
python evaluate_cbm.py --dataset cub --strategy backbone_prototype --backbone resnet18        --clip_name ViT-B/16 --n_experiences 20

# ViT-B/16-IN21K backbone
python main.py --dataset cub --strategy backbone_prototype --SAGA_lr 0.1 --n_iters 500 --lam 0.0002 --backbone ViT-B/16-IN21K --clip_name ViT-B/16 --batch_size 100 --n_experiences 5
python evaluate_cbm.py --dataset cub --strategy backbone_prototype --backbone ViT-B/16-IN21K --clip_name ViT-B/16 --n_experiences 5

python main.py --dataset cub --strategy backbone_prototype --SAGA_lr 0.1 --n_iters 500 --lam 0.0002 --backbone ViT-B/16-IN21K --clip_name ViT-B/16 --batch_size 100 --n_experiences 10
python evaluate_cbm.py --dataset cub --strategy backbone_prototype --backbone ViT-B/16-IN21K --clip_name ViT-B/16 --n_experiences 10

python main.py --dataset cub --strategy backbone_prototype --SAGA_lr 0.1 --n_iters 500 --lam 0.0002 --backbone ViT-B/16-IN21K --clip_name ViT-B/16 --batch_size 100 --n_experiences 20
python evaluate_cbm.py --dataset cub --strategy backbone_prototype --backbone ViT-B/16-IN21K --clip_name ViT-B/16 --n_experiences 20
```

### TinyImageNet (5 / 10 / 20 tasks)

```bash
python main.py --dataset tiny_imagenet --strategy backbone_prototype --SAGA_lr 0.1 --n_iters 1000 --backbone resnet18_places --clip_name ViT-B/16 --batch_size 100 --lam 0.0005 --n_experiences 5
python evaluate_cbm.py --dataset tiny_imagenet --strategy backbone_prototype --backbone resnet18_places --clip_name ViT-B/16 --n_experiences 5

python main.py --dataset tiny_imagenet --strategy backbone_prototype --SAGA_lr 0.1 --n_iters 1000 --backbone resnet18_places --clip_name ViT-B/16 --batch_size 100 --lam 0.0005 --n_experiences 10
python evaluate_cbm.py --dataset tiny_imagenet --strategy backbone_prototype --backbone resnet18_places --clip_name ViT-B/16 --n_experiences 10

python main.py --dataset tiny_imagenet --strategy backbone_prototype --SAGA_lr 0.1 --n_iters 1000 --backbone resnet18_places --clip_name ViT-B/16 --batch_size 100 --lam 0.0005 --n_experiences 20
python evaluate_cbm.py --dataset tiny_imagenet --strategy backbone_prototype --backbone resnet18_places --clip_name ViT-B/16 --n_experiences 20
```

### ImageNet-R / ImageNet-A (5 / 10 / 20 tasks)

```bash
# ImageNet-R
python main.py --dataset imagenet_r --strategy backbone_prototype --SAGA_lr 0.1 --n_iters 1000 --backbone ViT-B/16-IN21K --clip_name ViT-B/16 --batch_size 100 --lam 0.0005 --n_experiences 5
python evaluate_cbm.py --dataset imagenet_r --strategy backbone_prototype --backbone ViT-B/16-IN21K --clip_name ViT-B/16 --n_experiences 5

python main.py --dataset imagenet_r --strategy backbone_prototype --SAGA_lr 0.1 --n_iters 1000 --backbone ViT-B/16-IN21K --clip_name ViT-B/16 --batch_size 100 --lam 0.0005 --n_experiences 10
python evaluate_cbm.py --dataset imagenet_r --strategy backbone_prototype --backbone ViT-B/16-IN21K --clip_name ViT-B/16 --n_experiences 10

python main.py --dataset imagenet_r --strategy backbone_prototype --SAGA_lr 0.1 --n_iters 1000 --backbone ViT-B/16-IN21K --clip_name ViT-B/16 --batch_size 100 --lam 0.0005 --n_experiences 20
python evaluate_cbm.py --dataset imagenet_r --strategy backbone_prototype --backbone ViT-B/16-IN21K --clip_name ViT-B/16 --n_experiences 20

# ImageNet-A
python main.py --dataset imagenet_a --strategy backbone_prototype --SAGA_lr 0.1 --n_iters 1000 --backbone ViT-B/16-IN21K --clip_name ViT-B/16 --batch_size 100 --lam 0.0005 --n_experiences 5
python evaluate_cbm.py --dataset imagenet_a --strategy backbone_prototype --backbone ViT-B/16-IN21K --clip_name ViT-B/16 --n_experiences 5

python main.py --dataset imagenet_a --strategy backbone_prototype --SAGA_lr 0.1 --n_iters 1000 --backbone ViT-B/16-IN21K --clip_name ViT-B/16 --batch_size 100 --lam 0.0005 --n_experiences 10
python evaluate_cbm.py --dataset imagenet_a --strategy backbone_prototype --backbone ViT-B/16-IN21K --clip_name ViT-B/16 --n_experiences 10

python main.py --dataset imagenet_a --strategy backbone_prototype --SAGA_lr 0.1 --n_iters 1000 --backbone ViT-B/16-IN21K --clip_name ViT-B/16 --batch_size 100 --lam 0.0005 --n_experiences 20
python evaluate_cbm.py --dataset imagenet_a --strategy backbone_prototype --backbone ViT-B/16-IN21K --clip_name ViT-B/16 --n_experiences 20
```

### Places365 (5 / 10 / 20 tasks)

```bash
python main.py --dataset places365 --strategy backbone_prototype --SAGA_lr 0.1 --n_iters 80 --lam 0.0003 --backbone resnet18 --clip_name ViT-B/16 --batch_size 100 --n_experiences 5
python evaluate_cbm.py --dataset places365 --strategy backbone_prototype --backbone resnet18 --clip_name ViT-B/16 --n_experiences 5

python main.py --dataset places365 --strategy backbone_prototype --SAGA_lr 0.1 --n_iters 80 --lam 0.0003 --backbone resnet18 --clip_name ViT-B/16 --batch_size 100 --n_experiences 10
python evaluate_cbm.py --dataset places365 --strategy backbone_prototype --backbone resnet18 --clip_name ViT-B/16 --n_experiences 10

python main.py --dataset places365 --strategy backbone_prototype --SAGA_lr 0.1 --n_iters 80 --lam 0.0003 --backbone resnet18 --clip_name ViT-B/16 --batch_size 100 --n_experiences 20
python evaluate_cbm.py --dataset places365 --strategy backbone_prototype --backbone resnet18 --clip_name ViT-B/16 --n_experiences 20
```

### ImageNet (5 / 10 / 20 tasks)

```bash
python main.py --dataset imagenet --strategy backbone_prototype --SAGA_lr 0.1 --n_iters 80 --lam 0.0001 --backbone resnet18_places --clip_name ViT-B/16 --batch_size 100 --n_experiences 5
python evaluate_cbm.py --dataset imagenet --strategy backbone_prototype --backbone resnet18_places --clip_name ViT-B/16 --n_experiences 5

python main.py --dataset imagenet --strategy backbone_prototype --SAGA_lr 0.1 --n_iters 80 --lam 0.0001 --backbone resnet18_places --clip_name ViT-B/16 --batch_size 100 --n_experiences 10
python evaluate_cbm.py --dataset imagenet --strategy backbone_prototype --backbone resnet18_places --clip_name ViT-B/16 --n_experiences 10

python main.py --dataset imagenet --strategy backbone_prototype --SAGA_lr 0.1 --n_iters 80 --lam 0.0001 --backbone resnet18_places --clip_name ViT-B/16 --batch_size 100 --n_experiences 20
python evaluate_cbm.py --dataset imagenet --strategy backbone_prototype --backbone resnet18_places --clip_name ViT-B/16 --n_experiences 20
```

---

## Non-pretrained CIL (first-phase trained backbones)

These experiments use backbones trained on the first phase (e.g., FeTrIL, SelfPromptDeit) and then perform class-incremental learning on top.

### CIFAR100 (FeTrIL backbone, 5+1 / 10+1 / 20+1 / 60+1 tasks)

```bash
# 5+1 tasks
python main.py --seed 1993 --backbone resnet18_FeTrIL_cifar100_b50 --strategy backbone_prototype --SAGA_lr 0.1 --dataset cifar100 --half_split --n_experiences 5  --clip_name ViT-B/16 --batch_size 100
python evaluate_cbm.py --seed 1993 --backbone resnet18_FeTrIL_cifar100_b50 --strategy backbone_prototype --dataset cifar100 --half_split --n_experiences 5

# 10+1 tasks
python main.py --seed 1993 --backbone resnet18_FeTrIL_cifar100_b50 --strategy backbone_prototype --SAGA_lr 0.1 --dataset cifar100 --half_split --n_experiences 10 --clip_name ViT-B/16 --batch_size 100
python evaluate_cbm.py --seed 1993 --backbone resnet18_FeTrIL_cifar100_b50 --strategy backbone_prototype --dataset cifar100 --half_split --n_experiences 10

# 20+1 tasks
python main.py --seed 1993 --backbone resnet18_FeTrIL_cifar100_b40 --strategy backbone_prototype --SAGA_lr 0.1 --dataset cifar100 --half_split --n_experiences 20 --clip_name ViT-B/16 --batch_size 100
python evaluate_cbm.py --seed 1993 --backbone resnet18_FeTrIL_cifar100_b40 --strategy backbone_prototype --dataset cifar100 --half_split --n_experiences 20

# 60+1 tasks
python main.py --seed 1993 --backbone resnet18_FeTrIL_cifar100_b40 --strategy backbone_prototype --SAGA_lr 0.1 --dataset cifar100 --half_split --n_experiences 60 --clip_name ViT-B/16 --batch_size 100
python evaluate_cbm.py --seed 1993 --backbone resnet18_FeTrIL_cifar100_b40 --strategy backbone_prototype --dataset cifar100 --half_split --n_experiences 60
```

### TinyImageNet (FeTrIL backbone, 5+1 / 10+1 / 20+1 / 100+1 tasks)

```bash
python main.py --dataset tiny_imagenet --strategy backbone_prototype --SAGA_lr 0.1 --n_iters 1000 --lam 0.0005 --backbone resnet18_FeTrIL_tinyimagenet_b100 --half_split --n_experiences 5   --clip_name ViT-B/16 --batch_size 100
python evaluate_cbm.py --dataset tiny_imagenet --strategy backbone_prototype --backbone resnet18_FeTrIL_tinyimagenet_b100 --half_split --n_experiences 5   --clip_name ViT-B/16

python main.py --dataset tiny_imagenet --strategy backbone_prototype --SAGA_lr 0.1 --n_iters 1000 --lam 0.0005 --backbone resnet18_FeTrIL_tinyimagenet_b100 --half_split --n_experiences 10  --clip_name ViT-B/16 --batch_size 100
python evaluate_cbm.py --dataset tiny_imagenet --strategy backbone_prototype --backbone resnet18_FeTrIL_tinyimagenet_b100 --half_split --n_experiences 10  --clip_name ViT-B/16

python main.py --dataset tiny_imagenet --strategy backbone_prototype --SAGA_lr 0.1 --n_iters 1000 --lam 0.0005 --backbone resnet18_FeTrIL_tinyimagenet_b100 --half_split --n_experiences 20  --clip_name ViT-B/16 --batch_size 100
python evaluate_cbm.py --dataset tiny_imagenet --strategy backbone_prototype --backbone resnet18_FeTrIL_tinyimagenet_b100 --half_split --n_experiences 20  --clip_name ViT-B/16

python main.py --dataset tiny_imagenet --strategy backbone_prototype --SAGA_lr 0.1 --n_iters 1000 --lam 0.0005 --backbone resnet18_FeTrIL_tinyimagenet_b100 --half_split --n_experiences 100 --clip_name ViT-B/16 --batch_size 100
python evaluate_cbm.py --dataset tiny_imagenet --strategy backbone_prototype --backbone resnet18_FeTrIL_tinyimagenet_b100 --half_split --n_experiences 100 --clip_name ViT-B/16
```

### ImageNet-Subset (FeTrIL backbone, 5+1 / 10+1 / 20+1 / 60+1 tasks)

```bash
python main.py --dataset imagenetsubset --strategy backbone_prototype --SAGA_lr 0.1 --n_iters 1000 --lam 0.0005 --backbone resnet18_FeTrIL_imagenetsubset_b50 --half_split --n_experiences 5  --clip_name ViT-B/16 --batch_size 100
python evaluate_cbm.py --dataset imagenetsubset --strategy backbone_prototype --backbone resnet18_FeTrIL_imagenetsubset_b50 --half_split --n_experiences 5  --clip_name ViT-B/16

python main.py --dataset imagenetsubset --strategy backbone_prototype --SAGA_lr 0.1 --n_iters 1000 --lam 0.0005 --backbone resnet18_FeTrIL_imagenetsubset_b50 --half_split --n_experiences 10 --clip_name ViT-B/16 --batch_size 100
python evaluate_cbm.py --dataset imagenetsubset --strategy backbone_prototype --backbone resnet18_FeTrIL_imagenetsubset_b50 --half_split --n_experiences 10 --clip_name ViT-B/16

python main.py --dataset imagenetsubset --strategy backbone_prototype --SAGA_lr 0.1 --n_iters 1000 --lam 0.0005 --backbone resnet18_FeTrIL_imagenetsubset_b40 --half_split --n_experiences 20 --clip_name ViT-B/16 --batch_size 100
python evaluate_cbm.py --dataset imagenetsubset --strategy backbone_prototype --backbone resnet18_FeTrIL_imagenetsubset_b40 --half_split --n_experiences 20 --clip_name ViT-B/16

python main.py --dataset imagenetsubset --strategy backbone_prototype --SAGA_lr 0.1 --n_iters 1000 --lam 0.0005 --backbone resnet18_FeTrIL_imagenetsubset_b40 --half_split --n_experiences 60 --clip_name ViT-B/16 --batch_size 100
python evaluate_cbm.py --dataset imagenetsubset --strategy backbone_prototype --backbone resnet18_FeTrIL_imagenetsubset_b40 --half_split --n_experiences 60 --clip_name ViT-B/16
```

### SelfPromptDeit (APG) backbones

```bash
# CIFAR100, SelfPromptDeit_mytiny
python main.py --seed 1993 --backbone SelfPromptDeit_mytiny_cifar100_b50       --strategy backbone_prototype --SAGA_lr 0.1 --dataset cifar100      --half_split --n_experiences 5 --clip_name ViT-B/16 --batch_size 100
python evaluate_cbm.py --dataset cifar100      --strategy backbone_prototype --backbone SelfPromptDeit_mytiny_cifar100_b50       --clip_name ViT-B/16 --n_experiences 5 --half_split

# ImageNet-Subset, SelfPromptDeit_mytiny
python main.py --dataset imagenetsubset --strategy backbone_prototype --SAGA_lr 0.1 --n_iters 1000 --lam 0.0005 --backbone SelfPromptDeit_mytiny_imagenetsubset_b50 --half_split --n_experiences 5 --clip_name ViT-B/16 --batch_size 100
```