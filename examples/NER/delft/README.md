# Ontonotes V5.0 (CoNLL2012)

## train

RanNet-CNN-CRF

```
CUDA_VISIBLE_DEVICES=0 TF_KERAS=1 python3 delft/applications/nerTagger.py --dataset-type conll2012 train_eval --architecture RanNet_CNN_CRF --embedding glove-840B --use-ELMo
```

batch_size=128, lr=1e-3, dropout_rate=0.5, L300, head_num=4, head_size=256, f1 = 88.54 (onto-1)
batch_size=128, lr=1e-3, dropout_rate=0.35, L300, head_num=8, head_size=64, f1 = 88.53 (onto-2)


batch_size=150, lr=5e-4, dropout_rate=0.5, L300, head_num=8, head_size=256, f1 = 89.24  (onto-2)
**batch_size=150, lr=3e-4, dropout_rate=0.5, L300, head_num=8, head_size=256, f1 = 89.38**


# CoNLL2003


```
TF_KERAS=1 python3 delft/applications/nerTagger.py --dataset-type conll2003 train_eval --architecture RanNet_CNN_CRF --embedding glove-840B --use-ELMo
```

batch_size=150, lr=3e-4, dropout_rate=0.5, L300, head_num=8, head_size=256, f1 = 92.07 (col)
batch_size=150, lr=3e-4, dropout_rate=0.5, L300, head_num=8, head_size=128, f1 = 92.49 (col)
**batch_size=150, lr=3e-4, dropout_rate=0.5, L300, head_num=12, head_size=128, f1 = 92.68 (col)**
