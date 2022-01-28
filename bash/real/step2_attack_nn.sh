#!/bin/bash

# Train SVM and generate Adversarial Label Flip Attack examples
# We do NOT want to include 50% poison case! In a perfectly balanced dataset, 50% will lead to one class completely disappear.
python ./experiments/real/nn/step2_train_attack.py -f ./data/standard/abalone_subset_std.csv -o ./data/output -t 0.2 -s 0.05 -m 0.41
python ./experiments/real/nn/step2_train_attack.py -f ./data/standard/australian_std.csv -o ./data/output -t 0.2 -s 0.05 -m 0.41
python ./experiments/real/nn/step2_train_attack.py -f ./data/standard/banknote_std.csv -o ./data/output -t 0.2 -s 0.05 -m 0.41
python ./experiments/real/nn/step2_train_attack.py -f ./data/standard/breastcancer_std.csv -o ./data/output -t 0.2 -s 0.05 -m 0.41
python ./experiments/real/nn/step2_train_attack.py -f ./data/standard/cmc_std.csv -o ./data/output -t 0.2 -s 0.05 -m 0.41
python ./experiments/real/nn/step2_train_attack.py -f ./data/standard/htru2_subset_std.csv -o ./data/output -t 0.2 -s 0.05 -m 0.41
python ./experiments/real/nn/step2_train_attack.py -f ./data/standard/phoneme_subset_std.csv -o ./data/output -t 0.2 -s 0.05 -m 0.41
python ./experiments/real/nn/step2_train_attack.py -f ./data/standard/ringnorm_subset_std.csv -o ./data/output -t 0.2 -s 0.05 -m 0.41
python ./experiments/real/nn/step2_train_attack.py -f ./data/standard/texture_subset_std.csv -o ./data/output -t 0.2 -s 0.05 -m 0.41
python ./experiments/real/nn/step2_train_attack.py -f ./data/standard/yeast_subset_std.csv -o ./data/output -t 0.2 -s 0.05 -m 0.41
