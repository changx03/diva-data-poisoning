#!bin/bash

# Train SVM and generate Adversarial Label Flip Attack examples
# We do NOT want to include 50% poison case! In a perfectly balanced dataset, 50% will lead to one class completely disappear.
python ./experiments/real/nn/step2_train_attack.py -f ./data/preprocessed/abalone_subset.csv -o ./data/output -t 0.2 -s 0.05 -m 0.49
python ./experiments/real/nn/step2_train_attack.py -f ./data/preprocessed/australian.csv -o ./data/output -t 0.2 -s 0.05 -m 0.49
python ./experiments/real/nn/step2_train_attack.py -f ./data/preprocessed/banknote.csv -o ./data/output -t 0.2 -s 0.05 -m 0.49
python ./experiments/real/nn/step2_train_attack.py -f ./data/preprocessed/breastcancer.csv -o ./data/output -t 0.2 -s 0.05 -m 0.49
python ./experiments/real/nn/step2_train_attack.py -f ./data/preprocessed/cardiotocography.csv -o ./data/output -t 0.2 -s 0.05 -m 0.49
python ./experiments/real/nn/step2_train_attack.py -f ./data/preprocessed/cmc.csv -o ./data/output -t 0.2 -s 0.05 -m 0.49
python ./experiments/real/nn/step2_train_attack.py -f ./data/preprocessed/htru2_subset.csv -o ./data/output -t 0.2 -s 0.05 -m 0.49
python ./experiments/real/nn/step2_train_attack.py -f ./data/preprocessed/phoneme_subset.csv -o ./data/output -t 0.2 -s 0.05 -m 0.49
python ./experiments/real/nn/step2_train_attack.py -f ./data/preprocessed/ringnorm_subset.csv -o ./data/output -t 0.2 -s 0.05 -m 0.49
python ./experiments/real/nn/step2_train_attack.py -f ./data/preprocessed/texture.csv -o ./data/output -t 0.2 -s 0.05 -m 0.49
python ./experiments/real/nn/step2_train_attack.py -f ./data/preprocessed/yeast.csv -o ./data/output -t 0.2 -s 0.05 -m 0.49
