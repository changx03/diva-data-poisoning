#!bin/bash

# Train SVM and generate Adversarial Label Flip Attack examples
python ./experiments/real/nn/step2_train_attack.py -f ./data/preprocessed/abalone.csv -o ./data/output -t 0.2 -s 0.05
python ./experiments/real/nn/step2_train_attack.py -f ./data/preprocessed/australian.csv -o ./data/output -t 0.2 -s 0.05
python ./experiments/real/nn/step2_train_attack.py -f ./data/preprocessed/banknote.csv -o ./data/output -t 0.2 -s 0.05
python ./experiments/real/nn/step2_train_attack.py -f ./data/preprocessed/breastcancer.csv -o ./data/output -t 0.2 -s 0.05
python ./experiments/real/nn/step2_train_attack.py -f ./data/preprocessed/cardiotocography.csv -o ./data/output -t 0.2 -s 0.05
python ./experiments/real/nn/step2_train_attack.py -f ./data/preprocessed/cmc.csv -o ./data/output -t 0.2 -s 0.05
python ./experiments/real/nn/step2_train_attack.py -f ./data/preprocessed/htru2.csv -o ./data/output -t 0.2 -s 0.05
python ./experiments/real/nn/step2_train_attack.py -f ./data/preprocessed/phoneme.csv -o ./data/output -t 0.2 -s 0.05
python ./experiments/real/nn/step2_train_attack.py -f ./data/preprocessed/ringnorm.csv -o ./data/output -t 0.2 -s 0.05
python ./experiments/real/nn/step2_train_attack.py -f ./data/preprocessed/texture.csv -o ./data/output -t 0.2 -s 0.05
python ./experiments/real/nn/step2_train_attack.py -f ./data/preprocessed/yeast.csv -o ./data/output -t 0.2 -s 0.05
