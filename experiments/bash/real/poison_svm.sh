#!bin/bash

echo "For poisoned data..."
mkdir ./results/real_poison_svm
# Train SVM and generate Adversarial Label Flip Attack examples

# We do NOT want to include 50% poison case! In a perfectly balanced dataset, 50% will lead to one class completely disappear.
python ./experiments/real/svm/poison_svm.py -f ./data/standard/australian_std.csv -o ./data/output -t 0.2 -s 0.1 -m 0.41
Rscript ./experiments/real/step3_cmeasure.R "./data/output/svm/" "./results/real_poison_svm/australian_svm_poison.csv" "australian"

python ./experiments/real/svm/poison_svm.py -f ./data/standard/banknote_std.csv -o ./data/output -t 0.2 -s 0.1 -m 0.41
Rscript ./experiments/real/step3_cmeasure.R "./data/output/svm/" "./results/real_poison_svm/banknote_svm_poison.csv" "banknote"

python ./experiments/real/svm/poison_svm.py -f ./data/standard/breastcancer_std.csv -o ./data/output -t 0.2 -s 0.1 -m 0.41
Rscript ./experiments/real/step3_cmeasure.R "./data/output/svm/" "./results/real_poison_svm/breastcancer_svm_poison.csv" "breastcancer"

python ./experiments/real/svm/poison_svm.py -f ./data/standard/cmc_std.csv -o ./data/output -t 0.2 -s 0.1 -m 0.41
Rscript ./experiments/real/step3_cmeasure.R "./data/output/svm/" "./results/real_poison_svm/cmc_svm_poison.csv" "cmc"

python ./experiments/real/svm/poison_svm.py -f ./data/standard/texture_subset_std.csv -o ./data/output -t 0.2 -s 0.1 -m 0.41
Rscript ./experiments/real/step3_cmeasure.R "./data/output/svm/" "./results/real_poison_svm/texture_svm_poison.csv" "texture"

python ./experiments/real/svm/poison_svm.py -f ./data/standard/yeast_subset_std.csv -o ./data/output -t 0.2 -s 0.1 -m 0.41
Rscript ./experiments/real/step3_cmeasure.R "./data/output/svm/" "./results/real_poison_svm/yeast_svm_poison.csv" "yeast"

echo "Start running on larger datasets..."
python ./experiments/real/svm/poison_svm.py -f ./data/standard/abalone_subset_std.csv -o ./data/output -t 0.2 -s 0.1 -m 0.41
Rscript ./experiments/real/step3_cmeasure.R "./data/output/svm/" "./results/real_poison_svm/abalone_svm_poison.csv" "abalone"

python ./experiments/real/svm/poison_svm.py -f ./data/standard/htru2_subset_std.csv -o ./data/output -t 0.2 -s 0.1 -m 0.41
Rscript ./experiments/real/step3_cmeasure.R "./data/output/svm/" "./results/real_poison_svm/htru2_svm_poison.csv" "htru2"

python ./experiments/real/svm/poison_svm.py -f ./data/standard/phoneme_subset_std.csv -o ./data/output -t 0.2 -s 0.1 -m 0.41
Rscript ./experiments/real/step3_cmeasure.R "./data/output/svm/" "./results/real_poison_svm/phoneme_svm_poison.csv" "phoneme"

python ./experiments/real/svm/poison_svm.py -f ./data/standard/ringnorm_subset_std.csv -o ./data/output -t 0.2 -s 0.1 -m 0.41
Rscript ./experiments/real/step3_cmeasure.R "./data/output/svm/" "./results/real_poison_svm/ringnorm_svm_poison.csv" "ringnorm"
