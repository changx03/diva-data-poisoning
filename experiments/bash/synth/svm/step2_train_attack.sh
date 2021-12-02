#!bin/bash

# Train SVM and generate Adversarial Label Flip Attack examples
python ./experiments/synth/svm/step2_train_attack.py -p ./data/synth/ -s 0.05 -t 1000 -m 0.49
