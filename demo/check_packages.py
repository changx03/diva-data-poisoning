import art
import secml
import torch

if __name__ == '__main__':
    print(f'art expects [1.9.1] got [{art.__version__}]')
    print(f'secml expects [0.15] got [{secml.__version__}]')
    print(f'torch expects [1.10.1+cu113] got [{torch.__version__}]')
