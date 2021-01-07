# Mabs

This projects tries to improve Mnist classification, making it converge
with less examples, using different Mabs (Multi Arm Bandits) techniques

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.

```bash
pip install tensorflow==2.1.0
pip install requierements.txt
```

## Usage

```bash
python main.py --parallel
python main.py --type EXP3IX --gamma 0.01 --eta 0.1 --batch 100 --epochs 1000 
python main.py --type EpsilonGreedy --epsilon 0.1 --batch 100 --epochs 1000 

```
###### --type
Mab type: one of ['EpsilonGreedy', 'Random', 'UCB1', 'Thompson', 'EXP3', 'EXP3IX', 'FTL']
###### --parallel
If passed as flag, then run all Mab types in parallel
###### --batch
Batch size
###### --epochs
Epochs
###### --gamma --epsilon --eta
Parameters

