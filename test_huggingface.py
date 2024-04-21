import datasets

print(datasets.__version__)

from datasets import load_dataset

dataset_name = "JAAR90/best_practices"
dataset = load_dataset(dataset_name, split='train')


print("First few examples from the train dataset:")
for example in dataset.take(5):  
    print(example)


 