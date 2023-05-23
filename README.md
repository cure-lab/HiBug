# HiBug

HiBug is a data centric tool for investigating bugs in your model, such as rare cases in your dataset, model's correlation on visual feature.

## Features

- Utilizes large language models and vision-language models for bug discovery and repair.
- Provides a human-understandable debugging process.
- Equipped with a user interface.


## Installation

1. Clone the repository.
2. Set up the environment.
```
conda create -n HiBug python=3.7
conda activate HiBug
pip install -r requirements.txt
```

## Getting Started
1. Enter your ChatGPT API key in utils/gpt.py and run prepare.py.
    - ChatGPT provides potential attribute names to explore in the bug discovery process.
    - If you do not have an API key, you can also view the dataset through our user interface to come up with attribute names. An example of corpus_base.json is provided in exampleData/corpus_base.json.
2. Follow the steps in run.ipynb.
    - Fill the empty list with selected attribute names.
    - Ensure that the data types match the code specifications.