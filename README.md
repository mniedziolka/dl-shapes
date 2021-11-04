# dl-shapes
Using DL to solve shape classification and shapes counting problem.

## Task

1. **Shape classification**: network outputs six numbers:
   probabilities of a class being found on the image (in at least one copy).
2. **Geometric shapes counting**: network outputs 10 probabilities for each class,
   representing different numbers of objects of this class on the image.
   So the network should have 60 outputs. Outputs from 0 to 9 should sum up to 100%, so outputs from 10 to 19, and so on.
   
## Files

* [classification.ipynb](classification.ipynb) - Jupyter notebook with the report related to the first part
* [counter.ipynb](counter.ipynb) - Jupyter notebook with the report related to the second part
* [main.py](main.py) - python script for running the experiments
* [models](models) - directory with ML models for both tasks
* [datasets](datasets) - directory with python scripts needed for various operations on the dataset

## How to run
You'll need [pipenv](https://pypi.org/project/pipenv/).
```bash
pipenv install --dev
pipenv shell
```
Now you can run the main script.
```bash
python main.py -m classifier
```
You can see all arguments here:
```
usage: main.py [-h] [-n] -m MODEL [-e] [-f FILE]

Work with shapes networks.

optional arguments:
  -h, --help            show this help message and exit
  -n, --neptune         use neptune.ai for logging
  -m MODEL, --model MODEL
                        which model should be trained
  -e, --entropy         development or training environment
  -f FILE, --file FILE  file for storing output for plotting
```
