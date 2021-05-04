# dl-shapes
Using DL to solve shape classification and shapes counting problem.

## Task

1. **Shape classification**: network outputs six numbers:
   probabilities of a class being found on the image (in at least one copy).
   Loss function is the sum of loglosses over all classes:
    $$J = -\sum_{j=0}^5y_i*log(\hat y_i)+(1-y_i)*log(1-\hat y_i)$$
   *Notation:* $y_i$ is a ground truth, $\hat y_i$ is predicted probability of class $i$ on the image.
2. **Geometic shapes counting**: network outputs $10$ probabilities for each class,
   representing different numbers of objects of this class on the image.
   So the network should have $60$ outputs. Outputs $0..9$ should sum up to $100\%$, so outputs $10..19$, and so on.
   The loss function for the network is the sum of squared counting errors:
    $$J = \sum_{i=0}^5 \sum_{j=0}^9\hat y_j^i (j - r^i)^2$$
   *Notation:* $r^i$ is a ground truth, $\hat y_j^i$ is predicted probability of $j$ figures of class $i$ on the image.
   
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
Now you cen run the main script.
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
