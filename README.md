# Potential-Modification Of Gravity

Welcome to the open-source code supporting the initial work around P-MOG. An adjustment to special relativity which corrects for 


## Installation

You'll want to do this whichever way you're used to doing it. However, below is an example for ease.

### Create a new virtual environment
```
python3 -m venv pmog;
cd pmog;
source bin/activate;
```

### Clone the repo & install dependancies

Here you'll likely want to clone a fork of the repo if you're going to do anything beyond recreate the published results.

```
git clone git@github.com:timjdavey/pmog.git;
cd pmog;
python -m pip install --upgrade pip;
```

There's only a few dependancies, of which are listed below
```
python -m pip install scipy numpy jupyter pandas seaborn matplotlib arviz pymc3 theano;
```
Otherwise the requirements.txt is up to date
```
python -m pip install -r requirements.txt;
```

## Generate results

### Full published results
Recreating the full results will likely take a day or two on a reasonable machine. However, the process is entirely automated and can run in the background.
```
cd generations;
python gen_baseline.py;
python gen_pmog.py;
python gen_ratio.py;
```
This is with the exception of the mcmc tuning, which is done in a notebook.

Where `baseline` creates the standard, newtonian results. `pmog` creates the full `tau prime` adjustment from the paper. `ratio` creates just the simple `tau` (ratio of the potentials) adjustment.


### Quick versions
Since the above take a few days to generate, you can get a reasonable set of data using 2D representations of the galaxies in just an hour or so. To do this use the following command.
```
cd generations;
python gen_flats.py;
```
The results will then be located at `<folder>/51_1` by default e.g. `baseline/51_1`. This represents a 51x51 2D plane, flat to 1 in the z-axis.


### Alternative versions
You can generate more detailed versions by altering the `generations/params.py` file. Or of course writing whatever code you want to generate the results. Just make sure if you do this, you'll need to run baseline before pmog or ratio.


## Investigating further
Running `jupyter notebook` will launch an interactive environment where you can play with the results. Navigating to the folder `\notebooks` you'll find many examples and additional results beyond the paper to explore.


## Contributing to the framework
If you would like to use this framework in your own work and/or contribute fixes please do!

This framework hasn't been developed with collaboration at it's heart, instead it fundamentally evolved to support the ideas of P-MOG. Therefore, there is minimal documentation and no tests. There is however, a decent logical structure to the code, lots of helper functions to instropect the data (as demonstrated in the notebooks), plenty of examples and lost of explanatory comments.

If you would like to use the framework and have questions, please do contact me, nothing would make me happier.

