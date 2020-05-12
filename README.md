# Model opinion dynamics in an echo chamber

Authors: 
- [Christopher Currin](https://chriscurrin.com)

## 1. Create environment from requirements using [conda](https://docs.conda.io/en/latest/)

`conda env create -f environment.yaml`

Target Python version: >= 3.6

## 2. Play around with notebook

`jupter notebook opdynamics.ipynb`

## 3. Use as a module

`python -m opdynamics 1000 10 2 3 -beta 2 --activity negpowerlaw 2.1 1e-2 -r 0.5 -T 10 --plot summary --save -v`
`python -m opdynamics 1000 10 2 3 -D 0.01 -beta 2 --activity negpowerlaw 2.1 1e-2 -r 0.5 -T 10 --plot summary --save -v`

# Development

Note that the code is formatted using the [black](https://pypi.org/project/black/) Python module.

Update `environment.yml` using `conda env export --from-history > environment.yml`


