# Depolarisation of opinions

[Link to paper]()

Authors: 
- [Christopher Currin](https://chriscurrin.com)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ChrisCurrin/opinion_dynamics/master?filepath=opdynamics.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChrisCurrin/opinion_dynamics/blob/master/opdynamics.ipynb)

---

## 1. Clone repo

Options
- `git clone git@github.com:ChrisCurrin/opinion_dynamics.git`
- `git clone https://github.com/ChrisCurrin/opinion_dynamics.git`

```
cd opinion_dynamics
```

## 2. Create environment from requirements using [conda](https://docs.conda.io/en/latest/)

`conda env create -f environment.yml`

`conda activate opinion_dynamics`

Target Python version: >= 3.6

## 3. Play around with notebook

`jupter notebook opdynamics.ipynb`

`jupter lab opdynamics.ipynb`

## 4. Run desired scripts
   
`python -m scripts.<script name>`

## 5. Use as a module from the terminal

e.g.
 
```bash
python -m opdynamics --plot=all
```
Run a simulation with noise with the ``-D <value>`` parameter.
```bash
python -m opdynamics 1000 10 2 3 -D 0.01 -beta 2 --activity negpowerlaw 2.1 1e-2 -r 0.5 -T 10 --plot summary --save -v
```
Compare noise parameters by passing multiple values to ``-D``
```bash
python -m opdynamics 1000 10 2 3 -D 0.01 0.1 -beta 2 --activity negpowerlaw 2.1 1e-2 -r 0.5 -T 10 --plot summary --save -v
```

---
# Notes
This module uses Euler-Maruyama to solve sochastic differential equations.

> **[TODO]**
>
>    Additional methods can be specified using https://pypi.org/project/diffeqpy/, but require extra steps to set up (e.g. installing Julia).
>   
>    1. Download Julia and add `julia` to system path.
>    2. `pip install diffeqpy numba`
>    
>    3. Install Python compatibility
>       ```python
>       import julia
>       julia.install()
>       import diffeqpy
>       diffeqpy.install()
>       ```
>   4. Test it works
>      ```python
>      from diffeqpy import de
>      ```
>      
>   5. Re-specify equations in Julia.

---
# Development

Note that the code is formatted using the [black](https://pypi.org/project/black/) Python module.

Update `environment.yml` using `conda env export --from-history > environment.yml`

---
## TODO:
- [x] Allow `simulation.run_product` to be run asyncronously.
- [x] Create metric for determining whether opinions are polarised or uniform. 
- [x] Add `sample_size` to `SampleChamber.__repr__` 
- [ ] Test more efficient `get_social_interaction`
- [ ] Test more efficient `get_connection_probabilities`
- [x] Test and plot results for "internal noise".
- [x] Restrict internal noise to be from an agent of opposite opinion.
- [x] Implement noise as coming from agent <img src="https://latex.codecogs.com/svg.latex?D(x_i - x_k)"/> - "internal noise".
    - [x] with sigmoid transformation (tanh):
        * <img src="https://latex.codecogs.com/svg.latex?D(\tanh(x_i - x_k))"/>
        * <img src="https://latex.codecogs.com/svg.latex?D(x_i - \tanh(x_k))"/>
* plot <img src="https://latex.codecogs.com/svg.latex?D"/> vs <img src="https://latex.codecogs.com/svg.latex?x"/>
    - [x] <img src="https://latex.codecogs.com/svg.latex?D"/> from <img src="https://latex.codecogs.com/svg.latex?t_0"/>
    - [x] <img src="https://latex.codecogs.com/svg.latex?D"/> from <img src="https://latex.codecogs.com/svg.latex?t_10"/>
    - [x] with different <img src="https://latex.codecogs.com/svg.latex?K"/> values
    - [x] with different <img src="https://latex.codecogs.com/svg.latex?\beta"/> values
    - [x] with different <img src="https://latex.codecogs.com/svg.latex?\alpha"/> values
    - [x] with different <img src="https://latex.codecogs.com/svg.latex?dt"/> values
- [ ] [low priority] Implement other SDE solvers using https://pypi.org/project/diffeqpy/
- [x] change density activity vs opinion plot to be density per activity
- [x] Apply noise after time <img src="https://latex.codecogs.com/svg.latex?t"/>
- [x] Calculate <img src="https://latex.codecogs.com/svg.latex?v=\frac{dx_i}{dt}"> and <img src="https://latex.codecogs.com/svg.latex?\left \langle  v \right \rangle">


