# Speeding Logistic Regression In Cython and Numpy

This is a Cython implementation of `speedglm` package of `R`.

# Installation and Usage

Install using pip

``` shell
pip install git+github.com:ChingChuan-Chen/Logistic_regression_Cython
```

# Example

``` python
from speed_logit_reg import SpeedLogitReg
import numpy as np
np.random.seed(0)
n = 1000
p = 9
x = np.random.rand(n, p)
y = np.repeat(1.-np.arange(0, 2, dtype=np.float64), n//2)
print(SpeedLogitReg().fit(x, y)._coef)
```
