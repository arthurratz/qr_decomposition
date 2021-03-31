# QR Decomposition
#### _QR Decomposition In Python 3.9 & Numpy 1.20.2_

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Leverage the QR Decomposition of large-sized **real** and **complex** matrices of an arbitrary shape with the variety of methods: **Gram-Schmidt Orthogonalization**, **Schwarz-Rutishauser Algorithm**, **Householder Reflections**, surveying its performance. 

Source codes in **Python 3.9.x (64-bit)** / **Intel® Distribution for Python 2021.1.1** using the latest **Numpy 1.20.2** library, **Microsoft Visual Studio 2019** Python's project:

## Usage:

To perform QR decomposition of a randomly generated matrix A of an arbitrary shape, download the project and run the following code in your Python-environment:

Please don't forget to import the following Py's-modules to your project, and define a `real` or `complex` matrix A, such as:

**example.py**:
```
import numpy as np
import numpy.linalg as lin

from qr_gschmidt import *
from qr_gs_schwrt import *
from qr_householder import *

# An example of a `real` matrix A of shape (5,4):

A = [[1.0,7.0,2.0,4.0],
     [2.0,3.0,6.0,8.0],
     [1.0,9.0,5.0,4.0],
     [2.0,6.0,8.0,3.0],
     [3.0,5.0,2.0,7.0]]

# An example of a `complex` matrix A of shape (5,4):

A = [[2.5+3.7j,7.0-2.2j,1.8-2.6j,4.1+1.8j],
     [6.4-2.1j,3.1+1.7j,3.3+1.4j,1.3+1.1j],
     [8.5-4.6j,9.2-6.4j,5.8+3.6j,4.6+2.7j],
     [2.0-3.2j,4.4-5.1j,8.1+1.1j,3.5+4.4j],
     [0.0+6.8j,5.0+8.9j,5.5+4.6j,7.1+6.8j]]
```

Use the following sub-routines below in your code to perform QR decomposition of a matrix A into an orthogonal matrix Q and upper triangular matrix R:

**QR Decomposition Using `Gram-Schmidt Orthogonalization`**
```
Q, R = qr_gram_schmidt(A)
```
**QR Decomposition Using `Schwarz-Rutishauser Algorithm`**
```
Q, R = qr_gram_schmidt_mod(A)
```
**QR Decomposition Using `Householder Reflections`**
```
Q, R = qr_householder(A)
```

## License

CPOL (C) 2021 Arthur V. Ratz

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
   [john gruber]: <http://daringfireball.net>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>
