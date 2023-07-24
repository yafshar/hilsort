# Hilbert-related calculations plus sorting points in Euclidean space using space-filling curves.

A space-filling curve is a curve whose trajectory traverses and covers every
point within a higher-dimensional region, often exemplified by the unit square
(or, in more general cases, an n-dimensional unit hypercube). These curves have
been heavily researched for their locality-preserving properties and are
especially useful for dimensionality reduction.

The current module is a Python binding for Doug Moore's Fast Hilbert Curve
Generation plus extra functionality to use the methods for sorting points in
Euclidean space using space-filling curves. The number of nbits bits determines
the precision of the curve, and the algorithm work under the constraint
`ndims * nbits <= 64`.

## Fast Hilbert Curve Generation, Sorting, and Range Queries

Text from the [original web page](http://www.tiac.net/~sw/2008/10/Hilbert/moore/):

```
## Fast Hilbert Curve Generation, Sorting, and Range Queries

Hilbert curves are one of a class of space filling curves - continuous, nonsmooth
curves that pass arbitrarily close to every point in space of arbitrary dimension
- and much information about planar Hilbert curves is on the web - see here for
example - but little information on efficient generation of Hilbert curves and
related activities like Hilbert sorting, Hilbert range queries, and so on, is
available. The common recursive generation of planar Hilbert curves does not easily
lead to fast nonrecursive algorithms for related problems, nor to higher-dimensional
generalizations. A rather opaque description of a nonrecursive algorithm by Butz 
("Alternative Algorithm for Hilbert's Space-Filling Curve", IEEE Trans. Comp.,
April, 1971, pp 424-426) led to an implementation by Spencer Thomas that appeared
in Graphics Gems and has been widely circulated on the net.

I began with his implementation, and began rearranging it to eliminate the need
for various precomputed tables. In the process, I found that the basic
index-to-point and point-to-index calculations could be made quite terse, without
any supplementary tables. I extended those algorithms to allow the comparison of
two points to see which falls first on a Hilbert curve. Although this comparison
can be achieved by converting each point to its integer index and comparing
integers, there can be a problem with regard to the size of the resulting integers
overflowing conventional integer types. My implementation of the comparison is
less sensitive to this problem. Further extensions led to algorithms for finding
the first vertex of a box to appear on the Hilbert curve, finding the first point
(not necessarily a vertex) of a box to lie on the curve, and for finding the
first point after a given point to lie in the box.

The purpose of this work is to enable efficient multidimensional searches and to
associate spatial locality in searches with spatial locality in the ordering of
items in a B-tree based database. With the ability to find the first point in a
box after a given point, it is possible to enumerate all the points in the
database within a box without examining every point in the database. Likewise, it
would be possible to compute efficiently "spatial joins" - that is, all pairs of
points that both lie in some small box.

At this point, I can explain the algorithms best by offering the source code, so
here is the C source and here is the corresponding header file.

Please let me know if you find this implementation of nonrecursive multidimensional
Hilbert curve methods useful or entertaining.

Doug Moore (dougm@...edu)
```

A spatial sorting algorithm in this work, sorts points according to the order in
which they would be visited by a space-filling curve.

## Motivation for the sorting algorithm

When trying to store/pack up spatial data efficiently or find nearest neighbors
of spatial data. We need a way of sorting the data in such a way that (p1, p2)
data located in nearby regions will live closer in memory to one another. This
approach speeds up access to the data in memory.

## Usage

### Convert an index into a Hilbert curve to a set of coordinates 

using `hilbert_i2c(ndims, nbits, index) -> np.ndarray`, where

```sh
ndims: Number of coordinate axes
nbits: Number of bits/coordinate
index: The index, contains ndims*nbits bits (so ndims*nbits must be <= 64)
```

the returned coordinates have the `dtype=np.uint64` data type.

E.g.,

```py
from hilsort import *

coord = hilbert_i2c(4, 8, 10)  # [1, 1, 1, 1]
```

### Convert coordinates of a point on a Hilbert curve to its index

using `hilbert_c2i(nbits, coord) -> int`, where 

```sh
nbits: Number of bits/coordinate.
coord: Numpy array of n nbits-bit coordinates. shape (ndims)
```

The input coordinates must have the `dtype=np.int64` or `dtype=np.uint64`
data type, otherwise `TypeError` for incompatible function arguments will raise.

E.g.,

```py
from hilsort import *

coord = hilbert_i2c(4, 8, 10)  # coord [1, 1, 1, 1]
index = hilbert_c2i(8, coord)  # index 10
```
### Determine which of two points lies further along the Hilbert curve

using `hilbert_cmp(nbits, coord1, coord2) -> int`, where 

```sh
nbits:  Number of bits/coordinate.
coord1: Numpy array of n nbits-bit coordinates. shape (ndims)
coord2: Numpy array of n nbits-bit coordinates. shape (ndims)  
```

The function will return any of `-1`, `0`, or `1` according to whether 
`coord1<coord2`, `coord1==coord2`, or `coord1>coord2`.

The input coordinates must have the `int` data type (any of 
`np.int32, np.int64, np.unint32, np.uint64`).

E.g.,

```py
from hilsort import *

coord1 = hilbert_i2c(4, 8, 10)        # coord1 [1, 1, 1, 1]
coord2 = hilbert_i2c(4, 8, 10)        # coord2 [1, 1, 1, 1]
cmp = hilbert_cmp(8, coord1, coord2)  # cmp 0 (coord1==coord2)

coord2 = hilbert_i2c(4, 8, 11)        # coord2 [1, 1, 1, 0]
cmp = hilbert_cmp(8, coord1, coord2)  # cmp -1 (coord1<coord2)

coord2 = hilbert_i2c(4, 8, 5)         # coord2 [1, 1, 0, 1]
cmp = hilbert_cmp(8, coord1, coord2)  # cmp 1 (coord1>coord2)
```

If one wants to compare two points with floating data type, they can use
`hilbert_ieee_cmp(coord1, coord2) -> int`, where 

```sh
coord1: Numpy array coordinates. shape (ndims)
coord2: Numpy array coordinates. shape (ndims)  
```

E.g.,

```py
import numpy as np
from hilsort import *

coord1 = 2 * np.random.rand(2) - 1.0  # coord1 [0.45893551, 0.02698878]
cmp = hilbert_ieee_cmp(coord1, coord1)  # cmp 0 (coord1==coord1)

coord2 = 2 * np.random.rand(2) - 1.0  # coord2 [0.41206415, 0.7392873]

cmp = hilbert_ieee_cmp(8, coord1, coord2)  # cmp -1 (coord1<coord2)
```

### Determine the first/last vertex of a box to lie on a Hilbert curve

To find the first/last vertex of a box to appear on the Hilbert curve using 

`hilbert_min_box_vtx(nbits, coord1, coord2) -> np.ndarray`,  
`hilbert_max_box_vtx(nbits, coord1, coord2) -> np.ndarray`, where

```sh
nbits:  Number of bits/coordinate.
coord1: Array of ndims nbytes-byte coordinates - one corner of box
coord2: Array of ndims nbytes-byte coordinates - opposite corner
```

The input coordinates must have the `int` data type (any of 
`np.int32, np.int64, np.unint32, np.uint64`).

or if the input coordinates have the `double` data type

`hilbert_ieee_min_box_vtx(coord1, coord2) -> np.ndarray`,  
`hilbert_ieee_max_box_vtx(coord1, coord2) -> np.ndarray`, where

```sh
coord1: Array of ndims double coordinates - one corner of box
coord2: Array of ndims double coordinates - opposite corner
```

### Determine the first/last point of a box to lie on a Hilbert curve

To find the first/last point (not necessarily a vertex) of a box to lie on the
curve, using 

`hilbert_min_box_pt(nbits, coord1, coord2) -> np.ndarray`,  
`hilbert_max_box_pt(nbits, coord1, coord2) -> np.ndarray`, where

```sh
nbits:  Number of bits/coordinate.
coord1: Array of ndims nbytes-byte coordinates - one corner of box
coord2: Array of ndims nbytes-byte coordinates - opposite corner
```

The input coordinates must have the `int` data type (any of 
`np.int32, np.int64, np.unint32, np.uint64`).

or if the input coordinates have the `double` data type

`hilbert_ieee_min_box_pt(coord1, coord2) -> np.ndarray`,  
`hilbert_ieee_max_box_pt(coord1, coord2) -> np.ndarray`, where

```sh
coord1: Array of ndims double coordinates - one corner of box
coord2: Array of ndims double coordinates - opposite corner
```
 
### Finding the first point after a given point to lie in the box

To find the first point after a given point to lie in the box using

`hilbert_nextinbox(nbits, find_prev, coord1, coord2, point) -> int`, where

```sh
nbits:     Number of bits/coordinate.
find_prev: Is the previous point sought?
coord1:    Array of ndims nbytes-byte coordinates - one corner of box
coord2:    Array of ndims nbytes-byte coordinates - opposite corner
point:     Array of ndims nbytes-byte coordinates - lower bound on point returned
```

The function will return `0` or `1`. If it returns `1`, then `coord1` and `coord2`
modified to refer to least point after `point` in the box. If it returns `0` then
there is no arguments change and the input `point` is beyond the last point of
the box.

The input coordinates, and point must have the `int` data type (any of 
`np.int32, np.int64, np.unint32, np.uint64`).

### Advance from one point to its successor on a Hilbert curve 

using `hilbert_incr(nbits, coord) -> np.ndarray`.

E.g.,

```py
from hilsort import *

coord = hilbert_i2c(2, 4, 0)    # [0, 0]

coord = hilbert_incr(4, coord)  # [0, 1]
coord = hilbert_incr(4, coord)  # [1, 1]
coord = hilbert_incr(4, coord)  # [1, 0]
```

### Sorting spatial data

Sorting 3D data using both `hilbert_sort(nbits, data)` and
`hilbert_sort_3d(data)`.

E.g.,

```py
import numpy as np
from hilsort import *

data = np.random.rand(10000, 3)

sorted_data = hilbert_sort(8, data)

sorted_data_3d = hilbert_sort_3d(data)
```

The `hilbert_sort(nbits, data)` interface is used to sort ndims dimensional data
points with  `(N, ndims)` shape based on a Hilbert curve using nbits number of
bits/coordinate. The number of bits nbits determines the precision of the curve
and the algorithm works under the constraint of  `ndims * nbits <= 64`.

The `hilbert_sort_3d(data)` interface is used to sort 3D data points with 
`(N, 3)` shape based on a Hilbert curve using `8` number of bits/coordinate.

E.g.,

```py
import numpy as np
from hilsort import *

xv, yv = np.meshgrid(np.arange(2), np.arange(2), indexing="ij")
data = np.vstack([xv.ravel(), yv.ravel()]).T
data = np.ascontiguousarray(data)

# >>> print(data)
# [[0 0]
#  [0 1]
#  [1 0]
#  [1 1]]

sorted_data = hilbert_sort(4, data)

# >>> print(sorted_data)
# [[0 0]
#  [1 0]
#  [1 1]
#  [0 1]]
```

## Installing hilsort

### Python requirements

You need Python 3.8 or later to run `hilsort`. You can have multiple Python
versions (2.x and 3.x) installed on the same system without problems.

To install Python 3 for different Linux flavors, macOS and Windows, packages
are available at\
[https://www.python.org/getit/](https://www.python.org/getit/)

### Using pip

[![PyPI](https://img.shields.io/pypi/v/hilsort.svg)](https://pypi.python.org/pypi/hilsort)

**pip** is the most popular tool for installing Python packages, and the one
included with modern versions of Python.

`hilsort` can be installed with `pip`:

```sh
pip install hilsort
```

**Note:**

Depending on your Python installation, you may need to use `pip3` instead of
`pip`.

```sh
pip3 install hilsort
```

Depending on your configuration, you may have to run `pip` like this:

```sh
python3 -m pip install hilsort
```

### Using pip (GIT Support)

`pip` currently supports cloning over `git`

```sh
pip install git+https://github.com/yafshar/hilsort.git
```

For more information and examples, see the
[pip install](https://pip.pypa.io/en/stable/reference/pip_install/#id18)
reference.

### Using conda

[![Anaconda-Server Badge](https://img.shields.io/conda/vn/conda-forge/hilsort.svg)](https://anaconda.org/conda-forge/hilsort)
[![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/hilsort.svg)](https://anaconda.org/conda-forge/hilsort)

**conda** is the package management tool for Anaconda Python installations.

Installing `hilsort` from the `conda-forge` channel can be achieved by
adding `conda-forge` to your channels with:

```sh
conda config --add channels conda-forge
```

Once the `conda-forge` channel has been enabled, `hilsort` can be
installed with:

```sh
conda install hilsort
```

It is possible to list all of the versions of `hilsort` available on your platform
with:

```sh
conda search hilsort --channel conda-forge
```

## References

<a name="more_1998"></a>

1. Moore, Doug. "Fast Hilbert Curve Generation, Sorting, and Range Queries,"
[C source](http://www.tiac.net/~sw/2008/10/Hilbert/moore/) (1998)

## Contributing

Copyright (c) 2023.\
All rights reserved.

Contributors:\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Yaser Afshar

## License

This source code is available to everyone under the standard 
[LGPLv2.1](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html) 
license.

Hilbert Curve implementation copyright 1998, Rice University

```cpp
/* LICENSE
 *
 * This software is copyrighted by Rice University.  It may be freely copied,
 * modified, and redistributed, provided that the copyright notice is
 * preserved on all copies.
 *
 * There is no warranty or other guarantee of fitness for this software,
 * it is provided solely "as is".  Bug reports or fixes may be sent
 * to the author, who may or may not act on them as he desires.
 *
 * You may include this software in a program or other software product,
 * but must display the notice:
 *
 * Hilbert Curve implementation copyright 1998, Rice University
 *
 * in any place where the end-user would see your own copyright.
 *
 * If you modify this software, you should include a notice giving the
 * name of the person performing the modification, the date of modification,
 * and the reason for such modification.
 */
```

