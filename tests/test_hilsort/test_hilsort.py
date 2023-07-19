#
# Copyright (c) 2023,
# All rights reserved.
#
# Contributors:
#    Yaser Afshar
#

# CORRECTNESS CHECKS FOR HILSORT.

import numpy as np
import unittest

from hilsort import *


square_grid_2_2 = ((0, 0), (1, 0), (1, 1), (0, 1))
square_grid_4_4 = (
    (0, 0),
    (0, 1),
    (1, 1),
    (1, 0),
    (2, 0),
    (3, 0),
    (3, 1),
    (2, 1),
    (2, 2),
    (3, 2),
    (3, 3),
    (2, 3),
    (1, 3),
    (1, 2),
    (0, 2),
    (0, 3),
)

total_test: int = 0
test_fails: int = 0


class HILSORTModule:
    """Test hilsort module components."""

    def test_hilbert(self):
        """A: HILBERT TEST"""

        def result(coord, old_coord, msg: str) -> bool:
            global total_test
            global test_fails

            total_test += 1

            miscount = np.sum(np.abs(coord.astype(int) - old_coord.astype(int)))

            if miscount != 1:
                test_fails += 1
                print(msg)
                print(f"\t: miscount -> coord {coord}, old coord {old_coord}")
                return False

            return True

        def cmp_result(expected, ans, msg: str) -> bool:
            global total_test
            global test_fails

            total_test += 1

            if expected != ans:
                test_fails += 1
                print(msg)
                print(f"\t: expected {expected}, ans {ans}")
                return False

            return True

        def index_result(r1, r2, msg: str) -> bool:
            global total_test
            global test_fails

            total_test += 1

            if r1 != r2:
                test_fails += 1
                print(msg)
                print(f"\t: c2i(i2c({r1})) != {r2} does not match!")
                return False

            return True

        nbits_list = [4, 8, 16, 32]
        ndims_list = list(range(2, 20))

        for nbits in nbits_list:
            for ndims in ndims_list:
                if ndims * nbits > 8 * np.dtype(np.uint64).itemsize:
                    continue

                old_coord = hilbert_i2c(ndims, nbits, 0)

                for r in range(1, 100):
                    coord = hilbert_i2c(ndims, nbits, r)
                    self.assertTrue(result(coord, old_coord, ""))

                    old_coord[:] = coord[:]

                    r1 = hilbert_c2i(nbits, coord)
                    self.assertTrue(index_result(r, r1, ""))

                coord = hilbert_i2c(ndims, nbits, 100)

                for r in range(1, 100):
                    coord1 = hilbert_i2c(ndims, nbits, r)
                    ans = hilbert_cmp(nbits, coord, coord1)
                    self.assertTrue(cmp_result(1, ans, ""))

                coord1 = hilbert_i2c(ndims, nbits, 100)
                ans = hilbert_cmp(nbits, coord, coord1)
                self.assertTrue(cmp_result(0, ans, ""))

                for r in range(101, 200):
                    coord1 = hilbert_i2c(ndims, nbits, r)
                    ans = hilbert_cmp(nbits, coord, coord1)
                    self.assertTrue(cmp_result(-1, ans, ""))

                coord = hilbert_i2c(ndims, nbits, 100)
                coord = coord.astype(np.int32)

                for r in range(1, 100):
                    coord1 = hilbert_i2c(ndims, nbits, r)
                    coord1 = coord1.astype(np.int32)
                    ans = hilbert_cmp(nbits, coord, coord1)
                    self.assertTrue(cmp_result(1, ans, ""))

                coord1 = hilbert_i2c(ndims, nbits, 100)
                coord1 = coord1.astype(np.int32)
                ans = hilbert_cmp(nbits, coord, coord1)
                self.assertTrue(cmp_result(0, ans, ""))

                for r in range(101, 200):
                    coord1 = hilbert_i2c(ndims, nbits, r)
                    coord1 = coord1.astype(np.int32)
                    ans = hilbert_cmp(nbits, coord, coord1)
                    self.assertTrue(cmp_result(-1, ans, ""))

                coord = hilbert_i2c(ndims, nbits, 100)
                coord = coord.astype(np.int64)

                for r in range(1, 100):
                    coord1 = hilbert_i2c(ndims, nbits, r)
                    coord1 = coord1.astype(np.int64)
                    ans = hilbert_cmp(nbits, coord, coord1)
                    self.assertTrue(cmp_result(1, ans, ""))

                coord1 = hilbert_i2c(ndims, nbits, 100)
                coord1 = coord1.astype(np.int64)
                ans = hilbert_cmp(nbits, coord, coord1)
                self.assertTrue(cmp_result(0, ans, ""))

                for r in range(101, 200):
                    coord1 = hilbert_i2c(ndims, nbits, r)
                    coord1 = coord1.astype(np.int64)
                    ans = hilbert_cmp(nbits, coord, coord1)
                    self.assertTrue(cmp_result(-1, ans, ""))

                old_coord = hilbert_i2c(ndims, nbits, 0)

                for r in range(1, 100):
                    coord = hilbert_i2c(ndims, nbits, r)
                    self.assertTrue(result(coord, old_coord, ""))

                    old_coord[:] = coord[:]

                    r1 = hilbert_c2i(nbits, coord)
                    self.assertTrue(index_result(r, r1, ""))

        nbits_list = [8, 16, 32]
        ndims_list = list(range(2, 20))

        for nbits in nbits_list:
            for ndims in ndims_list:
                if ndims * nbits > 8 * np.dtype(np.uint64).itemsize:
                    continue

                old_coord = hilbert_i2c(ndims, nbits, 999)

                for r in range(1000, 1100):
                    coord = hilbert_i2c(ndims, nbits, r)
                    self.assertTrue(result(coord, old_coord, ""))

                    old_coord[:] = coord[:]

                    r1 = hilbert_c2i(nbits, coord)
                    self.assertTrue(index_result(r, r1, ""))

                coord = hilbert_i2c(ndims, nbits, 1100)

                for r in range(1000, 1100):
                    coord1 = hilbert_i2c(ndims, nbits, r)
                    ans = hilbert_cmp(nbits, coord, coord1)
                    self.assertTrue(cmp_result(1, ans, ""))

                coord = hilbert_i2c(ndims, nbits, 1100)
                coord = coord.astype(np.int32)

                for r in range(1000, 1100):
                    coord1 = hilbert_i2c(ndims, nbits, r)
                    coord1 = coord1.astype(np.int32)
                    ans = hilbert_cmp(nbits, coord, coord1)
                    self.assertTrue(cmp_result(1, ans, ""))

    def test_hilbert_ieee(self):
        """A: HILBERT IEEE TEST"""

        def cmp_result(expected, ans, msg: str) -> bool:
            global total_test
            global test_fails

            total_test += 1

            if expected != ans:
                test_fails += 1
                print(msg)
                print(f"\t: expected {expected}, ans {ans}")
                return False

            return True

        ndims_list = list(range(2, 20))

        for ndims in ndims_list:
            coord = np.random.choice([-1, 1], size=[ndims]).astype(np.float64)
            ans = hilbert_ieee_cmp(coord, coord)
            self.assertTrue(cmp_result(0, ans, ""))

    def test_hilbert_vtx(self):
        """A: HILBERT VTX TEST"""

        def cmp_result(expected, ans, msg: str) -> bool:
            global total_test
            global test_fails

            total_test += 1

            if expected != ans:
                test_fails += 1
                print(msg)
                print(f"\t: expected {expected}, ans {ans}")
                return False

            return True

        first_index = 0
        nbits = 8
        ndims_list = list(range(2, 20))
        for ndims in ndims_list:
            if ndims * nbits > 8 * np.dtype(np.uint64).itemsize:
                continue

            last_index = np.power(2, ndims) - 1
            onelast_index = last_index - 1

            coord0 = hilbert_i2c(ndims, nbits, first_index)
            coord1 = hilbert_i2c(ndims, nbits, last_index)
            coord2 = hilbert_i2c(ndims, nbits, onelast_index)

            coord = hilbert_min_box_vtx(nbits, coord0, coord1)
            ans = hilbert_cmp(nbits, coord, coord0)
            self.assertTrue(cmp_result(0, ans, ""))

            coord = hilbert_max_box_vtx(nbits, coord0, coord1)
            ans = hilbert_cmp(nbits, coord, coord1)
            self.assertTrue(cmp_result(0, ans, ""))

            coord = hilbert_max_box_vtx(nbits, coord0, coord2)
            ans = hilbert_cmp(nbits, coord, coord1)
            self.assertTrue(cmp_result(0, ans, ""))

    def test_hilbert_ieee_vtx(self):
        """A: HILBERT VTX IEEE TEST"""

        ndims_list = list(range(2, 20))
        for ndims in ndims_list:
            coords = 2 * np.random.rand(2, ndims) - 1.0

            lo = hilbert_ieee_min_box_vtx(coords[0], coords[1])
            hi = hilbert_ieee_max_box_vtx(coords[0], coords[1])

            cornerlo = np.array(coords[0], copy=True)
            work = np.array(coords[1], copy=True)

            hilbert_ieee_box_vtx(1, cornerlo, work)
            self.assertTrue(np.allclose(cornerlo, lo))

            work = np.array(coords[0], copy=True)
            cornerhi = np.array(coords[1], copy=True)

            hilbert_ieee_box_vtx(0, work, cornerhi)
            self.assertTrue(np.allclose(cornerhi, hi))

    def test_hilbert_incr(self):
        """A: HILBERT INCR TEST"""

        nbits = 8
        ndims_list = list(range(2, 20))
        for ndims in ndims_list:
            if ndims * nbits > 8 * np.dtype(np.uint64).itemsize:
                continue

            last_index = np.power(2, ndims) - 1
            onelast_index = last_index - 1

            coord1 = hilbert_i2c(ndims, nbits, last_index)
            coord2 = hilbert_i2c(ndims, nbits, onelast_index)

            coord = hilbert_incr(nbits, coord2)

            self.assertTrue(np.allclose(coord, coord1))

    def test_hilbert_sort(self):
        """A: HILBERT SORT TEST"""
        global total_test
        global test_fails

        def result(array, hilbert_array, msg: str) -> bool:
            global total_test
            global test_fails

            total_test += 1

            array = np.array(array, copy=False)
            hilbert_array = np.array(hilbert_array, copy=False)

            if array.shape != hilbert_array.shape:
                print(msg)
                test_fails += 1
                print("Wrong input!")
                return False

            if not np.allclose(array, hilbert_array):
                print(msg)
                test_fails += 1
                for i in range(len(array)):
                    if not np.allclose(array[i], hilbert_array[i]):
                        print(f"-- ROW {i}")
                        print(f"   ANSWER: {array[i]}")
                        print(f"   CORRECT: {hilbert_array[i]}")
                return False

            return True

        print("\n")
        xv, yv = np.meshgrid(np.arange(2), np.arange(2), indexing="ij")
        data = np.vstack([xv.ravel(), yv.ravel()]).T
        data = np.ascontiguousarray(data)

        sorted_data = hilbert_sort(4, data)
        self.assertTrue(
            result(
                sorted_data,
                square_grid_2_2,
                "The basic element of a Hilbert curve. A 2 x 2 square grid with 4 bits/coordinate",
            )
        )

        sorted_data = hilbert_sort(4, square_grid_2_2)
        self.assertTrue(
            result(
                sorted_data,
                square_grid_2_2,
                "The basic element of a Hilbert curve. A 2 x 2 square grid with 4 bits/coordinate",
            )
        )

        sorted_data = hilbert_sort(8, data)
        self.assertTrue(
            result(
                sorted_data,
                square_grid_2_2,
                "The basic element of a Hilbert curve. A 2 x 2 square grid with 8 bits/coordinate",
            )
        )

        sorted_data = hilbert_sort(8, square_grid_2_2)
        self.assertTrue(
            result(
                sorted_data,
                square_grid_2_2,
                "The basic element of a Hilbert curve. A 2 x 2 square grid with 8 bits/coordinate",
            )
        )

        sorted_data = hilbert_sort(16, data)
        self.assertTrue(
            result(
                sorted_data,
                square_grid_2_2,
                "The basic element of a Hilbert curve. A 2 x 2 square grid with 16 bits/coordinate",
            )
        )

        sorted_data = hilbert_sort(16, square_grid_2_2)
        self.assertTrue(
            result(
                sorted_data,
                square_grid_2_2,
                "The basic element of a Hilbert curve. A 2 x 2 square grid with 16 bits/coordinate",
            )
        )

        xv, yv = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
        data = np.vstack([xv.ravel(), yv.ravel()]).T
        data = np.ascontiguousarray(data)

        sorted_data = hilbert_sort(4, data)
        self.assertTrue(
            result(
                sorted_data,
                square_grid_4_4,
                "A 4 x 4 square grid with 4 bits/coordinate",
            )
        )

        sorted_data = hilbert_sort(8, data)
        self.assertTrue(
            result(
                sorted_data,
                square_grid_4_4,
                "A 4 x 4 square grid with 8 bits/coordinate",
            )
        )

        sorted_data = hilbert_sort(16, data)
        self.assertTrue(
            result(
                sorted_data,
                square_grid_4_4,
                "A 4 x 4 square grid with 16 bits/coordinate",
            )
        )

        data = np.random.rand(10000, 3)
        sorted_data = hilbert_sort(8, data)
        sorted_data_3d = hilbert_sort_3d(data)

        self.assertTrue(np.allclose(sorted_data, sorted_data_3d))
        total_test += 1


class TestHILSORTModule(HILSORTModule, unittest.TestCase):
    @classmethod
    def tearDownClass(HILSORTModule):
        print("\nTotal number of tests = {}".format(total_test))
        print("{:8} tests failed".format(test_fails))
        print("{:8} tests passed successfuly.".format(total_test - test_fails))
        print("\nDONE\n")

    pass
