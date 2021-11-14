"""Unit tests for the Special Linear group."""

import warnings

import geomstats.backend as gs
import geomstats.tests
import tests.helper as helper
from geomstats.geometry.special_linear import SpecialLinear, SpecialLinearLieAlgebra


class TestSpecialLinear(geomstats.tests.TestCase):
    def setUp(self):
        gs.random.seed(1234)
        self.n = 3
        self.n_samples = 2
        self.group = SpecialLinear(n=self.n)
        self.algebra = SpecialLinearLieAlgebra(n=self.n)

        warnings.simplefilter("ignore", category=ImportWarning)

    def test_belongs(self):
        mat = gs.eye(3)
        result = self.group.belongs(mat)
        expected = True
        self.assertAllClose(result,expected)

        mat = gs.ones(3)
        result = self.group.belongs(mat)
        expected = False
        self.assertAllClose(result, expected) 

        mat = gs.ones((3,3))
        result = self.group.belongs(mat)
        expected = False
        self.assertAllClose(result, expected)

    def test_belongs_vectorization(self):
        mats = gs.array([gs.eye(3), gs.ones((3, 3))])
        result = self.group.belongs(mats)
        expected = gs.array([True, False])
        self.assertAllClose(result, expected)

    def test_random_and_belongs(self):
        sample = self.group.random_point(n_samples = self.n_samples)
        result = self.group.belongs(sample)
        self.assertTrue(result)

    def test_projection_and_belongs(self):
        shape = (self.n_samples, self.n, self.n)
        result = helper.test_projection_and_belongs(self.group, shape)
        for res in result:
            self.assertTrue(res)

    def test_belongs_algebra(self):
        mat = gs.array([[3.,2.,1.],[5.,2.,5.],[3.,2.,-5.]])
        result = self.algebra.belongs(mat)
        expected = True
        self.assertAllClose(result,expected)

        mat = gs.ones(3)
        result = self.algebra.belongs(mat)
        expected = False
        self.assertAllClose(result, expected)

        mat = gs.ones((3,3))
        result = self.algebra.belongs(mat)
        expected = False
        self.assertAllClose(result, expected)

    def test_random_and_belongs_algebra(self):
        sample=self.algebra.random_point(n_samples= self.n_samples)
        result = self.algebra.belongs(sample)
        self.assertTrue(result)

    def test_projection_and_belongs_algebra(self):
        shape = (self.n_samples, self.n, self.n)
        result = helper.test_projection_and_belongs(self.algebra, shape)
        for res in result:
            self.assertTrue(res)
