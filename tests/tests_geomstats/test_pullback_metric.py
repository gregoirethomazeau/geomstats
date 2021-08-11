"""Unit tests for the pull-back metrics."""

import warnings

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.pullback_metric import PullbackMetric


class TestPullbackMetric(geomstats.tests.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=UserWarning)
        gs.random.seed(0)
        self.dim = 2
        self.sphere = Hypersphere(dim=self.dim)
        self.sphere_metric = self.sphere.metric

        def _sphere_immersion(spherical_coords):
            theta = spherical_coords[..., 0]
            phi = spherical_coords[..., 1]
            return gs.array([
                gs.cos(phi) * gs.sin(theta), 
                gs.sin(phi) * gs.sin(theta),
                gs.cos(theta)])

        self.immersion = _sphere_immersion
        self.pullback_metric = PullbackMetric(
            dim=self.dim,
            embedding_dim=self.dim + 1,
            immersion=self.immersion
        )

    def test_immersion(self):
        expected = gs.array([0., 0., 1.])
        result = self.immersion(gs.array([0., 0.]))
        self.assertAllClose(result, expected)

        expected = gs.array([0., 0., -1.])
        result = self.immersion(gs.array([gs.pi, 0.]))
        self.assertAllClose(result, expected)

        expected = gs.array([-1., 0., 0.])
        result = self.immersion(gs.array([gs.pi / 2., gs.pi]))
        self.assertAllClose(result, expected)

    def test_immersion_and_spherical_to_extrinsic(self):
        point = gs.array([0., 0.])
        expected = self.immersion(point)
        result = self.sphere.spherical_to_extrinsic(point)
        self.assertAllClose(result, expected)
        
        point = gs.array([0.2, 0.1])
        expected = self.immersion(point)
        result = self.sphere.spherical_to_extrinsic(point)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_jacobian_immersion(self):
        def _expected_jacobian_immersion(point):
            theta = point[..., 0]
            phi = point[..., 1]
            jacobian = gs.array([
                [gs.cos(phi) * gs.cos(theta), - gs.sin(phi) * gs.sin(theta)],
                [gs.sin(phi) * gs.cos(theta), gs.cos(phi) * gs.sin(theta)],
                [-gs.sin(theta), 0.]
            ])
            return jacobian

        pole = gs.array([0., 0.])
        result = self.pullback_metric.jacobian_immersion(pole)
        expected = _expected_jacobian_immersion(pole)
        self.assertAllClose(result, expected)
        
        base_point = gs.array([0.22, 0.1])
        result = self.pullback_metric.jacobian_immersion(base_point)
        expected = _expected_jacobian_immersion(base_point)
        self.assertAllClose(result, expected)
        
        base_point = gs.array([0.1, 0.88])
        result = self.pullback_metric.jacobian_immersion(base_point)
        expected = _expected_jacobian_immersion(base_point)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_tangent_immersion(self):
        point = gs.array([gs.pi / 2., gs.pi / 2.])

        tangent_vec = gs.array([1., 0.])
        result = self.pullback_metric.tangent_immersion(
            tangent_vec, point)
        expected = gs.array([0., 0., -1.])
        self.assertAllClose(result, expected)

        tangent_vec = gs.array([0., 1.])
        result = self.pullback_metric.tangent_immersion(
            tangent_vec, point)
        expected = gs.array([-1., 0., 0.])
        self.assertAllClose(result, expected)
        
        point = gs.array([gs.pi / 2., 0.])
        
        tangent_vec = gs.array([1., 0.])
        result = self.pullback_metric.tangent_immersion(
            tangent_vec, point)
        expected = gs.array([0., 0., -1.])
        self.assertAllClose(result, expected)
        
        tangent_vec = gs.array([0., 1.])
        result = self.pullback_metric.tangent_immersion(
            tangent_vec, point)
        expected = gs.array([0., 1., 0.])
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_metric_matrix(self):
        def _expected_metric_matrix(point):
            theta = point[..., 0]
            mat = gs.array([
                [1., 0.],
                [0., gs.sin(theta) ** 2]
            ])
            return mat

        base_point = gs.array([0., 0.])
        result = self.pullback_metric.metric_matrix(base_point)
        expected = _expected_metric_matrix(base_point)
        self.assertAllClose(result, expected)
        
        base_point = gs.array([1., 1.])
        result = self.pullback_metric.metric_matrix(base_point)
        expected = _expected_metric_matrix(base_point)
        self.assertAllClose(result, expected)
        
        base_point = gs.array([0.3, 0.8])
        result = self.pullback_metric.metric_matrix(base_point)
        expected = _expected_metric_matrix(base_point)
        self.assertAllClose(result, expected)


    @geomstats.tests.np_only
    def test_inverse_metric_matrix(self):
        def _expected_inverse_metric_matrix(point):
            theta = point[..., 0]
            mat = gs.array([
                [1., 0.],
                [0., gs.sin(theta) ** (-2)]
            ])
            return mat
        
        base_point = gs.array([.6, -1.])
        result = self.pullback_metric.inner_product_inverse_matrix(
            base_point)
        expected = _expected_inverse_metric_matrix(base_point)
        self.assertAllClose(result, expected)
        
        base_point = gs.array([0.8, -0.8])
        result = self.pullback_metric.inner_product_inverse_matrix(
            base_point)
        expected = _expected_inverse_metric_matrix(base_point)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_inner_product_and_sphere_inner_product(self):
        """Test consistency between sphere's inner-products.

        The inner-product of the class Hypersphere is
        defined in terms of extrinsic coordinates.

        The inner-product of pullback_metric is defined in terms
        of the spherical coordinates.
        """
        tangent_vec_a = gs.array([0., 1.])
        tangent_vec_b = gs.array([0., 1.])
        base_point = gs.array([gs.pi / 2., 0.])
        immersed_base_point = self.immersion(base_point)
        jac_immersion = self.pullback_metric.jacobian_immersion(
            base_point)
        immersed_tangent_vec_a = gs.matmul(
            jac_immersion, tangent_vec_a)
        immersed_tangent_vec_b = gs.matmul(
            jac_immersion, tangent_vec_b)

        result = self.pullback_metric.inner_product(
            tangent_vec_a, 
            tangent_vec_b, 
            base_point=base_point)
        expected = self.sphere_metric.inner_product(
            immersed_tangent_vec_a, 
            immersed_tangent_vec_b, 
            base_point=immersed_base_point)
        self.assertAllClose(result, expected)
        
        tangent_vec_a = gs.array([0.4, 1.])
        tangent_vec_b = gs.array([0.2, 0.6])
        base_point = gs.array([gs.pi / 2., 0.1])
        immersed_base_point = self.immersion(base_point)
        jac_immersion = self.pullback_metric.jacobian_immersion(
            base_point)
        immersed_tangent_vec_a = gs.matmul(
            jac_immersion, tangent_vec_a)
        immersed_tangent_vec_b = gs.matmul(
            jac_immersion, tangent_vec_b)

        result = self.pullback_metric.inner_product(
            tangent_vec_a, 
            tangent_vec_b, 
            base_point=base_point)
        expected = self.sphere_metric.inner_product(
            immersed_tangent_vec_a, 
            immersed_tangent_vec_b, 
            base_point=immersed_base_point)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_christoffels_and_sphere_christoffels(self):
        """Test consistency between sphere's christoffels.

        The christoffels of the class Hypersphere are
        defined in terms of spherical coordinates.

        The christoffels of pullback_metric are also defined
        in terms of the spherical coordinates.
        """
        base_point = gs.array([0.1, 0.2])
        result = self.pullback_metric.christoffels(base_point=base_point)
        expected = self.sphere_metric.christoffels(point=base_point)
        self.assertAllClose(result, expected)

        
        base_point = gs.array([0.1, 0.233])
        result = self.pullback_metric.christoffels(base_point=base_point)
        expected = self.sphere_metric.christoffels(point=base_point)
        self.assertAllClose(result, expected)


    @geomstats.tests.np_only
    def test_exp_and_sphere_exp(self):
        """Test consistency between sphere's Riemannian exp.

        The exp map of the class Hypersphere is
        defined in terms of extrinsic coordinates.

        The exp map of pullback_metric is defined
        in terms of the spherical coordinates.
        """
        base_point = gs.array([gs.pi / 2., 0.])
        tangent_vec_a = gs.array([0., 1.])
        immersed_base_point = self.immersion(base_point)
        jac_immersion = self.pullback_metric.jacobian_immersion(
            base_point)
        immersed_tangent_vec_a = gs.matmul(jac_immersion, tangent_vec_a)
        result = self.pullback_metric.exp(
            tangent_vec_a, 
            base_point=base_point)
        result = self.sphere.spherical_to_extrinsic(result)
        expected = self.sphere.metric.exp(
            immersed_tangent_vec_a, 
            base_point=immersed_base_point)
        self.assertAllClose(result, expected)
        
        
        tangent_vec_a = gs.array([0.4, 1.])
        base_point = gs.array([gs.pi / 2., 0.1])
        immersed_base_point = self.immersion(base_point)
        jac_immersion = self.pullback_metric.jacobian_immersion(
            base_point)
        immersed_tangent_vec_a = gs.matmul(jac_immersion, tangent_vec_a)
        result = self.pullback_metric.exp(
            tangent_vec_a, 
            base_point=base_point)
        result = self.sphere.spherical_to_extrinsic(result)
        expected = self.sphere.metric.exp(
            immersed_tangent_vec_a, 
            base_point=immersed_base_point)

        self.assertAllClose(result, expected, atol=1e-1)
    

     
    # def test_cometric_matrix(self):
    #     base_point = gs.array([0., 1.])

    #     result = self.euc_metric.inner_product_inverse_matrix(base_point)
    #     expected = gs.eye(self.dim)

    #     self.assertAllClose(result, expected)

    # @geomstats.tests.np_only
    # def test_metric_derivative_euc_metric(self):
    #     base_point = gs.array([5., 1.])

    #     result = self.euc_metric.inner_product_derivative_matrix(base_point)
    #     expected = gs.zeros((self.dim,) * 3)

    #     self.assertAllClose(result, expected)

    # @geomstats.tests.np_only
    # def test_metric_derivative_new_euc_metric(self):
    #     base_point = gs.array([5., 1.])

    #     result = self.new_euc_metric.inner_product_derivative_matrix(
    #         base_point)
    #     expected = gs.zeros((self.dim,) * 3)

    #     self.assertAllClose(result, expected)

    # def test_inner_product_new_euc_metric(self):
    #     base_point = gs.array([0., 1.])
    #     tangent_vec_a = gs.array([0.3, 0.4])
    #     tangent_vec_b = gs.array([0.1, -0.5])
    #     expected = -0.17

    #     result = self.new_euc_metric.inner_product(
    #         tangent_vec_a, tangent_vec_b, base_point=base_point
    #     )

    #     self.assertAllClose(result, expected)

    # def test_inner_product_new_sphere_metric(self):
    #     base_point = gs.array([gs.pi / 3., gs.pi / 5.])
    #     tangent_vec_a = gs.array([0.3, 0.4])
    #     tangent_vec_b = gs.array([0.1, -0.5])
    #     expected = -0.12

    #     result = self.new_sphere_metric.inner_product(
    #         tangent_vec_a, tangent_vec_b, base_point=base_point
    #     )

    #     self.assertAllClose(result, expected)

    # @geomstats.tests.np_only
    # def test_christoffels_eucl_metric(self):
    #     base_point = gs.array([0.2, -.9])

    #     result = self.euc_metric.christoffels(base_point)
    #     expected = gs.zeros((self.dim,) * 3)

    #     self.assertAllClose(result, expected)

    # @geomstats.tests.np_only
    # def test_christoffels_new_eucl_metric(self):
    #     base_point = gs.array([0.2, -.9])

    #     result = self.new_euc_metric.christoffels(base_point)
    #     expected = gs.zeros((self.dim,) * 3)

    #     self.assertAllClose(result, expected)

    # @geomstats.tests.np_only
    # def test_christoffels_sphere_metrics(self):
    #     base_point = gs.array([0.3, -.7])

    #     expected = self.sphere_metric.christoffels(base_point)
    #     result = self.new_sphere_metric.christoffels(base_point)

    #     self.assertAllClose(result, expected)

    # @geomstats.tests.np_only
    # def test_exp_new_eucl_metric(self):
    #     base_point = gs.array([7.2, -8.9])
    #     tan = gs.array([-1., 4.5])

    #     expected = base_point + tan
    #     result = self.new_euc_metric.exp(tan, base_point)
    #     self.assertAllClose(result, expected)

    # @geomstats.tests.np_only
    # def test_log_new_eucl_metric(self):
    #     base_point = gs.array([7.2, -8.9])
    #     point = gs.array([-3., 1.2])

    #     expected = point - base_point
    #     result = self.new_euc_metric.log(point, base_point)
    #     self.assertAllClose(result, expected)

    # @geomstats.tests.np_only
    # def test_exp_new_sphere_metric(self):
    #     base_point = gs.array([gs.pi / 10., gs. pi / 10.])
    #     tan = gs.array([gs.pi / 2., 0.])

    #     expected = gs.array([gs.pi / 10. + gs.pi / 2., gs.pi / 10.])
    #     result = self.new_sphere_metric.exp(tan, base_point)
    #     self.assertAllClose(result, expected)