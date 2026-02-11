import unittest
import numpy as np

from DynamicsMDH import transform_inertial_to_mdh


class DynamicsMdhTests(unittest.TestCase):
    def test_identity_frame_preserves_com_and_inertia(self) -> None:
        mass = 1.0
        T_link_inertial = np.eye(4)
        I = np.array(
            [
                [0.1, 0.002, -0.001],
                [0.002, 0.2, 0.003],
                [-0.001, 0.003, 0.3],
            ],
            dtype=float,
        )
        out = transform_inertial_to_mdh(
            mass=mass,
            T_link_inertial=T_link_inertial,
            I_inertial_about_origin=I,
            T_base_link=np.eye(4),
            T_base_mdh_joint=np.eye(4),
            inertia_about="com",
        )

        self.assertTrue(np.allclose(out["r_J_com"], np.zeros(3), atol=1e-12))
        self.assertTrue(np.allclose(out["I_J"], I, atol=1e-12))


if __name__ == "__main__":
    unittest.main()
