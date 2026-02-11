import unittest
import tempfile
import os
import csv
import numpy as np

from DynamicsMDH import transform_inertial_to_mdh, save_dynamics_to_csv


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

    def test_save_dynamics_to_csv(self) -> None:
        results = {
            "link1": {
                "joint_name": "joint0",
                "joint_index": 0,
                "base_frame_name": "base_link",
                "mass": 2.0,
                "com_base": np.array([1.1, 1.2, 1.3], dtype=float),
                "inertia_base_about_com": np.array(
                    [
                        [4.0, 0.4, 0.5],
                        [0.4, 5.0, 0.6],
                        [0.5, 0.6, 6.0],
                    ],
                    dtype=float,
                ),
                "com": np.array([0.1, 0.2, 0.3], dtype=float),
                "inertia": np.array(
                    [
                        [1.0, 0.1, 0.2],
                        [0.1, 2.0, 0.3],
                        [0.2, 0.3, 3.0],
                    ],
                    dtype=float,
                ),
                "inertia_about": "com",
                "raw_urdf": {
                    "xyz": np.array([0.01, 0.02, 0.03], dtype=float),
                    "rpy": np.array([0.0, 0.0, 0.0], dtype=float),
                    "inertia_matrix_in_inertial": np.array(
                        [
                            [1.0, 0.0, 0.0],
                            [0.0, 2.0, 0.0],
                            [0.0, 0.0, 3.0],
                        ],
                        dtype=float,
                    ),
                },
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "dyn.csv")
            save_dynamics_to_csv(results, csv_path)

            self.assertTrue(os.path.exists(csv_path))
            with open(csv_path, "r", encoding="utf-8", newline="") as fp:
                rows = list(csv.DictReader(fp))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["link_name"], "link1")
            self.assertEqual(rows[0]["joint_name"], "joint0")
            self.assertEqual(rows[0]["base_frame_name"], "base_link")
            self.assertEqual(rows[0]["inertia_about"], "com")
            self.assertAlmostEqual(float(rows[0]["mass"]), 2.0)
            self.assertAlmostEqual(float(rows[0]["com_base_x"]), 1.1)
            self.assertAlmostEqual(float(rows[0]["I_base_zz"]), 6.0)
            self.assertAlmostEqual(float(rows[0]["com_x"]), 0.1)
            self.assertAlmostEqual(float(rows[0]["I_zz"]), 3.0)


if __name__ == "__main__":
    unittest.main()
