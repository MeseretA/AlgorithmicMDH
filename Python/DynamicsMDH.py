from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import xml.etree.ElementTree as ET

import numpy as np

try:
    from urdfpy import URDF  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    URDF = None

from ForwardKinematics import ForwardKinematics


ArrayLike = Union[np.ndarray, Sequence[float]]
TransformMapInput = Union[Sequence[np.ndarray], Mapping[Union[int, str], np.ndarray]]


@dataclass(frozen=True)
class ParsedInertial:
    """Raw inertial values parsed from URDF.

    URDF units are assumed SI: mass [kg], distances [m], inertia [kg*m^2].
    """

    link_name: str
    mass: float
    xyz: np.ndarray
    rpy: np.ndarray
    T_link_inertial: np.ndarray
    I_inertial: np.ndarray


@dataclass(frozen=True)
class JointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    axis: np.ndarray
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray


def _as_float_vec(text: Optional[str], size: int, default: float = 0.0) -> np.ndarray:
    if text is None:
        return np.full((size,), float(default), dtype=float)
    vals = [float(x) for x in text.strip().split() if x]
    if len(vals) != size:
        raise ValueError(f"Expected {size} values, got {len(vals)} from '{text}'")
    return np.asarray(vals, dtype=float)


def rpy_to_rotation_matrix(rpy: ArrayLike) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw to rotation matrix.

    Convention used here: R = Rz(yaw) @ Ry(pitch) @ Rx(roll), matching URDF/ROS fixed-axis XYZ RPY.
    """

    roll, pitch, yaw = [float(v) for v in rpy]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz @ ry @ rx


def make_transform(xyz: ArrayLike, rpy: ArrayLike) -> np.ndarray:
    t = np.eye(4, dtype=float)
    t[:3, :3] = rpy_to_rotation_matrix(rpy)
    t[:3, 3] = np.asarray(xyz, dtype=float).reshape(3)
    return t


def _parse_joint_specs(root: ET.Element) -> List[JointSpec]:
    joints: List[JointSpec] = []
    for joint_elem in root.findall("joint"):
        joint_type = joint_elem.attrib.get("type", "fixed")
        parent_elem = joint_elem.find("parent")
        child_elem = joint_elem.find("child")
        if parent_elem is None or child_elem is None:
            continue

        origin_elem = joint_elem.find("origin")
        xyz = _as_float_vec(origin_elem.attrib.get("xyz") if origin_elem is not None else None, 3, default=0.0)
        rpy = _as_float_vec(origin_elem.attrib.get("rpy") if origin_elem is not None else None, 3, default=0.0)

        axis_elem = joint_elem.find("axis")
        axis = _as_float_vec(axis_elem.attrib.get("xyz") if axis_elem is not None else None, 3, default=0.0)

        joints.append(
            JointSpec(
                name=joint_elem.attrib["name"],
                joint_type=joint_type,
                parent=parent_elem.attrib["link"],
                child=child_elem.attrib["link"],
                axis=axis,
                origin_xyz=xyz,
                origin_rpy=rpy,
            )
        )
    return joints


def _collect_link_names(root: ET.Element) -> List[str]:
    return [elem.attrib["name"] for elem in root.findall("link") if "name" in elem.attrib]


def parse_urdf_inertials(urdf_path: str) -> Dict[str, ParsedInertial]:
    """Parse link inertial data from URDF.

    Returns only links that contain an `<inertial>` element.
    """

    root = ET.parse(urdf_path).getroot()
    inertials: Dict[str, ParsedInertial] = {}

    for link_elem in root.findall("link"):
        link_name = link_elem.attrib.get("name")
        if not link_name:
            continue
        inertial_elem = link_elem.find("inertial")
        if inertial_elem is None:
            continue

        mass_elem = inertial_elem.find("mass")
        inertia_elem = inertial_elem.find("inertia")
        if mass_elem is None or inertia_elem is None:
            raise ValueError(f"Link '{link_name}' inertial is missing mass or inertia sub-elements")

        mass = float(mass_elem.attrib["value"])

        origin_elem = inertial_elem.find("origin")
        xyz = _as_float_vec(origin_elem.attrib.get("xyz") if origin_elem is not None else None, 3, default=0.0)
        rpy = _as_float_vec(origin_elem.attrib.get("rpy") if origin_elem is not None else None, 3, default=0.0)

        ixx = float(inertia_elem.attrib["ixx"])
        ixy = float(inertia_elem.attrib["ixy"])
        ixz = float(inertia_elem.attrib["ixz"])
        iyy = float(inertia_elem.attrib["iyy"])
        iyz = float(inertia_elem.attrib["iyz"])
        izz = float(inertia_elem.attrib["izz"])
        I = np.array(
            [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]],
            dtype=float,
        )

        inertials[link_name] = ParsedInertial(
            link_name=link_name,
            mass=mass,
            xyz=xyz,
            rpy=rpy,
            T_link_inertial=make_transform(xyz, rpy),
            I_inertial=I,
        )

    return inertials


def _infer_root_link(link_names: Sequence[str], joints: Sequence[JointSpec]) -> str:
    child_names = {j.child for j in joints}
    roots = [name for name in link_names if name not in child_names]
    if not roots:
        raise ValueError("Could not infer URDF root/base link")
    return roots[0]


def _compute_T_root_link_xml(urdf_path: str) -> Tuple[Dict[str, np.ndarray], str, List[JointSpec]]:
    root = ET.parse(urdf_path).getroot()
    joints = _parse_joint_specs(root)
    link_names = _collect_link_names(root)
    root_link = _infer_root_link(link_names, joints)

    parent_to_children: Dict[str, List[JointSpec]] = {}
    for joint in joints:
        parent_to_children.setdefault(joint.parent, []).append(joint)

    T_root_link: Dict[str, np.ndarray] = {root_link: np.eye(4, dtype=float)}
    queue: List[str] = [root_link]

    while queue:
        parent = queue.pop(0)
        T_root_parent = T_root_link[parent]
        for joint in parent_to_children.get(parent, []):
            # At zero joint configuration, parent->child equals the URDF origin transform.
            T_parent_child = make_transform(joint.origin_xyz, joint.origin_rpy)
            T_root_link[joint.child] = T_root_parent @ T_parent_child
            queue.append(joint.child)

    return T_root_link, root_link, joints


def _compute_joint_specs_from_urdfpy(robot: Any) -> List[JointSpec]:
    joint_specs: List[JointSpec] = []
    for joint in robot.joints:
        origin = np.asarray(joint.origin, dtype=float)
        xyz = origin[:3, 3]
        R = origin[:3, :3]
        # Reconstruct rpy only for traceability (not used in computation from URDF object).
        pitch = np.arcsin(-R[2, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
        axis = np.asarray(joint.axis if joint.axis is not None else [0.0, 0.0, 0.0], dtype=float)
        joint_specs.append(
            JointSpec(
                name=joint.name,
                joint_type=joint.joint_type,
                parent=joint.parent,
                child=joint.child,
                axis=axis,
                origin_xyz=xyz,
                origin_rpy=np.array([roll, pitch, yaw], dtype=float),
            )
        )
    return joint_specs


def compute_T_base_link(
    urdf_path: str,
    base_link_name: str,
) -> Tuple[Dict[str, np.ndarray], List[JointSpec], List[JointSpec]]:
    """Compute base->link transforms at zero joint configuration.

    For consistency with existing mDH extraction, this prefers `urdfpy.link_fk()` when available.
    Fallback uses XML joint origins at zero configuration.

    Returns:
    - T_base_link for all links
    - actuated_joints ordered list
    - all_joints ordered list
    """

    if URDF is not None:
        robot = URDF.load(urdf_path)
        fk_world = robot.link_fk()
        by_name_world: Dict[str, np.ndarray] = {link.name: np.asarray(T, dtype=float) for link, T in fk_world.items()}
        if base_link_name not in by_name_world:
            raise ValueError(f"base_link_name '{base_link_name}' not found in URDF links")
        T_world_base = by_name_world[base_link_name]
        T_base_world = np.linalg.inv(T_world_base)
        T_base_link = {name: T_base_world @ T_world for name, T_world in by_name_world.items()}

        actuated = [
            JointSpec(
                name=j.name,
                joint_type=j.joint_type,
                parent=j.parent,
                child=j.child,
                axis=np.asarray(j.axis if j.axis is not None else [0.0, 0.0, 0.0], dtype=float),
                origin_xyz=np.asarray(j.origin[:3, 3], dtype=float),
                origin_rpy=np.zeros(3, dtype=float),
            )
            for j in robot.actuated_joints
        ]
        all_joints = _compute_joint_specs_from_urdfpy(robot)
        return T_base_link, actuated, all_joints

    T_root_link, root_link, all_joints = _compute_T_root_link_xml(urdf_path)
    if base_link_name not in T_root_link:
        raise ValueError(
            f"base_link_name '{base_link_name}' not found in URDF links (root link: '{root_link}')"
        )

    T_root_base = T_root_link[base_link_name]
    T_base_root = np.linalg.inv(T_root_base)
    T_base_link = {name: T_base_root @ T_root for name, T_root in T_root_link.items()}

    actuated = [j for j in all_joints if j.joint_type != "fixed"]
    return T_base_link, actuated, all_joints


def _normalize_mdh_frames(mdh_frames: TransformMapInput) -> Dict[Union[int, str], np.ndarray]:
    out: Dict[Union[int, str], np.ndarray] = {}
    if isinstance(mdh_frames, Mapping):
        items = mdh_frames.items()
    else:
        items = enumerate(mdh_frames)
    for key, value in items:
        T = np.asarray(value, dtype=float)
        if T.shape != (4, 4):
            raise ValueError(f"mDH frame transform for key '{key}' must be 4x4, got {T.shape}")
        out[key] = T
    return out


def get_T_base_mdh_frame(
    mdh: Any,
    joint_type_list: Sequence[int],
    T_base_mdh0: Optional[np.ndarray] = None,
    thetalist: Optional[Sequence[float]] = None,
) -> Dict[int, np.ndarray]:
    """Compute base->mDH frame transforms from existing MDH parameters.

    `joint_type_list`: 0 for revolute, 1 for prismatic.
    """

    if thetalist is None:
        thetalist = [0.0] * len(joint_type_list)
    fk = ForwardKinematics(mdh)
    T_list = fk.calc_T_list(thetalist, list(joint_type_list))

    if T_base_mdh0 is None:
        T_base_mdh0 = np.eye(4, dtype=float)
    T_base_mdh0 = np.asarray(T_base_mdh0, dtype=float)
    if T_base_mdh0.shape != (4, 4):
        raise ValueError("T_base_mdh0 must be a 4x4 transform")

    return {i: T_base_mdh0 @ T for i, T in enumerate(T_list)}


def _symmetrize_inertia(I: np.ndarray, atol: float = 1e-12) -> Tuple[np.ndarray, float]:
    asym = np.linalg.norm(I - I.T, ord=np.inf)
    I_sym = 0.5 * (I + I.T)
    if asym > atol:
        # Keep deterministic behavior while tolerating minor formatting/noise asymmetry in URDF files.
        return I_sym, asym
    return I_sym, asym


def _parallel_axis(I_about_C: np.ndarray, mass: float, d_C_minus_P: np.ndarray) -> np.ndarray:
    d = np.asarray(d_C_minus_P, dtype=float).reshape(3)
    return I_about_C + mass * ((float(d @ d) * np.eye(3, dtype=float)) - np.outer(d, d))


def transform_inertial_to_mdh(
    mass: float,
    T_link_inertial: np.ndarray,
    I_inertial_about_origin: np.ndarray,
    T_base_link: np.ndarray,
    T_base_mdh_joint: np.ndarray,
    inertia_about: str = "com",
    com_offset_in_inertial: Optional[ArrayLike] = None,
) -> Dict[str, np.ndarray]:
    """Transform inertial parameters from URDF inertial frame to mDH joint frame.

    `inertia_about`:
    - "com": return inertia about CoM in mDH frame.
    - "origin": return inertia about mDH frame origin.

    `com_offset_in_inertial` supports the general case where inertial origin != CoM.
    For standard URDF inertials this is typically [0, 0, 0].
    """

    if inertia_about not in {"com", "origin"}:
        raise ValueError("inertia_about must be 'com' or 'origin'")

    T_link_inertial = np.asarray(T_link_inertial, dtype=float)
    I_i_O, asym = _symmetrize_inertia(np.asarray(I_inertial_about_origin, dtype=float))
    T_base_link = np.asarray(T_base_link, dtype=float)
    T_base_J = np.asarray(T_base_mdh_joint, dtype=float)

    if T_link_inertial.shape != (4, 4) or T_base_link.shape != (4, 4) or T_base_J.shape != (4, 4):
        raise ValueError("All transforms must be 4x4")
    if I_i_O.shape != (3, 3):
        raise ValueError("Inertia matrix must be 3x3")

    c_i = np.zeros(3, dtype=float) if com_offset_in_inertial is None else np.asarray(com_offset_in_inertial, dtype=float)

    R_bl = T_base_link[:3, :3]
    p_bl = T_base_link[:3, 3]
    R_li = T_link_inertial[:3, :3]
    p_li = T_link_inertial[:3, 3]

    p_bi = p_bl + R_bl @ p_li
    R_bi = R_bl @ R_li

    I_b_O = R_bi @ I_i_O @ R_bi.T
    p_bC = p_bi + R_bi @ c_i
    I_b_C = I_b_O - mass * ((float((p_bC - p_bi) @ (p_bC - p_bi)) * np.eye(3)) - np.outer(p_bC - p_bi, p_bC - p_bi))

    R_bJ = T_base_J[:3, :3]
    p_bJ = T_base_J[:3, 3]
    R_Jb = R_bJ.T

    r_J_com = R_Jb @ (p_bC - p_bJ)
    I_J_C = R_Jb @ I_b_C @ R_Jb.T
    I_J_C, _ = _symmetrize_inertia(I_J_C)

    if inertia_about == "com":
        I_J = I_J_C
    else:
        I_J = _parallel_axis(I_J_C, mass, r_J_com)
        I_J, _ = _symmetrize_inertia(I_J)

    return {
        "r_J_com": r_J_com,
        "I_J": I_J,
        "I_J_about_com": I_J_C,
        "inertia_asymmetry_inf_norm": np.array([asym], dtype=float),
    }


def extract_dynamics_in_mdh_frames(
    urdf_path: str,
    mdh_frames: TransformMapInput,
    base_link_name: str,
    inertia_about: str = "com",
) -> Dict[str, Dict[str, Any]]:
    """Extract and transform URDF inertials into mDH joint frames.

    API intentionally simple:
    `extract_dynamics_in_mdh_frames(urdf_path, mdh_frames, base_link_name, inertia_about='com'|'origin')`

    `mdh_frames` may be:
    - dict keyed by joint index, joint name, or child link name -> 4x4 base->mDH transform
    - list/tuple of 4x4 transforms indexed by joint index

    Output is keyed by child link name for each actuated joint that has inertial data.
    """

    inertial_by_link = parse_urdf_inertials(urdf_path)
    T_base_link, actuated_joints, _ = compute_T_base_link(urdf_path, base_link_name)
    mdh_frame_map = _normalize_mdh_frames(mdh_frames)

    results: Dict[str, Dict[str, Any]] = {}

    for joint_index, joint in enumerate(actuated_joints):
        link_name = joint.child
        if link_name not in inertial_by_link:
            continue

        inertial = inertial_by_link[link_name]
        T_base_J = None
        for frame_key in (link_name, joint.name, joint_index, str(joint_index)):
            if frame_key in mdh_frame_map:
                T_base_J = mdh_frame_map[frame_key]
                break
        if T_base_J is None:
            raise KeyError(
                f"No mDH frame found for joint '{joint.name}' (index {joint_index}, child link '{link_name}')"
            )

        transformed = transform_inertial_to_mdh(
            mass=inertial.mass,
            T_link_inertial=inertial.T_link_inertial,
            I_inertial_about_origin=inertial.I_inertial,
            T_base_link=T_base_link[link_name],
            T_base_mdh_joint=T_base_J,
            inertia_about=inertia_about,
        )

        results[link_name] = {
            "joint_name": joint.name,
            "joint_index": joint_index,
            "mass": inertial.mass,
            "com": transformed["r_J_com"],
            "inertia": transformed["I_J"],
            "inertia_about": inertia_about,
            "raw_urdf": {
                "T_link_inertial": inertial.T_link_inertial,
                "inertia_matrix_in_inertial": inertial.I_inertial,
                "mass": inertial.mass,
                "xyz": inertial.xyz,
                "rpy": inertial.rpy,
            },
        }

    return results


def save_dynamics_to_csv(results: Mapping[str, Mapping[str, Any]], csv_path: str) -> None:
    """Save transformed dynamic parameters to CSV.

    Columns include mass, CoM in mDH frame, inertia in mDH frame, and raw URDF inertial values.
    """

    fieldnames = [
        "link_name",
        "joint_name",
        "joint_index",
        "mass",
        "inertia_about",
        "com_x",
        "com_y",
        "com_z",
        "I_xx",
        "I_xy",
        "I_xz",
        "I_yx",
        "I_yy",
        "I_yz",
        "I_zx",
        "I_zy",
        "I_zz",
        "raw_xyz_x",
        "raw_xyz_y",
        "raw_xyz_z",
        "raw_rpy_roll",
        "raw_rpy_pitch",
        "raw_rpy_yaw",
        "raw_I_xx",
        "raw_I_xy",
        "raw_I_xz",
        "raw_I_yx",
        "raw_I_yy",
        "raw_I_yz",
        "raw_I_zx",
        "raw_I_zy",
        "raw_I_zz",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for link_name, vals in results.items():
            com = np.asarray(vals["com"], dtype=float).reshape(3)
            inertia = np.asarray(vals["inertia"], dtype=float).reshape(3, 3)
            raw = vals.get("raw_urdf", {})
            raw_xyz = np.asarray(raw.get("xyz", np.zeros(3)), dtype=float).reshape(3)
            raw_rpy = np.asarray(raw.get("rpy", np.zeros(3)), dtype=float).reshape(3)
            raw_I = np.asarray(raw.get("inertia_matrix_in_inertial", np.zeros((3, 3))), dtype=float).reshape(3, 3)

            writer.writerow(
                {
                    "link_name": link_name,
                    "joint_name": vals.get("joint_name", ""),
                    "joint_index": vals.get("joint_index", ""),
                    "mass": float(vals.get("mass", 0.0)),
                    "inertia_about": vals.get("inertia_about", ""),
                    "com_x": com[0],
                    "com_y": com[1],
                    "com_z": com[2],
                    "I_xx": inertia[0, 0],
                    "I_xy": inertia[0, 1],
                    "I_xz": inertia[0, 2],
                    "I_yx": inertia[1, 0],
                    "I_yy": inertia[1, 1],
                    "I_yz": inertia[1, 2],
                    "I_zx": inertia[2, 0],
                    "I_zy": inertia[2, 1],
                    "I_zz": inertia[2, 2],
                    "raw_xyz_x": raw_xyz[0],
                    "raw_xyz_y": raw_xyz[1],
                    "raw_xyz_z": raw_xyz[2],
                    "raw_rpy_roll": raw_rpy[0],
                    "raw_rpy_pitch": raw_rpy[1],
                    "raw_rpy_yaw": raw_rpy[2],
                    "raw_I_xx": raw_I[0, 0],
                    "raw_I_xy": raw_I[0, 1],
                    "raw_I_xz": raw_I[0, 2],
                    "raw_I_yx": raw_I[1, 0],
                    "raw_I_yy": raw_I[1, 1],
                    "raw_I_yz": raw_I[1, 2],
                    "raw_I_zx": raw_I[2, 0],
                    "raw_I_zy": raw_I[2, 1],
                    "raw_I_zz": raw_I[2, 2],
                }
            )


def _self_check_identity_case() -> None:
    """Quick internal consistency check requested in task description."""

    mass = 2.5
    T_link_inertial = np.eye(4, dtype=float)
    I_inertial = np.array([[0.4, 0.01, -0.02], [0.01, 0.3, 0.005], [-0.02, 0.005, 0.2]], dtype=float)
    T_base_link = np.eye(4, dtype=float)
    T_base_J = np.eye(4, dtype=float)

    out = transform_inertial_to_mdh(
        mass=mass,
        T_link_inertial=T_link_inertial,
        I_inertial_about_origin=I_inertial,
        T_base_link=T_base_link,
        T_base_mdh_joint=T_base_J,
        inertia_about="com",
    )

    if not np.allclose(out["r_J_com"], np.zeros(3), atol=1e-12):
        raise AssertionError("Identity-case self-check failed for CoM")
    if not np.allclose(out["I_J"], I_inertial, atol=1e-12):
        raise AssertionError("Identity-case self-check failed for inertia")


if __name__ == "__main__":
    _self_check_identity_case()
    print("DynamicsMDH self-check passed.")
