# This  is urdf demo file 
# Author: Minchang Sung 
# E-mail: wdrac331@hanyang.ac.kr
# Date: November 30, 2023      

from Line import Line
from LineGeometry import LineGeometry
from URDF2Line import URDF2Line
from AlgorithmicMDH import AlgorithmicMDH
from Plotter import Plotter
from PbSim import PbSim
from DynamicsMDH import extract_dynamics_in_mdh_frames, get_T_base_mdh_frame, save_dynamics_to_csv
import numpy as np

# you can change urdf_name
# urdf_name = "urdf/indy7/indy7.urdf"
#urdf_name = "urdf/kuka_iiwa/kuka_iiwa.urdf"
#urdf_name = "urdf/ur5/ur5.urdf"
#urdf_name = "urdf/kuka_lwr/kuka.urdf"
#urdf_name = "urdf/softrobot_urdf/softarm_urdf(p)_test.urdf"
#urdf_name ="urdf/scara/scara.urdf"
# urdf_name ="urdf/posco_dual/mmr_steel_torso_frame_fixer.urdf"
urdf_name ="urdf/qdd_va_dual/SF-Simple_Assy_SLDASM_meshes.urdf"
urdf=URDF2Line(urdf_name)

ADH = AlgorithmicMDH(urdf.z_list,urdf.p_list,urdf.z_tcf,urdf.p_tcf,urdf.x_tcf,urdf.x_0)
print("=================MDH RESULT================")
print(ADH.MDH)

# Dynamic parameters in mDH frames (SI units: kg, m, kg*m^2).
joint_type_map = {"revolute": 0, "prismatic": 1}
joint_types = [joint_type_map.get(j.joint_type, 0) for j in urdf.robot.actuated_joints]
mdh_frames = get_T_base_mdh_frame(ADH.MDH, joint_types, T_base_mdh0=np.eye(4))
base_link_name = urdf.base_link_name

dynamics = extract_dynamics_in_mdh_frames(
    urdf_path=urdf_name,
    mdh_frames=mdh_frames,
    base_link_name=base_link_name,
    inertia_about="com",
)
csv_path = "dynamics_in_mdh_frames.csv"
save_dynamics_to_csv(dynamics, csv_path)
print("=================DYNAMICS IN MDH FRAMES================")
print(f"CSV saved: {csv_path}")
print(f"Base frame for dynamics (same as mDH extraction): {base_link_name}")
if len(dynamics) == 0:
    print("No <inertial> elements were found for actuated child links.")
else:
    for link_name, vals in dynamics.items():
        print(f"[{vals['joint_index']}] {vals['joint_name']} -> {link_name}")
        print(f"  mass = {vals['mass']:.6f} kg")
        print(f"  com (mDH frame) = {vals['com']}")
        print(f"  inertia ({vals['inertia_about']}) =\\n{vals['inertia']}")

PbSim(ADH.MDH,urdf_name)
