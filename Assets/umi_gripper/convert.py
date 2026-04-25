from mjcf_urdf_simple_converter import convert
convert("Assets/umi_gripper/umi_gripper.xml", "Assets/umi_gripper/umi_gripper.urdf")
# or, if you are using it in your ROS package and would like for the mesh directories to be resolved correctly, set meshfile_prefix, for example:
# convert("Assets/umi_gripper/umi_gripper.xml", "Assets/umi_gripper/umi_gripper.urdf", asset_file_prefix="package://your_package_name/model/")