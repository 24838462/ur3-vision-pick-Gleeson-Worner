# This is building the UR3 from a model I pulled off the internet at this link 

from pathlib import Path # Assists in cross file work making it easier and neater

import shutil # This loads tools for copying files and folders

import tempfile # This will let us create temporary working folders that Python can automatically remove afterwards.
# This will allow us to convert the ur3 model files into MuJoCo friendly versions.

import xml.etree.ElementTree as ET # Loads an XML reader/editor and give it the short name ET. This will be used as robot descriptions use XML tags such as <link> and <joint>

import xacro # This will load the tool which expands the xacro templates into a complete robot description

import mujoco # Loads the simulator's Python tools

MODELS = Path(__file__).resolve().parent / "models" # This will allow it to always find and launch the models script.
CONFIG = MODELS / "ur3_config" # This will find the config folder within models.

with tempfile.TemporaryDirectory() as temp_dir:
    temp_root = Path(temp_dir)
    shutil.copytree(MODELS / "urdf", temp_root / "urdf")

    # Replace ROS package references with a local folder.
    for file in (temp_root / "urdf").rglob("*.xacro"):
        text = file.read_text(encoding="utf-8")
        text = text.replace(
            "$(find ur_description)", temp_root.as_posix()
        )
        file.write_text(text, encoding="utf-8")

    # Select the UR3 and its configuration files.
    settings = {
        "name": "ur3",
        "ur_type": "ur3",
        "joint_limit_params":
            (CONFIG / "joint_limits.yaml").as_posix(),
        "kinematics_params":
            (CONFIG / "default_kinematics.yaml").as_posix(),
        "physical_params":
            (CONFIG / "physical_parameters.yaml").as_posix(),
        "visual_params":
            (CONFIG / "visual_parameters.yaml").as_posix(),
        "force_abs_paths": "false",
    }

    document = xacro.process_file(
        str(temp_root / "urdf" / "ur.urdf.xacro"),
        mappings=settings,
    )
    robot = ET.fromstring(document.toxml())


# Remove the DAE visual geometry.
# The STL collision geometry will also provide the visible shape.
for link in robot.findall("link"):
    for visual in link.findall("visual"):
        link.remove(visual)

# Point each remaining mesh at your STL folder.
for mesh in robot.iter("mesh"):
    filename = mesh.get("filename").split("/")[-1]
    mesh_path = MODELS / "ur3_meshes" / "collision" / filename

    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)

    mesh.set("filename", f"ur3_meshes/collision/{filename}")


# Preserve mesh paths and fixed bodies, including the tool frame.
extension = ET.SubElement(robot, "mujoco")
ET.SubElement(
    extension, "compiler",
    strippath="false",
    fusestatic="false",
)

# Save the expanded robot description.
urdf_path = MODELS / "ur3.urdf"
ET.indent(robot)
ET.ElementTree(robot).write(
    str(urdf_path), encoding="utf-8", xml_declaration=True
)

# Import into MuJoCo and export its native XML format.
model = mujoco.MjModel.from_xml_path(str(urdf_path))
xml_path = MODELS / "ur3.xml"
mujoco.mj_saveLastXML(str(xml_path), model)

# Check that the exported model loads successfully.
reloaded = mujoco.MjModel.from_xml_path(str(xml_path))
print(f"Created: {urdf_path.name} and {xml_path.name}")
print(f"Arm joints: {reloaded.njnt}")
print(f"Actuators: {reloaded.nu}")




