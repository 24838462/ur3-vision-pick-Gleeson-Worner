from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET


PROJECT = Path(__file__).resolve().parent
MODELS = PROJECT / "models"

robot_path = MODELS / "ur3_collision_fixed.xml"
gripper_path = MODELS / "robotiq_2f85" / "2f85.xml"
scene_path = MODELS / "ur3_pick_scene.xml"

robot_output = MODELS / "ur3_with_gripper.xml"
scene_output = MODELS / "ur3_pick_gripper_scene.xml"

robot = ET.parse(robot_path).getroot()
gripper = ET.parse(gripper_path).getroot()
scene = ET.parse(scene_path).getroot()

tool = robot.find(".//body[@name='tool0']")
if tool is None:
    raise ValueError("Could not find tool0 in the UR3 model.")

# Give every mesh an explicit name before changing its file path.
for mesh in gripper.findall("./asset/mesh"):
    mesh.attrib.setdefault("name", Path(mesh.get("file")).stem)
    mesh.set(
        "file",
        (
            Path("robotiq_2f85")
            / "assets"
            / mesh.get("file")
        ).as_posix(),
    )

# Prefix names and references to avoid clashes with the UR3.
reference_attributes = {
    "name", "class", "childclass",
    "mesh", "material", "texture",
    "joint", "joint1", "joint2",
    "body", "body1", "body2",
    "site", "site1", "site2",
    "tendon", "tendon1", "tendon2",
    "actuator",
}

for element in gripper.iter():
    for attribute in reference_attributes:
        value = element.get(attribute)
        if value is not None:
            element.set(attribute, "gripper_" + value)

# Keep compiler settings in the combined robot.
# Mesh paths above are relative to the models folder.
compiler = robot.find("compiler")
if compiler is None:
    compiler = ET.SubElement(robot, "compiler")
compiler.set("autolimits", "true")

# Preserve the gripper's contact solver settings.
option = robot.find("option")
if option is None:
    option = ET.SubElement(robot, "option")
option.set("cone", "elliptic")
option.set("impratio", "10")

# Merge assets, default classes, constraints and actuator.
for section_name in (
    "asset", "default", "contact",
    "tendon", "equality", "actuator",
):
    source_section = gripper.find(section_name)
    if source_section is None:
        continue

    destination = robot.find(section_name)
    if destination is None:
        destination = ET.SubElement(robot, section_name)

    for child in source_section:
        destination.append(deepcopy(child))

# Attach the full gripper body tree to the tool frame.
mount = gripper.find("./worldbody/body")
if mount is None:
    raise ValueError("Could not find the gripper mounting body.")

tool.append(deepcopy(mount))

# Create a scene that loads the combined robot.
robot_include = next(
    (
        element
        for element in scene.findall("include")
        if Path(element.get("file", "")).name
        == robot_path.name
    ),
    None,
)

if robot_include is None:
    raise ValueError(
        "Scene must include ur3_collision_fixed.xml."
    )

robot_include.set("file", robot_output.name)

for root, output in (
    (robot, robot_output),
    (scene, scene_output),
):
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(
        output,
        encoding="utf-8",
        xml_declaration=True,
    )

print("Created:", robot_output)
print("Created:", scene_output)
print("Attached gripper to tool0.")
print("Gripper command: 0 = open, 255 = close.")