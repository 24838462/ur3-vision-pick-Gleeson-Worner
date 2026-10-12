from copy import deepcopy
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"

ROBOT_XML = MODELS / "ur3_with_gripper.xml"
CAMERA_XML = MODELS / "realsense_d435i" / "d435i.xml"

PREFIX = "wrist_d435i_"


def attach_camera():
    robot = ET.parse(ROBOT_XML).getroot()
    camera = ET.parse(CAMERA_XML).getroot()

    tool = robot.find(".//body[@name='tool0']")
    if tool is None:
        raise ValueError("Could not find the tool0 body.")

    # Back up the source model once.
    backup = ROBOT_XML.with_name("ur3_with_gripper_before_camera.xml")
    if not backup.exists():
        shutil.copy2(ROBOT_XML, backup)

    # Give meshes explicit names before prefixing references.
    for mesh in camera.findall("./asset/mesh"):
        if mesh.get("name") is None:
            mesh.set("name", Path(mesh.get("file")).stem)

        mesh.set(
            "file",
            "realsense_d435i/assets/" + mesh.get("file"),
        )

    # Avoid conflicts with the gripper's visual/collision classes.
    reference_attributes = {
        "name",
        "class",
        "childclass",
        "mesh",
        "material",
        "texture",
    }

    for element in camera.iter():
        for attribute in reference_attributes:
            value = element.get(attribute)
            if value is not None:
                element.set(attribute, PREFIX + value)

    # Remove the old viewpoint and any previous camera installation.
    for parent in robot.iter():
        for child in list(parent):
            name = child.get("name", "")

            if (
                child.tag == "camera"
                and name == "wrist_camera"
            ) or (
                child.tag == "body"
                and name == "wrist_camera_mount"
            ):
                parent.remove(child)

    # Remove previously imported camera assets/defaults on reruns.
    for section_name in ("asset", "default"):
        section = robot.find(section_name)
        if section is not None:
            for child in list(section):
                identifiers = (
                    child.get("name", ""),
                    child.get("class", ""),
                )
                if any(
                    value.startswith(PREFIX)
                    for value in identifiers
                ):
                    section.remove(child)

    # Merge camera assets and named defaults.
    for section_name in ("asset", "default"):
        source = camera.find(section_name)
        destination = robot.find(section_name)

        if destination is None:
            destination = ET.SubElement(robot, section_name)

        for child in source:
            destination.append(deepcopy(child))

    # Preserve the original camera model's primitive mesh fitting.
    compiler = robot.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(robot, "compiler")
    compiler.set("fitaabb", "true")

    # Initial mounting position, relative to tool0.
    # Move this body to adjust housing and viewpoint together.
    mount = ET.SubElement(
        tool,
        "body",
        name="wrist_camera_mount",
        pos="0 0.065 0.04",
    )

    ET.SubElement(
        mount,
        "geom",
        name="wrist_camera_bracket",
        type="box",
        pos="0 -0.0325 -0.028",
        size="0.008 0.0325 0.003",
        mass="0.015",
        rgba="0.35 0.35 0.4 1",
        group="2",
        contype="1",
        conaffinity="1",
    )

    housing = deepcopy(camera.find("./worldbody/body"))
    mount.append(housing)

    # RGB pupil mesh centre is approximately:
    # [0.0325, 0.000001, -0.003605] metres.
    # Put the viewpoint just ahead of its surface.
    ET.SubElement(
        housing,
        "camera",
        name="wrist_camera",
        mode="fixed",
        pos="0.0325 0.000001 -0.0033",
        quat="0 1 0 0",
        fovy="60",
    )

    ET.indent(robot, space="  ")
    ET.ElementTree(robot).write(
        ROBOT_XML,
        encoding="utf-8",
        xml_declaration=True,
    )

    print("Camera attached to:", ROBOT_XML)
    print("Backup:", backup)
    print("Next: python .\\compile_scene.py")


if __name__ == "__main__":
    attach_camera()