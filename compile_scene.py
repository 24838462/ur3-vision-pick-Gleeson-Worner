# run this script to generate the combined robot, gripper and scene XML when the original XML is updated.
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco


ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
SOURCE = MODELS / "ur3_pick_gripper_scene.xml"
COMPLETE = MODELS / "ur3_complete.xml"

# Full cube widths: 3, 4 and 5 cm.
HALF_SIZES = [0.015, 0.020, 0.025]
COLOURS = [
    "0.9 0.1 0.1 1",
    "0.1 0.8 0.2 1",
    "0.1 0.3 0.9 1",
]


def expand_includes(path):
    """Flatten the top-level includes used by our scene files."""
    root = ET.parse(path).getroot()

    for element in list(root):
        if element.tag != "include":
            continue

        included = expand_includes(
            path.parent / element.get("file")
        )
        index = list(root).index(element)
        root.remove(element)

        for offset, child in enumerate(list(included)):
            root.insert(index + offset, child)

    return root


def compile_scene():
    """Build and validate the combined robot, gripper and scene XML."""
    root = expand_includes(SOURCE)
    root.set("model", "ur3_complete")

    for world in root.findall("worldbody"):
        for body in list(world.findall("body")):
            name = body.get("name", "")
            if name == "target_cube" or name.startswith("cube_"):
                world.remove(body)

    world = root.find("worldbody")
    if world is None:
        raise ValueError("Scene has no worldbody.")

    for index, half_size in enumerate(HALF_SIZES):
        body = ET.SubElement(
            world,
            "body",
            name=f"cube_{index}",
            pos=(
                f"{0.25 + index * 0.06} 0.15 "
                f"{-0.01 + half_size + 0.002}"
            ),
        )

        ET.SubElement(
            body,
            "freejoint",
            name=f"cube_{index}_joint",
        )

        ET.SubElement(
            body,
            "geom",
            name=f"cube_{index}_geom",
            type="box",
            size=f"{half_size} {half_size} {half_size}",
            mass="0.05",
            rgba=COLOURS[index],
            friction="0.8 0.005 0.0001",
            contype="1",
            conaffinity="1",
        )

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(
        COMPLETE,
        encoding="utf-8",
        xml_declaration=True,
    )

    model = mujoco.MjModel.from_xml_path(str(COMPLETE))
    mujoco.mj_saveLastXML(str(COMPLETE), model)

    print("Created:", COMPLETE)
    print("Launch with: python .\\main.py")


if __name__ == "__main__":
    compile_scene()