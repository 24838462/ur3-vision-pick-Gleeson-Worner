from pathlib import Path
import copy
import json
import sys
import xml.etree.ElementTree as ET
import numpy as np

import coacd
import trimesh


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "models" / "ur3_controlled.xml"
OUTPUT = ROOT / "models" / "ur3_collision_fixed.xml"

SECTIONS = [
    "base", "shoulder", "upperarm", "forearm",
    "wrist1", "wrist2", "wrist3",
]

tree = ET.parse(SOURCE)
model = tree.getroot()
asset = model.find("asset")

if asset is None:
    raise ValueError("Missing asset section.")

compiler = model.find("compiler")
meshdir = compiler.get("meshdir", "") if compiler is not None else ""
mesh_root = SOURCE.parent / meshdir

convex_folder = mesh_root / "ur3_meshes" / "convex"
convex_folder.mkdir(parents=True, exist_ok=True)
manifest_path = convex_folder / "parts_manifest.json"

mesh_assets = {
    mesh.get("name"): mesh
    for mesh in asset.findall("mesh")
}


def save_manifest():
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(completed, indent=2))
    temporary.replace(manifest_path)


if manifest_path.exists():
    completed = json.loads(manifest_path.read_text())
else:
    completed = {}

    # Recover previously generated pieces from the existing fixed XML.
    if OUTPUT.exists():
        old_model = ET.parse(OUTPUT).getroot()
        used_meshes = {
            geom.get("mesh")
            for geom in old_model.findall(".//geom")
            if geom.get("group") == "3"
            and geom.get("contype", "1") != "0"
        }

        for section in SECTIONS:
            recovered = []

            for mesh in old_model.findall("./asset/mesh"):
                name = mesh.get("name", "")
                filename = mesh.get("file", "")

                if (
                    name.startswith(f"{section}_convex_")
                    and name in used_meshes
                    and (mesh_root / filename).is_file()
                ):
                    recovered.append({
                        "name": name,
                        "file": filename,
                    })

            if recovered:
                completed[section] = recovered
                print(
                    f"Recovered {section}: "
                    f"{len(recovered)} pieces."
                )

    save_manifest()


def generate_section(section):
    if section in completed:
        print(
            f"{section} already completed: "
            f"{len(completed[section])} pieces."
        )
        answer = input("Regenerate this section? [y/N]: ")
        if answer.strip().lower() != "y":
            return

    mesh_asset = mesh_assets[section]

    if mesh_asset.get("scale") is not None:
        raise ValueError(
            "This script expects the unscaled imported STL meshes."
        )

    mesh = trimesh.load_mesh(
        mesh_root / mesh_asset.get("file")
    )

    print(f"Splitting {section}...", flush=True)
    coacd.set_log_level("warn")

    parts = coacd.run_coacd(
        coacd.Mesh(mesh.vertices, mesh.faces),
        threshold=0.05,
        preprocess_resolution=30,
        resolution=1000,
        mcts_nodes=10,
        mcts_iterations=20,
        mcts_max_depth=2,
        max_convex_hull=32,
        seed=7,
    )

    if not parts:
        raise RuntimeError("No collision pieces were generated.")

    section_folder = convex_folder / section
    section_folder.mkdir(exist_ok=True)

    records = []

    for index, (vertices, faces) in enumerate(parts):
        name = f"{section}_convex_{index:02d}"
        path = section_folder / f"{name}.stl"

        # Prevent decomposition pieces extending beyond source bounds.
        lower = mesh.bounds[0].copy()
        upper = mesh.bounds[1].copy()

        # Small clearance only at the base–shoulder mating plane.
        # These directions match your imported mesh coordinate frames.
        seam_clearance = 0.00005  # 0.05 mm per surface

        if section == "base":
            upper[2] -= seam_clearance

        elif section == "shoulder":
            lower[2] += seam_clearance

        vertices = np.clip(
            np.asarray(vertices),
            lower,
            upper,
        )

        # Rebuild a valid convex solid after clipping the vertices.
        piece = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=False,
        ).convex_hull

        piece.export(path)

        records.append({
            "name": name,
            "file": path.relative_to(mesh_root).as_posix(),
        })

    # Mark complete only after every piece has been exported.
    completed[section] = records
    save_manifest()

    print(f"Saved {section}: {len(records)} pieces.")
    print("The simulation XML has not been changed.")


def assemble_xml():
    missing = [
        section for section in SECTIONS
        if section not in completed
    ]

    if missing:
        raise ValueError(
            "Generate these sections first: " + ", ".join(missing)
        )

    # Check saved pieces exist before updating the XML.
    for records in completed.values():
        for record in records:
            if not (mesh_root / record["file"]).is_file():
                raise FileNotFoundError(record["file"])

    for section in SECTIONS:
        for record in completed[section]:
            ET.SubElement(
                asset,
                "mesh",
                name=record["name"],
                file=record["file"],
            )

    for body in model.findall(".//body"):
        for geom in list(body.findall("geom")):
            section = geom.get("mesh")

            if section not in SECTIONS:
                continue

            if body.find("inertial") is None:
                raise ValueError(
                    f"{body.get('name')} needs an explicit inertial "
                    "element before its geometry is replaced."
                )

            template = copy.deepcopy(geom)

            # Original mesh remains visible, without duplicate collisions.
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
            geom.set("group", "1")

            for index, record in enumerate(completed[section]):
                collision = copy.deepcopy(template)
                collision.set(
                    "name",
                    f"{body.get('name')}_{section}_collision_{index:02d}",
                )
                collision.set("mesh", record["name"])
                collision.set("contype", "1")
                collision.set("conaffinity", "1")
                collision.set("group", "3")
                collision.set("rgba", "0.9 0.3 0.1 0.35")
                collision.set("density", "0")
                collision.attrib.pop("mass", None)
                body.append(collision)

    diagnostic_pairs = {
        frozenset(("base_link_inertia", "shoulder_link")),
        frozenset(("forearm_link", "wrist_2_link")),
    }

    for contact in model.findall("contact"):
        for exclusion in list(contact.findall("exclude")):
            pair = frozenset((
                exclusion.get("body1"),
                exclusion.get("body2"),
            ))

            if pair in diagnostic_pairs:
                contact.remove(exclusion)

    ET.indent(tree, space="  ")

    temporary = OUTPUT.with_suffix(".tmp")
    tree.write(temporary, encoding="unicode")
    temporary.replace(OUTPUT)

    print(f"Created complete model: {OUTPUT}")
    print("Both diagnostic collision exclusions were removed.")


if len(sys.argv) != 2:
    print("Usage:")
    print("  python fix_collision_meshes.py base")
    print("  python fix_collision_meshes.py status")
    print("  python fix_collision_meshes.py assemble")

else:
    command = sys.argv[1]

    if command in SECTIONS:
        generate_section(command)

    elif command == "status":
        for section in SECTIONS:
            if section in completed:
                print(
                    f"{section}: DONE "
                    f"({len(completed[section])} pieces)"
                )
            else:
                print(f"{section}: pending")

    elif command == "assemble":
        assemble_xml()

    else:
        raise ValueError(f"Unknown command: {command}")