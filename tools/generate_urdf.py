#!/usr/bin/env python3
"""Generate RH56DFX left/right URDF files for the external Isaac Lab project."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


DEFAULT_CONTAINER_MESH_ROOT = "/workspace/external/rh56dfx_description"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RH56DFX URDF files from xacro.")
    parser.add_argument(
        "--description-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "rh56dfx_description",
        help="Path to the rh56dfx_description repository on the host.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "urdf",
        help="Directory where generated URDF files should be written.",
    )
    parser.add_argument(
        "--mesh-root",
        type=str,
        default=DEFAULT_CONTAINER_MESH_ROOT,
        help="Mesh path prefix baked into the generated URDF files.",
    )
    parser.add_argument(
        "--side",
        action="append",
        choices=("left", "right"),
        help="Hand side to generate. Repeat to generate multiple sides.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate both left and right hand URDF files.",
    )
    return parser.parse_args()


def _selected_sides(args: argparse.Namespace) -> list[str]:
    if args.all or not args.side:
        return ["left", "right"]
    return args.side


def _require_xacro() -> str:
    xacro_bin = shutil.which("xacro")
    if xacro_bin is None:
        raise RuntimeError("Could not find `xacro` in PATH. Install it first, for example with `pip install xacro`.")
    return xacro_bin


def _write_wrapper_xacro(path: Path, macro_path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                '<?xml version="1.0"?>',
                '<robot name="rh56dfx" xmlns:xacro="http://www.ros.org/wiki/xacro">',
                '  <xacro:arg name="side" default="left"/>',
                '  <xacro:arg name="prefix" default=""/>',
                '  <xacro:property name="side_s" value="$(arg side)"/>',
                '  <xacro:property name="prefix_p" value="$(arg prefix)"/>',
                f'  <xacro:include filename="{macro_path}"/>',
                '  <xacro:rh56dfx_hand side="${side_s}" prefix="${prefix_p}"/>',
                "</robot>",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _sanitize_generated_urdf(urdf_text: str, side: str) -> str:
    """Remove the empty kinematic anchor root that breaks Isaac Sim USD visuals."""
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    root = ET.fromstring(urdf_text, parser=parser)

    anchor_link_name = f"{side}_hand_root"
    anchor_joint_name = f"{side}_hand_root_joint"
    for child in list(root):
        if child.tag == "link" and child.attrib.get("name") == anchor_link_name:
            root.remove(child)
        elif child.tag == "joint" and child.attrib.get("name") == anchor_joint_name:
            root.remove(child)

    # Strip stray non-whitespace text emitted near top-level nodes by xacro.
    for child in root:
        if child.tail and not child.tail.isspace():
            child.tail = "\n"

    root.text = "\n"
    ET.indent(root, space="  ")
    return '<?xml version="1.0" ?>\n' + ET.tostring(root, encoding="unicode") + "\n"


def generate_urdf(
    *,
    xacro_bin: str,
    description_root: Path,
    output_dir: Path,
    mesh_root: str,
    side: str,
) -> Path:
    macro_src = description_root / "urdf" / "rh56dfx_macro.xacro"
    if not macro_src.is_file():
        raise FileNotFoundError(f"Macro xacro not found: {macro_src}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"rh56dfx_{side}.urdf"
    prefix = f"{side}_"

    with tempfile.TemporaryDirectory(prefix="rh56dfx_urdf_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        macro_dst = temp_dir_path / "rh56dfx_macro_patched.xacro"
        wrapper_path = temp_dir_path / "rh56dfx_wrapper.xacro"

        macro_text = macro_src.read_text(encoding="utf-8")
        macro_text = macro_text.replace("package://rh56dfx_description/", mesh_root.rstrip("/") + "/")
        macro_dst.write_text(macro_text, encoding="utf-8")
        _write_wrapper_xacro(wrapper_path, macro_dst)

        result = subprocess.run(
            [xacro_bin, str(wrapper_path), f"side:={side}", f"prefix:={prefix}"],
            check=True,
            text=True,
            capture_output=True,
        )
        output_path.write_text(_sanitize_generated_urdf(result.stdout, side), encoding="utf-8")

    return output_path


def main() -> int:
    args = _parse_args()
    xacro_bin = _require_xacro()
    description_root = args.description_root.resolve()
    output_dir = args.output_dir.resolve()

    print(f"[INFO] Using rh56dfx_description: {description_root}")
    print(f"[INFO] Writing URDFs to: {output_dir}")
    print(f"[INFO] Mesh root baked into URDFs: {args.mesh_root}")

    generated_paths = []
    for side in _selected_sides(args):
        path = generate_urdf(
            xacro_bin=xacro_bin,
            description_root=description_root,
            output_dir=output_dir,
            mesh_root=args.mesh_root,
            side=side,
        )
        generated_paths.append(path)
        print(f"[OK] Generated {path}")

    print("")
    for path in generated_paths:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
