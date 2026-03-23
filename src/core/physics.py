"""Physics property estimation from aligned real-world-scale meshes."""

import numpy as np
import trimesh


# Material density lookup: label -> (material, density_kg_m3, fill_ratio)
# fill_ratio accounts for hollow objects: solid mesh volume * fill_ratio = material volume
MATERIAL_TABLE = {
    "mug":    ("ceramic",         2400, 0.42),   # hollow, thick walls + handle
    "cup":    ("glass",           2500, 0.12),   # hollow, thin walls
    "fork":   ("stainless steel", 8000, 0.08),   # thin flat tines
    "bottle": ("plastic (PET)",   1380, 0.05),   # hollow, thin walls
    "phone":  ("glass/aluminum",  2700, 0.35),   # mostly solid slab
}

DEFAULT_MATERIAL = ("unknown", 1000, 0.5)


def estimate_physics(label: str, mesh_path: str, partial_points: np.ndarray = None) -> dict:
    """Estimate physical properties from an aligned, real-world-scale mesh.

    Args:
        label: Object label for material lookup.
        mesh_path: Path to aligned OBJ mesh (real-world scale, meters).
        partial_points: Optional partial point cloud (N x 3) for stats.

    Returns:
        Dict with height_cm, width_cm, depth_cm, volume_cm3,
        surface_area_cm2, weight_g, material, partial_points_count,
        mesh_faces_count.
    """
    mesh = trimesh.load(mesh_path, force='mesh')

    # Oriented bounding box for dimensions
    obb = mesh.bounding_box_oriented
    extents = sorted(obb.extents, reverse=True)  # longest first
    # Convention: height = longest, width = second, depth = third
    height_m, width_m, depth_m = extents

    # Volume
    if mesh.is_watertight:
        volume_m3 = abs(mesh.volume)
    else:
        volume_m3 = mesh.convex_hull.volume

    # Surface area
    surface_area_m2 = mesh.area

    # Weight from material density, adjusted by fill ratio for hollow objects
    material_name, density, fill_ratio = MATERIAL_TABLE.get(label, DEFAULT_MATERIAL)
    material_volume_m3 = volume_m3 * fill_ratio
    weight_g = density * material_volume_m3 * 1000  # kg/m3 * m3 = kg -> * 1000 = g

    return {
        "height_cm": round(height_m * 100, 1),
        "width_cm": round(width_m * 100, 1),
        "depth_cm": round(depth_m * 100, 1),
        "volume_cm3": round(volume_m3 * 1e6, 1),
        "surface_area_cm2": round(surface_area_m2 * 1e4, 1),
        "weight_g": round(weight_g, 1),
        "material": material_name,
        "partial_points_count": len(partial_points) if partial_points is not None else 0,
        "mesh_faces_count": len(mesh.faces),
    }
