import numpy as np
import trimesh
import pyvista
import argparse
import glob
import pdb

def get_mesh_faces_in_direction(mesh, direction_normal, tol_dot=0.01):
    face_idxs = []
    for face_idx, face_normal in enumerate(mesh.face_normals):
        face_normal = face_normal / np.linalg.norm(face_normal)
        face_dir_dot = np.dot(face_normal, direction_normal)
        if face_dir_dot > tol_dot:  # face normal in same direction ?
            face_idxs.append(face_idx)
    return mesh.faces[face_idxs]


def extrude_convex_mesh(mesh, direction):
    # assert mesh.is_convex, "Can only extrude convex meshes"

    normal = direction / np.linalg.norm(direction)

    faces_to_extrude = get_mesh_faces_in_direction(mesh, normal)

    # code slightly adapted from `trimesh.creation.extrude_triangulation`

    # stack the (n,3) faces into (3*n, 2) edges
    edges = trimesh.geometry.faces_to_edges(faces_to_extrude)
    edges_sorted = np.sort(edges, axis=1)
    # edges which only occur once are on the boundary of the polygon
    # since the triangulation may have subdivided the boundary of the
    # shapely polygon, we need to find it again
    edges_unique = trimesh.grouping.group_rows(
        edges_sorted, require_count=1)

    # (n, 3, 2) set of line segments (positions, not references)
    boundary = mesh.vertices[edges[edges_unique]]

    # we are creating two vertical triangles for every 3D line segment
    # on the boundary of the 3D triangulation
    vertical = np.tile(boundary.reshape((-1, 3)), 2).reshape((-1, 3))
    vertical[1::2] += direction
    vertical_faces = np.tile([3, 1, 2, 2, 1, 0],
                             (len(boundary), 1))
    vertical_faces += np.arange(len(boundary)).reshape((-1, 1)) * 4
    vertical_faces = vertical_faces.reshape((-1, 3))

    # reversed order of vertices, i.e. to flip face normals
    bottom_faces_seq = faces_to_extrude[:, ::-1]

    top_faces_seq = faces_to_extrude.copy()

    # a sequence of zero- indexed faces, which will then be appended
    # with offsets to create the final mesh
    faces_seq = [bottom_faces_seq,
                 top_faces_seq,
                 vertical_faces]
    vertices_seq = [mesh.vertices,
                    mesh.vertices.copy() + direction,
                    vertical]

    # append sequences into flat nicely indexed arrays
    vertices, faces = trimesh.util.append_faces(vertices_seq, faces_seq)

    # create mesh object
    extruded_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    extruded_mesh.fix_normals() # somehow face normals are inverted for some `direction`, fixing it here
    return extruded_mesh


def modify_mesh(path):
    mesh_path = f'{path}/arti_pred.obj'
    mesh = trimesh.load(mesh_path)
    norm_path = f'{path}/normal.npy'
    plane_norm = np.load(norm_path)
    thick_mesh = extrude_convex_mesh(mesh, plane_norm*0.05)
    return thick_mesh

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input video series name")
    args = parser.parse_args()
    input_path = f'{args.input}/*/*/frame_*'
    paths = glob.glob(input_path)

    for path in paths:
        # pdb.set_trace()
        mesh = modify_mesh(path)
        output_path = f'{path}/extrude_pred.obj'
        _ = mesh.export(output_path)
        # pdb.set_trace()

if __name__ == '__main__':
    main()