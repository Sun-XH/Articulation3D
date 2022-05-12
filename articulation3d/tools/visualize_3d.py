import open3d as o3d
import numpy as np
import json
import math

def getCamera(transformation, fx, fy, cx, cy):
    # Return the camera and its corresponding frustum framework
    camera = o3d.geometry.TriangleMesh.create_coordinate_frame()
    camera.transform(transformation)
    # Add origin and four corner points in image plane
    points = []
    camera_origin = np.array([0, 0, 0, 1])
    points.append(np.dot(transformation, camera_origin)[0:3])
    # Calculate the four points for of the image plane
    magnitude = (cy**2 + cx**2 + fx**2) ** 0.5
    plane_points = [[-cy, -cx, -fx], [-cy, cx, -fx],
                    [cy, -cx, -fx], [cy, cx, -fx]]
    for point in plane_points:
        point = list(np.array(point) / magnitude)
        temp_point = np.array(point + [1])
        points.append(np.dot(transformation, temp_point)[0:3])
    # Draw the camera framework
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 2],
        [2, 4],
        [1, 3],
        [3, 4]
    ]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )

    return [camera, line_set]


# Borrow ideas and codes from H. Sánchez's answer
# https://stackoverflow.com/questions/59026581/create-arrows-in-open3d
import open3d as o3d
import numpy as np


def get_arrow(origin=[0, 0, 0], end=None, color=[0, 0, 0]):
    """
    Creates an arrow from an origin point to an end point,
    or create an arrow from a vector vec starting from origin.
    Args:
        - end (): End point. [x,y,z]
        - vec (): Vector. [i,j,k]
    """
    vec_Arr = np.array(end) - np.array(origin)
    vec_len = np.linalg.norm(vec_Arr)
    mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(
        cone_height=0.2 * vec_len,
        cone_radius=0.001,
        cylinder_height=0.8 * vec_len,
        cylinder_radius=0.001,
    )
    mesh_arrow.paint_uniform_color(color)
    rot_mat = caculate_align_mat(vec_Arr)
    mesh_arrow.rotate(rot_mat, center=np.array([0, 0, 0]))
    mesh_arrow.translate(np.array(origin))
    return mesh_arrow


def get_cross_prod_mat(pVec_Arr):
    # pVec_Arr shape (3)
    qCross_prod_mat = np.array([
        [0, -pVec_Arr[2], pVec_Arr[1]],
        [pVec_Arr[2], 0, -pVec_Arr[0]],
        [-pVec_Arr[1], pVec_Arr[0], 0],
    ])
    return qCross_prod_mat


def caculate_align_mat(pVec_Arr):
    scale = np.linalg.norm(pVec_Arr)
    pVec_Arr = pVec_Arr / scale
    # must ensure pVec_Arr is also a unit vec.
    z_unit_Arr = np.array([0, 0, 1])
    z_mat = get_cross_prod_mat(z_unit_Arr)

    z_c_vec = np.matmul(z_mat, pVec_Arr)
    z_c_vec_mat = get_cross_prod_mat(z_c_vec)
    if np.dot(z_unit_Arr, pVec_Arr) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, pVec_Arr) == 1:   
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat,
                                                        z_c_vec_mat)/(1 + np.dot(z_unit_Arr, pVec_Arr))
    qTrans_Mat *= scale
    return qTrans_Mat


if __name__ == "__main__":

    
    obj_mesh_1 = o3d.io.read_triangle_mesh("/localhome/xsa55/Xiaohao/Articulation3D/articulation3d/output/test_split_fridge/frame_0005/arti_pred.obj")
    obj_mesh_1.compute_vertex_normals()

    # plane_mesh_1 = o3d.io.read_triangle_mesh("test_video/output3d_pad_test/frame_0019/arti_pred.obj", enable_post_processing=True)
    # plane_mesh_1.compute_vertex_normals()

    # plane_mesh_2 = o3d.io.read_triangle_mesh("test_video/_Archive/output3d_test_trans_517/frame_0018/arti_pred.obj", enable_post_processing=True)
    # plane_mesh_2.compute_vertex_normals()

    # hum_mesh_opt = o3d.io.read_triangle_mesh("../results/refrigerator_b005-0001/00009_smpl.obj")
    # T = np.eye(4)
    # T[0, 0] = -1
    # T[2, 2] = -1
    # hum_mesh_opt.transform(T)
    
    # hum_mesh_opt.scale(1./100, np.array([0., 0., 0.]))
    # hum_mesh_opt.compute_vertex_normals()

    # verts = np.asarray(hum_mesh_opt.vertices)
    # hand = verts[5466]

    # # hum_mesh = o3d.io.read_triangle_mesh("../../d3d-hoi/dataset/d3dhoi_video_data/refrigerator/b005-0001/smplmesh/smplmesh-0010.obj")
    # hum_mesh = o3d.io.read_triangle_mesh("/localhome/sha279/Desktop/neurips22/eft/mocap_output/mocap/scene_00000018.obj")
    # # hum_mesh.scale(1./100, np.array([0., 0., 0.]))
    # T = np.eye(4)
    # # T[0, 0] = -1
    # T[1, 1] = -1
    # T[2, 2] = -1
    # hum_mesh.transform(T)
    # hum_mesh.compute_vertex_normals()
    # hum_mesh.paint_uniform_color([1., 0., 0.])

    # hum_mesh2 = o3d.io.read_triangle_mesh("/localhome/sha279/Desktop/neurips22/eft/mocap_output/mocap/scene_00000010_img.obj")
    # hum_mesh2.scale(1./100, np.array([0., 0., 0.]))
    # T1 = np.eye(4)
    # # T[0, 0] = -1
    # T1[1, 1] = -1
    # T1[2, 2] = -1
    # T1[:3, 3] = [0.2, 0., 0]
    # hum_mesh2.transform(T1)
    # hum_mesh2.compute_vertex_normals()
    # hum_mesh2.paint_uniform_color([0., 1., 0.])
    # print(hand)
    # print(hand-np.array([0, 0, -2]), hand+np.array([0, 0, 2]))
    # arr = get_arrow(origin=[0., 0., 0.], end=hand)

    print(obj_mesh_1)
    # Visualize the world coordinate
    world = o3d.geometry.TriangleMesh.create_coordinate_frame()

    # test = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # test.translate([ 0.33785772,  0.42967508, -0.47955493], relative=False)
    # Visualize the camera coordinate
    camera = getCamera(np.eye(4), 983, 983, 720/2, 1280/2)

    # Final Visualization
    o3d.visualization.draw_geometries([obj_mesh_1, world] + camera)
    # o3d.visualization.draw_geometries([obj_mesh])