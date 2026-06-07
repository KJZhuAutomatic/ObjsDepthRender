import os
import numpy as np
import scipy.io as sio
from utils import mesh_utils, vis_utils
import open3d as o3d

scene_path = "/opt/data/private/graspnet_dataset/scenes/scene_0000/realsense/"
poses_paths = os.listdir(scene_path + 'meta/')
cam_pos = np.load(scene_path + 'cam0_wrt_table.npy')
extrinsic_mat = np.linalg.inv(cam_pos).tolist()
meshes_path = "/opt/data/private/graspnet_dataset/models/"
meshes_name = "nontextured.ply"
object_render_dir = "/opt/data/private/ObjsDepthRender/objects_render/"

def vis_obj():

    for i_img in range(len(poses_paths)):
        meshes = []
        pose_idx = poses_paths[i_img].split('.')[0]
        pose_path = scene_path + 'meta/' + poses_paths[i_img]
        mat = sio.loadmat(pose_path)
        poses = mat['poses'] # (3, 4, N_obj)
        idx_meshes = mat['cls_indexes'] # (1, N_obj)
        num_meshes = poses.shape[2]
        for i_mesh in range(num_meshes):
            idx_mesh = str(idx_meshes[0][i_mesh] - 1).zfill(3)
            mesh_path = meshes_path + idx_mesh + '/' + meshes_name
            mesh = mesh_utils.load_mesh(mesh_path)
            # meshes.append(mesh)
            # o3d.visualization.draw_geometries([mesh])
            pose = poses[:, :, i_mesh]

            homog_term = np.array([[0., 0., 0., 1.]])
            pose = np.concatenate((pose, homog_term), axis=0)
            mesh = mesh.transform(pose)

            if not cam_pos.any() == None:
                mesh = mesh.transform(cam_pos)

            # o3d.visualization.draw_geometries([mesh])

            vis = o3d.visualization.Visualizer()
            vis.create_window(width=320, height=240, visible=False)
            vis.add_geometry(mesh)

            vis.poll_events()
            depth_name = object_render_dir + str(idx_meshes[0][i_mesh]).zfill(3) + ".png"
            vis.capture_screen_image(depth_name, do_render=True)
            

if __name__ == "__main__":
    vis_obj()