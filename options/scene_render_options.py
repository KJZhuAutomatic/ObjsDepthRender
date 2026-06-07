from .base_options import BaseOptions

class SceneRenderOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--phase', type=str, default='render', help='builder')

        parser.add_argument('--is_table', type=bool, default=True, help='whether to render table, if True render with fixed mesh and dynamic camera, else render with dynamic mesh and camera without extrinsic')
        parser.add_argument('--meshes_path', type=str, default='/opt/data/private/graspnet_dataset/models/', help='path to one example')
        parser.add_argument('--meshes_name', type=str, default='nontextured.ply', help='obj_xx.ply')
        
        parser.add_argument('--is_offscreen', type=bool, default=False, help='is offscreen render')

        parser.add_argument('--one_scene_path', type=str, default='E:/Datasets/LineMod/Linemod_preprocessed/data/02/', help='path to one example')
        return parser