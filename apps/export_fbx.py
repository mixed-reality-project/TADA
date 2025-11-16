import os
import json
import pickle as pkl
import random

os.environ['PYOPENGL_PLATFORM'] = 'osmesa'  # Uncommnet this line while running remotely
import argparse
import cv2
import torch
import smplx
import trimesh
import imageio
from PIL import Image
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm

from lib.common.obj import Mesh
from lib.common.remesh import subdivide_inorder
from lib.common.utils import SMPLXSeg
from lib.common.lbs import warp_points
from lib.common.obj import normalize_vert, compute_normal

from fbx import *
from apps.FbxCommon import *

def get_motion_diffusion_pose(file_path):
    data = np.load(file_path, allow_pickle=True).item()
    x_translations = data['motion'][-1, :3]
    motion = data['motion'][:-1]

    motion = torch.as_tensor(motion).permute(2, 0, 1)
    x_translations = torch.as_tensor(x_translations).permute(1, 0)
    from lib.common.rotation_conversions import rotation_6d_to_matrix, matrix_to_euler_angles, matrix_to_axis_angle
    rotations = rotation_6d_to_matrix(motion) # [540, 24, 3, 3]
    rotations = matrix_to_axis_angle(rotations.reshape(-1, 3, 3)).reshape(-1, 24, 3)

    return rotations, x_translations


def build_new_mesh(v, f, vt, ft):
    # build a correspondences dictionary from the original mesh indices to the (possibly multiple) texture map indices
    f_flat = f.flatten()
    ft_flat = ft.flatten()
    correspondences = {}

    # traverse and find the corresponding indices in f and ft
    for i in range(len(f_flat)):
        if f_flat[i] not in correspondences:
            correspondences[f_flat[i]] = [ft_flat[i]]
        else:
            if ft_flat[i] not in correspondences[f_flat[i]]:
                correspondences[f_flat[i]].append(ft_flat[i])

    # build a mesh using the texture map vertices
    new_v = np.zeros((v.shape[0], vt.shape[0], 3))
    for old_index, new_indices in correspondences.items():
        for new_index in new_indices:
            new_v[:, new_index] = v[:, old_index]

    # define new faces using the texture map faces
    f_new = ft
    return new_v, f_new

class Animation:
    def __init__(self, render_res=512):
        self.device = torch.device("cuda")

        # load data
        init_data = np.load('./data/init_body/data.npz')
        self.dense_faces = torch.as_tensor(init_data['dense_faces'], device=self.device)
        self.dense_lbs_weights = torch.as_tensor(init_data['dense_lbs_weights'], device=self.device)
        self.unique = init_data['unique']
        self.vt = init_data['vt']
        self.ft = init_data['ft']

        model_params = dict(
            model_path="./data/smplx/SMPLX_NEUTRAL_2020.npz",
            model_type='smplx',
            create_global_orient=True,
            create_body_pose=True,
            create_betas=True,
            create_left_hand_pose=True,
            create_right_hand_pose=True,
            create_jaw_pose=True,
            create_leye_pose=True,
            create_reye_pose=True,
            create_expression=True,
            create_transl=False,
            use_pca=False,
            flat_hand_mean=False,
            num_betas=300,
            num_expression_coeffs=100,
            num_pca_comps=12,
            dtype=torch.float32,
            batch_size=1,
        )
        self.body_model = smplx.create(**model_params).to(device='cuda')
        self.smplx_face = self.body_model.faces.astype(np.int32)

    def load_ckpt_data(self, ckpt_file):
        # model_data = torch.load(ckpt_file)["model"]
        model_data = torch.load(ckpt_file)
        self.expression = model_data["expression"] if "expression" in model_data else None
        self.jaw_pose = model_data["jaw_pose"] if "jaw_pose" in model_data else None
        # self.raw_albedo = torch.sigmoid(model_data['raw_albedo'])
        self.betas = model_data['betas']
        self.v_offsets = model_data['v_offsets']
        self.v_offsets[SMPLXSeg.eyeball_ids] = 0.
        self.v_offsets[SMPLXSeg.hands_ids] = 0.

        # tex to trimesh texture
        vt = self.vt.copy()
        vt[:, 1] = 1 - vt[:, 1]
        # albedo = T.ToPILImage()(self.raw_albedo.permute(2, 0, 1))
        self.trimesh_visual = trimesh.visual.TextureVisuals(
            uv=vt,
            # image=albedo,
            material=trimesh.visual.texture.SimpleMaterial(
                # image=albedo,
                diffuse=[255, 255, 255, 255],
                ambient=[255, 255, 255, 255],
                specular=[0, 0, 0, 255],
                glossiness=0)
        )

    def forward_talkshow(self, pose_file, video_save_path, interval=5):
        self.v_offsets[SMPLXSeg.lips_ids] = 0
        smplx_params = np.load(pose_file)
        scan_v_posed = []
        smplx_v_posed = []
        smplx_v_canon = []

        for i in tqdm(range(0, smplx_params.shape[0], interval)):
            params_batch = torch.as_tensor(smplx_params[i:i + 1], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                output = self.body_model(
                    betas=self.betas,
                    jaw_pose=params_batch[:, 0:3],
                    global_orient=params_batch[:, 9:12],
                    body_pose=params_batch[:, 12:75].view(-1, 21, 3),
                    left_hand_pose=params_batch[:, 75:120],
                    right_hand_pose=params_batch[:, 120:165],
                    expression=params_batch[:, 165:265],
                    return_verts=True
                )

            v_cano = output.v_posed[0]
            # re-mesh
            v_cano_dense = subdivide_inorder(v_cano, self.smplx_face[SMPLXSeg.remesh_mask], self.unique).squeeze(0)
            # add offsets
            vn = compute_normal(v_cano_dense, self.dense_faces)[0]
            v_cano_dense += self.v_offsets * vn
            # do LBS
            v_posed_dense = warp_points(v_cano_dense, self.dense_lbs_weights, output.joints_transform[:, :55])

            smplx_v_posed.append(output.vertices)
            scan_v_posed.append(v_posed_dense)
            smplx_v_canon.append(output.vertices)

        scan_v_posed = torch.cat(scan_v_posed).detach().cpu().numpy()
        smplx_v_posed = torch.cat(smplx_v_posed).detach().cpu().numpy()
        smplx_v_canon = torch.cat(smplx_v_canon).detach().cpu().numpy()

        new_scan_v_posed, new_face = build_new_mesh(scan_v_posed, self.dense_faces, self.vt, self.ft)

        out_frames = []
        for idx in tqdm(range(0, scan_v_posed.shape[0])):
            mesh = trimesh.Trimesh(new_scan_v_posed[idx], new_face, visual=self.trimesh_visual, process=False)

        os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
        imageio.mimsave(video_save_path, out_frames, fps=30 // interval, quality=8, macro_block_size=1)
        print("save to", video_save_path)

    def forward_aist(self, video_save_path, aist_dir="../data/aist", interval=5):
        os.makedirs(os.path.dirname(video_save_path), exist_ok=True)

        mapping = list(open(f"{aist_dir}/cameras/mapping.txt", 'r').read().splitlines())
        motion_setting_dict = {}
        for pairs in mapping:
            motion, setting = pairs.split(" ")
            motion_setting_dict[motion] = setting
        # motion_name = random.choice(os.listdir(f"{aist_dir}/motion/"))
        motion_name = "gHO_sBM_cAll_d19_mHO0_ch04.pkl"

        # load camera data
        setting = motion_setting_dict[motion_name[:-4]]
        camera_path = open(f"{aist_dir}/cameras/{setting}.json", 'r')
        camera_params = json.load(camera_path)[0]
        rvec = np.array(camera_params['rotation'])
        tvec = np.array(camera_params['translation'])
        matrix = np.array(camera_params['matrix']).reshape((3, 3))
        distortions = np.array(camera_params['distortions'])

        # load motion
        smpl_data = pkl.load(open(f"{aist_dir}/motion/{motion_name}", 'rb'))
        poses = smpl_data['smpl_poses']  # (N, 24, 3)
        scale = smpl_data['smpl_scaling']  # (1,)
        trans = smpl_data['smpl_trans']  # (N, 3)
        poses = torch.from_numpy(poses).view(-1, 24, 3).float()
        # interval = poses.shape[0] // 400
        poses = poses[::interval]
        trans = trans[::interval]
        print("NUM pose", poses.shape, trans.shape)

        scan_v_posed = []
        for i, (pose, t) in tqdm(enumerate(zip(poses, trans))):
            body_pose = torch.as_tensor(pose[None, 1:22].view(1, 21, 3), device=self.device)
            global_orient = torch.as_tensor(pose[None, :1], device=self.device)
            output = self.body_model(
                betas=self.betas,
                global_orient=global_orient,
                jaw_pose=self.jaw_pose,
                body_pose=body_pose,
                expression=self.expression,
                return_verts=True)
            v_cano = output.v_posed[0]
            # re-mesh
            v_cano_dense = subdivide_inorder(v_cano, self.smplx_face[SMPLXSeg.remesh_mask], self.unique).squeeze(0)
            # add offsets
            vn = compute_normal(v_cano_dense, self.dense_faces)[0]
            v_cano_dense += self.v_offsets * vn
            # do LBS
            v_posed_dense = warp_points(v_cano_dense, self.dense_lbs_weights, output.joints_transform[:, :55])
            scan_v_posed.append(v_posed_dense)

        scan_v_posed = torch.cat(scan_v_posed).detach().cpu().numpy()
        new_scan_v_posed, new_face = build_new_mesh(scan_v_posed, self.dense_faces, self.vt, self.ft)

        for i, (posed_points, t) in tqdm(enumerate(zip(new_scan_v_posed, trans))):
            posed_points = posed_points * scale + t[None, :]

            pts2d = cv2.projectPoints(
                posed_points, rvec=rvec, tvec=tvec, cameraMatrix=matrix, distCoeffs=distortions)[0][:, 0]
            posed_points = np.concatenate([pts2d, posed_points[:, 2:]], axis=1)

            posed_points[:, 0] -= 420
            posed_points[:, 1] = 1080 - posed_points[:, 1]
            posed_points = posed_points / 1080 * 2 - 1

            posed_points[:, 1] += 0.35

            mesh = trimesh.Trimesh(posed_points, new_face, visual=self.trimesh_visual, process=False)

    def forward_mdm(self, mdm_file_path, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        mdm_body_pose, translate = get_motion_diffusion_pose(mdm_file_path)
        translate = translate.to(self.device)

        for i, (pose, t) in tqdm(enumerate(zip(mdm_body_pose, translate))):
            body_pose = torch.as_tensor(pose[None, 1:22].view(1, 21, 3), device=self.device)
            global_orient = torch.as_tensor(pose[None, :1], device=self.device)
            output = self.body_model(
                betas=self.betas,
                global_orient=global_orient,
                jaw_pose=self.jaw_pose,
                body_pose=body_pose,
                expression=self.expression,
                return_verts=True
            )
            v_cano = output.v_posed[0]
            # re-mesh
            v_cano_dense = subdivide_inorder(v_cano, self.smplx_face[SMPLXSeg.remesh_mask], self.unique).squeeze(0)
            # add offsets
            vn = compute_normal(v_cano_dense, self.dense_faces)[0]
            v_cano_dense += self.v_offsets * vn
            # do LBS
            v_posed_dense = warp_points(v_cano_dense, self.dense_lbs_weights, output.joints_transform[:, :55])
            # translate
            v_posed_dense += t - translate[0]

            mesh = Mesh(v_posed_dense[0].detach(), self.dense_faces,
                        vt=torch.as_tensor(self.vt),
                        ft=torch.as_tensor(self.ft),
                        albedo=self.raw_albedo)
            mesh.auto_normal()
            mesh.write(f"{save_dir}/{i:03d}/mesh.obj")

    # FBX creation
    num_vertices = 25193
    num_faces = 50080
    num_edges = 75500
    JointsSize = 1 / 3
    SkeletonWeights = []
    nodeDict = {}
    Num2Joints = {5: 'R_Calf', 8: 'R_Foot', 22: 'Jaw', 23: 'L_Eye', 6: 'Spine1', 18: 'L_ForeArm', 9: 'Spine2',
                  1: 'L_Thigh', 52: 'R_Thumb1', 54: 'R_Thumb3', 53: 'R_Thumb2', 12: 'Neck', 21: 'R_Hand', 15: 'Head',
                  24: 'R_Eye', 35: 'L_Ring2', 34: 'L_Ring1', 20: 'L_Hand', 16: 'L_UpperArm', 39: 'L_Thumb3',
                  38: 'L_Thumb2',
                  37: 'L_Thumb1', 50: 'R_Ring2', 51: 'R_Ring3', 49: 'R_Ring1', 27: 'L_Index3', 26: 'L_Index2',
                  25: 'L_Index1', 46: 'R_Pinky1', 17: 'R_UpperArm', 31: 'L_Pinky1', 3: 'Spine', 14: 'R_Shoulder',
                  42: 'R_Index3', 41: 'R_Index2', 36: 'L_Ring3', 40: 'R_Index1', 19: 'R_ForeArm', 10: 'L_Toes',
                  45: 'R_Middle3', 44: 'R_Middle2', 43: 'R_Middle1', 7: 'L_Foot', 32: 'L_Pinky2', 33: 'L_Pinky3',
                  28: 'L_Middle1', 30: 'L_Middle3', 29: 'L_Middle2', 47: 'R_Pinky2', 0: 'Root', 48: 'R_Pinky3',
                  2: 'R_Thigh', 13: 'L_Shoulder', 4: 'L_Calf', 11: 'R_Toes'}

    Joints2Num = {'R_Calf': 5, 'R_Foot': 8, 'Jaw': 22, 'L_Eye': 23, 'Spine1': 6, 'L_ForeArm': 18, 'Spine2': 9,
                  'L_Thigh': 1,
                  'R_Thumb1': 52, 'R_Thumb3': 54, 'R_Thumb2': 53, 'Neck': 12, 'R_Hand': 21, 'Head': 15, 'R_Eye': 24,
                  'L_Ring2': 35, 'L_Ring1': 34, 'L_Hand': 20, 'L_UpperArm': 16, 'L_Thumb3': 39, 'L_Thumb2': 38,
                  'L_Thumb1': 37, 'R_Ring2': 50, 'R_Ring3': 51, 'R_Ring1': 49, 'L_Index3': 27, 'L_Index2': 26,
                  'L_Index1': 25, 'R_Pinky1': 46, 'R_UpperArm': 17, 'L_Pinky1': 31, 'Spine': 3, 'R_Shoulder': 14,
                  'R_Index3': 42, 'R_Index2': 41, 'L_Ring3': 36, 'R_Index1': 40, 'R_ForeArm': 19, 'L_Toes': 10,
                  'R_Middle3': 45, 'R_Middle2': 44, 'R_Middle1': 43, 'L_Foot': 7, 'L_Pinky2': 32, 'L_Pinky3': 33,
                  'L_Middle1': 28, 'L_Middle3': 30, 'L_Middle2': 29, 'R_Pinky2': 47, 'Root': 0, 'R_Pinky3': 48,
                  'R_Thigh': 2, 'L_Shoulder': 13, 'L_Calf': 4, 'R_Toes': 11}

    def CreateScene(self, pSdkManager, pScene):
        # Create scene info
        lSceneInfo = FbxDocumentInfo.Create(pSdkManager, "SceneInfo")
        lSceneInfo.mTitle = "SMPL-X"
        lSceneInfo.mSubject = "SMPL-X model with weighted skin"
        lSceneInfo.mAuthor = "ExportScene01.exe sample program."
        lSceneInfo.mRevision = "rev. 1.0"
        lSceneInfo.mKeywords = "weighted skin"
        lSceneInfo.mComment = "no particular comments required."
        pScene.SetSceneInfo(lSceneInfo)

        lMeshNode = FbxNode.Create(pScene, "meshNode")
        smplxNeutralMesh = self.CreateMesh(pSdkManager, "Mesh")
        lControlPoints = smplxNeutralMesh.GetControlPoints()
        lMeshNode.SetNodeAttribute(smplxNeutralMesh)
        lSkeletonRoot = self.CreateSkeleton(pSdkManager, "Skeleton")

        pScene.GetRootNode().AddChild(lMeshNode)
        pScene.GetRootNode().AddChild(lSkeletonRoot)

        # weightsInfo = open("SkinWeights.txt", "r")
        for i in range(0, 55):
            # self.SkeletonWeights.append(weightsInfo.readline())
            self.SkeletonWeights.append(self.dense_lbs_weights[:,i])
        lSkin = FbxSkin.Create(pSdkManager, "")
        # self.LinkMeshToSkeleton(lSdkManager, lMeshNode, lSkin)
        self.LinkMeshToSkeleton(pSdkManager, lMeshNode, lSkin)
        self.AddShape(pScene, lMeshNode)
        # AnimateSkeleton(pSdkManager, pScene, lSkeletonRoot)

    def AddShape(pScene, node):
        lBlendShape = FbxBlendShape.Create(pScene, "BlendShapes")

        shapeInfo = open("vertexLoc.txt", "r")
        for j in range(0, 1):
            lBlendShapeChannel = FbxBlendShapeChannel.Create(pScene, "ShapeChannel" + str(j))
            lShape = FbxShape.Create(pScene, "Shape" + str(j))
            lShape.InitControlPoints(10475)
            for i in range(0, 10475):
                ctrlPInfo = shapeInfo.readline().split(" ")
                lShape.SetControlPointAt(FbxVector4(0, 0, 0), i)
            lBlendShapeChannel.AddTargetShape(lShape)
            lBlendShape.AddBlendShapeChannel(lBlendShapeChannel)
        node.GetMesh().AddDeformer(lBlendShape)

    # Create the mesh
    def CreateMesh(self, pSdkManager, pName):
        # preparation
        lMesh = FbxMesh.Create(pSdkManager, pName)
        lMesh.InitControlPoints(10475)
        lControlPoints = lMesh.GetControlPoints()
        vertexLocList = []
        # read
        # geoInfo = open("ImprovedGeometryInfo.txt", "r")
        output = self.body_model(
            betas=self.betas,
            jaw_pose=self.jaw_pose,
            expression=self.expression,
            return_verts=True
        )

        v_cano = output.v_posed[0]
        v_cano_dense = subdivide_inorder(v_cano, self.smplx_face[SMPLXSeg.remesh_mask], self.unique).squeeze(0)
        # add offsets
        vn = compute_normal(v_cano_dense, self.dense_faces)[0]
        v_cano_dense += self.v_offsets * vn

        for i in range(0, self.num_vertices):
            # vertexInput = geoInfo.readline().split(' ')
            # locX = vertexInput[0]
            # locY = vertexInput[1]
            # locZ = vertexInput[2]
            # vertexLoc = FbxVector4(float(locX), float(locY), float(locZ))
            # vertexLocList.append(vertexLoc)
            # lControlPoints[i] = vertexLoc

            vertexInput = output.v_posed[i]
            locX = vertexInput[0]
            locY = vertexInput[1]
            locZ = vertexInput[2]
            vertexLoc = FbxVector4(float(locX), float(locY), float(locZ))
            vertexLocList.append(vertexLoc)
            lControlPoints[i] = vertexLoc

        for i in range(0, self.num_faces):
            # fragmentInput = geoInfo.readline().split(' ')
            # fragIndex1 = int(fragmentInput[0]) - 1
            # fragIndex2 = int(fragmentInput[1]) - 1
            # fragIndex3 = int(fragmentInput[2]) - 1
            # lMesh.BeginPolygon(i)  # Material index.
            # lMesh.AddPolygon(fragIndex1)
            # lMesh.AddPolygon(fragIndex2)
            # lMesh.AddPolygon(fragIndex3)
            # # Control point index.
            # lMesh.EndPolygon()

            fragmentInput = self.dense_faces[i]
            fragIndex1 = int(fragmentInput[0]) - 1
            fragIndex2 = int(fragmentInput[1]) - 1
            fragIndex3 = int(fragmentInput[2]) - 1
            lMesh.BeginPolygon(i)  # Material index.
            lMesh.AddPolygon(fragIndex1)
            lMesh.AddPolygon(fragIndex2)
            lMesh.AddPolygon(fragIndex3)
            # Control point index.
            lMesh.EndPolygon()
        for i in range(0, self.num_vertices):
            lMesh.SetControlPointAt(lControlPoints[i], i)
        return lMesh

    # create 55 skeletons for SMPL-X model
    def CreateSkeleton(self, pSdkManager, pName):
        # 2021.4.8 shape influence are not supported now
        # read
        # jointsLoc = open("AdjustedJointsLoc.txt", "r")
        output = self.body_model(
            betas=self.betas,
            jaw_pose=self.jaw_pose,
            expression=self.expression,
            return_verts=True
        )
        jointsLoc = output.joints_transform[0]

        # lSkeletonRootAttribute = FbxSkeleton.Create(lSdkManager, "Root")
        lSkeletonRootAttribute = FbxSkeleton.Create(pSdkManager, "Root")
        lSkeletonRootAttribute.SetSkeletonType(FbxSkeleton.eLimbNode)
        lSkeletonRootAttribute.Size.Set(self.JointsSize)
        # lSkeletonRoot = FbxNode.Create(lSdkManager, "Root")
        lSkeletonRoot = FbxNode.Create(pSdkManager, "Root")
        lSkeletonRoot.SetNodeAttribute(lSkeletonRootAttribute)
        rootInfo = jointsLoc[0]
        lSkeletonRoot.LclTranslation.Set(FbxDouble3(float(rootInfo[0,3]), float(rootInfo[1,3]), float(rootInfo[2,3])))

        self.nodeDict[0] = lSkeletonRoot
        locDict = {0: (float(rootInfo[0,3]), float(rootInfo[1,3]), float(rootInfo[2,3]))}

        for i in range(1, 55):
            skeletonInfo = jointsLoc[i]
            skeletonName = self.Num2Joints[i]
            # skeletonAtrribute = FbxSkeleton.Create(lSdkManager, skeletonName)
            skeletonAtrribute = FbxSkeleton.Create(pSdkManager, skeletonName)
            skeletonAtrribute.SetSkeletonType(FbxSkeleton.eLimbNode)
            skeletonAtrribute.Size.Set(self.JointsSize)
            # skeletonNode = FbxNode.Create(lSdkManager, skeletonName)
            skeletonNode = FbxNode.Create(pSdkManager, skeletonName)
            skeletonNode.SetNodeAttribute(skeletonAtrribute)
            self.nodeDict[i] = skeletonNode
            locDict[i] = (float(skeletonInfo[0,3]), float(skeletonInfo[1,3]), float(skeletonInfo[2,3]))
            # skeletonFather = int(skeletonInfo[0])
            skeletonFather = int(self.body_model.parents[i])
            fatherNode = self.nodeDict[skeletonFather]
            skeletonNode.LclTranslation.Set(
                FbxDouble3(float(float(skeletonInfo[0,3]) - float(locDict[skeletonFather][0,3])),
                           float(float(skeletonInfo[1,3]) - float(locDict[skeletonFather][1,3])),
                           float(float(skeletonInfo[2,3]) - float(locDict[skeletonFather][2,3]))))
            fatherNode.AddChild(skeletonNode)

        return lSkeletonRoot

    def LinkMeshToSkeleton(self, pSdkManager, pMeshNode, lSkin):
        for i in range(0, 55):
            skeletonNode = self.nodeDict[i]
            skeletonName = skeletonNode.GetName()
            skeletonNum = self.Joints2Num[str(skeletonName)]
            # skeletonWeightsInfo = self.SkeletonWeights[skeletonNum].split(' ')
            skeletonWeightsInfo = self.SkeletonWeights[skeletonNum]
            skeletonCluster = FbxCluster.Create(pSdkManager, "")
            skeletonCluster.SetLink(skeletonNode)
            skeletonCluster.SetLinkMode(FbxCluster.eNormalize)
            for j in range(0, self.num_vertices):
                skeletonCluster.AddControlPointIndex(j, float(skeletonWeightsInfo[j]))

            # Now we have the Mesh and the skeleton correctly positioned,
            # set the Transform and TransformLink matrix accordingly.
            lXMatrix = FbxAMatrix()
            lScene = pMeshNode.GetScene()
            if lScene:
                lXMatrix = lScene.GetAnimationEvaluator().GetNodeGlobalTransform(pMeshNode)
            skeletonCluster.SetTransformMatrix(lXMatrix)
            lScene = skeletonNode.GetScene()
            if lScene:
                lXMatrix = lScene.GetAnimationEvaluator().GetNodeGlobalTransform(skeletonNode)
            skeletonCluster.SetTransformLinkMatrix(lXMatrix)
            # Add the clusters to the Mesh by creating a skin and adding those clusters to that skin.
            # After add that skin.
            lSkin.AddCluster(skeletonCluster)

        pMeshNode.GetNodeAttribute().AddDeformer(lSkin)

    def print(self):
        output = self.body_model(
            betas=self.betas,
            jaw_pose=self.jaw_pose,
            expression=self.expression,
            return_verts=True
        )

        v_cano = output.v_posed[0]
        v_cano_dense = subdivide_inorder(v_cano, self.smplx_face[SMPLXSeg.remesh_mask], self.unique).squeeze(0)
        # add offsets
        vn = compute_normal(v_cano_dense, self.dense_faces)[0]
        v_cano_dense += self.v_offsets * vn

        print(output.v_posed.size())
        print (v_cano_dense.size())
        print(output.joints_transform.size())
        print(output.joints_transform[0])

if __name__ == '__main__':
    try:
        import apps.FbxCommon
        from fbx import *
    except ImportError:
        print("Error: module FbxCommon and/or fbx failed to import.\n")
        print(
            "Copy the files located in the compatible sub-folder lib/python<version> into your python interpreter site-packages folder.")
        import platform

        if platform.system() == 'Windows' or platform.system() == 'Microsoft':
            print('For example: copy ..\\..\\lib\\Python27_x64\\* C:\\Python27\\Lib\\site-packages')
        elif platform.system() == 'Linux':
            print('For example: cp ../../lib/Python27_x64/* /usr/local/lib/python2.7/site-packages')
        elif platform.system() == 'Darwin':
            print(
                'For example: cp ../../lib/Python27_x64/* /Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages')
        sys.exit(1)

    ckpt_file = f"Avatar-100/Abraham Lincoln/params.pt"
    assert os.path.exists(ckpt_file)
    animator = Animation()
    animator.load_ckpt_data(ckpt_file)

    animator.print()

    # # Prepare the FBX SDK.
    # (lSdkManager, lScene) = FbxCommon.InitializeSdkObjects()
    #
    # # Create the scene.
    # lResult = animator.CreateScene(lSdkManager, lScene)
    #
    # if lResult == False:
    #     print("\n\nAn error occurred while creating the scene...\n")
    #     lSdkManager.Destroy()
    #     sys.exit(1)
    #
    # lSdkManager.GetIOSettings().SetBoolProp(EXP_FBX_EMBEDDED, True)
    # lFileFormat = lSdkManager.GetIOPluginRegistry().GetNativeWriterFormat()
    #
    # # Save the scene.
    # # The example can take an output file name as an argument.
    # if len(sys.argv) > 1:
    #     lResult = FbxCommon.SaveScene(lSdkManager, lScene, sys.argv[1])
    # # A default output file name is given otherwise.
    # else:
    #     SAMPLE_FILENAME = "ExportShape.fbx"
    #     lResult = FbxCommon.SaveScene(lSdkManager, lScene, SAMPLE_FILENAME)
    #
    # if lResult == False:
    #     print("\n\nAn error occurred while saving the scene...\n")
    #     lSdkManager.Destroy()
    #     sys.exit(1)
    #
    # # Destroy all objects created by the FBX SDK.
    # lSdkManager.Destroy()

    sys.exit(0)

