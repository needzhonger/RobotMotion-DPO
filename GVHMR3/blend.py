import os
import bpy
import numpy as np
import pickle as pkl
import math
import joblib
from scipy.spatial.transform import Rotation as R
from mathutils import Vector, Quaternion

# from transforms import quaternion_to_matrix_np, matrix_to_quaternion_np, yup_to_zup_rotation_matrices


byd_joint_names = ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint',
    'left_hip_roll_joint', 'right_hip_roll_joint', 'waist_roll_joint','left_hip_yaw_joint',
    'right_hip_yaw_joint', 'waist_pitch_joint', 'left_knee_joint', 'right_knee_joint',
    'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint',
     'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_ankle_roll_joint', 
     'right_ankle_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 
     'left_elbow_joint', 'right_elbow_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint',
      'left_wrist_pitch_joint', 'right_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint']
mujoco_joint_names = [
    'left_hip_pitch', 'left_hip_roll', 'left_hip_yaw', 'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
    'right_hip_pitch', 'right_hip_roll', 'right_hip_yaw', 'right_knee', 'right_ankle_pitch', 'right_ankle_roll',
    'waist_yaw', 'waist_roll', 'waist_pitch',
    'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw', 'left_elbow', 'left_wrist_roll', 'left_wrist_pitch', 'left_wrist_yaw',
    'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw', 'right_elbow', 'right_wrist_roll', 'right_wrist_pitch', 'right_wrist_yaw',
]
byd_joint_to_mujoco_joint = [byd_joint_names.index(joint_name+'_joint') for joint_name in mujoco_joint_names]
mujoco_joint_to_byd_joint = [mujoco_joint_names.index(joint_name[:-6]) for joint_name in byd_joint_names]


G1_JOINT_NAME = [
    'pelvis',
    'left_hip_pitch', 'left_hip_roll', 'left_hip_yaw', 'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
    'right_hip_pitch', 'right_hip_roll', 'right_hip_yaw', 'right_knee', 'right_ankle_pitch', 'right_ankle_roll',
    'waist_yaw', 'waist_roll', 'waist_pitch',
    'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw', 'left_elbow', 'left_wrist_roll', 'left_wrist_pitch', 'left_wrist_yaw',
    'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw', 'right_elbow', 'right_wrist_roll', 'right_wrist_pitch', 'right_wrist_yaw',
]

HIP_KNEE_NAME_LEFT = ['left_hip_pitch', 'left_knee', 'left_hip_yaw', 'left_hip_roll']
HIP_KNEE_NAME_RIGHT = ['right_hip_pitch', 'right_knee', 'right_hip_yaw', 'right_hip_roll']

def animation_data_clear(obj):

    obj.animation_data_clear()
    obj.data.animation_data_clear()

def yup_to_zup_quat(quat):
    """
    Simpler conversion method
    Args:
        quat: numpy array of shape (b, 4), xyzw
    Returns:
        converted quaternions of shape (b, 4)
    """
    # return np.stack([quat[..., 0], quat[..., 1], -quat[..., 3], quat[..., 2]], axis=-1)
    return np.stack([quat[..., 0], -quat[..., 2], quat[..., 1], quat[..., 3]], axis=-1)

def zup_to_yup_quat(quat):
    """
    Simpler conversion method
    Args:
        quat: numpy array of shape (b, 4), xyzw
    Returns:
        converted quaternions of shape (b, 4)
    """
    # return np.stack([quat[..., 0], quat[..., 1], quat[..., 3], -quat[..., 2]], axis=-1)
    return np.stack([quat[..., 0], quat[..., 2], -quat[..., 1], quat[..., 3]], axis=-1)

def load_g1_animation_fast(armature, root_pos, root_rot, dof_pos, bonename_list=None, is_keyframe=None, is_keyframe_foot=None, scale_waist=False):
    '''
    :param armature: bpy.types.Object
    :param root_pos: y-up, N x 3 numpy array
    :param root_rot: quaternion, xyzw, y-up, N x 4 numpy array
    :param dof_pos: euler around y-axis, N x 29 numpy array
    '''

    assert len(root_pos) == len(root_rot) == len(dof_pos)

    animation_data_clear(armature)

    bpy.context.view_layer.objects.active = armature  # mesh needs to be active object for recalculating joint locations
    num_keyframes = len(root_pos)

    armature.pose.bones['pelvis'].rotation_mode = 'QUATERNION'
    root_rot = zup_to_yup_quat(root_rot)
    root_rot = root_rot[..., [3, 0, 1, 2]]

    root_pos = root_pos.copy() # avoid changing root_pos outside
    root_pos[..., [1, 2]] = root_pos[..., [2, 1]].copy()
    root_pos[..., 2] *= -1

    bpy.ops.object.mode_set(mode='OBJECT')

    animation_data = armature.animation_data_create()
    action = animation_data.action = bpy.data.actions.new(f'{armature.name}Action')

    for i in range(3):
        fcurve = action.fcurves.new('pose.bones["pelvis"].location', index=i)
        fcurve.keyframe_points.add(count=num_keyframes)
        fcurve.keyframe_points.foreach_set("co", [x for co in zip(range(1, num_keyframes+1), root_pos[:, i]) for x in co])
        fcurve.update()
    for i in range(4):
        fcurve = action.fcurves.new('pose.bones["pelvis"].rotation_quaternion', index=i)
        fcurve.keyframe_points.add(count=num_keyframes)
        fcurve.keyframe_points.foreach_set("co", [x for co in zip(range(1, num_keyframes + 1), root_rot[:, i]) for x in co])
        fcurve.update()

    # prefix_range = 356
    prefix_range = 85
    subfix_range = 0
    arm_init_reserve_idx = prefix_range + 13

    foot_init_blend_idx_list = [85, 86, 87, 88, 89, 90, 91]

    if is_keyframe_foot is not None:
        is_keyframe_foot_left = [i for idx, x in enumerate(is_keyframe_foot) if idx%2==1 for i in range(x-1, x+6)] + foot_init_blend_idx_list
        is_keyframe_foot_right = [i for idx, x in enumerate(is_keyframe_foot) if idx%2==0 for i in range(x-1, x+6)] + list(range(prefix_range-1, prefix_range+6))

    for idx, bone_name in enumerate(G1_JOINT_NAME[1:]):
        if bonename_list is None or bone_name not in bonename_list:
            if bone_name in HIP_KNEE_NAME_LEFT and is_keyframe_foot is not None:
                dof = dof_pos[:, idx]
                reserve_idx = [i for i in range(num_keyframes) if i not in is_keyframe_foot_left]

                fcurve = action.fcurves.new(f'pose.bones["{bone_name}"].rotation_euler', index=1)
                fcurve.keyframe_points.add(count=len(reserve_idx))
                co = [x for co in zip(range(1, num_keyframes + 1), dof) if co[0]-1 not in is_keyframe_foot_left for x in co]
                fcurve.keyframe_points.foreach_set("co", co)
                fcurve.update()
            elif bone_name in HIP_KNEE_NAME_RIGHT and is_keyframe_foot is not None:
                dof = dof_pos[:, idx]
                reserve_idx = [i for i in range(num_keyframes) if i not in is_keyframe_foot_right]

                fcurve = action.fcurves.new(f'pose.bones["{bone_name}"].rotation_euler', index=1)
                fcurve.keyframe_points.add(count=len(reserve_idx))
                co = [x for co in zip(range(1, num_keyframes + 1), dof) if co[0]-1 not in is_keyframe_foot_right for x in co]
                fcurve.keyframe_points.foreach_set("co", co)
                fcurve.update()
            else:
                dof = dof_pos[:, idx]
                fcurve = action.fcurves.new(f'pose.bones["{bone_name}"].rotation_euler', index=1)
                fcurve.keyframe_points.add(count=num_keyframes)
                co = [x for co in zip(range(1, num_keyframes + 1), dof) for x in co]
                # print(co)
                # breakpoint()
                fcurve.keyframe_points.foreach_set("co", co)
                fcurve.update()
        else:
            # arm blending
            dof = dof_pos[:, idx]
            reserve_idx = [i for i in range(num_keyframes) if i in is_keyframe or i < prefix_range or i >= num_keyframes - subfix_range]

            fcurve = action.fcurves.new(f'pose.bones["{bone_name}"].rotation_euler', index=1)
            fcurve.keyframe_points.add(count=len(reserve_idx))
            co = [x for co in zip(range(1, num_keyframes+1), dof) if co[0]-1 in is_keyframe or co[0] < prefix_range or co[0] >= num_keyframes - subfix_range or co[0] == arm_init_reserve_idx for x in co]

            fcurve.keyframe_points.foreach_set("co", co)
            fcurve.update()

    bpy.context.scene.frame_set(1)

    if scale_waist:
        armature_name = armature.name
        bone_name = "waist_pitch"
        scalar = 0.7  # Scaling factor - 1.5 will make movements 50% larger
        straight_value = -10.0 / 180 * math.pi  # The reference value (usually 0 for rotation, but could be different)

        scale_bone_fcurve(armature_name, bone_name, scalar, straight_value)



    return {'FINISHED'}




def scale_bone_fcurve(armature_name, bone_name, scalar, straight_value):
    """
    Scale the FCurve of a specified bone based on a scalar value and a straight value.

    Parameters:
    - armature_name: Name of the armature
    - bone_name: Name of the bone to modify
    - scalar: Scaling factor for the FCurve
    - straight_value: Base value to adjust the scaling around
    """
    # Get the armature object
    if armature_name not in bpy.data.objects:
        print(f"Error: Armature '{armature_name}' not found")
        return False

    armature = bpy.data.objects[armature_name]

    # Check if it's an armature
    if armature.type != 'ARMATURE':
        print(f"Error: '{armature_name}' is not an armature")
        return False

    # Check if the bone exists
    if bone_name not in armature.pose.bones:
        print(f"Error: '{bone_name}' bone not found in the armature")
        return False

    # Get animation data
    if not (armature.animation_data and armature.animation_data.action):
        print("No animation data found on the armature")
        return False

    action = armature.animation_data.action
    data_path = f'pose.bones["{bone_name}"].rotation_euler'

    # Find FCurves for this bone's rotation
    fcurves = [fc for fc in action.fcurves if fc.data_path == data_path]

    if not fcurves:
        print(f"No FCurves found for '{bone_name}' bone rotation")
        return False

    # Store the current frame
    current_frame = bpy.context.scene.frame_current

    # Process each FCurve (x, y, z rotation)
    for fc in fcurves:
        # Identify which axis this fcurve controls (0=X, 1=Y, 2=Z)
        axis_index = fc.array_index
        print(axis_index)
        if axis_index != 1:
            continue

        # Scale keyframe values on this curve
        for keyframe in fc.keyframe_points:
            # Get the original value
            original_value = keyframe.co[1]

            # Calculate the offset from the straight_value
            offset = original_value - straight_value

            # Scale the offset and add back to the straight_value
            new_value = straight_value + (offset * scalar)

            # Update the keyframe value
            keyframe.co[1] = new_value
            keyframe.handle_left[1] = new_value
            keyframe.handle_right[1] = new_value

        # Update the FCurve
        fc.update()

    # Restore the original current frame
    bpy.context.scene.frame_set(current_frame)

    print(f"Scaled '{bone_name}' FCurves by factor {scalar} around value {straight_value}")
    return True


if __name__ == "__main__":
    
    filename = '''
/home/eerrr/GVHMR1/outputs/infer_g1/BMLrub_rub108_0018_lifting_light2_stageii_g1.npz
'''.replace('\n', '')
    #filename = '''
#/home/eerrr/GVHMR1/outputs/g1_fk_check.npz
#'''.replace('\n', '')
    #filename = '''
#/home/eerrr/GVHMR1/g1_paired/val/g1_test/DFaust_50009_50009_hips_stageii.npz
#'''.replace('\n', '')
    # np load
#    with open(filename, 'rb') as f:
#        output_data = np.load(f, allow_pickle=True).item()
    
    # pkl load
#    with open(filename, 'rb') as f:
#        output_data = pkl.load(f)#['data']
        
    # joblib load
#    with open(filename, 'rb') as f:
#        output_data = joblib.load(f)
#        data_key = list(output_data.keys())[0]
#        output_data = output_data[data_key]
    
    # unitree lafan1 csv load
#    data = np.loadtxt(filename, delimiter=',')
#    output_data = {}
#    output_data['root_pos'] = data[:, :3]
#    output_data['root_rot'] = data[:, 3:7]
#    output_data['dof_pos'] = data[:, 7:]
    
    # bones-seed csv load
#    data = np.loadtxt(filename, delimiter=',', skiprows=1)[:, 1:]
#    data[:, :3] /= 100.   # cm to m
#    data[:, 3:] = data[:, 3:] / 180 * np.pi
#    output_data = {}
#    output_data['root_pos'] = data[:, :3]
#    quats = R.from_euler('xyz', data[:, 3:6]).as_quat()
#    output_data['root_rot'] = quats
#    output_data['dof_pos'] = data[:, 6:]

       # npz load
    data = np.load(filename, allow_pickle=True)
    output_data = {}
    #output_data['root_pos'] = data['body_pos_w'][:, 0, :]
    #quat_wxyz = data['body_quat_w'][:, 0, :]
    output_data['root_pos'] = data['root_pos_w']
    quat_wxyz = data['root_quat_w']
    output_data['root_rot'] = np.concatenate([quat_wxyz[:, 1:], quat_wxyz[:, :1]], axis=1)
    output_data['dof_pos'] = data['joint_pos'][:, byd_joint_to_mujoco_joint] 
    


    min_len = 3000
    
    load_g1_animation_fast(armature=bpy.data.objects[f'g1'],
                       root_pos=output_data['root_pos'][:min_len],
                       root_rot=output_data['root_rot'][:min_len],
                       dof_pos=output_data['dof_pos'][:min_len]) 
                       
                       
#    load_g1_animation_fast(armature=bpy.data.objects[f'g1'],
#                       root_pos=output_data['root_trans_offset'][:min_len],
#                       root_rot=output_data['root_rot'][:min_len],
#                       dof_pos=output_data['dof'][:min_len])
                       