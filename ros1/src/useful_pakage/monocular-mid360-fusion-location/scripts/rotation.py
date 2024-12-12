import numpy as np

def rotation_matrix_from_euler(roll, pitch, yaw):
    # 计算旋转矩阵
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    R = R_z @ R_y @ R_x
    return R
    
roll = np.deg2rad(0)
pitch = np.deg2rad(0)
yaw = np.deg2rad(0)

R = rotation_matrix_from_euler(roll, pitch, yaw)
print(R)