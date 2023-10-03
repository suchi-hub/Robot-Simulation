! pip install modern_robotics
import scipy
import numpy as np
import pandas as pd
import modern_robotics as mr
np.set_printoptions(formatter={'float_kind':'{:.5f}'.format})
import matplotlib.pyplot as plt

"""MILESTONE 1 START

---


"""

def NextState(current_config, joint_speed, delta_t, speed_limit):

  current_config = current_config
  joint_speed = joint_speed
  delta_t = delta_t
  speed_limit = speed_limit
  joint_speed[joint_speed > speed_limit] = speed_limit
  joint_speed[joint_speed < -speed_limit] = -speed_limit
  F = np.array([[-2.5974, 2.5974, 2.5974, -2.5974],
                [1,1,1,1],
                [-1,1,-1,1]])*(0.0475/4)

  new_joint_angles = current_config[3:8] + joint_speed[:5]*delta_t
  new_wheel_angles = current_config[8:] + joint_speed[5:]*delta_t
  delta_wheel_angles = joint_speed[5:]*delta_t
  twist_v = np.matmul(F, delta_wheel_angles)
  twist_v6 = np.insert(twist_v, 0, [0,0], axis=0)
  twist_v6 = np.append(twist_v6, [0], axis=0)
  bracket_twist_v6 = mr.VecTose3(twist_v6)
  T_k_kplus1 = mr.MatrixExp6(bracket_twist_v6)
  T_s_k = np.array([[np.cos(current_config[0]), -np.sin(current_config[0]), 0, current_config[1]],
                    [np.sin(current_config[0]), np.cos(current_config[0]), 0, current_config[2]],
                    [0, 0, 1, 0.0963],
                    [0, 0, 0, 1]])
  T_s_kplus1 = np.matmul(T_s_k, T_k_kplus1)
  q_kplus1 = np.array([np.arccos(T_s_kplus1[0,0]),T_s_kplus1[0,3], T_s_kplus1[1,3]])

  new_chasis_config = np.concatenate((q_kplus1, new_joint_angles, new_wheel_angles), axis=0)
  return new_chasis_config

# for loop to create the robot motion csv file
current_config = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
joint_speed = np.array([0.1,0.1,0.1,0.1,0.1,-10,10,10,-10])
delta_t = 0.01
speed_limit = 10

csv = current_config.reshape(1,-1)
time = 0
while time < 1:
  iter = NextState(current_config, joint_speed, delta_t, speed_limit)
  # print(iter,'\n',time)
  time = time + delta_t
  csv = np.concatenate((csv, iter.reshape(1,-1)), axis=0)
  current_config = iter

csv = np.concatenate((csv, np.zeros((csv.shape[0],1))), axis=1)
np.savetxt('robot motion.csv', csv, fmt='%.6f', delimiter=',')

"""---

MILESTONE 1 END

MILESTONE 2 START - TRAJECTORY GENERATOR

---
"""

def TrajectoryGenerator(T_s_e, T_s_cube_initial, T_s_cube_final, T_cube_e_grasp, T_cube_e_standoff, reference_config_k):

  # Trajectory1 - to standoff position
  Xstart = T_s_e
  Xend = np.matmul(T_s_cube_initial, T_cube_e_standoff)
  X_time = (np.sqrt(np.sum((Xstart[[0,2],3]-Xend[[0,2],3])**2)))/0.1
  gripper_position = 0
  trajectory_1 = mr.ScrewTrajectory(Xstart, Xend, X_time, X_time/(delta_t*reference_config_k), 3)
  segment_1 = (np.insert(trajectory_1[0][0:3,3], 0, (trajectory_1[0][0:3,0:3]).reshape(-1))).reshape(1,-1)
  for x in trajectory_1:
    segment_1 = np.concatenate((segment_1, (np.insert(x[0:3,3], 0, (x[0:3,0:3]).reshape(-1)).reshape(1,-1))), axis=0)
  segment_1 = np.concatenate((segment_1, np.ones((segment_1.shape[0],1))*gripper_position), axis=1)

  # Trajectory2 - to grasp position
  Xstart = np.matmul(T_s_cube_initial, T_cube_e_standoff)
  Xend = np.matmul(T_s_cube_initial, T_cube_e_grasp)
  X_time = (np.sqrt(np.sum((Xstart[[0,2],3]-Xend[[0,2],3])**2)))/(0.1*0.5)
  gripper_position = 0
  trajectory_2 = mr.ScrewTrajectory(Xstart, Xend, X_time, X_time/(delta_t*reference_config_k), 3)
  segment_2 = (np.insert(trajectory_2[0][0:3,3], 0, (trajectory_2[0][0:3,0:3]).reshape(-1))).reshape(1,-1)
  for x in trajectory_2:
    segment_2 = np.concatenate((segment_2, (np.insert(x[0:3,3], 0, (x[0:3,0:3]).reshape(-1)).reshape(1,-1))), axis=0)
  segment_2 = np.concatenate((segment_2, np.ones((segment_2.shape[0],1))*gripper_position), axis=1)

  # Trajectory3 - grasping cube
  segment_3 = np.zeros((65,13))
  segment_3[:] = segment_2[-1]
  segment_3[:,-1] = 1

  # Trajectory4 - to standoff position with cube
  Xstart = np.matmul(T_s_cube_initial, T_cube_e_grasp)
  Xend = np.matmul(T_s_cube_initial, T_cube_e_standoff)
  X_time = (np.sqrt(np.sum((Xstart[[0,2],3]-Xend[[0,2],3])**2)))/(0.1*0.5)
  gripper_position = 1
  trajectory_4 = mr.ScrewTrajectory(Xstart, Xend, X_time, X_time/(delta_t*reference_config_k), 3)
  segment_4 = (np.insert(trajectory_4[0][0:3,3], 0, (trajectory_4[0][0:3,0:3]).reshape(-1))).reshape(1,-1)
  for x in trajectory_4:
    segment_4 = np.concatenate((segment_4, (np.insert(x[0:3,3], 0, (x[0:3,0:3]).reshape(-1)).reshape(1,-1))), axis=0)
  segment_4 = np.concatenate((segment_4, np.ones((segment_4.shape[0],1))*gripper_position), axis=1)

  # Trajectory5 - to cube final standoff position with cube
  Xstart = np.matmul(T_s_cube_initial, T_cube_e_standoff)
  Xend = np.matmul(T_s_cube_final, T_cube_e_standoff)
  X_time = (np.sqrt(np.sum((Xstart[[0,2],3]-Xend[[0,2],3])**2)))/0.1
  gripper_position = 1
  trajectory_5 = mr.ScrewTrajectory(Xstart, Xend, X_time, X_time/(delta_t*reference_config_k), 3)
  segment_5 = (np.insert(trajectory_5[0][0:3,3], 0, (trajectory_5[0][0:3,0:3]).reshape(-1))).reshape(1,-1)
  for x in trajectory_5:
    segment_5 = np.concatenate((segment_5, (np.insert(x[0:3,3], 0, (x[0:3,0:3]).reshape(-1)).reshape(1,-1))), axis=0)
  segment_5 = np.concatenate((segment_5, np.ones((segment_5.shape[0],1))*gripper_position), axis=1)

  # Trajectory6 - to cube final grasp position with cube
  Xstart = np.matmul(T_s_cube_final, T_cube_e_standoff)
  Xend = np.matmul(T_s_cube_final, T_cube_e_grasp)
  X_time = (np.sqrt(np.sum((Xstart[[0,2],3]-Xend[[0,2],3])**2)))/(0.1*0.5)
  gripper_position = 1
  trajectory_6 = mr.ScrewTrajectory(Xstart, Xend, X_time, X_time/(delta_t*reference_config_k), 3)
  segment_6 = (np.insert(trajectory_6[0][0:3,3], 0, (trajectory_6[0][0:3,0:3]).reshape(-1))).reshape(1,-1)
  for x in trajectory_6:
    segment_6 = np.concatenate((segment_6, (np.insert(x[0:3,3], 0, (x[0:3,0:3]).reshape(-1)).reshape(1,-1))), axis=0)
  segment_6 = np.concatenate((segment_6, np.ones((segment_6.shape[0],1))*gripper_position), axis=1)

  # Trajectory7 - ungrasping cube
  segment_7 = np.zeros((65,13))
  segment_7[:] = segment_6[-1]
  segment_7[:,-1] = 0

  # Trajectory8 - to  final standoff position
  Xstart = np.matmul(T_s_cube_final, T_cube_e_grasp)
  Xend = np.matmul(T_s_cube_final, T_cube_e_standoff)
  X_time = (np.sqrt(np.sum((Xstart[[0,2],3]-Xend[[0,2],3])**2)))/(0.1*0.5)
  gripper_position = 0
  trajectory_8 = mr.ScrewTrajectory(Xstart, Xend, X_time, X_time/(delta_t*reference_config_k), 3)
  segment_8 = (np.insert(trajectory_8[0][0:3,3], 0, (trajectory_8[0][0:3,0:3]).reshape(-1))).reshape(1,-1)
  for x in trajectory_8:
    segment_8 = np.concatenate((segment_8, (np.insert(x[0:3,3], 0, (x[0:3,0:3]).reshape(-1)).reshape(1,-1))), axis=0)
  segment_8 = np.concatenate((segment_8, np.ones((segment_8.shape[0],1))*gripper_position), axis=1)

  end_effector_path = np.concatenate((segment_1, segment_2, segment_3, segment_4, segment_5, segment_6, segment_7, segment_8), axis=0)
  np.savetxt('end effector path.csv', end_effector_path, fmt='%.6f', delimiter=',')
  return end_effector_path

# Requirements for input
T_s_e = np.array([[0,0,1,0],
                  [0,1,0,0],
                  [-1,0,0,0.5],
                  [0,0,0,1]])
T_s_cube_initial = np.array([[1,0,0,1],
                             [0,1,0,0],  # new task change y from 0 to 1
                             [0,0,1,0.025],
                             [0,0,0,1]])
T_s_cube_final = np.array([[0,1,0,0],    # new task change x from 0 to -0.5
                           [-1,0,0,-1],
                           [0,0,1,0.025],
                           [0,0,0,1]])
T_cube_e_grasp = np.array([[np.cos(-0.75*np.pi), 0, -np.sin(-0.75*np.pi), 0],
                           [0, 1, 0, 0],
                           [np.sin(-0.75*np.pi), 0, np.cos(-0.75*np.pi), 0],
                           [0,0,0,1]])
T_cube_e_standoff = np.array([[np.cos(-0.75*np.pi), 0, -np.sin(-0.75*np.pi), 0],
                           [0, 1, 0, 0],
                           [np.sin(-0.75*np.pi), 0, np.cos(-0.75*np.pi), 0.1],
                           [0,0,0,1]])
reference_config_k = 1
# calling function
end_effector_path = TrajectoryGenerator(T_s_e, T_s_cube_initial, T_s_cube_final, T_cube_e_grasp, T_cube_e_standoff, reference_config_k)

"""---
MILESTONE 2 END

MILESTONE 3 START - FEEDBACK CONTROL

---
"""

# inputs
M_o_e = np.array([[1,0,0,0.033],
                  [0,1,0,0],
                  [0,0,1,0.6546],
                  [0,0,0,1]])
screw_b_list = np.array([[0,0,1,0,0.033,0],
                         [0,-1,0,-0.5076,0,0],
                         [0,-1,0,-0.3526,0,0],
                         [0,-1,0,-0.2176,0,0],
                         [0,0,1,0,0,0]]).T
F_base = np.array([[0,0,0,0],
              [0,0,0,0],
              [-2.5974, 2.5974, 2.5974, -2.5974],
              [1,1,1,1],
              [-1,1,-1,1],
              [0,0,0,0]])*(0.0475/4)
T_b_o = np.array([[1,0,0,0.1662],
                  [0,1,0,0],
                  [0,0,1,0.0026],
                  [0,0,0,1]])
gain_k_p = np.diag((1,1,1,1,1,1))*0                           #input kP
gain_k_i = np.diag((1,1,1,1,1,1))*0                           #input kI
delta_t = 0.01                                                #input delta_t

robot_config = np.array([0,0,0,0,0,0.2,-1.6,0,0,0,0,0])             # input q, theta values to calculate T_s_e

J_arm = mr.JacobianBody(screw_b_list, robot_config[3:8])
T_o_e = mr.FKinBody(M_o_e, screw_b_list, robot_config[3:8])
J_base = np.matmul(mr.Adjoint(np.matmul(mr.TransInv(T_o_e), mr.TransInv(T_b_o))), F_base)
J_base_plus_arm = np.concatenate((J_base, J_arm), axis=1)
T_s_b = np.array([[np.cos(robot_config[0]), -np.sin(robot_config[0]), 0, robot_config[1]],
                    [np.sin(robot_config[0]), np.cos(robot_config[0]), 0, robot_config[2]],
                    [0, 0, 1, 0.0963],
                    [0, 0, 0, 1]])
T_b_e = np.matmul(T_b_o, T_o_e)
current_actual_e_config = np.matmul(T_s_b, T_b_e)             #input T_s_e

current_ref_e_config = np.array([[0,0,1,0.5],                 #input T_d
                                 [0,1,0,0],
                                 [-1,0,0,0.5],
                                 [0,0,0,1]])
current_refplus1_e_config = np.array([[0,0,1,0.6],            #input T_d+1
                                      [0,1,0,0],
                                      [-1,0,0,0.3],
                                      [0,0,0,1]])

# defining function FeedbackControl
def FeedbackControl(current_actual_e_config, current_ref_e_config, current_refplus1_e_config, gain_k_p, gain_k_i, delta_t, x_error_integral=0):
  twist_e_desired = (mr.se3ToVec(mr.MatrixLog6(np.matmul(mr.TransInv(current_ref_e_config), current_refplus1_e_config))))/delta_t
  x_error = mr.se3ToVec(mr.MatrixLog6(np.matmul(mr.TransInv(current_actual_e_config), current_ref_e_config)))

  twist_e_commanded = (np.matmul(mr.Adjoint(np.matmul(mr.TransInv(current_actual_e_config), current_ref_e_config)), twist_e_desired)
                      + np.matmul(gain_k_p, x_error) + np.matmul(gain_k_i, (x_error_integral + x_error*delta_t)))
  x_error_integral = x_error_integral + x_error*delta_t
  J_inv_base_plus_arm = np.around(np.linalg.pinv(J_base_plus_arm, rcond=1e-4), decimals=6)
  # J_inv_base_plus_arm = np.linalg.pinv(J_base_plus_arm)
  calculated_controls_u_thetadot = np.matmul(J_inv_base_plus_arm, twist_e_commanded)
  return (calculated_controls_u_thetadot, x_error, x_error_integral)

# calling function
(calculated_controls_u_thetadot, x_error, x_error_integral) = FeedbackControl(current_actual_e_config, current_ref_e_config, current_refplus1_e_config, gain_k_p, gain_k_i, delta_t)

"""---
MILESTONE 3 END

FINAL LOOP START - PUTTING EVERYTHING TOGETHER

---
"""

# Main program - running everything learnt till now in loop
robot_config = np.array([-0.5, 0.5, -0.5, -0.2, -0.2, -0.2, -0.2, -0.2, 0, 0, 0, 0])  # initial robot config - change this go add initial error
# robot_config = np.array([2, -0.5, -0.5, -0.2, -0.2, -0.2, -0.2, -0.2, 0, 0, 0, 0])
M_o_e = np.array([[1,0,0,0.033],
                  [0,1,0,0],
                  [0,0,1,0.6546],
                  [0,0,0,1]])
screw_b_list = np.array([[0,0,1,0,0.033,0],
                         [0,-1,0,-0.5076,0,0],
                         [0,-1,0,-0.3526,0,0],
                         [0,-1,0,-0.2176,0,0],
                         [0,0,1,0,0,0]]).T
F_base = np.array([[0,0,0,0],
              [0,0,0,0],
              [-2.5974, 2.5974, 2.5974, -2.5974],
              [1,1,1,1],
              [-1,1,-1,1],
              [0,0,0,0]])*(0.0475/4)
T_b_o = np.array([[1,0,0,0.1662],
                  [0,1,0,0],
                  [0,0,1,0.0026],
                  [0,0,0,1]])
gain_k_p = np.diag((1,1,1,1,1,1))*1                          #input kP
gain_k_i = np.diag((1,1,1,1,1,1))*0.01                           #input kI
delta_t = 0.01                                                #input delta_t
speed_limit = 10
x_error_integral = 0

log_robot_config = np.append(robot_config, 0)
log_x_error = np.array([[0,0,0,0,0,0]])
for i in np.arange(end_effector_path.shape[0] -1):            #for loop to call the funtions in milestone 1,2 and 3 to get q and theta values
  J_arm = mr.JacobianBody(screw_b_list, robot_config[3:8])
  T_o_e = mr.FKinBody(M_o_e, screw_b_list, robot_config[3:8])
  J_base = np.matmul(mr.Adjoint(np.matmul(mr.TransInv(T_o_e), mr.TransInv(T_b_o))), F_base)
  J_base_plus_arm = np.concatenate((J_base, J_arm), axis=1)
  T_s_b = np.array([[np.cos(robot_config[0]), -np.sin(robot_config[0]), 0, robot_config[1]],
                      [np.sin(robot_config[0]), np.cos(robot_config[0]), 0, robot_config[2]],
                      [0, 0, 1, 0.0963],
                      [0, 0, 0, 1]])
  T_b_e = np.matmul(T_b_o, T_o_e)
  current_actual_e_config = np.matmul(T_s_b, T_b_e)             #input T_s_e
  current_ref_e_config = np.concatenate((end_effector_path[i][:9].reshape(3,3), end_effector_path[i][9:12].reshape(3,-1)), axis=1)            #input T_d
  current_ref_e_config = np.concatenate((current_ref_e_config, [[0,0,0,1]]), axis=0)
  current_refplus1_e_config = np.concatenate((end_effector_path[i+1][:9].reshape(3,3), end_effector_path[i+1][9:12].reshape(3,-1)), axis=1)   #input T_d+1
  current_refplus1_e_config = np.concatenate((current_refplus1_e_config, [[0,0,0,1]]), axis=0)
  (calculated_controls_u_thetadot, x_error, x_error_integral) = FeedbackControl(current_actual_e_config, current_ref_e_config, current_refplus1_e_config, gain_k_p, gain_k_i, delta_t, x_error_integral)
  log_x_error = np.concatenate((log_x_error, x_error.reshape(1,-1)), axis=0)
  joint_speed = np.concatenate((calculated_controls_u_thetadot[4:], calculated_controls_u_thetadot[:4]))
  robot_config = NextState(robot_config, joint_speed, delta_t, speed_limit)
  log_robot_config = np.append(log_robot_config, np.append(robot_config, end_effector_path[i][-1]))

log_robot_config = log_robot_config.reshape(-1,13)
np.savetxt('log_robot_config.csv', log_robot_config, fmt='%.6f', delimiter=',')    #saving final configuration with gripper state
np.savetxt('log_x_error.csv', log_x_error, fmt='%.6f', delimiter=',')              #saving x_error values

"""---
FINAL LOOP END

X_ERROR PLOT

---
"""

plt.plot(log_x_error)
plt.legend

