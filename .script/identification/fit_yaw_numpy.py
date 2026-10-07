#!/usr/bin/env python3
import csv
import numpy as np

def read_csv(path):
    t, angle, omega, torque = [], [], [], []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t.append(float(row['index']))
                angle.append(float(row['/gimbal/yaw/angle']))
                omega.append(float(row['/gimbal/yaw/velocity_imu']))
                torque.append(float(row['/gimbal/yaw/control_torque']))
            except:
                pass
    return np.array(t), np.array(angle), np.array(omega), np.array(torque)

def smooth(x, window=31):
    return np.convolve(x, np.ones(window)/window, mode='valid')

def fit_least_squares(omega, torque, dt):
    domega_dt = np.diff(omega) / dt
    A = np.column_stack([domega_dt, omega[:-1], np.ones_like(domega_dt)])
    b = torque[:-1]
    result = np.linalg.lstsq(A, b, rcond=None)
    J, B, c = result[0]
    omega_pred = omega[:-1] + (torque[:-1] - B * omega[:-1] - c) / J * dt
    error = omega[1:] - omega_pred
    ss_res = np.sum(error ** 2)
    ss_tot = np.sum((omega[1:] - np.mean(omega[1:])) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return J, B, c, r2

def main():
    import sys
    if len(sys.argv) < 2:
        print('用法：python3 fit_yaw_numpy.py 数据.csv')
        return
    csv_path = sys.argv[1]
    t, angle, omega, torque = read_csv(csv_path)
    
    print(f'数据点数量：{len(omega)}')
    print()
    
    dt = 0.005
    omega_smooth = smooth(smooth(omega, window=31), window=11)
    torque_smooth = smooth(smooth(torque, window=31), window=11)
    
    # 确保两个数组长度一样
    n = min(len(omega_smooth), len(torque_smooth))
    omega_smooth = omega_smooth[:n]
    torque_smooth = torque_smooth[:n]
    
    print(f'平滑后数据点数量：{len(omega_smooth)}')
    print()
    
    J, B, c, r2 = fit_least_squares(omega_smooth, torque_smooth, dt)
    
        # 多步预测
    omega_pred_multistep = np.zeros_like(omega_smooth)
    omega_pred_multistep[0] = omega_smooth[0]
    for i in range(1, len(omega_smooth)):
        theta_dd = (torque_smooth[i-1] - B * omega_pred_multistep[i-1] - c) / J
        omega_pred_multistep[i] = omega_pred_multistep[i-1] + theta_dd * dt
    
    error_multistep = omega_smooth - omega_pred_multistep
    ss_res = np.sum(error_multistep ** 2)
    ss_tot = np.sum((omega_smooth - np.mean(omega_smooth)) ** 2)
    r2_multistep = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    rmse = np.sqrt(np.mean(error_multistep ** 2))
    mae = np.mean(np.abs(error_multistep))

    
    print('=' * 40)
    print('yaw 轴云台模型拟合结果')
    print('=' * 40)
    print(f'转动惯量 J = {J:.6f}')
    print(f'粘性阻尼 B = {B:.6f}')
    print(f'库仑摩擦 c = {c:.6f}')
    print(f'拟合优度 R² = {r2_multistep:.4f}')
    print(f'均方根误差 RMSE = {rmse:.6f}')
    print(f'平均绝对误差 MAE = {mae:.6f}')
    print('=' * 40)
    print(f'模型方程：{J:.6f}·θ\'\' + {B:.6f}·θ\' + {c:.6f} = τ')

if __name__ == '__main__':
    main()
