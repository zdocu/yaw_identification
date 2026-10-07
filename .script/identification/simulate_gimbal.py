import numpy as np

J = 0.124684
B = 0.534220
c = 0.015445
dt = 0.005

ref_kp_vel = 13.0
ref_ki_vel = 0.02
ref_kp_angle = 10.0

class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.last_error = 0
    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.last_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.last_error = error
        return output

def evaluate_pid(kp_vel, ki_vel, kp_angle, target=0.523):
    angle = 0
    omega = 0
    angle_pid = PID(kp_angle, 0, 0)
    vel_pid = PID(kp_vel, ki_vel, 0)
    overshoot = 0
    rise_time = 0
    settled = False
    for i in range(2000):
        angle_error = target - angle
        target_vel = angle_pid.update(angle_error, dt)
        vel_error = target_vel - omega
        tau = vel_pid.update(vel_error, dt)
        
        # 保护：如果tau太大，直接返回很差的分数
        if abs(tau) > 100:
            return 99999
        
        theta_dd = (tau - B * omega - c) / J
        omega += theta_dd * dt
        angle += omega * dt
        
        # 保护：如果omega太大，直接返回很差的分数
        if abs(omega) > 100:
            return 99999
        
        if angle > target and not settled:
            rise_time = i
            settled = True
        if angle > overshoot:
            overshoot = angle
    
    # 评分：上升时间 + 超调 + PID误差
    score = rise_time + (overshoot - target) * 50
    score += abs(kp_vel - ref_kp_vel) * 100
    score += abs(ki_vel - ref_ki_vel) * 10000
    score += abs(kp_angle - ref_kp_angle) * 100
    
    return score

best_score = 99999
best_params = None

for kp_vel in np.linspace(5, 20, 30):
    for ki_vel in np.linspace(0.01, 0.1, 20):
        for kp_angle in np.linspace(5, 20, 20):
            score = evaluate_pid(kp_vel, ki_vel, kp_angle)
            if score < best_score:
                best_score = score
                best_params = (kp_vel, ki_vel, kp_angle)

if best_params is None:
    print('没找到合适的PID参数')
else:
    kp_vel, ki_vel, kp_angle = best_params
    print('最优PID参数：')
    print('yaw_velocity_kp: {:.4f}'.format(kp_vel))
    print('yaw_velocity_ki: {:.4f}'.format(ki_vel))
    print('yaw_angle_kp: {:.4f}'.format(kp_angle))
