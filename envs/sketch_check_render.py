from cv2 import imwrite

from envs.drone_env import DroneWorldEnv

if __name__ == '__main__':
    env = DroneWorldEnv(num_drones=12, num_obstacles=30, render_mode="rgb_array")
    obs, _ = env.reset()

    for k in range(200):
        # one random action per drone
        frame = env.render()  # returns (H,W,3) uint8
        imwrite(f"screens/snapshot{k}.png", frame)
        obs, _ = env.reset()
        print(k)
    env.close()
