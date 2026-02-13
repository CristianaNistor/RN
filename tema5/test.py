import gymnasium as gym
import flappy_bird_gymnasium
import torch
import torch.nn as nn
import numpy as np
import cv2
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAME_SIZE = 84

env = gym.make("FlappyBird-v0", render_mode="rgb_array")

def process_screen(screen):
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    h, w = screen.shape
    screen = screen[int(h*0.1):int(h*0.85), :]
    screen = cv2.resize(screen, (FRAME_SIZE, FRAME_SIZE))
    return (screen / 255.0).astype(np.float32)

class DuelingDQN(nn.Module):
    def __init__(self, outputs):
        super(DuelingDQN, self).__init__()
        
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.state_value_stream = nn.Sequential(
            nn.Linear(3136, 512), 
            nn.ReLU(), 
            nn.Linear(512, 1)
        )
        
        self.action_advantage_stream = nn.Sequential(
            nn.Linear(3136, 512), 
            nn.ReLU(), 
            nn.Linear(512, outputs)
        )

    def forward(self, x):
        features = self.convolutional_layers(x)
        val = self.state_value_stream(features)
        adv = self.action_advantage_stream(features)
        return val + (adv - adv.mean(dim=1, keepdim=True))

model = DuelingDQN(env.action_space.n).to(DEVICE)
try:
    model.load_state_dict(torch.load("best_flappy_pixels_2026-01-10_11-18-10.pth", map_location=DEVICE))
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: The .pth file was not found. Check the filename path.")
    exit()
except RuntimeError as e:
    print(f"Runtime Error: {e}")
    exit()

model.eval() 

for i in range(10):
    env.reset()
    
    raw_screen = env.render()
    img = process_screen(raw_screen)
    state = np.stack([img] * 4, axis=0)
    
    total_reward = 0
    while True:
        show_screen = cv2.cvtColor(raw_screen, cv2.COLOR_RGB2BGR)
        cv2.imshow("Agent View", show_screen)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            env.close()
            cv2.destroyAllWindows()
            exit()
        
        state_tensor = torch.tensor(state).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            action = model(state_tensor).argmax().item()
            
        _, reward, term, trunc, info = env.step(action)
        
        raw_screen = env.render()
        next_img = process_screen(raw_screen)
        
        state = np.concatenate((state[1:], np.expand_dims(next_img, 0)), axis=0)
        total_reward += reward
        
        time.sleep(0.03)
        
        if term or trunc:
            print(f"Episode {i+1} finished. Score: {info.get('score', 0)}")
            break

env.close()
cv2.destroyAllWindows()