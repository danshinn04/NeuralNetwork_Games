import pygame
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(12, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SnakeGame:
    def __init__(self, dimension, bomb_frequency):
        pygame.init()
        self.dimension = dimension
        self.cell_size = 20
        self.width = self.dimension * self.cell_size
        self.height = self.dimension * self.cell_size
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Snake RL Training")
        self.clock = pygame.time.Clock()
        self.bomb_frequency = bomb_frequency
        self.reset()

    def reset(self):
        self.snake = [(self.dimension // 2, self.dimension // 2)]
        self.direction = 'right'
        self.score = 0
        self.bombs = []
        for _ in range(self.bomb_frequency):
            self.bombs.append(self._place_food(True))
        self.food_pos = self._place_food()
        self.frame_iteration = 0

    def _place_food(self, placing_bombs=False):
        while True:
            position = (random.randint(0, self.dimension - 1), random.randint(0, self.dimension - 1))
            if position not in self.snake and (placing_bombs or position not in self.bombs):
                return position

    def play_step(self, action):
        direction_vectors = {'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)}
        directions = ['up', 'down', 'left', 'right']
        current_direction = directions[action]

        head_x, head_y = self.snake[0]
        move_x, move_y = direction_vectors[current_direction]
        new_head = (head_x + move_x, head_y + move_y)

        if new_head in self.snake or new_head[0] < 0 or new_head[0] >= self.dimension or new_head[1] < 0 or new_head[1] >= self.dimension or new_head in self.bombs:
            return False, -10, True, self.score  # Penalize and end game for collision

        if new_head == self.food_pos:
            self.snake.insert(0, new_head)
            self.food_pos = self._place_food()
            self.score += 1  # Increase the score when an apple is eaten
            return True, 15, False, self.score  # Reward for eating an apple
        
        self.snake.insert(0, new_head)  # Move the snake
        self.snake.pop()
        return False, 0.1, False, self.score  # Small reward for moving

    def get_state(self):
        head_x, head_y = self.snake[0]
        points = [
            (head_x, head_y - 1), (head_x, head_y + 1),
            (head_x - 1, head_y), (head_x + 1, head_y)
        ]
        danger = [((px, py) in self.snake or
                   px < 0 or px >= self.dimension or
                   py < 0 or py >= self.dimension or
                   (px, py) in self.bombs) for px, py in points]
        
        food_dir = [
            self.food_pos[0] < head_x,  # Food left
            self.food_pos[0] > head_x,  # Food right
            self.food_pos[1] < head_y,  # Food up
            self.food_pos[1] > head_y   # Food down
        ]

        state = np.array(danger + food_dir + [self.direction == 'up', self.direction == 'down', self.direction == 'left', self.direction == 'right'], dtype=int)
        return state

    def render(self):
        self.display.fill((0, 0, 0))
        for x, y in self.snake:
            pygame.draw.rect(self.display, (0, 255, 0), (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))
        pygame.draw.rect(self.display, (255, 0, 0), (self.food_pos[0] * self.cell_size, self.food_pos[1] * self.cell_size, self.cell_size, self.cell_size))
        for x, y in self.bombs:
            pygame.draw.rect(self.display, (0, 0, 255), (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))
        pygame.display.flip()

def train():
    dimension = 20
    game = SnakeGame(dimension, 5)
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    gamma = 0.99  # Discount factor for future rewards
    episodes = 1000
    previous_scores = []

    for episode in range(episodes):
        game.reset()
        state = game.get_state()
        total_reward = 0
        done = False
        count = 0
        score = 0

        while not done:
            game.render()
            time.sleep(0.1)

            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()  # Exploitation

            scored, reward, done, score = game.play_step(action)
            if reward == 0.1:
                count += 1
            if count > 30 and reward == 0.1:
                reward = -0.1
            elif count > 30 and reward > 0.1:
                count = 0
                print("Score increased!")

            if count > 150: 
                done = True

            total_reward += reward

            new_state = game.get_state()
            new_state_tensor = torch.FloatTensor(new_state).unsqueeze(0)
            future_q_values = model(new_state_tensor)
            max_future_q = torch.max(future_q_values).item()

            # Compute the target Q-value
            target_q_value = reward + gamma * max_future_q
            target_q_values = q_values.clone().detach()
            target_q_values[0, action] = target_q_value  # Only update the action taken

            # Calculate loss
            loss = nn.MSELoss()(q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = new_state

        # Adjust the reward based on the last few episodes
        if len(previous_scores) > 10:
            average_score = sum(previous_scores[-10:]) / 10
            if score < average_score:
                if not total_reward < 0:
                    total_reward = -5  # Penalize if performance decreases
            else:
                total_reward += 5  # Reward if performance improves

        previous_scores.append(score)

        print(f'Episode {episode + 1} finished with score: {score}, total reward: {total_reward}')

if __name__ == "__main__":
    train()
