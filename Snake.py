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
        # Adjust input size based on the expected state vector length
        self.fc1 = nn.Linear(19, 128)  # Adjust as needed based on state vector calculations
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)

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
            return False, -10, True, self.score

        if new_head == self.food_pos:
            self.snake.insert(0, new_head)
            self.food_pos = self._place_food()
            self.score += 1
            return True, 15, False, self.score
        
        self.snake.insert(0, new_head)
        self.snake.pop()
        return False, 0.1, False, self.score

    def get_state(self):
        head_x, head_y = self.snake[0]

        # Calculate angle to food and encode direction
        angle_to_food = np.arctan2(self.food_pos[1] - head_y, self.food_pos[0] - head_x) / np.pi
        direction_encoding = [self.direction == 'up', self.direction == 'down', self.direction == 'left', self.direction == 'right']

        # Distance to each bomb
        bomb_distances = []
        for (bomb_x, bomb_y) in self.bombs:
            distance_x = (bomb_x - head_x) / self.dimension
            distance_y = (bomb_y - head_y) / self.dimension
            bomb_distances.extend([distance_x, distance_y])

        # Body danger based on current direction
        body_danger = []
        direction_vectors = {'up': (0, -1), 'left': (-1, 0), 'down': (0, 1), 'right': (1, 0)}
        for dx, dy in direction_vectors.values():
            next_x, next_y = head_x + dx, head_y + dy
            if (next_x, next_y) in self.snake or next_x < 0 or next_x >= self.dimension or next_y < 0 or next_y >= self.dimension:
                body_danger.append(1)
            else:
                body_danger.append(0)

        state = np.array([angle_to_food] + direction_encoding + bomb_distances + body_danger, dtype=float)
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
    bomb_frequency = 5
    game = SnakeGame(dimension, bomb_frequency)
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    gamma = 0.99
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
            time.sleep(0.001)

            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()

            scored, reward, done, score = game.play_step(action)
            total_reward += reward

            new_state = game.get_state()
            new_state_tensor = torch.FloatTensor(new_state).unsqueeze(0)
            future_q_values = model(new_state_tensor)
            max_future_q = torch.max(future_q_values).item()

            target_q_value = reward + gamma * max_future_q
            target_q_values = q_values.clone().detach()
            target_q_values[0, action] = target_q_value

            loss = nn.MSELoss()(q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = new_state

        if len(previous_scores) > 10:
            average_score = sum(previous_scores[-10:]) / 10
            if score < average_score:
                total_reward = -5
            else:
                total_reward += 5

        previous_scores.append(score)

        print(f'Episode {episode + 1} finished with score: {score}, total reward: {total_reward}')

if __name__ == "__main__":
    train()
