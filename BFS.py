import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import pygame
from collections import deque
import time

# Neural Network Model
class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Maze Environment
class Maze:
    def __init__(self, maze, start, end, cell_size=10):
        self.maze = maze
        self.start = start
        self.end = end
        self.current_pos = start
        self.visited = set()
        self.visited_order = []  # Track the order of visited cells
        self.score = 0  # Initialize the score attribute
        self.cell_size = cell_size
        pygame.init()
        self.width = self.maze.shape[1] * self.cell_size
        self.height = self.maze.shape[0] * self.cell_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

    def reset(self):
        self.current_pos = self.start
        self.visited = set()
        self.visited_order = [self.start]  # Reset the visited order and start with the start position
        self.score = 0  # Reset the score
        return self.get_state()

    def get_state(self):
        state = np.zeros_like(self.maze, dtype=np.float32)
        state[self.current_pos] = 1
        return state.flatten()

    def step(self, action):
        x, y = self.current_pos
        if action == 0:  # Up
            new_pos = (x - 1, y)
        elif action == 1:  # Down
            new_pos = (x + 1, y)
        elif action == 2:  # Left
            new_pos = (x, y - 1)
        elif action == 3:  # Right
            new_pos = (x, y + 1)

        if new_pos[0] < 0 or new_pos[0] >= self.maze.shape[0] or new_pos[1] < 0 or new_pos[1] >= self.maze.shape[1] or self.maze[new_pos] == 1:
            # Hit a wall or out of bounds
            return self.get_state(), -5, False, self.score

        if new_pos == self.end:
            # Reached the end
            self.current_pos = new_pos
            self.score += 1
            return self.get_state(), 10, True, self.score

        if new_pos in self.visited:
            # Revisiting a cell (potential backtrack)
            if self.visited_order and self.visited_order[-1] == new_pos:
                # Valid backtrack (just backtracked to the previous cell)
                self.visited_order.pop()  # Remove the last visited cell as it backtracked correctly
                reward = -0.1
            else:
                # Unnecessary backtrack, end the run
                return self.get_state(), -10, True, self.score
        else:
            reward = 1  # Reward for discovering a new cell
            self.visited.add(new_pos)
            self.visited_order.append(new_pos)  # Add to visited order

        self.current_pos = new_pos
        return self.get_state(), reward, False, self.score

    def render(self):
        self.screen.fill((0, 0, 0))
        for x in range(self.maze.shape[0]):
            for y in range(self.maze.shape[1]):
                if self.maze[x, y] == 1:
                    pygame.draw.rect(self.screen, (255, 255, 255), (y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size))
        for pos in self.visited:
            pygame.draw.rect(self.screen, (0, 255, 0), (pos[1] * self.cell_size, pos[0] * self.cell_size, self.cell_size, self.cell_size))
        pygame.draw.rect(self.screen, (0, 0, 255), (self.start[1] * self.cell_size, self.start[0] * self.cell_size, self.cell_size, self.cell_size))
        pygame.draw.rect(self.screen, (255, 0, 0), (self.end[1] * self.cell_size, self.end[0] * self.cell_size, self.cell_size, self.cell_size))
        pygame.draw.rect(self.screen, (255, 255, 0), (self.current_pos[1] * self.cell_size, self.current_pos[0] * self.cell_size, self.cell_size, self.cell_size))
        pygame.display.flip()

# BFS to find the shortest path in the maze
def bfs_shortest_path(maze, start, end):
    queue = deque([(start, 0)])
    visited = set([start])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

    while queue:
        (current, dist) = queue.popleft()
        if current == end:
            return dist

        for direction in directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])
            if (0 <= neighbor[0] < maze.shape[0] and 0 <= neighbor[1] < maze.shape[1] and
                maze[neighbor] == 0 and neighbor not in visited):
                queue.append((neighbor, dist + 1))
                visited.add(neighbor)

    return float('inf')  # Return infinity if no path is found

# Random Maze Generation using Recursive Backtracking
def generate_random_maze(dim):
    maze = np.ones((dim, dim), dtype=np.int8)
    start = (0, 0)
    end = (dim - 1, dim - 1)

    def carve_passages_from(cx, cy, maze):
        directions = [(cx-2, cy), (cx+2, cy), (cx, cy-2), (cx, cy+2)]
        random.shuffle(directions)
        for (nx, ny) in directions:
            if 0 <= nx < dim and 0 <= ny < dim and maze[nx, ny] == 1:
                maze[cx + (nx - cx)//2, cy + (ny - cy)//2] = 0
                maze[nx, ny] = 0
                carve_passages_from(nx, ny, maze)

    maze[start] = 0
    carve_passages_from(start[0], start[1], maze)
    maze[end] = 0
    return maze, start, end

def train():
    dimension = 21  # Larger dimension for the maze
    model = Net(dimension * dimension, 4)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    gamma = 0.99  # Discount factor for future rewards
    episodes = 50000
    epsilon = 0.1  # Exploration rate
    previous_scores = []

    for episode in range(episodes):
        maze_array, start, end = generate_random_maze(dimension)
        env = Maze(maze_array, start, end)
        state = env.reset()
        total_reward = 0
        done = False
        count = 0
        score = 0

        bfs_distance = bfs_shortest_path(maze_array, start, end)
        max_steps = bfs_distance * 10  # Maximum steps allowed for the AI

        episode_rewards = []  # List to store rewards for the episode
        initial_visited_len = len(env.visited)

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, 3)  # Explore
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = model(state_tensor)
                    action = torch.argmax(q_values).item()  # Exploit

            next_state, reward, done, score = env.step(action)
            env.render()
            time.sleep(0.000001)  # Short delay to make the simulation faster
            count += 1

            if count > max_steps:
                done = True
                reward = -10  # Penalize for exceeding maximum steps

            total_reward += reward
            episode_rewards.append(reward)

            new_state = env.get_state()
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            new_state_tensor = torch.FloatTensor(new_state).unsqueeze(0)
            q_values = model(state_tensor)
            future_q_values = model(new_state_tensor)
            max_future_q = torch.max(future_q_values)

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

            # Check for unnecessary backtracking
            if len(env.visited_order) > 1 and next_state == env.visited_order[-2]:
                done = True
                reward = -10  # Penalize for unnecessary backtracking

        # Adjust the reward based on performance relative to BFS
        if count <= bfs_distance * 1.2:
            total_reward += 2000  # High reward for being within 20% of BFS
        elif count <= bfs_distance * 1.5:
            total_reward += 1000  # Moderate reward for being within 50% of BFS
        elif count <= bfs_distance * 2:
            total_reward += 500  # Small reward for being within 100% of BFS
        else:
            total_reward -= 500  # Penalty for taking more than 2x BFS

        # Reward for discovering new cells
        discovered_cells = len(env.visited) - initial_visited_len
        total_reward += discovered_cells * 10

        # Cap the penalty to avoid excessive negative rewards
        total_reward = max(total_reward, -3000)

        previous_scores.append(score)

        print(f'Episode {episode + 1} finished with score: {score}, total reward: {total_reward}')

if __name__ == "__main__":
    train()
    pygame.quit()
