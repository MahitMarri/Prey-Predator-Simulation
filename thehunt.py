import pygame
import random
import sys
import math
import numpy as np

pygame.init()

info = pygame.display.Info()
screen_width, screen_height = info.current_w, info.current_h
screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
pygame.display.set_caption("Predator-Prey Evolution")
clock = pygame.time.Clock()
FPS = 30
GRID_SIZE = 6
font = pygame.font.SysFont(None, 24)

def wrap_distance(x1, y1, x2, y2, max_x, max_y):
    dx = x2 - x1
    dy = y2 - y1
    if abs(dx) > max_x / 2:
        dx -= math.copysign(max_x, dx)
    if abs(dy) > max_y / 2:
        dy -= math.copysign(max_y, dy)
    return dx, dy, math.hypot(dx, dy)

def clamp(v, a, b):
    return max(a, min(b, v))

class SimpleNN:
    def __init__(self, inputSize=4, hiddenSize=8, outputSize=2):
        self.w1 = np.random.randn(hiddenSize, inputSize) * (1.0 / math.sqrt(inputSize))
        self.b1 = np.zeros((hiddenSize,)) 
        self.w2 = np.random.randn(outputSize, hiddenSize) * (1.0 / math.sqrt(hiddenSize))
        self.b2 = np.zeros((outputSize,))

    def forward(self, inputs):
        h = np.tanh(np.dot(self.w1, inputs) + self.b1)
        return np.tanh(np.dot(self.w2, h) + self.b2)

    def copy_and_mutate(self, mutation_rate=0.08):
        child = SimpleNN()
        child.w1 = self.w1 + np.random.randn(*self.w1.shape) * mutation_rate
        child.b1 = self.b1 + np.random.randn(*self.b1.shape) * mutation_rate
        child.w2 = self.w2 + np.random.randn(*self.w2.shape) * mutation_rate
        child.b2 = self.b2 + np.random.randn(*self.b2.shape) * mutation_rate
        return child

class Prey:
    def __init__(self, screen_width, screen_height, grid_size, traits=None):
        self.grid_size = grid_size
        max_x = screen_width / grid_size
        max_y = screen_height / grid_size

        self.x = random.uniform(0, max_x)
        self.y = random.uniform(0, max_y)
        self.size = grid_size
        self.birth_time = pygame.time.get_ticks()
        self.reproduction_age = traits["reproduction_age"] if traits and "reproduction_age" in traits else random.randint(60000, 70000)
        self.speed = traits["speed"] if traits and "speed" in traits else random.uniform(0.6, 1.2)
        self.color = (0, 220, 0)
        self.vision_range = 180 / grid_size 
        self.brain = SimpleNN(4, 8, 2)
        self.angle = random.uniform(0, 2 * math.pi)
        self.vx = self.vy = self.rotation = 0.0

    def move(self, screen_width, screen_height, predators, prey_list):
        max_x = screen_width / self.grid_size
        max_y = screen_height / self.grid_size

        visible = []
        for p in predators:
            dx, dy, dist = wrap_distance(self.x, self.y, p.x, p.y, max_x, max_y)
            if dist <= self.vision_range:
                angle_to = math.atan2(dy, dx)
                angle_diff = ((angle_to - self.angle + math.pi) % (2 * math.pi)) - math.pi
                if abs(angle_diff) <= math.radians(120):
                    visible.append((dist, angle_diff, p))

        closest = min(visible, default=None, key=lambda t: t[0])
        if closest:
            dist, rel_angle, _ = closest
            nn_input = np.array([dist / self.vision_range, rel_angle / math.pi, self.x / max_x, self.y / max_y])
            out = self.brain.forward(nn_input)
            self.rotation += out[0] * 0.12
            self.rotation *= 0.82
            self.rotation = clamp(self.rotation, -0.45, 0.45)
            self.angle = (self.angle + self.rotation) % (2 * math.pi)
            thrust = clamp(out[1], 0.0, 1.0)
            self.vx = math.cos(self.angle) * thrust * self.speed
            self.vy = math.sin(self.angle) * thrust * self.speed
        else:
            nn_input = np.array([0.0, 0.0, self.x / max_x, self.y / max_y])
            out = self.brain.forward(nn_input)
            self.angle += random.uniform(-0.05, 0.05)
            self.rotation += out[0] * 0.12
            self.rotation *= 0.82
            self.rotation = clamp(self.rotation, -0.45, 0.45)
            self.angle = (self.angle + self.rotation) % (2 * math.pi)
            thrust = clamp(out[1], 0.0, 1.0) * 0.3
            self.vx = math.cos(self.angle) * thrust * self.speed
            self.vy = math.sin(self.angle) * thrust * self.speed

        new_x = (self.x + self.vx) % max_x
        new_y = (self.y + self.vy) % max_y

        for other in prey_list:
            if other is not self:
                _, _, odst = wrap_distance(new_x, new_y, other.x, other.y, max_x, max_y)
                if odst < 0.6:
                    away_angle = math.atan2(new_y - other.y, new_x - other.x)
                    new_x += math.cos(away_angle) * 0.2
                    new_y += math.sin(away_angle) * 0.2

        self.x, self.y = new_x, new_y

    def draw(self, screen, camera, screen_width, screen_height):
        px, py, zoom = camera.apply(self.x, self.y, self.grid_size, screen_width, screen_height)
        size = int(self.size * zoom)
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.rect(surf, self.color, (0, 0, size, size))
        rot = pygame.transform.rotate(surf, -math.degrees(self.angle))
        rect = rot.get_rect(center=(px + size // 2, py + size // 2))
        screen.blit(rot, rect)

    def ready_to_reproduce(self):
        now = pygame.time.get_ticks()
        return (now - self.birth_time) >= round(self.reproduction_age * (len(prey_list) / 200))

class Predator:
    def __init__(self, screen_width, screen_height, grid_size, traits=None):
        self.grid_size = grid_size
        max_x = screen_width / grid_size
        max_y = screen_height / grid_size

        self.x = random.uniform(0, max_x)
        self.y = random.uniform(0, max_y)
        self.size = grid_size * 1.5
        self.color = (220, 40, 40)
        self.speed = traits["speed"] if traits and "speed" in traits else random.uniform(0.6, 1.0)
        self.angle = random.uniform(0, 2 * math.pi)
        self.killCount = 0
        self.energy = traits["Energy"] if traits and "Energy" in traits else random.randint(200, 240)
        self.energy_decay = traits["EnergyDecay"] if traits and "EnergyDecay" in traits else random.uniform(0.25, 0.5)
        self.energy_gain = traits["EnergyGain"] if traits and "EnergyGain" in traits else random.uniform(55.0, 65.0)
        self.reproduceRule = traits["reproduceAmount"] if traits and "reproduceAmount" in traits else random.randint(4,6)
        self.vision_range = 350 / grid_size
        self.vision_angle = math.radians(170)
        self.brain = SimpleNN(4, 8, 2)
        self.rotation = 0.0
        self.fitness = 0.0

    def move(self, prey_list, predator_list, screen_width, screen_height):
        max_x = screen_width / self.grid_size
        max_y = screen_height / self.grid_size

        visible = []
        for prey in prey_list:
            dx, dy, dist = wrap_distance(self.x, self.y, prey.x, prey.y, max_x, max_y)
            if dist <= self.vision_range:
                angle_to = math.atan2(dy, dx)
                angle_diff = ((angle_to - self.angle + math.pi) % (2 * math.pi)) - math.pi
                if abs(angle_diff) <= self.vision_angle / 2:
                    visible.append((dist, angle_diff, prey))

        if visible:
            dist, rel_angle, prey_obj = min(visible, key=lambda t: t[0])
            nn_input = np.array([dist / self.vision_range, rel_angle / math.pi, self.x / max_x, self.y / max_y])
            out = self.brain.forward(nn_input)
            self.rotation += out[0] * 0.12
            self.rotation *= 0.8
            self.rotation = clamp(self.rotation, -0.6, 0.6)
            self.angle = (self.angle + self.rotation) % (2 * math.pi)
            thrust = clamp(out[1], 0.0, 1.0)
            dx = math.cos(self.angle)
            dy = math.sin(self.angle)
            new_x = (self.x + dx * thrust * self.speed) % max_x
            new_y = (self.y + dy * thrust * self.speed) % max_y
        else:
            self.angle = (self.angle + random.uniform(-0.2, 0.2)) % (2 * math.pi)
            dx = math.cos(self.angle)
            dy = math.sin(self.angle)
            new_x = (self.x + dx * self.speed * 0.5) % max_x
            new_y = (self.y + dy * self.speed * 0.5) % max_y

        blocked = False
        for o in predator_list:
            if o is not self:
                _, _, odst = wrap_distance(new_x, new_y, o.x, o.y, max_x, max_y)
                if odst < 0.8:
                    away_angle = math.atan2(new_y - o.y, new_x - o.x)
                    new_x += math.cos(away_angle) * 0.3
                    new_y += math.sin(away_angle) * 0.3
                    blocked = True
        self.x, self.y = new_x, new_y

        self.energy -= self.energy_decay
        self.fitness *= 0.995

    def draw(self, screen, camera, screen_width, screen_height):
        px, py, zoom = camera.apply(self.x, self.y, self.grid_size, screen_width, screen_height)
        size = int(self.size * zoom)
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.rect(surf, self.color, (0, 0, size, size))
        rot = pygame.transform.rotate(surf, -math.degrees(self.angle))
        rect = rot.get_rect(center=(px + size // 2, py + size // 2))
        screen.blit(rot, rect)

    def on_eat_prey(self):
        self.killCount += 1
        self.energy += self.energy_gain
        self.fitness += 0.5

class Camera:
    def __init__(self):
        self.target = None
        self.zoom = 2.0
        self.default_zoom = 1.0

    def apply(self, x, y, grid_size, screen_width, screen_height):
        zoom = self.zoom if self.target else self.default_zoom
        px = int(x * grid_size * zoom)
        py = int(y * grid_size * zoom)
        if self.target:
            tx = int(self.target.x * grid_size * zoom)
            ty = int(self.target.y * grid_size * zoom)
            px = px - tx + screen_width // 2
            py = py - ty + screen_height // 2
        return px, py, zoom

camera = Camera()
INITIAL_PREY = 200
INITIAL_PREDATORS = 50
prey_list = [Prey(screen_width, screen_height, GRID_SIZE) for _ in range(INITIAL_PREY)]
predator_list = [Predator(screen_width, screen_height, GRID_SIZE) for _ in range(INITIAL_PREDATORS)]

paused = False
show_help = False
show_minimap = True


running = True
while running:
    clock.tick(FPS)
    screen.fill((10, 10, 10))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.VIDEORESIZE:
            screen_width, screen_height = event.w, event.h
            screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            clicked = None
            for predator in predator_list:
                px, py, zoom = camera.apply(predator.x, predator.y, GRID_SIZE, screen_width, screen_height)
                if pygame.Rect(px, py, int(predator.size*zoom), int(predator.size*zoom)).collidepoint(mx,my):
                    clicked = predator
                    break
            if not clicked:
                for prey in prey_list:
                    px, py, zoom = camera.apply(prey.x, prey.y, GRID_SIZE, screen_width, screen_height)
                    if pygame.Rect(px, py, int(prey.size*zoom), int(prey.size*zoom)).collidepoint(mx,my):
                        clicked = prey
                        break
            camera.target = clicked if clicked else None
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p: 
                paused = not paused
            elif event.key == pygame.K_h:
                show_help = not show_help
            elif event.key == pygame.K_m:
                show_minimap = not show_minimap
            elif event.key == pygame.K_EQUALS:
                camera.zoom += 0.1
            elif event.key == pygame.K_MINUS:
                camera.zoom = max(0.2, camera.zoom - 0.1)
            elif event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()

    if not paused:
        new_prey, new_predators, prey_to_remove = [], [], []
        for predator in predator_list:
            predator.move(prey_list, predator_list, screen_width, screen_height)
        for prey in prey_list:
            prey.move(screen_width, screen_height, predator_list, prey_list)
            eaten = False
            for predator in predator_list:
                if wrap_distance(prey.x, prey.y, predator.x, predator.y, screen_width/GRID_SIZE, screen_height/GRID_SIZE)[2] < 1.0:
                    eaten = True
                    prey_to_remove.append(prey)
                    predator.on_eat_prey()
                    break
            if eaten: continue
            if prey.ready_to_reproduce():
                child_traits = {
                    "reproduction_age": max(1500, int(prey.reproduction_age + random.randint(-800, 800))) if random.randint(1,10) > 9 else prey.reproduction_age,
                    "speed": min(max(0.08, prey.speed + random.uniform(-0.06, 0.06)), 1.8) if random.randint(1,10) > 9 else prey.speed,
                }
                child = Prey(screen_width, screen_height, GRID_SIZE, traits=child_traits)
                child.x, child.y = prey.x+random.choice([-1,0,1]), prey.x+random.choice([-1,0,1])
                child.brain = prey.brain.copy_and_mutate(0.08)
                new_prey.append(child)
                prey.birth_time = pygame.time.get_ticks()
        for predator in predator_list:
            if predator.killCount >= predator.reproduceRule:
                base_mutation, fitness_factor = 0.06, max(0.01, 1.0 - (predator.fitness * 0.05))
                mutation_rate = base_mutation * fitness_factor
                child_traits = {
                    "speed": min(max(0.08, predator.speed + random.uniform(-0.05, 0.1)), 1.5) if random.randint(1,10) > 9 else predator.speed,
                    "reproduceAmount": max(1, predator.reproduceRule + random.randint(-1,1)) if random.randint(1,10) > 9 else predator.reproduceRule,
                    "EnergyGain": max(20.0, predator.energy_gain + random.uniform(-5,5)) if random.randint(1,10) > 9 else predator.energy_gain,
                    "Energy": max(100, predator.energy + random.randint(-20,20)) if random.randint(1,10) > 9 else predator.energy,
                    "EnergyDecay": max(0.2, predator.energy_decay + random.uniform(-0.02,0.04)) if random.randint(1,10) > 9 else predator.energy_decay
                }
                child = Predator(screen_width, screen_height, GRID_SIZE, traits=child_traits)
                child.x, child.y = predator.x+random.choice([-1,0,1]), predator.y+random.choice([-1,0,1])
                child.brain = predator.brain.copy_and_mutate(mutation_rate)
                new_predators.append(child)
                predator.killCount, predator.energy, predator.fitness = 0, predator.energy-60, predator.fitness*0.7
        if prey_to_remove:
            prey_list = [p for p in prey_list if p not in prey_to_remove]
        predator_list = [p for p in predator_list if p.energy > 0.0]
        prey_list.extend(new_prey)
        predator_list.extend(new_predators)

    for predator in predator_list: predator.draw(screen, camera, screen_width, screen_height)
    for prey in prey_list: prey.draw(screen, camera, screen_width, screen_height)

    prey_count_text = font.render(f"Prey: {len(prey_list)}", True, (220,220,220))
    predator_count_text = font.render(f"Predators: {len(predator_list)}", True, (255,150,150))
    screen.blit(prey_count_text, (10,10))
    screen.blit(predator_count_text, (10,30))

    if show_minimap:
        minimap_w, minimap_h = 200, 150
        minimap = pygame.Surface((minimap_w, minimap_h))
        minimap.fill((30,30,30))
        for prey in prey_list:
            px = int((prey.x / (screen_width/GRID_SIZE)) * minimap_w)
            py = int((prey.y / (screen_height/GRID_SIZE)) * minimap_h)
            pygame.draw.circle(minimap, (0,255,0), (px,py), 2)
        for pred in predator_list:
            px = int((pred.x / (screen_width/GRID_SIZE)) * minimap_w)
            py = int((pred.y / (screen_height/GRID_SIZE)) * minimap_h)
            pygame.draw.circle(minimap, (255,0,0), (px,py), 2)
        screen.blit(minimap, (screen_width-minimap_w-10, 10))


    if show_help:
        help_lines = [
            "Controls:",
            "Mouse: Click entity to follow",
            "P: Pause/Resume",
            "+/-: Zoom (when following)",
            "H: Toggle this help",
            "M: Toggle minimap",
            "ESC: Quit"
        ]
        for i,line in enumerate(help_lines):
            screen.blit(font.render(line, True, (200,200,200)), (10, 60+i*20))

    pygame.display.flip()

pygame.quit()
sys.exit()