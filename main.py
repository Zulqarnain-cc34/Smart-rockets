import sys
import pygame
# import time
import numpy as np
import random
import math
# from scipy.ndimage.interpolation import rotate
from utils import mapRange

# screen
WIDTH, HEIGHT = (800, 600)
TARGET = np.array([[WIDTH / 2], [100]])
LIFESPAN = 300

# Color
BLACK = (0, 0, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)


class Population(object):
    def __init__(self):
        self.rockets = []
        self.pop_max = 40

        self.mating_pool = []

        for _ in range(self.pop_max):
            self.rockets.append(Rocket())

    def run(self, win):
        for i in range(self.pop_max):
            self.rockets[i].update()
            self.rockets[i].show(win)

    def evaluate(self):
        # print("Evaluation\n")
        max_fitness = 0

        # print("\n                 Fitness              ")
        for i in range(self.pop_max):
            self.rockets[i].calcfitness()
            # print(self.rockets[i].fitness)
            if self.rockets[i].fitness > max_fitness:
                max_fitness = self.rockets[i].fitness
        # print("\n")

        for i in range(self.pop_max):
            self.rockets[i].fitness /= max_fitness

        self.mating_pool = []

        for i in range(self.pop_max):
            # rocket_count = 0
            n = math.floor(self.rockets[i].fitness * 100)
            for _ in range(n):
                self.mating_pool.append(self.rockets[i])
                # rocket_count += 1
            # print("Rocket ", i, " has ", rocket_count, " rockets in Mating Pool")
        print("Mating Pool: ", len(self.mating_pool))

    def natural_selection(self):
        # print("Selection\n")
        new_rockets = []
        for _ in range(len(self.rockets)):
            parentA = np.random.choice(self.mating_pool).dna
            parentB = np.random.choice(self.mating_pool).dna
            child = parentA.crossover(parentB)
            child.mutation()

            # print(child.genes)
            # with open("force.txt", "a") as log:
            #     log.write(f"New Force Vector : {child.genes} \n")

            new_rockets.append(Rocket(child))
        self.rockets = new_rockets


class DNA(object):
    lifespan = 100

    def __init__(self, genes=[], num_thrusters=1):

        self.mag = 0.2
        self.num_thrusters = num_thrusters

        if genes != []:
            self.genes = genes
        else:
            self.genes = np.random.randn(self.num_thrusters, 2)
            # self.genes = [[(random.random()-0.5)*2,0] for _ in range(self.num_thrusters)]
            # self.genes = np.array(self.genes)
            self.genes[0:self.num_thrusters, 1] = 0

            # print(self.genes.shape)
            # print(self.genes)

    def crossover(self, partner):
        # print("Crossover")
        newgenes = np.zeros((len(self.genes), 2))

        # print("New genes", newgenes)
        # print(self.genes," ",partner.genes)

        mid = np.random.randint(len(self.genes))
        for i in range(len(self.genes)):
            if i > mid:
                newgenes[i] = self.genes[i]
            else:
                newgenes[i] = partner.genes[i]

        # print("-------------Crossover Genes------------")
        # print(newgenes)
        # print("\n")

        return DNA(newgenes, self.num_thrusters)

    def mutation(self):
        # print("Mutation")
        for i in range(len(self.genes)):
            # if random number less than 0.01, new gene is then random vector
            if random.random() < 0.01:
                mutated_gene = np.random.randn(1, 2)
                mutated_gene[0][1] = 0

                self.genes[i] = mutated_gene

        # print(self.genes)
        # print("-------------Mutated Genes------------")


class Rocket():
    count = 0

    def __init__(self, dna=None, theta=30):
        # Accelartion, Velocity and Position Vectors
        self.acc = np.zeros((2, 1))
        self.vel = np.zeros((2, 1))

        self.rocket_width = 30
        self.rocket_height = 30

        self.pos = np.array([[WIDTH / 2], [HEIGHT - self.rocket_height - 100]])

        # Angular velocity
        self.theta_dot = 0

        self.fitness = 0

        self.color = WHITE
        self.theta = np.radians(theta)
        self.mag = 0.01

        self.crashed = False
        self.completed = False

        self.TRUSTER_1 = np.array([self.rocket_width / 2, self.rocket_height / 6])
        self.TRUSTER_2 = np.array([self.rocket_width / 2, -self.rocket_height / 6])
        self.TRUSTER_3 = np.array([-self.rocket_width / 2, self.rocket_height / 6])
        self.TRUSTER_4 = np.array([-self.rocket_width / 2, -self.rocket_height / 6])

        # # Loading the image for rockets object from assets folder
        # self.image = pygame.Surface((320, 200), pygame.SRCALPHA)
        # pygame.draw.polygon(self.image, pygame.Color('dodgerblue1'), ((1, 0), (120, 35), (1, 70)))
        #
        self.image = pygame.image.load("./assets/rocket.png")
        self.image = pygame.transform.scale(self.image, (self.rocket_height, self.rocket_height))

        # self.origin_image = self.image
        # self.rect = self.image.get_rect(center=(self.pos[0][0], self.pos[1][0]))

        if dna is not None:
            self.dna = dna
        else:
            self.dna = DNA(num_thrusters=4)

        self.TRUST_1 = self.dna.genes[0]
        self.TRUST_2 = self.dna.genes[1]
        self.TRUST_3 = self.dna.genes[2]
        self.TRUST_4 = self.dna.genes[3]

    def log(self):
        with open("force.txt", "a") as log:
            # log.write(f"_________________THRUSTERS______________\n\n")
            # log.write(f"Thruster 1: {self.TRUSTER_1}\n")
            # log.write(f"Thruster 2: {self.TRUSTER_2}\n")
            # log.write(f"Thruster 3: {self.TRUSTER_3}\n")
            # log.write(f"Thruster 4: {self.TRUSTER_4}\n\n")

            log.write("__________________THRUSTS_______________\n\n")
            log.write(f"Thrust 1: {self.TRUST_1}\n")
            log.write(f"Thrust 2: {self.TRUST_2}\n")
            log.write(f"Thrust 3: {self.TRUST_3}\n")
            log.write(f"Thrust 4: {self.TRUST_4}\n\n")

            # log.write(f"Velocity    : {self.vel}\n")
            # log.write(f"Position    : {self.pos}\n")
            # log.write(f"Acceleration: {self.acc}\n")
            # log.write(f"Theta       : {self.theta}\n")
            # log.write(f"Theta dot   : {self.theta_dot}\n")

    def calcfitness(self):
        # d = math.sqrt(sum([(a - b) ** 2 for a, b in zip(self.pos, TARGET)]))
        d = self.dist(self.pos, TARGET)

        # print(self.dist(self.pos, TARGET))

        # print("------------------------")
        # print("distance ", d)

        max_dist = math.sqrt((TARGET[0][0]) ** 2 + (HEIGHT - TARGET[1][0]) ** 2)
        self.fitness = mapRange(d, 0, WIDTH, max_dist, 0)
        # print("Fitness ", self.fitness)
        # print("------------------------")

    def dist(self, a, b):
        t_x = b[0][0]
        t_y = b[1][0]
        pos_x = a[0][0]
        pos_y = a[1][0]

        return math.sqrt((t_y - pos_y) ** 2 + (t_x - pos_x) ** 2)
        # print(t_x,t_y,pos_x,pos_y)

    def collision(self):
        pos_x = self.pos[0][0]
        pos_y = self.pos[1][0]

        # # Rocket hit the barrier
        # if (pos_x > WIDTH and  pos_x < WIDTH + self.rocket_width and  pos_y > HEIGHT and pos_y < HEIGHT + self.rocket_height):
        #     self.crashed = True;

        # Rocket has hit left or right of window
        if (pos_x + self.rocket_width > WIDTH or pos_x < 0):
            # print("Collision")
            self.crashed = True

        # Rocket has hit top or bottom of window
        if (pos_y + self.rocket_height > HEIGHT or pos_y < 0):
            self.crashed = True
            # print("Collision")

    def show(self, win):
        """ Shows the image for the rocket"""
        self.draw_img(win, self.image, self.pos[0][0], self.pos[1][0], self.theta)
        # self.rectRotated(win, WHITE, self.pos, 12, self.angle)

    def draw_img(self, screen, image, x, y, angle):
        """ Rotates the image for rocket by the given angle and draws the rotated image"""
        rotated_image = pygame.transform.rotate(image, angle)
        screen.blit(rotated_image, rotated_image.get_rect(center=image.get_rect(topleft=(x, y)).center).topleft)

    def rectRotated(self, surface, color, pos, fill, angle):
        """ Rotates the Rectangle for rocket by the given angle and draws the rotated rectangle"""
        max_area = max(self.rocket_width, self.rocket_height)
        s = pygame.Surface((max_area, max_area))
        s = s.convert_alpha()
        s.fill((0, 0, 0, 0))
        pygame.draw.rect(s, color, (0, 0, self.rocket_width, self.rocket_height), fill)
        # surface.blit(self.rocket_image, (pos[0][0], pos[1][0]))
        s = pygame.transform.rotate(s, angle)
        surface.blit(s, (pos[0][0], pos[1][0]))

    def torque(self, r, force):
        """ Calculates the Torque given Force """
        return np.cross(r, force)

    def sum_of_all_torques(self):
        sum = self.torque(self.TRUSTER_1, self.TRUST_1) + self.torque(self.TRUSTER_2, self.TRUST_2) + self.torque(self.TRUSTER_3, self.TRUST_3) + self.torque(self.TRUSTER_4, self.TRUST_4)

        # with open("log.txt","a") as log:
        #     log.write(f"Torques Force   : {sum}\n")

        return self.mag * np.asscalar(sum)

    def sum_of_all_forces(self):
        sum_of_forces = self.TRUST_1 + self.TRUST_2 + self.TRUST_3 + self.TRUST_4
        # print(sum_of_forces)

        # print("Sum " ,sum_of_forces)
        c, s = np.cos(self.theta), np.sin(self.theta)
        rotational_matrix = np.array(((c, -s), (s, c)))

        global_forces = np.matmul(rotational_matrix, sum_of_forces.reshape(2, 1))

        # with open("log.txt","a") as log:
        #     log.write(f"\nGlobal Force   : {global_forces}\n")
        return global_forces

    def calculate(self, canonical_forces, initial_pos, initial_vel):
        self.acc = canonical_forces * 0.01

        final_vel = self.acc + initial_vel
        final_pos = initial_pos + (final_vel + initial_vel) / 2

        return final_vel, final_pos

    def update_forces(self):
        self.TRUST_1 = self.dna.genes[0]
        self.TRUST_2 = self.dna.genes[1]
        self.TRUST_3 = self.dna.genes[2]
        self.TRUST_4 = self.dna.genes[3]

    def update(self):
        """ Calculates the new velocity, position, angular velocity and theta based on the ΣF and Στ 
            Also check for collision and updates the new Forces """
        self.collision()

        if not self.crashed:
            self.vel, self.pos = self.calculate(self.sum_of_all_forces(), self.pos, self.vel)
            self.theta_dot, self.theta = self.calculate(self.sum_of_all_torques(), self.theta, self.theta_dot)

        self.update_forces()


# updates the screen with every main game loop
def redrawwindow(win, rocket):
    win.fill(BLACK)
    # win.blit(BG_IMG,(0,0))
    draw_circle(win)
    rocket.run(win)
    pygame.display.update()


def draw_circle(win):
    # print("Circle Printed")
    pygame.draw.circle(win, RED, (TARGET[0][0], TARGET[1][0]), 5)


def Mainloop():
    popl = Population()
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Smart Rockets Game")
    clock = pygame.time.Clock()

    count = 0
    Fps = 200

    run = True
    while run:
        clock.tick(Fps)
        keys = pygame.key.get_pressed()

        for event in pygame.event.get():
            if event.type == keys[pygame.QUIT] or keys[pygame.K_ESCAPE]:
                run = False
                sys.exit()

        popl.run(win)

        count += 1
        if count == LIFESPAN:
            popl.evaluate()
            popl.natural_selection()
            count = 0

        redrawwindow(win, popl)
    pygame.QUIT
    quit()


Mainloop()
