import sys
import pygame
# import time
import numpy as np
import random
import math
import time
# from scipy.ndimage.interpolation import rotate
from utils import mapRange
from functools import reduce

# screen
WIDTH, HEIGHT = (1200, 800)
TARGET = np.array([WIDTH / 2, 100])
LIFESPAN = 500

# Color
BLACK = (0, 0, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)


class Population(object):
    def __init__(self):
        self.rockets = []
        self.pop_max = 25

        self.mating_pool = []

        for _ in range(self.pop_max):
            self.rockets.append(Rocket())

    def run(self, win, counter):
        for i in range(self.pop_max):
            self.rockets[i].update(win, counter)
            # self.rockets[i].show(win)

    def evaluate(self):
        # print("Evaluation\n")
        max_fitness = 0

        for i in range(self.pop_max):
            self.rockets[i].calcfitness()
            if self.rockets[i].fitness > max_fitness:
                max_fitness = self.rockets[i].fitness

        print(max_fitness)
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
            # print("parentA", parentA.genes)
            # print("parentB", parentB.genes)

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
            self.genes = np.random.rand(LIFESPAN, self.num_thrusters)
            # self.genes[:, :, 0] = 0
            # print("Genes Shape: ", self.genes.shape)
            # print(self.genes)

    def crossover(self, partner):
        # print("Crossover")
        newgenes = np.zeros((len(self.genes), 4))

        # print(self.genes, " ", partner.genes)

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

                mutated_gene = np.random.randn(self.num_thrusters)
                # print("Mutated genes", mutated_gene)
                self.genes[i] = mutated_gene

        # print(self.genes)
        # print("-------------Mutated Genes------------")


class Rocket():
    count = 0

    def __init__(self, dna=None, theta=180):
        # Accelartion, Velocity and Position Vectors
        self.acc = np.zeros((1, 2))[0]
        self.vel = np.zeros((1, 2))[0]

        self.rocket_width = 30
        self.rocket_height = 30

        self.pos = np.array([WIDTH / 2, HEIGHT / 2])

        # Angular velocity
        self.theta_dot = 0

        self.fitness = 0

        self.color = WHITE
        self.theta = np.radians(theta)
        self.mag = 0.01

        self.crashed = False
        self.completed = False

        self.booster_size = 20

        self.image = pygame.image.load("./assets/rocket.png")
        self.image = pygame.transform.scale(self.image, (self.rocket_height, self.rocket_height))

        self.booster = pygame.image.load("./assets/loud.png")
        self.booster = pygame.transform.scale(self.booster, (self.booster_size, self.booster_size))

        if dna is not None:
            self.dna = dna
        else:
            self.dna = DNA(num_thrusters=4)

        # FIX: Convert Trusters into an array
        self.TRUSTER_1 = np.array([self.rocket_width / 4, -self.rocket_height / 2])
        self.TRUSTER_2 = np.array([self.rocket_width / 2, -self.rocket_height / 2])
        self.TRUSTER_3 = np.array([- self.rocket_width / 4, -self.rocket_height / 2])
        self.TRUSTER_4 = np.array([- self.rocket_width / 2, self.rocket_height / 2])

        self.num_thrusters = 4

        self.thrusters = [self.TRUSTER_1, self.TRUSTER_2, self.TRUSTER_3, self.TRUSTER_4]
        self.forces = self.dna.genes

        # self.TRUST_1 = self.dna.genes[0]
        # self.TRUST_2 = self.dna.genes[1]
        # self.TRUST_3 = self.dna.genes[2]
        # self.TRUST_4 = self.dna.genes[3]

    def global_coords(self, x, y):
        return self.pos + np.array([x, y])

    def show(self, win):
        """ Shows the image for the rocket"""
        self.draw_img(win, self.image, self.pos[0], self.pos[1], np.rad2deg(self.theta))

        # for i in range(self.num_thrusters):
        #     self.draw_boosters(win, self.booster, self.thrusters[i][0], self.thrusters[i][1])

    def draw_img(self, screen, image, x, y, angle):
        """ Rotates the image for rocket by the given angle and draws the rotated image"""
        rotated_image = pygame.transform.rotate(image, angle)
        screen.blit(rotated_image, rotated_image.get_rect(center=image.get_rect(topleft=(x, y)).center).topleft)

    def draw_boosters(self, screen, image, x, y):
        """ Rotates the boosters for rocket by the pos and draws the rotated boosters"""
        # screen.blit(rotated_image, rotated_image.get_rect(center=image.get_rect(topleft=(x, y)).center).topleft)
        global_booster_coords = self.global_coords(x, y)
        screen.blit(image, image.get_rect(center=(global_booster_coords[0], global_booster_coords[1])))

    def torque(self, r, force):
        """ Calculates the Torque of the given Force """
        return r[0] * force

    def calcfitness(self):
        d = self.dist(self.pos, TARGET)

        # max_dist = math.sqrt((TARGET[0]) ** 2 + (HEIGHT - TARGET[1]) ** 2)
        self.fitness = mapRange(d, 0, WIDTH, 5000, 0)

    def collision(self):
        # Rocket has hit left or right of window
        if (self.pos[0] + self.rocket_width > WIDTH or self.pos[0] < 0):
            self.crashed = True

        # Rocket has hit top or bottom of window
        if (self.pos[1] + self.rocket_height > HEIGHT or self.pos[1] < 0):
            self.crashed = True

    def dist(self, a, b):
        t_x = b[0]
        t_y = b[1]
        pos_x = a[0]
        pos_y = a[1]

        # print(t_x, t_y)
        # print(pos_x, pos_y)
        return math.sqrt((t_y - pos_y) ** 2 + (t_x - pos_x) ** 2)

    def sum_of_all_torques(self, forces):
        """  Calculates the Torque from all the force Vectors and calculates it sum"""
        # print(" Force in sum_of_torques", forces)

        sum = self.torque(self.TRUSTER_1, forces[0]) + self.torque(self.TRUSTER_2, forces[1]) + self.torque(self.TRUSTER_3, forces[2]) + self.torque(self.TRUSTER_4, forces[3])

        return self.mag * np.asscalar(sum)

    def sum_of_all_forces(self, force, theta):
        """  Calculates the sum of all the force Vectors"""
        sum_of_forces = np.array(reduce(lambda a, b: a + b, force))

        # print("Sum of all forces", sum_of_forces)

        global_forces = np.array([-sum_of_forces * np.sin(theta), sum_of_forces * np.cos(theta)])
        return global_forces

    def calculate(self, canonical_forces, initial_pos, initial_vel):
        self.acc = canonical_forces * 0.01

        # print("\n -------------------")
        # print("canonical_forces: ", self.acc)
        # print("Initial Velocity ", initial_vel)
        final_vel = self.acc + initial_vel
        # print("final Velocity ", final_vel)

        final_pos = initial_pos + (final_vel + initial_vel) / 2
        # print("Sum: ", initial_pos)
        # print("final Posiition ", final_pos)
        #
        return final_vel, final_pos

    def update(self, win, i):
        """ Calculates the new velocity, position, angular velocity and theta based on the ΣF and Στ 
            Also check for collision and updates the new Forces """
        # count = 0
        # self.collision()

        # if not self.crashed:
        # print("Current Position", self.pos, "\n")
        # print("------------------------------------")
        # print("Counter", count)
        self.vel, self.pos = self.calculate(self.sum_of_all_forces(self.forces[i], self.theta), self.pos, self.vel)
        self.theta_dot, self.theta = self.calculate(self.sum_of_all_torques(self.forces[i]), self.theta, self.theta_dot)
        # print("Update Complete")
        self.show(win)


# updates the screen with every main game loop
def redrawwindow(win, rocket, counter):
    win.fill(BLACK)
    # win.blit(BG_IMG,(0,0))
    draw_circle(win)
    rocket.run(win, counter)
    pygame.display.update()


def draw_circle(win):
    pygame.draw.circle(win, RED, (TARGET[0], TARGET[1]), 5)


def Mainloop():
    LIFE = 0
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

        # popl.run(win, LIFE)
        LIFE += 1
        if LIFE == LIFESPAN:
            LIFE = 0

        count += 1
        if count == LIFESPAN:
            popl.evaluate()
            popl.natural_selection()
            count = 0

        # time.sleep(0.01)
        redrawwindow(win, popl, LIFE)
    pygame.QUIT
    quit()


Mainloop()
