import gym
from gym import spaces

import numpy as np
import cv2
from timer import Timer

from config import *

M = 0
L = 1
R = 2
a_dir = {
        M : 0,
        L : -1,
        R : 1,
        }

class Ball(object):
    def __init__(self, i, j, d):
        self.i = i
        self.j = j
        self.dir = d
        self.warn = True
        self.age = 0

    def step(self):
        if self.age > 5:
            self.warn = False
        if not self.warn:
            self.j += self.dir * BALL_SPEED
        self.age += 1

    def render(self, frame):
        if self.warn:
            if self.dir > 0:
                pt1 = (self.j, self.i - BALL_R)
                pt2 = (self.j, self.i + BALL_R)
                pt3 = (self.j+BALL_R, self.i)
            else:
                pt1 = (self.j, self.i - BALL_R)
                pt2 = (self.j, self.i + BALL_R)
                pt3 = (self.j-BALL_R, self.i)

            #cv2.fillConvexPoly(frame, [pt1,pt2,pt3], 255)
            cv2.fillPoly(frame, np.array([[pt1,pt2,pt3]]), 255)

        else:
            cv2.circle(frame, (self.j, self.i), BALL_R, 255, -1)

def collisionCheck(cir, rect):
    cy,cx,cr = cir
    ry,rx,rh,rw = rect
    dx = cx - max(rx, min(cx, rx + rw))
    dy = cy - max(ry, min(cy, ry + rh))
    return dx*dx + dy*dy < cr*cr

class Gravitron(gym.Env):
    metadata = {
            'render.modes' : ['human']
            }
    def __init__(self, w=WIN_W, h=WIN_H):
        self.w, self.h = w,h
        self.action_space = spaces.Discrete(2) # left / right
        low = np.zeros((self.h, self.w), dtype=np.uint8)
        high = np.full((self.h, self.w), 255, dtype=np.uint8)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low, high)

        self.reset()
        self.drawing = False 

    def _reset(self):
        self.time = 0
        self.frame = np.zeros((self.h, self.w), dtype=np.uint8)
        self.balls = []
        self.i, self.j = self.h/2, self.w/2
        self.dir = 1
        self.draw()
        return self.state()
    def _step(self, action):

        if self.time % BALL_SPAWN_PERIOD == 0:
            i = np.random.randint(6) * (BALL_R*2 + BALL_GAP) + BALL_R
            j = np.random.choice((0,self.w-1))
            self.balls.append(Ball(i,j,(-1 if j else 1)))


        # MOVE CHAR 
        self.i += self.dir * CHAR_SPEED_Y
        self.j += a_dir[action] * CHAR_SPEED_X

        if self.i >= self.h:
            self.dir = -self.dir
            self.i = self.h - 1
        elif self.i <= 0:
            self.dir = -self.dir
            self.i = 0

        self.j %= self.w

        # MOVE BALLS
        self.balls = filter(lambda b: (b.j>=0 and b.j<self.w), self.balls)

        p1_x = self.j - CHAR_W/2
        p2_x = self.j + CHAR_W/2
        p1_y = self.i - CHAR_H/2
        p2_y = self.i + CHAR_H/2
        done = False
        for b in self.balls:
            b.step()
            if collisionCheck((b.i,b.j,BALL_R),(p1_y, p1_x, CHAR_H, CHAR_W)):
                done = True
        # DRAW

        self.draw()
        self.time += 1

        reward = 0.001 #-1 if done else 0.001#?
        return self.state(), reward, done, {}

    def _render(self, mode='human', close=False):
        if close:
            self.drawing = False
            cv2.destroyAllWindows()
            return
        if not self.drawing:
            self.drawing = True
            cv2.namedWindow('Gravitron')
            cv2.moveWindow('Gravitron', 100,50)
        cv2.imshow('Gravitron', self.frame)

    def draw(self):
        self.frame.fill(0)
        p1_x = self.j - CHAR_W/2
        p2_x = self.j + CHAR_W/2
        p1_y = self.i - CHAR_H/2
        p2_y = self.i + CHAR_H/2
        cv2.rectangle(self.frame, (p1_x,p1_y), (p2_x,p2_y), 255, -1)
        for b in self.balls:
            b.render(self.frame)
    def state(self):
        return np.expand_dims((self.frame == 255).astype(np.float32), axis=2)

def main():
    W,A,S,D = 119,97,115,100
    env = Gravitron()
    d = False
    s0 = env.reset()
    a = M
    with Timer('run'):
        while not d:
            env.render()
            k = cv2.waitKey(30)

            if k == A:
                a = L
            elif k == D:
                a = R
            elif k == 255:
                pass
            else:
                a = M
            s, r, d, _ = env.step(a)
        print env.time

if __name__ == "__main__":
    main()
