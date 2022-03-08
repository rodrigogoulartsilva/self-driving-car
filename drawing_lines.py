import pygame
import math
import numpy as np
import pickle
import os
from os import walk
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque

cwd = os.getcwd()+'/'

tracks_path = cwd+'/tracks/'

imgs_path = cwd+'/images/'

pygame.init()

clock = pygame.time.Clock()

track_load = 0

display_width = 1200
display_height = 800

gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption('Building Car')

#RGB colors
black = (0,0,0)
white = (255,255,255)
red = (255,0,0)
green = (50,205,50)
blue = (0,0,255)
silver = (192,192,192)
grey = (128,128,128)
grass_green = (34,139,34)

button_x = 10
button_y = 10
button_width = 150
button_height = 30

lines_visible=True

class car(object):

    def __init__(self):
        self.x = (display_width * 0.7)
        self.y = (display_height * 0.8)
        self.current_speed = 0
        self.acceleration = 1
        self.friction = self.acceleration*1.7
        self.max_speed = 15
        self.degree = 0
        self.degree_change = (self.max_speed/3)
        self.crash = False
        self.last_reward_gate = -1

    def image(self,img_file):
        self.image_obj = pygame.image.load(cwd+img_file)
        self.car_rect = self.image_obj.get_rect(topleft=(self.x,self.y))

        self.width = self.car_rect.w

        self.car_orig_center = self.car_rect.center

        self.box_lines = {'center':self.car_rect.center,
                            'left_side':(self.car_rect.bottomleft,self.car_rect.topleft),
                            'top_side':(self.car_rect.topleft,self.car_rect.topright),
                            'right_side':(self.car_rect.topright,self.car_rect.bottomright),
                            'botom_side':(self.car_rect.bottomright,self.car_rect.bottomleft),
                            'positive_diagonal':(self.car_rect.bottomleft,self.car_rect.topright),
                            'negative_diagonal':(self.car_rect.topleft,self.car_rect.bottomright)}

        num_of_sensors = 8

        self.sensors = {}

        self.sensor_radius = 130

        self.sensors[0] = ((self.car_rect.center[0]-self.sensor_radius,self.car_rect.midleft[1]),
                            self.car_rect.center)

        add_degree = 360/num_of_sensors

        sensor_degree = 360/num_of_sensors

        for sensor_n in range(num_of_sensors-1):
            sensor_n = sensor_n + 1
            self.sensors[sensor_n] = update_lines(self.sensors[sensor_n-1],self.car_rect.center,0,0,-sensor_degree)
            sensor_degree += add_degree

        self.sensors_distances = np.array([np.linalg.norm(np.array(self.sensors[s][0])-np.array(self.sensors[s][1])) for s in self.sensors.keys()])


    def drive(self,right_turn,left_turn,forward,backward,prev_direction,track):

        reward = 0

        start_degree=self.degree

        global lines_visible

        if forward or backward:
            if right_turn:
                self.degree += self.degree_change
                while self.degree > 359:
                    self.degree -= 360

            elif left_turn:
                self.degree -= self.degree_change
                while self.degree < 0:
                    self.degree += 360

        degree_delta = self.degree-start_degree

        dx = math.sin(math.radians(self.degree))
        dy = math.cos(math.radians(self.degree))

        if forward:
            if self.current_speed+self.acceleration <= self.max_speed:
                self.current_speed += self.acceleration
            self.y -= int(self.current_speed * dx)
            self.x -= int(self.current_speed * dy)
        elif backward:
            if self.current_speed+self.acceleration <= self.max_speed:
                self.current_speed += self.acceleration
            self.y += int(self.current_speed * dx)
            self.x += int(self.current_speed * dy)

        else:
            if self.current_speed-self.friction >= 0:
                self.current_speed -= self.friction
            else:
                self.current_speed = 0

            if prev_direction == 'forward':
                self.y -= int(self.current_speed * dx)
                self.x -= int(self.current_speed * dy)
            elif prev_direction == 'backward':
                self.y += int(self.current_speed * dx)
                self.x += int(self.current_speed * dy)

        center_before = self.car_rect.center

        car2 = pygame.transform.rotate(self.image_obj,-self.degree).convert_alpha()

        self.car_rect = car2.get_rect(topleft=(self.x,self.y))

        center_after = self.car_rect.center

        x_change = center_after[0]-center_before[0]
        y_change = center_after[1]-center_before[1]

        new_center = line_intersection(self.box_lines['positive_diagonal'],self.box_lines['negative_diagonal'])
        self.box_lines['center'] = (new_center[0]+x_change,new_center[1]+y_change)

        for box_side in ['left_side','top_side','right_side','botom_side','positive_diagonal','negative_diagonal']:
            self.box_lines[box_side] = update_lines(self.box_lines[box_side],self.box_lines['center'],x_change,y_change,-degree_delta)
            for track_line in track.side_lines:
                intersect = line_intersection(self.box_lines[box_side],track_line)
                if intersect !=None:
                    if self.check_touch(intersect,self.box_lines[box_side],track_line):
                        if lines_visible:
                            pygame.draw.line(gameDisplay, red, self.box_lines[box_side][0], self.box_lines[box_side][1])
                        reward = -1
                        break
                    else:
                        if lines_visible:
                            pygame.draw.line(gameDisplay, black, self.box_lines[box_side][0], self.box_lines[box_side][1])

            for gate_num, gate_line in enumerate(track.reward_gates):
                intersect = line_intersection(self.box_lines[box_side],gate_line)
                if intersect !=None:
                    if self.check_touch(intersect,self.box_lines[box_side],gate_line) and gate_num == self.last_reward_gate+1:
                        reward = 1

                        if gate_num+1 < len(track.reward_gates):
                            self.last_reward_gate = gate_num

                        else:
                            self.max_speed += 1
                            #self.degree_change = (self.max_speed/3)
                            self.last_reward_gate = -1

                        break

        for sensor in self.sensors:
            self.sensors[sensor] = update_lines(self.sensors[sensor],self.box_lines['center'],x_change,y_change,-degree_delta)
            if lines_visible:
                pygame.draw.line(gameDisplay, black, self.sensors[sensor][0], self.sensors[sensor][1])
            intersections = []

            for track_line in track.side_lines:
                intersect = line_intersection(self.sensors[sensor],track_line)
                overlap = self.check_touch(intersect,self.sensors[sensor],track_line)
                if intersect != None and overlap:
                    intersect = (int(intersect[0]),int(intersect[1]))
                    intersections.append(intersect)

            if len(intersections)>0:
                intersections = [(a,np.linalg.norm(np.array(a)-np.array(self.sensors[sensor][1]))) for a in intersections]
                intersections.sort(key=lambda tup: tup[1])
                closest_intersect = intersections[0][0]
                self.sensors_distances[sensor] = intersections[0][1]
                if lines_visible:
                    pygame.draw.circle(gameDisplay, black, closest_intersect,5)

            else:
                self.sensors_distances[sensor] = np.linalg.norm(np.array(self.sensors[sensor][0])-np.array(self.sensors[sensor][1]))

        gameDisplay.blit(car2,(self.x,self.y))

        return self.sensors_distances, reward

    def display_only(self):
        gameDisplay.blit(self.image_obj,(self.x,self.y))


    def check_touch(self,intersect,car_line,track_line):
        intersect = (int(intersect[0]),int(intersect[1]))
        A_x_min = int(np.min(np.array((car_line[0][0],car_line[1][0]))))
        A_x_max = int(np.max(np.array((car_line[0][0],car_line[1][0]))))
        A_y_min = int(np.min(np.array((car_line[0][1],car_line[1][1]))))
        A_y_max = int(np.max(np.array((car_line[0][1],car_line[1][1]))))
        B_x_min = int(np.min(np.array((track_line[0][0],track_line[1][0]))))
        B_x_max = int(np.max(np.array((track_line[0][0],track_line[1][0]))))
        B_y_min = int(np.min(np.array((track_line[0][1],track_line[1][1]))))
        B_y_max = int(np.max(np.array((track_line[0][1],track_line[1][1]))))

        if intersect == None:
            return False

        elif A_x_min<=intersect[0]<=A_x_max and A_y_min<=intersect[1]<=A_y_max and B_x_min<=intersect[0]<=B_x_max and B_y_min<=intersect[1]<=B_y_max:
            return True

        else:
            return False


def update_lines(line,center,x_chg,y_chg,degree):
    new_line = ((line[0][0]+x_chg,line[0][1]+y_chg),
                (line[1][0]+x_chg,line[1][1]+y_chg))
    new_line = (rotate_vec(center[0],center[1],new_line[0][0],new_line[0][1],degree),
                rotate_vec(center[0],center[1],new_line[1][0],new_line[1][1],degree))
    return new_line

def rotate_vec(x_origin,y_origin,x,y,degree):
    x_rotated = ((x - x_origin) * math.cos(math.radians(degree))) - ((y_origin - y) * math.sin(math.radians(degree))) + x_origin
    y_rotated = ((y - y_origin) * math.cos(math.radians(degree))) - ((x - x_origin) * math.sin(math.radians(degree))) + y_origin
    return (x_rotated,y_rotated)

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def get_angle(p0, p1=np.array([0,0]), p2=None):
    if p2 is None:
        p2 = p1 + np.array([1, 0])

    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)
    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))

    return np.degrees(angle)

def text_objects(text,font,color):
    textSurface = font.render(text,True,color)
    return textSurface, textSurface.get_rect()

def button(msg,x,y,width,height,act_color,inact_color,mouse_click,action=None):
    global track_load
    global track_obj
    global screen_mode
    global ai_agent
    global num_of_actions
    global num_of_states
    mouse = pygame.mouse.get_pos()

    if x+width > mouse[0] > x and y+height > mouse[1] > y:
        pygame.draw.rect(gameDisplay,act_color,(x,y,width,height))

        if mouse_click and action != None:
            if action == 'draw_track':
                draw_track()
            elif action == 'done_draw':
                game_loop()
            elif action == 'add_reward_gate':
                track_obj.track_reward_mode = 'reward_mode'
            elif action == 'back_to_track':
                track_obj.track_reward_mode = 'track_mode'

            elif action == 'undo_track':
                if len(track_obj.lines)>0:
                    track_obj.remove_line()

            elif action == 'undo_rewardl':
                if len(track_obj.reward_gates)>0:
                    track_obj.remove_reward_gate()

            elif action == 'save_track':
                track_num = len([f for f in list(walk(tracks_path))[0][2] if f.endswith('.pickle')])
                file_name = 'track_{}.pickle'.format(track_num)
                with open(tracks_path+file_name, 'wb') as handle:
                    pickle.dump(track_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

            elif action == 'load_track':
                available_tracks = [f for f in list(walk(tracks_path))[0][2] if f.endswith('.pickle')]

                if len(available_tracks) < track_load+1:
                    track_load = 0

                if len(available_tracks)>0:
                    file_name = 'track_{}.pickle'.format(track_load)
                    with open(tracks_path+file_name, 'rb') as handle:
                        track_obj = pickle.load(handle)
                    track_load+=1

            elif action == 'RL_mode':
                screen_mode = 'learning_mode'

            elif action == 'game_mode':
                screen_mode = 'game'

            elif action == 'learn_now':
                ai_agent = learning_agent(num_of_states,num_of_actions)
                

    else:

        pygame.draw.rect(gameDisplay,inact_color,(x,y,width,height))

    largeText = pygame.font.Font('freesansbold.ttf',15)
    TextSurf, TextRect = text_objects(msg, largeText,black)
    TextRect.center = ((x+(width/2)),(y+(height/2)))
    gameDisplay.blit(TextSurf, TextRect)

class track(object):

    def __init__(self):
        self.lines = []
        self.side_lines = []
        self.polygons = []
        self.reward_gates = []
        self.track_reward_mode = 'track_mode'
        self.bushes = []

    def add_line(self,position1,position2):
        self.lines.append((position1,position2))

    def add_track(self,polygon):
        self.polygons.append(polygon)

    def remove_line(self):
        if len(self.lines)>0:
            del self.lines[-1]
            del self.polygons[-1]

    def draw_polygon(self,mouse_line,rect_width=30):
        rect_height = np.linalg.norm(np.array(mouse_line[0])-np.array(mouse_line[1]))
        rect_bottomcenter = mouse_line[0]
        rect_topcenter = (mouse_line[0][0],mouse_line[0][1]+rect_height)
        rect_center = (mouse_line[0][0],mouse_line[0][1]+rect_height/2)
        rect_bottomleft = (mouse_line[0][0]-rect_width/2,mouse_line[0][1])
        rect_bottomright = (mouse_line[0][0]+rect_width/2,mouse_line[0][1])
        rect_topleft = (mouse_line[0][0]+rect_width/2,mouse_line[0][1]+rect_height)
        rect_topright = (mouse_line[0][0]-rect_width/2,mouse_line[0][1]+rect_height)
        degree_diff = get_angle(mouse_line[1], mouse_line[0],rect_topcenter)
        points = [rect_bottomleft,rect_bottomright,rect_topleft,rect_topright]
        rotated_middle = np.array([rotate_vec(rect_bottomcenter[0],rect_bottomcenter[1],i[0],i[1],degree_diff) for i in points])
        
        return rotated_middle

    def finish_track(self):
        for polygon in self.polygons:
            self.side_lines.append((polygon[1],polygon[2]))
            self.side_lines.append((polygon[0],polygon[3]))

        n_bushes = 50

        #for i in range(n_bushes):
            #bushes_imgs = [f for f in list(walk(imgs_path))[0][2] if f.startswith('bush')]
            #bush_img_file = random.choice(bushes_imgs)
            #bush_img = pygame.image.load(imgs_path+bush_img_file)
            #bush_x = random.randint(0,display_width)
            #bush_y = random.randint(button_height+10,display_height)
            #self.bushes.append((bush_img,bush_x,bush_y ))

    def add_reward_gate(self,position1,position2):
        self.reward_gates.append((position1,position2))

    def remove_reward_gate(self):
        if len(self.reward_gates)>0:
            del self.reward_gates[-1]


def display_rewards(count,try_num,best_score):
    font = pygame.font.SysFont(None,25)
    text = font.render('Score: {0} Tries: {1} Best Score:{2}'.format(count,try_num,best_score),True,black)
    gameDisplay.blit(text,(0,button_height+button_y+10))

def draw_track():

    init_point = None
    first_point = None
    r_init_point=None
    track_obj.lines = []
    track_obj.side_lines = []
    track_obj.reward_gates = []
    track_obj.polygons = []
    
    track_change = 10
    max_track_width = 200
    min_track_width = car_obj.width+30
    track_width = min_track_width

    while True:

        gameDisplay.fill(grass_green)
        mouse = pygame.mouse.get_pos()
        mouse_click = False


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_click = True

                    if mouse[1] > button_height+button_y:
                        if track_obj.track_reward_mode == 'track_mode':
                            if init_point == None:
                                init_point = mouse
                                first_point = mouse
                                track_points = track_obj.draw_polygon((init_point,mouse),track_width)
                                init_line = (track_points[0],track_points[1])
                                first_line = (track_points[0],track_points[1])
                            elif len(track_obj.lines)>1 and first_point != None and np.linalg.norm(np.array(mouse)-np.array(first_point)) <= 10:
                                track_obj.add_line(init_point,first_point)
                                track_obj.add_track((track_obj.polygons[-1][3],track_obj.polygons[-1][2],track_obj.polygons[0][1],track_obj.polygons[0][0]))
                                track_obj.finish_track()
                                first_point=None
                                init_point=None

                            else:
                                track_obj.add_line(init_point,mouse)
                                if len(track_obj.polygons)>0:
                                    track_obj.polygons[-1] = (track_obj.polygons[-1][0],track_obj.polygons[-1][1],track_points[1],track_points[0])
                                track_obj.add_track(track_points)
                                init_line = (track_points[2],track_points[3])
                                init_point = mouse

                        elif track_obj.track_reward_mode == 'reward_mode':
                            if r_init_point == None:
                                r_init_point = mouse

                            else:
                                track_obj.add_reward_gate(r_init_point,mouse)
                                r_init_point = None

            if event.type == pygame.KEYDOWN:             
                if event.key == pygame.K_s:
                    if track_width+track_change <= max_track_width:
                        track_width += track_change
                elif event.key == pygame.K_x:
                    if track_width+track_change >= min_track_width:
                        track_width -= track_change


        display_track()

        if init_point != None and track_obj.track_reward_mode == 'track_mode':

            track_points = track_obj.draw_polygon((init_point,mouse),track_width)
            pygame.draw.polygon(gameDisplay,grey,track_points)
            pygame.draw.line(gameDisplay, blue, init_point, mouse)

        if r_init_point != None and track_obj.track_reward_mode == 'reward_mode':
            pygame.draw.line(gameDisplay, green, r_init_point, mouse)

        button("Done",button_x,button_y,button_width,button_height,silver,grey,mouse_click,'done_draw')

        if track_obj.track_reward_mode == 'track_mode':
            button("Add Reward Gate",button_width+button_x,button_y,button_width,button_height,silver,grey,mouse_click,'add_reward_gate')
        else:
            button("Draw Track",button_width+button_x,button_y,button_width,button_height,silver,grey,mouse_click,'back_to_track')

        rem_line = len(track_obj.lines)

        #Need to make undo reward
        if track_obj.track_reward_mode == 'track_mode':
            button("Undo",button_width*2+button_x,button_y,button_width,button_height,silver,grey,mouse_click,'undo_track')

            if rem_line > len(track_obj.lines):
                init_point=track_obj.lines[-1][1]
        else:
            button("Undo",button_width*2+button_x,button_y,button_width,button_height,silver,grey,mouse_click,'undo_rewardl')

        button("Save track",button_width*3+button_x,button_y,button_width,button_height,silver,grey,mouse_click,'save_track')

        car_obj.display_only()

        pygame.display.update()

        clock.tick(60)

def display_track():

    for bush in track_obj.bushes:
        gameDisplay.blit(bush[0],(bush[1],bush[2]))

    for polygon in track_obj.polygons:
        pygame.draw.polygon(gameDisplay,grey,polygon)

    for line in track_obj.lines+track_obj.side_lines:
        pygame.draw.line(gameDisplay, white, line[0], line[1])

    if lines_visible:
        for line in track_obj.reward_gates:
            pygame.draw.line(gameDisplay, green, line[0], line[1])

class learning_agent(object):
    def __init__(self,state_size,action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.0001
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(30, input_dim=self.state_size, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((np.array(state).reshape(1, -1), action, reward, np.array(next_state).reshape(1, -1), done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)

        max_score = np.max(act_values[0])

        actions_max_score = np.where(act_values[0]== max_score)[0]

        a = random.choice(actions_max_score)
        
        return a #np.argmax(act_values[0])

    def replay(self, batch_size):

        if batch_size > len(self.memory):
            batch_size = len(self.memory)

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward

            if not done:
                target = reward + self.gamma * \
                    np.amax(self.model.predict(next_state)[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


track_obj = track()
car_obj = car()
car_obj.image('car.png')
screen_mode = 'game'
ai_agent = None
num_of_states = len(car_obj.sensors.keys())+1 #+1 for speed
#actions = ['forward','backward','forward_left','forward_right','backward_left','backward_right']
actions = ['forward','forward_left','forward_right','nothing']
#actions = ['forward','forward_left','forward_right']
num_of_actions = len(actions)

def game_loop():

    left_turn = False
    right_turn = False
    forward = False
    backward = False
    prev_direction = None
    global car_obj
    global screen_mode
    car_obj.x = (display_width * 0.45)
    car_obj.y = (display_height * 0.8)
    total_rewards=0
    best_score = 0
    state = list(car_obj.sensors_distances) + list([car_obj.current_speed])
    state = np.array(state).reshape(1, -1)
    action = 0
    done = False
    max_tries = 10000
    car_still = 0

    for try_num in range(max_tries):
    
        while True:

            #print(screen_mode,ai_agent)

            mouse_click = False

            gameDisplay.fill(grass_green)

            for event in pygame.event.get():

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        mouse_click=True

                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

                if screen_mode == 'game':
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_LEFT:
                            left_turn = True
                        elif event.key == pygame.K_RIGHT:
                            right_turn = True
                        if event.key == pygame.K_UP:
                            forward = True
                        elif event.key == pygame.K_DOWN:
                            backward = True

                    if event.type == pygame.KEYUP:
                        if event.key == pygame.K_LEFT:
                            left_turn = False
                        if event.key == pygame.K_RIGHT:
                            right_turn = False
                        if event.key == pygame.K_UP:
                            forward = False
                        if event.key == pygame.K_DOWN:
                            backward = False

            if screen_mode == 'learning_mode' and ai_agent != None:

                action = ai_agent.act(state)

                print(actions[action],ai_agent.epsilon)

                if actions[action] == 'forward':
                    right_turn = False
                    left_turn = False
                    forward = True
                    backward = False

                elif actions[action] == 'backward':
                    right_turn = False
                    left_turn = False
                    forward = False
                    backward = True

                elif actions[action] == 'forward_left':
                    right_turn = False
                    left_turn = True
                    forward = True
                    backward = False

                elif actions[action] == 'forward_right':
                    right_turn = True
                    left_turn = False
                    forward = True
                    backward = False

                elif actions[action] == 'backward_left':
                    right_turn = False
                    left_turn = True
                    forward = False
                    backward = True

                elif actions[action] == 'backward_right':
                    right_turn = True
                    left_turn = False
                    forward = False
                    backward = True

                elif actions[action] == 'nothing' and (prev_direction == 'forward' or prev_direction == 'backward'):
                    right_turn = False
                    left_turn = False
                    forward = False
                    backward = False



            button("Draw Track",button_x,button_y,button_width,button_height,silver,grey,mouse_click,'draw_track')

            button("Load Track",button_width*1+button_x,button_y,button_width,button_height,silver,grey,mouse_click,'load_track')

            if screen_mode == 'game':
                button("Learning Mode",button_width*2+button_x,button_y,button_width,button_height,silver,grey,mouse_click,'RL_mode')
            else:
                button("Gaming Mode",button_width*2+button_x,button_y,button_width,button_height,silver,grey,mouse_click,'game_mode')
                #***************
                #*Do this later*
                #***************
                button("Load Trained Agent",button_width*3+button_x,button_y,button_width,button_height,silver,grey,mouse_click,'RL_mode')

                button("Learn!",button_width*4+button_x,button_y,button_width,button_height,silver,grey,mouse_click,'learn_now')
            

            display_track()

            sensor_data,reward = car_obj.drive(right_turn,left_turn,forward,backward,prev_direction,track_obj)

            total_rewards += reward

            if screen_mode == 'learning_mode' and ai_agent != None:

                #if reward == 1:
                    #reward = total_rewards

                next_state = list(sensor_data) + list([car_obj.current_speed])

                if reward == -1:
                    done = True
                else:
                    done = False

                ai_agent.remember(state,action,reward,next_state,done)

                state = next_state

                state = np.array(state).reshape(1, -1)

            if car_obj.current_speed == 0 and ai_agent != None:
                car_still += 1

                if car_still > 10:
                    reward = -1

            else:
                car_still = 0

            if reward == -1:
                left_turn = False
                right_turn = False
                forward = False
                backward = False
                prev_direction = None
                car_obj = car()
                car_obj.image('car.png')
                car_obj.x = (display_width * 0.45)
                car_obj.y = (display_height * 0.8)
                total_rewards=0

                ai_agent.replay(500)

                break
            

            if total_rewards>best_score:
                best_score=total_rewards

            display_rewards(total_rewards,try_num,best_score)

            if forward or prev_direction == 'forward':
                prev_direction = 'forward'

            if backward or prev_direction == 'backward':
                prev_direction = 'backward'

            pygame.display.update()

            clock.tick(60)

game_loop()
