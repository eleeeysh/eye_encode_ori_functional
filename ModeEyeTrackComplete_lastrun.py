#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on February 21, 2025, at 14:00
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from codePrepareComponents
import random
import math

MAX_DRAWINGS = 3
MAX_STROKES = 10
CLICK_REGION_FACTOR = 3

# define some helper functions
# functions for ori comparison
def funcComputeOriDiff(s1, s2):
    diff = (s1 - s2 + 90) % 180 - 90
    if diff == -90:
        diff = 90
    return diff

# functions to create buttons
class SelfDefinedButton:
    def __init__(self, window, text, pos, size):
        # load background
        img_path = "resources/button.png"
        self.img = visual.ImageStim(
            win=window, image=img_path,
            size=size, pos=pos)
        
        # load text
        text_height = size[1] / 2
        self.text = visual.TextStim(
            win=window, text=text, pos=pos, color='black',
            height=text_height, anchorHoriz='center', anchorVert='center')
            
        # get stats
        self.left = pos[0] - size[0] / 2
        self.right = pos[0] + size[0] / 2
        self.top = pos[1] + size[1] / 2
        self.bottom = pos[1] - size[1] / 2
        
        # set auto draw
        self.setAutoDraw(True)
        
    def setAutoDraw(self, autoDraw):
        self.img.setAutoDraw(autoDraw)
        self.text.setAutoDraw(autoDraw)
        
    def contains(self, pos):
        px = pos[0]
        py = pos[1]
        on_button = (
            (px > self.left) & (px < self.right) & (
            py > self.bottom) & (py < self.top))
        return on_button

def funcCreateButton(window, text, pos, size):
    button = SelfDefinedButton(window, text, pos, size)
    return button
    
def createPatch(window, pos, radius, fill_color='white', line_color='black'):
    scaled_border = radius * 10 / 0.2
    patch = visual.Circle(
        window, pos=pos, radius=radius,
        fillColor=fill_color, lineColor=line_color, colorSpace='rgb',
        lineWidth=scaled_border)
    return patch
    
def createEnabledDrawPatch(window, pos, radius):
    patch = createPatch(
        window, pos, radius, 
        fill_color='white', line_color='black')
    return patch
    
def createDisabledDrawPatch(window, pos, radius):
    gray_color = [-0.3, -0.3, -0.3]
    patch = createPatch(
        window, pos, radius, 
        fill_color=gray_color, line_color=gray_color)
    return patch

# define the function to display the stimuli
def funcDrawStim(window, ori, loc, radius):
    # ori: 0-180 (degrees)
    # loc: position of the grating
    grating = visual.GratingStim(
        window, tex='sin', mask='gauss', 
        sf=5.0, size=radius*2, 
        ori=ori, pos=loc)
    grating.autoDraw = True
    return grating

# define the function to make report
# define the function to process mouse response
class WheelClickRegister:
    def __init__(self, wheel, pos_point, opp_point, wheel_pos):
        self.wheel = wheel
        self.wheel_c = wheel_pos[0]
        self.wheel_r = wheel_pos[1]
        self.click_radius = wheel_pos[2]
        self.point_click_selected = pos_point
        self.opposite_point = opp_point
        self.clicks_x = []
        self.clicks_y = []
        self.clicks_time = []
        
        # flag for whether there are any response
        self.is_valid = True
        self.ever_have_response = False
        
        # initialize only the wheel
        self.wheel.setAutoDraw(True)
        self.point_click_selected.setAutoDraw(False)
        self.opposite_point.setAutoDraw(False)
   
    def register_click(self, mouse_position, mouse_time):
        mouse_x = mouse_position[0]
        mouse_y = mouse_position[1]
        mouse_r = math.sqrt(
            (mouse_x-self.wheel_c[0]) ** 2 +  
            (mouse_y-self.wheel_c[1]) ** 2)

        click_min_r = self.wheel_r - self.click_radius * CLICK_REGION_FACTOR
        click_max_r = self.wheel_r + self.click_radius * CLICK_REGION_FACTOR
        if mouse_r >= click_min_r and mouse_r <= click_max_r:
            # the click is treated as a valid response
            self.ever_have_response = True
            
            pos_x_scaled = (mouse_x-self.wheel_c[0]) / mouse_r
            pos_y_scaled = (mouse_y-self.wheel_c[1]) / mouse_r
            pos_select_x = self.wheel_c[0] + pos_x_scaled * self.wheel_r
            pos_select_y = self.wheel_c[1] + pos_y_scaled * self.wheel_r
            opp_select_x = self.wheel_c[0] - pos_x_scaled * self.wheel_r
            opp_select_y = self.wheel_c[1] - pos_y_scaled * self.wheel_r
            
            if len(self.clicks_x) == 0:
                # initialize the two points
                self.point_click_selected.autoDraw = True
                self.opposite_point.autoDraw = True
            # update the position of two_points
            self.point_click_selected.setPos(
                (pos_select_x, pos_select_y))
            self.opposite_point.setPos(
                (opp_select_x, opp_select_y))
            # add the selected
            self.clicks_x.append(pos_x_scaled)
            self.clicks_y.append(pos_y_scaled)
            self.clicks_time.append(mouse_time)
            
    def setAutoDraw(self, autoDraw):
        self.wheel.setAutoDraw(autoDraw)
        self.point_click_selected.setAutoDraw(autoDraw)
        self.opposite_point.setAutoDraw(autoDraw)
        
    def compute_response(self):
        # convert response to orientation
        if self.ever_have_response:
            x_resp = self.clicks_x[-1]
            y_resp = self.clicks_y[-1]
            rad = np.arctan2(y_resp, x_resp)
            deg = np.rad2deg(rad)
            deg = deg % 180
            deg = 90 - deg
            deg = deg % 180
            return deg
        else:
            return None
            
class FakeWheelClickRegister:
    def __init__(self, wheel, wheel_pos):
        self.wheel = wheel
        self.wheel_c = wheel_pos[0]
        self.wheel_r = wheel_pos[1]
        self.click_radius = wheel_pos[2]
        
        # if it's valid
        self.is_valid = False
        self.has_response = False
        
        # change the color of some
        
        # initialize
        self.setAutoDraw(True)
   
    def register_click(self, mouse_position, mouse_time):
        mouse_x = mouse_position[0]
        mouse_y = mouse_position[1]
        mouse_r = math.sqrt(
            (mouse_x-self.wheel_c[0]) ** 2 +  
            (mouse_y-self.wheel_c[1]) ** 2)

        click_min_r = self.wheel_r - self.click_radius * CLICK_REGION_FACTOR
        click_max_r = self.wheel_r + self.click_radius * CLICK_REGION_FACTOR
        if mouse_r >= click_min_r and mouse_r <= click_max_r:
            # the click is within the valid region
            self.has_response = True
            
    def setAutoDraw(self, autoDraw):
        self.wheel.setAutoDraw(autoDraw)

# define the function to collect free-form drawing
class CanvasDrawRegister:
    def __init__(self, canvas, strokes, redo_button,
            canvas_pos, redo_button_pos):
        self.canvas = canvas
        self.strokes = strokes
        self.redo_button = redo_button
        self.canvas_c = canvas_pos[0]
        self.canvas_r = canvas_pos[1]
        self.redo_button_pos = redo_button_pos # left, right, top, bottom
        
        # this is a valid response object
        self.is_valid = True
        
        # temporary inner states
        self.all_drawings_x = []
        self.all_drawings_y = []
        self.all_drawings_time = []
        self.drawing_index = 0
        self.initialize_drawing()
        
        # flag for whether there are any response
        self.ever_have_response = False
        
        # if we have strokes not saved yet
        self.has_unsaved_drawing = False
        self.has_unsaved_stroke = False
        
        # start drawing
        self.canvas.setAutoDraw(True)
        self.redo_button.setAutoDraw(True)
    
    def initialize_drawing(self):
        self.stroke_time = []
        self.stroke_data_x = []
        self.stroke_data_y = []
        self.stroke_index = 0
        self.current_vertices = []
        self.current_stroke_x = []
        self.current_stroke_y = []
        self.current_stroke_time = []
        self.has_started_drawing = False
        self.has_stopped_drawing = True
        
        self.last_x = None
        self.last_y = None
        
        # initialize all strokes
        for i in range(len(self.strokes)):
            stroke = self.strokes[i]
            stroke.autoDraw = False
            stroke.vertices = [(0, 0)]
        
    def register_mouse(self, mouse_position, mouse_time):
        has_redo = False # sent back to timer
        stroke = self.strokes[self.stroke_index]
        if self.stroke_index < len(self.strokes):
            # only process the first max_strokes strokes
            # Get the current mouse position and state
            # check if it's within the region to draw
            mouse_x = mouse_position[0]
            mouse_y = mouse_position[1]
            mouse_r = mouse_r = math.sqrt(
                (mouse_x-self.canvas_c[0]) ** 2 +  
                (mouse_y-self.canvas_c[1]) ** 2)
        
            if mouse_r < self.canvas_r:
                # the click is treated as a valid response
                self.ever_have_response = True
                
                # if within the drawing region
                if not self.has_started_drawing:
                    # initialize a stroke
                    stroke.autoDraw = True
                    self.current_vertices = []
                    self.current_stroke_x = []
                    self.current_stroke_y = []
                    self.current_stroke_time = []
                    self.has_started_drawing = True
                    self.has_stopped_drawing = False
                
                if self.last_x == None or abs(self.last_x - mouse_x) > 0.01 or abs(self.last_y - mouse_y) > 0.01:
                    # Update 
                    pos_x_scaled = (mouse_x-self.canvas_c[0]) / self.canvas_r
                    pos_y_scaled = (mouse_y-self.canvas_c[1]) / self.canvas_r
                    self.current_vertices.append(mouse_position)
                    self.current_stroke_x.append(pos_x_scaled)
                    self.current_stroke_y.append(pos_y_scaled)
                    self.current_stroke_time.append(mouse_time)
                    stroke.vertices = self.current_vertices
                
                    # flag that there are unsaved drawings
                    self.has_unsaved_stroke = True
                    self.has_unsaved_drawing = True
                    
                    self.last_x = mouse_x
                    self.last_y = mouse_y
            else:
                # if out of the region
                if self.has_started_drawing:
                    # if have already start drawing, stop
                    self.deregister_mouse(mouse_position, mouse_time)
                    self.has_stopped_drawing = False
                else:
                    # if over the buttons and click and has unsaved
                    if self.has_stopped_drawing:
                        on_redo = (
                            (mouse_x > self.redo_button_pos[0]) 
                            & (mouse_x < self.redo_button_pos[1])) & (
                            (mouse_y > self.redo_button_pos[3]) 
                            & (mouse_y < self.redo_button_pos[2]))
                        
                        if on_redo and self.has_unsaved_drawing:
                            # only allow redo if it does not exceed the limit:
                            if self.drawing_index < MAX_DRAWINGS:
                                # save the drawings
                                self.save_drawing()
                                # clear the canvas
                                self.initialize_drawing()
                                # later signal to the timer
                                has_redo = True
                            
        return has_redo
    
    def deregister_mouse(self, mouse_position, mouse_time):
        self.has_stopped_drawing = True
        if self.has_started_drawing:
            self.save_stroke()
            self.has_started_drawing = False
            
    def save_stroke(self):
        # only save valid stroke
        if len(self.current_stroke_time) > 0:
            self.stroke_time.append(self.current_stroke_time)
            self.stroke_data_x.append(self.current_stroke_x)
            self.stroke_data_y.append(self.current_stroke_y)
            self.stroke_index += 1
        self.has_unsaved_stroke = False
            
    def save_drawing(self):
        # only save valid drawing
        if len(self.stroke_time) > 0:
            self.all_drawings_time.append(self.stroke_time)
            self.all_drawings_x.append(self.stroke_data_x)
            self.all_drawings_y.append(self.stroke_data_y)
            self.drawing_index += 1
        
        # mark that everything has veen saved
        self.has_unsaved_drawing = False
        
    def setAutoDraw(self, autoDraw):
        self.canvas.autoDraw = autoDraw
        for i in range(len(self.strokes)):
            self.strokes[i].autoDraw = autoDraw
        self.redo_button.setAutoDraw(autoDraw)
        
    def compute_response(self):
        # convert response to orientation
        if len(self.all_drawings_time) > 0:
            x_to_analyze = self.all_drawings_x[-1]
            y_to_analyze = self.all_drawings_y[-1]
            
            # the easiest way: start to end
            x_resp = x_to_analyze[-1][-1] - x_to_analyze[0][0]
            y_resp = y_to_analyze[-1][-1] - y_to_analyze[0][0]
            
            # compute all the rest
            rad = np.arctan2(y_resp, x_resp)
            deg = np.rad2deg(rad)
            deg = deg % 180
            deg = 90 - deg
            deg = deg % 180
            return deg
        else:
            return None
            
# define the function to collect free-form drawing
class FakeCanvasDrawRegister:
    def __init__(self, canvas, redo_button, canvas_pos, redo_button_pos):
        self.canvas = canvas
        self.redo_button = redo_button
        self.canvas_c = canvas_pos[0]
        self.canvas_r = canvas_pos[1]
        self.redo_button_pos = redo_button_pos # left, right, top, bottom
        
        # variable to keep track if there are unnecessary drawing
        self.is_valid = False
        self.has_response = False
        
        # initialize
        self.canvas.setAutoDraw(True)
        self.redo_button.setAutoDraw(False)
        
    def register_mouse(self, mouse_position, mouse_time):
        # register clicks as unvalid
        mouse_x = mouse_position[0]
        mouse_y = mouse_position[1]
        mouse_r = mouse_r = math.sqrt(
            (mouse_x-self.canvas_c[0]) ** 2 +  
            (mouse_y-self.canvas_c[1]) ** 2)
        
        if mouse_r < self.canvas_r:
            # the click is within drawing
            self.has_response = True
            
        return False # place holder
    
    def deregister_mouse(self, mouse_position, mouse_time):
        pass
            
    def setAutoDraw(self, autoDraw):
        self.canvas.autoDraw = autoDraw
        self.redo_button.setAutoDraw(autoDraw)
        

def funcDrawAdjustResponse(window, loc, draw_radius, click_radius, is_valid):
    # set the drawing region
    ## for the wheel
    draw_area_x = loc[0]
    draw_area_y = loc[1]
    draw_area_bottom_y = draw_area_y - draw_radius
    
    wheel_pos = [
        (loc[0], loc[1]),
        draw_radius, click_radius]
    wheel_color = 'black'
    if is_valid:
        wheel_color = 'white'
    wheel = createPatch(
        window, pos=(draw_area_x, draw_area_y), radius=draw_radius,
        fill_color='gray', line_color=wheel_color)
    
    response_wheel_obj = None
    if is_valid:
        # create the point to represent orientation
        # initialize the two points
        point_click_selected = createPatch(
            window, pos=(draw_area_x, draw_area_y), radius=click_radius,
            fill_color='black', line_color='black')
        opposite_point = createPatch(
            window, pos=(draw_area_x, draw_area_y), radius=click_radius,
            fill_color='black', line_color='black')
    
        # register mouse click function
        response_wheel_obj = WheelClickRegister(
            wheel, pos_point=point_click_selected, 
            opp_point=opposite_point, wheel_pos=wheel_pos)
    else:
        response_wheel_obj = FakeWheelClickRegister(
            wheel, wheel_pos=wheel_pos)
    return response_wheel_obj

def funcDrawDrawResponse(window, loc, draw_radius, is_valid):
    # set the drawing region
    draw_area_x = loc[0]
    draw_area_y = loc[1]
    draw_area_bottom_y = draw_area_y - draw_radius
    # first define the canvas
    canvas_pos = [(loc[0], loc[1]), draw_radius]
    canvas = None
    if is_valid:
        canvas = createEnabledDrawPatch(
            window, pos=(draw_area_x, draw_area_y), radius=draw_radius)
    else:
        canvas = createDisabledDrawPatch(
            window, pos=(draw_area_x, draw_area_y), radius=draw_radius)
    canvas.setAutoDraw(True)
        
    # define the confirm and redo button
    button_width = draw_radius / 1.8 * 1.0
    button_height = draw_radius / 1.8 * 0.5
    button_x = draw_area_x
    button_y = draw_area_bottom_y - button_height / 2 - 0.01
    
    # redo button
    redo_button_pos = [
        button_x - button_width / 2, # left
        button_x + button_width / 2, # right
        button_y + button_height / 2, # top
        button_y - button_height / 2 # bottom
    ]
    redo_button = funcCreateButton(
        window, 'redo', 
        pos=(button_x, button_y),
        size=(button_width, button_height))
        
    response_canvas_obj = None
    if is_valid:
        # this is a real drawing object
        # then define the strokes (on top of background)
        strokes = []
        for i in range(MAX_STROKES):
            new_stroke = visual.ShapeStim(
                window, vertices=[(0, 0)], 
                fillColor=None, lineWidth=5, 
                lineColor='black', closeShape=False)
            new_stroke.autoDraw = False
            strokes.append(new_stroke)
        
        response_canvas_obj = CanvasDrawRegister(
            canvas, strokes, redo_button,
            canvas_pos, redo_button_pos)
    else:
        response_canvas_obj = FakeCanvasDrawRegister(
            canvas, redo_button,
            canvas_pos, redo_button_pos)
        
    return response_canvas_obj

# define the function to display cue
class SingleCueObject:
    def __init__(self, window, loc, cue_code, radius, cue_color=None):
        cue_r = radius * 0.3
        cue_type = int(cue_code)
        cue_ori = 180 * random.random()
        self.cue_object = self.load_report_cue(
            window, loc, (cue_r, cue_r), cue_ori, cue_type, cue_color)
            
        self.setAutoDraw(True)
            
    def load_report_cue(self, window, loc, cue_size, cue_ori, cue_type, cue_color):
        cue_img_path = None
        color_info = '' if cue_color is None else f'_{cue_color}'
        if cue_type == 99:
            cue_img_path = f"resources/dot{color_info}.png"
        elif cue_type==1:
            cue_img_path = f"resources/star{color_info}.png"
        elif cue_type==0:
            cue_img_path = f"resources/empty_star{color_info}.png"
        else:
            raise ValueError(f'Unknow cue type {cue_type}')
        cue_obj = visual.ImageStim(
            win=window, image=cue_img_path,
            ori=cue_ori, size=cue_size, pos=loc)
        return cue_obj
            
    def setAutoDraw(self, auto_draw):
        self.cue_object.setAutoDraw(auto_draw)
            
    
class CueObject:
    def __init__(self, window, loc, cue_codes, radius):
        self.cue_objects = []
        cue_r = radius * 0.3
        
        # create the center bar
        bar_object = self.load_report_cue(
            window, loc, (cue_r*1.2, cue_r*0.4), 0, 0)
        self.cue_objects.append(bar_object)
        
        # create the cues
        cue_locs = [
            (loc[0]-cue_r, loc[1]),
            (loc[0]+cue_r, loc[1])]
        for i in range(2):
            cue_object = SingleCueObject(
                window, cue_locs[i], cue_codes[i], radius)
            self.cue_objects.append(cue_object)
                
        self.setAutoDraw(True)

    def load_report_cue(self, window, loc, cue_size, cue_ori, cue_type):
        cue_img_path = None
        if cue_type==0:
            cue_img_path = "resources/bar.png"
        else:
            raise ValueError(f'Unknow cue type {cue_type}')
        cue_obj = visual.ImageStim(
            win=window, image=cue_img_path,
            ori=cue_ori, size=cue_size, pos=loc)
        return cue_obj
        
    def setAutoDraw(self, auto_draw):
        for obj in self.cue_objects:
            obj.setAutoDraw(auto_draw)

# define the function to display implicit cue
class ImplicitCueObject:
    def __init__(self, window, loc, radius, n_split, regions, offset):
        self.n_split = n_split
        self.cue_objects = []
        cue_r = radius * 0.5
        
        # create the background
        back_object = self.load_report_cue(
            window, loc, [cue_r, cue_r], 0, offset)
        self.cue_objects.append(back_object)
        
        # highlight the region to report
        for i in range(len(regions)):
            region_id = regions[i]
            cue_ori = 180 / self.n_split * region_id
            # cue_width = abs(cue_r * math.sin(math.radians(180 / self.n_split + offset)))
            cue_object = self.load_report_cue(
                window, loc, [cue_r, cue_r], 1, cue_ori + offset)
            self.cue_objects.append(cue_object)
                
        self.setAutoDraw(True)

    def load_report_cue(self, window, loc, cue_size, cue_type, cue_ori=0):
        cue_img_path = None
        if cue_type==0:
            cue_img_path = f"resources/implicit_cue_back_{self.n_split}.png"
        elif cue_type==1:
            cue_img_path = f"resources/implicit_cue_front_{self.n_split}.png"
        cue_obj = visual.ImageStim(
            win=window, image=cue_img_path,
            ori=cue_ori, size=cue_size, pos=loc)
        return cue_obj
        
    def setAutoDraw(self, auto_draw):
        for obj in self.cue_objects:
            obj.setAutoDraw(auto_draw)

def funcDrawArrow(window, radius, start_loc, end_loc):
    # compute orientation
    x_dir = end_loc[0] - start_loc[0]
    y_dir = end_loc[1] - start_loc[1]
    arrow_rad = np.arctan2(-y_dir, x_dir)
    arrow_ori = np.rad2deg(arrow_rad)
    arrow_pos = [
        (end_loc[0] + start_loc[0])/2, 
        (end_loc[1] + start_loc[1])/2]
    cue_size = radius * 0.3
    arrow_img_path = "resources/arrow.png"
    arrow = visual.ImageStim(
        win=window, image=arrow_img_path,
        ori=arrow_ori, pos=arrow_pos,
        size=(cue_size, cue_size))
    arrow.setAutoDraw(True)
    return arrow

def funcCheckNoPending(objs):
    no_unanswered = True
    for i in range(len(objs)):
        obj = objs[i]
        if obj.is_valid:
            no_unanswered = no_unanswered and obj.ever_have_response
    return no_unanswered

# other components for feedback and training
class ClickAnswerObject:
    def __init__(self, window, ori, loc, radius, click_radius):
        ## for the wheel
        self.components = []
        self.wheel_r = radius
        self.wheel_c = [loc[0], loc[1]]
        self.click_radius = click_radius
        self.wheel = createPatch(
            window, pos=self.wheel_c, radius=self.wheel_r,
            fill_color='gray', line_color='white')
        self.components.append(self.wheel)
        
        ## create the point to represent orientation
        if ori is not None:
            converted_pos = self.convert_ori_to_pos(ori)
            ans_point_pos = converted_pos[0]
            opp_point_pos = converted_pos[1]
            self.point_click_selected = createPatch(
                window, pos=ans_point_pos, 
                radius=self.click_radius,
                fill_color='black', line_color='black')
            self.opposite_point = createPatch(
                window, pos=opp_point_pos, 
                radius=self.click_radius,
                fill_color='black', line_color='black')
            self.components.append(self.point_click_selected)
            self.components.append(self.opposite_point)
    
        # make all objects visible
        self.setAutoDraw(True)

    def setAutoDraw(self, autoDraw):
        n_comp = len(self.components)
        for i in range(n_comp):
            comp = self.components[i]
            comp.setAutoDraw(autoDraw)
        
    def convert_ori_to_pos(self, ori):
        # convert orientation to x,y
        deg = 90 - ori
        deg = deg % 180
        rad = np.deg2rad(deg)
        
        # adjust point position
        x_ans = self.wheel_c[0] + np.cos(rad) * self.wheel_r
        y_ans = self.wheel_c[1] + np.sin(rad) * self.wheel_r
        x_opp = self.wheel_c[0] - np.cos(rad) * self.wheel_r
        y_opp = self.wheel_c[1] - np.sin(rad) * self.wheel_r
        return (x_ans, y_ans), (x_opp, y_opp)

        
class DrawAnswerObject:
    def __init__(self, window, response, is_ori, loc, radius):
        self.components = []
        self.strokes = []
        
        # set the drawing region
        self.canvas_r = radius
        self.canvas_c = (loc[0], loc[1])
        self.canvas = canvas = createEnabledDrawPatch(
            window, pos=self.canvas_c, radius=self.canvas_r)
        self.components.append(self.canvas)

        # then set the strokes
        if response is not None:
            strokes = response
            if is_ori:
                # the 'answer' provided is orientation
                strokes = self.convert_ori_to_drawing(response)
            x_strokes = strokes[0]
            y_strokes = strokes[1]
            n_strokes = len(x_strokes)
            for i in range(n_strokes):
                xs = x_strokes[i]
                ys = y_strokes[i]
                n_vertices = len(xs)
                vertices = []
                for j in range(n_vertices):
                    # convert the x,y to new coords
                    vx = self.canvas_c[0] + xs[j] * self.canvas_r
                    vy = self.canvas_c[1] + ys[j] * self.canvas_r
                    vertices.append((vx, vy))
                new_stroke = visual.ShapeStim(
                    win=window, vertices=vertices, 
                    fillColor=None, lineWidth=5, 
                    lineColor='black', closeShape=False)
                self.strokes.append(new_stroke)
                self.components.append(new_stroke)
            
        # make all objects visible
        self.setAutoDraw(True)

    def setAutoDraw(self, autoDraw):
        n_comp = len(self.components)
        for i in range(n_comp):
            comp = self.components[i]
            comp.setAutoDraw(autoDraw)
            
    def convert_ori_to_drawing(self, ori):
        # convert orientation to x,y
        deg = 90 - ori
        deg = deg % 180
        rad = np.deg2rad(deg)
        
        # adjust point position
        r_scale =  0.7
        x1 = np.cos(rad) * r_scale
        x2 = - np.cos(rad) * r_scale
        y1 = np.sin(rad) * r_scale 
        y2 = - np.sin(rad) * r_scale 
        drawing = [[[x1, x2]], [[y1, y2]]]
        return drawing
        
class FeedbackText:
    def __init__(self, window, text, loc, radius):
        self.text = text
        self.text_object = visual.TextStim(
            win=window, text=self.text, 
            pos=(loc[0], loc[1]),
            height=radius * 0.2,
            color='black', colorSpace='rgb')
        # make components visible
        self.setAutoDraw(True)
    
    def setAutoDraw(self, autoDraw):
        self.text_object.setAutoDraw(autoDraw)
        
class TrainingCue:
    def __init__(self, window, text, target_loc, display_r):
        # get the location
        vec_x = (1 - 2 * (target_loc[0] < 0)) * display_r / math.sqrt(2)
        vec_y = (1 - 2 * (target_loc[1] < 0)) * display_r / math.sqrt(2)
        x_offset_scale = 1
        text_loc = [
            target_loc[0] + vec_x * (1.4+x_offset_scale),
            target_loc[1] + vec_y * 1.4]
        
        # create the text
        self.text = text
        self.text_object = visual.TextStim(
            win=window, text=self.text, 
            pos=text_loc,
            height=display_r * 0.2,
            color='white', colorSpace='rgb')
        # create the arrow
        arrow_start_loc = [
            target_loc[0] + vec_x * (1.2+x_offset_scale),
            target_loc[1] + vec_y * 1.2]
        arrow_end_loc = [
            target_loc[0] + vec_x * (1.02+x_offset_scale),
            target_loc[1] + vec_y * 1.02]
        self.arrow = funcDrawArrow(
            window, display_r * 0.5, 
            arrow_start_loc, arrow_end_loc)
            
        # auto draw --> true
        self.setAutoDraw(True)
        
    def setAutoDraw(self, autoDraw):
        self.text_object.setAutoDraw(autoDraw)
        self.arrow.setAutoDraw(autoDraw)
    
class NoisePatch:
    def __init__(self, window, loc, display_r):
        noise_img_path = "resources/noise.png"
        noise_ori = 180 * random.random()
        self.obj = visual.ImageStim(
            win=window, image=noise_img_path,
            ori=noise_ori, size=2*display_r, pos=loc) 
        
        self.setAutoDraw(True)
        
    def setAutoDraw(self, autoDraw):
        self.obj.setAutoDraw(autoDraw)
        
class DefaultFixation:
    def __init__(self, window):
        self.fixation = visual.ShapeStim(
            win=window, vertices='cross',
            size=(0.06, 0.06), lineWidth=1,
            lineColor='white', fillColor='white')
        
        self.setAutoDraw(True)
        
    def setAutoDraw(self, autoDraw):
        self.fixation.setAutoDraw(autoDraw)

# Run 'Before Experiment' code from codePrepareQuiz
class QuizDiagram:
    def __init__(self, window, loc, width, height, diagram_type, params):
        self.obj_display = []
        loc_x = loc[0]
        loc_y = loc[1]
        stim_r = height * 0.36
        if diagram_type == 'stims':
            stims = params['stims']
            n_stims = len(stims)
            unit_width = width / n_stims
            for i in range(n_stims):
                # compute the location of this stim
                stim_ori = stims[i]
                stim_x = loc_x + (2*i+1-n_stims) * unit_width / 2
                stim_y = loc_y
                new_stim_obj = funcDrawStim(
                    window, stim_ori, [stim_x, stim_y], stim_r)
                self.obj_display.append(new_stim_obj)
        
        elif diagram_type == 'answers':
            ans_mode = params['mode']
            answers = params['answers']
            n_answers = len(answers)
            unit_width = width / n_answers
            for i in range(n_answers):
                # compute the location of this stim
                ans_ori = answers[i]
                ans_x = loc_x + (2*i+1-n_answers) * unit_width / 2
                ans_y = loc_y
                ans_r = stim_r
                
                # create the correct answer
                new_ans_obj = None
                if ans_mode == 'click':
                    new_ans_obj = ClickAnswerObject(
                        window, ans_ori, 
                        [ans_x, ans_y], ans_r, 0.01)
                elif ans_mode == 'draw':
                    new_ans_obj = DrawAnswerObject(
                        window, ans_ori, True, 
                        [ans_x, ans_y], ans_r)
                self.obj_display.append(new_ans_obj)
                
        elif diagram_type == 'stims_with_cue':
            stims = params['stims']
            n_stims = len(stims)
            unit_width = width / (n_stims+1)
            
            stim_locs = []
            if n_stims == 2:
                stim_locs = [
                    [loc_x-unit_width, loc_y],
                    [loc_x+unit_width, loc_y]]
            else:
                raise NotImplementedError()
            
            for i in range(n_stims):
                # draw the stims
                stim_ori = stims[i]
                stim_loc = stim_locs[i]
                new_stim_obj = funcDrawStim(
                    window, stim_ori, stim_loc, stim_r)
                self.obj_display.append(new_stim_obj)
            
                # draw the cue
                cue_type = params['cue_type']
                color_code = params["color_code"][i]
                cue_obj = None
                if cue_type == "explicit":
                    to_report_code = params["to_report_code"][i]
                    cue_obj = SingleCueObject(
                        window, stim_loc, to_report_code, stim_r, color_code)
                self.obj_display.append(cue_obj)
            
    
    def setAutoDraw(self, auto_draw):
        for i in range(len(self.obj_display)):
            self.obj_display[i].setAutoDraw(auto_draw)

class singleSelectionButton:
    def __init__(self, window, selection_id, text, loc):
        self.selection_id = selection_id
        self.text = text
        
        # predefine
        height = 0.04
        
        # create the button
        self.check_box = visual.Rect(
            window, width=height, height=height,
            pos=loc, lineColor='black', fillColor='white')
        
        # create the text
        self.text_obj = visual.TextStim(
            window, text=text, pos=(loc[0]+height * 1.5, loc[1]),
            anchorHoriz='left', alignText='left', height=height)
            
        # make everything visible
        self.setAutoDraw(True)
        
    def contains(self, pos):
        return self.check_box.contains(pos)
        
    def to_select(self):
        self.check_box.setFillColor('black')
        
    def to_unselect(self):
        self.check_box.setFillColor('white')
        
    def setAutoDraw(self, auto_draw):
        self.check_box.setAutoDraw(auto_draw)
        self.text_obj.setAutoDraw(auto_draw)
        
class QuizSelectionObjects:
    def __init__(self, window, texts, locs):
        self.choice_objs = []
        self.selected = None
        n_choices = len(texts)
        self.choices = []
        for i in range(n_choices):
            one_choice = singleSelectionButton(window, i, texts[i], locs[i])
            self.choice_objs.append(one_choice)
    
    def check_click(self, pos):
        for i in range(len(self.choice_objs)):
            to_check = self.choice_objs[i]
            if to_check.contains(pos):
                # deselect previous one
                if self.selected is not None:
                    self.choice_objs[self.selected].to_unselect()
                self.selected = i
                to_check.to_select()
                return True # something got selected
        return False # nothing got selected
                
    def get_selected_text(self):
        if self.selected is not None:
            return self.choice_objs[self.selected].text
        else:
            return None
                
    def setAutoDraw(self, auto_draw):
        for i in range(len(self.choice_objs)):
            to_check = self.choice_objs[i]
            to_check.setAutoDraw(auto_draw)

class quizManagementObj:
    def __init__(self):
        self.obj_display = []
        self.question_diagram_objs = []
        self.choice_diagram_objs = []
        self.choices = None
        self.people_answer = None
        self.correct_answer_id = None
        self.correct_answer = None
        self.continue_button = None
        self.question_text = ''
        
        # layout
        self.quiz_title_loc = [0, 0.35] 
        self.quiz_title_width = 1.3
        self.question_diagram_loc = [0, 0]
        self.question_diagram_size = [0, 0]
        self.choice_diagram_loc = [0, 0]
        self.choice_diagram_size = [0, 0]
        self.question_prompt_loc = [0, 0]
        self.question_prompt_width = 0
        self.choices_loc = [0, 0]
        self.continue_button_loc = []
        self.answer_width = 0
        
    def initiate_quiz(self, window, question_type, params):
        if question_type == 'response_instruction':
            # given a stim, what is the correct response?
            # create the title
            title_text = "[Quiz] What is the correct response?"
            self.create_quiz_title(window, title_text)
            
            # create stimulus
            stims = [random.uniform(0, 180)]
            # create question diagram
            self.question_diagram_loc = [-0.3, 0.1]
            self.question_diagram_size = [0.36, 0.36]
            self.create_question_diagram(window, 
                'stims', {'stims': stims})
                
            # create choice diagram
            response_mode = params['mode']
            correct_ans = stims[0]
            wrong_ans = (correct_ans + 90) % 180
            left_correct = random.random() > 0.5
            choice_values = None
            if left_correct:
                self.correct_answer_id = 0
                choice_values = [correct_ans, wrong_ans]
            else:
                self.correct_answer_id = 1
                choice_values = [wrong_ans, correct_ans]
            self.choice_diagram_loc = [-0.3, -0.2]
            self.choice_diagram_size = [0.6, 0.36]
            self.create_choice_diagram(window, 
                'answers', {'answers': choice_values, 
                'mode': response_mode})
                
            # create question prompt
            self.question_prompt_loc = [0.2, 0.1]
            self.question_prompt_width = 0.7
            self.answer_width = 0.6
            self.question_text = "For the grating patch on the top, \n" +\
                "which of the response at the bottom is correct?"
            question_text_display = self.question_text + "\n Choose your answer and click 'continue' to check."
            self.question_prompt = visual.TextStim(
                window, text=question_text_display, height=0.04,
                pos=self.question_prompt_loc,
                wrapWidth=self.question_prompt_width)
            self.question_prompt.setAutoDraw(True)
            self.obj_display.append(self.question_prompt)
            
            # create choices
            self.choices_loc = [[0.1, -0.1], [0.1, -0.2]]
            choice_texts = ['the left one', 'the right one']
            self.correct_answer = choice_texts[self.correct_answer_id]
            self.choices = QuizSelectionObjects(
                window, choice_texts, self.choices_loc)
            self.obj_display.append(self.choices)
                
            # set continue button position
            self.continue_button_loc = [0.2, -0.35]
            self.continue_button = funcCreateButton(
                window, 'continue', self.continue_button_loc, [0.24, 0.1])
            self.obj_display.append(self.continue_button)
        elif question_type == 'cue':
            # how to respond to a implicit cue?
            ## create the title
            cue_type = params["cue_type"]
            title_text = "[Quiz] What to report when there is an " + cue_type + " cue?"
            self.create_quiz_title(window, title_text)
            
            ## set up question diagram
            ## create stimulus
            self.question_diagram_loc = [0, 0.15]
            self.question_diagram_size = [1, 0.5]
            sampler = params['stim_sampler']
            stim_region_ids, stims = sampler.sample_for_trial(
                allow_same_region=False)
            diagram_params = {
                "stims": stims, 
                "cue_type": cue_type}
                
            # decide the cue to report
            # report_type_id = random.randint(0, 2)
            report_type_id = sampler.n_trials % 3
            self.correct_answer_id = report_type_id
            if cue_type == "explicit":
                to_report = []
                if report_type_id == 0:
                    to_report = [True, False]
                elif report_type_id == 1:
                    to_report = [False, True]
                else:
                    to_report = [True, True]
                diagram_params["to_report_code"] = to_report
                
            # set color code
            diagram_params["color_code"] = [1, 2] if random.random() < 0.5 else [2, 1]
            
            self.create_question_diagram(
                window, 'stims_with_cue', params=diagram_params)

            # create question prompt
            self.question_prompt_loc = [-0.3, -0.15]
            self.question_prompt_width = 0.7
            self.answer_width = 0.6
            self.question_text = "Given the cue in the center, \n" +\
                "which of the stimuli you need to remember and report?"
            question_text_display = self.question_text + "\n Choose your answer and click 'continue' to check."
            self.question_prompt = visual.TextStim(
                window, text=question_text_display, height=0.04,
                pos=self.question_prompt_loc,
                wrapWidth=self.question_prompt_width)
            self.question_prompt.setAutoDraw(True)
            self.obj_display.append(self.question_prompt)
            
            # create choices
            self.choices_loc = [[0.1, -0.1], [0.1, -0.17], [0.1, -0.24]]
            choice_texts = ['the left one', 'the right one', 'both']
            self.correct_answer = choice_texts[self.correct_answer_id]
            self.choices = QuizSelectionObjects(
                window, choice_texts, self.choices_loc)
            self.obj_display.append(self.choices)
            
            # set continue button position
            self.continue_button_loc = [0, -0.35]
            self.continue_button = funcCreateButton(
                window, 'continue', self.continue_button_loc, [0.24, 0.1])
            self.obj_display.append(self.continue_button)

        else:
            pass
            
    def create_quiz_title(self, window, title_text):
        self.title = visual.TextStim(
            window, text=title_text, height=0.06,
            pos=self.quiz_title_loc,
            wrapWidth=self.quiz_title_width)
        self.title.setAutoDraw(True)
        self.obj_display.append(self.title)
            
    def create_question_diagram(self, window, diagram_type, params):
        self.question_diagram_objs = QuizDiagram(
            window, self.question_diagram_loc, 
            self.question_diagram_size[0],
            self.question_diagram_size[1], 
            diagram_type, params)
        self.obj_display.append(self.question_diagram_objs)
        
    def create_choice_diagram(self, window, diagram_type, params):
        self.choice_diagram_objs = QuizDiagram(
            window, self.choice_diagram_loc, 
            self.choice_diagram_size[0],
            self.choice_diagram_size[1], 
            diagram_type, params)
        self.obj_display.append(self.choice_diagram_objs)
            
    def collect_quiz_answer(self):
        self.people_answer = self.choices.get_selected_text()
        return self.people_answer is not None
        
    def create_quiz_feedback_obj(self, window):
        # remove choices
        self.choices.setAutoDraw(False)
        self.choices = []
        
        # change question
        self.question_prompt.setText(self.question_text)
        
        # display answer
        answer_msg = ''
        if self.people_answer == self.correct_answer:
            # the person get correct answer
            answer_msg += "Your are right! "
            answer_msg += "The correct answer is '" + self.correct_answer + "'\n"
        else:
            answer_msg += "Thanks for your input! However, "
            answer_msg += "the correct answer is '" + self.correct_answer + "'\n"
        answer_msg_width = self.answer_width
        answer_msg_x = self.choices_loc[0][0] + 0.2
        answer_msg_y = (self.choices_loc[0][1] + self.choices_loc[-1][1]) / 2
        self.answer_msg_obj = visual.TextStim(
            window, text=answer_msg, height=0.04,
            pos=[answer_msg_x, answer_msg_y], 
            wrapWidth=answer_msg_width)
        self.answer_msg_obj.setAutoDraw(True)
        self.obj_display.append(self.answer_msg_obj)
        
    def setAutoDraw(self, auto_draw):
        n_objs = len(self.obj_display)
        for i in range(n_objs):
            obj = self.obj_display[i]
            obj.setAutoDraw(auto_draw)

# Run 'Before Experiment' code from codePrepareExperiment
VALID_N_REGIONS = [4,]

class StimSampler:
    def __init__(self, n_sample_regions):
        if n_sample_regions not in VALID_N_REGIONS:
            raise ValueError(
                f'{VALID_N_REGIONS} is not a valid number of sample regions ')
        self.n_sample_regions = n_sample_regions
        self.count = 0
        
    def sample_sample_region(self, sample_strategy):
        result_regions = None
        if self.n_sample_regions == 4:
            if sample_strategy == 'single':
                result_regions = self.sample_single_sample_region()
            elif sample_strategy == 'adjacent':
                result_regions = self.sample_adjacent_sample_region()
            elif sample_strategy == 'orthogonal':
                result_regions = self.sample_orthogonal_sample_region()
            elif sample_strategy == 'all':
                result_regions = self.sample_all_sample_region()
            elif sample_strategy == 'same':
                result_regions = self.sample_same_sample_region()
            elif sample_strategy == 'opp':
                result_regions = self.sample_opp_sample_region()
            elif sample_strategy == 'mixed':
                result_regions = self.sample_mixed_sample_region()
            else:
                raise ValueError(
                    f'{sample_strategy} is not valid for '
                    f'for {self.n_sample_regions} regions')
        else:
            raise NotImplementedError(f'{sample_strategy} not implemented')
        return result_regions
        
    def sample_single_sample_region(self):
        region = random.randint(0, self.n_sample_regions - 1)
        return [region,]
    
    def sample_adjacent_sample_region(self):
        region_1 = random.randint(0, self.n_sample_regions - 1)
        region_2 = (region_1 + 1) % self.n_sample_regions
        return [region_1, region_2]
        
    def sample_orthogonal_sample_region(self):
        region_1 = random.randint(0, self.n_sample_regions - 1)
        region_2 = (region_1 + self.n_sample_regions // 2) % self.n_sample_regions
        return [region_1, region_2]
        
    def sample_all_sample_region(self):
        regions = []
        for i in range(self.n_sample_regions):
            regions.append(i)
        return regions
        
    def sample_same_sample_region(self):
        big_region_id = random.randint(0, 1)
        region_1 = big_region_id * 2 + random.randint(0, 1)
        region_2 = big_region_id * 2 + random.randint(0, 1)
        return [region_1, region_2]
        
    def sample_opp_sample_region(self):
        big_region_id = random.randint(0, 1)
        region_1 = big_region_id * 2 + random.randint(0, 1)
        region_2 = (1 - big_region_id) * 2 + random.randint(0, 1)
        return [region_1, region_2]
        
    def sample_mixed_sample_region(self):
        region_1 = random.randint(0, 3)
        region_2 = random.randint(0, 3)
        return [region_1, region_2]
        
    def sample_from_given_regions(self, regions, allow_same_region):
        # sample the regions to sample from
        stim_region_ids_selected = []
        if allow_same_region:
            for i in range(2):
                stim_region_id = random.choice(regions)
                stim_region_ids_selected.append(stim_region_id)
        else:
            if len(regions) < 2:
                raise ValueError('Unable to sample without duplicate twice')
            stim_region_ids_selected = random.sample(regions, 2)
        
        stims_selected = []
        for i in range(2):
            stim_region_id = stim_region_ids_selected[i]
            stim = random.uniform(
                stim_region_id * 180 / self.n_sample_regions,
                (stim_region_id+1) * 180 / self.n_sample_regions)
            stims_selected.append(stim)
            
        return stim_region_ids_selected, stims_selected


class StimSamplingScheduler:
    def __init__(self, n_sample_regions, stage_strategies):
        self.n_sample_regions = n_sample_regions
        self.stage_strategies = stage_strategies
        self.n_stages = len(self.stage_strategies)
        self.sampler = StimSampler(n_sample_regions)
        
        # keep a record
        self.strategy = None
        self.regions = None
        self.stage_idx = -1
        self.block_idx = -1
        self.block_within_stage_idx = -1
        self.n_trials = 0
    
    def initialize_stage(self):
        self.stage_idx += 1
        self.block_within_stage_idx = -1
        self.n_trials = 0
        self.strategy = self.stage_strategies[self.stage_idx]
        
    def initialize_block(self):
        self.block_idx += 1
        self.block_within_stage_idx += 1
        self.n_trials = 0
        self.regions = self.sampler.sample_sample_region(self.strategy)
        
    def sample_for_trial(self, allow_same_region):
        self.n_trials += 1
        selected_regions, selected_stims = self.sampler.sample_from_given_regions(
            self.regions, allow_same_region)
        # change regions every trial
        self.regions = self.sampler.sample_sample_region(self.strategy)
        return selected_regions, selected_stims
        
# Run 'Before Experiment' code from codePrepareAllPractice
# helper functions for creating feedbacks
def DrawStimsDiagram(oris, locs, cue_codes, colors, radius):
    objs = []
    for i in range(2):
        # draw the stimulus
        ori = oris[i]
        stim_loc = locs[i]
        stim_obj = funcDrawStim(win, ori, stim_loc, radius)
        objs.append(stim_obj)
        
        # the title
        stim_title = FeedbackText(
            win, f'stimulus {i+1}', 
            [stim_loc[0], stim_loc[1]+radius*1.2], radius)
        objs.append(stim_title)
        
        # also show the color code
        color_code, cue_code = colors[i], cue_codes[i]
        color_cue = SingleCueObject(
            win, stim_loc, cue_code, radius*0.6, color_code)
        objs.append(color_cue)
        
    # show the arrow
    stim_arrow = funcDrawArrow(win, radius, locs[0], locs[1])
    objs.append(stim_arrow)
    
    return objs
    

def DrawResponseCanvasDiagram(responses, locs, to_report_codes, colors, radius, h, w, mode, is_gt):
    objs = []
    
    # create the canvas
    canvas_loc = [
        (locs[0][0]+locs[1][0])/2,
        (locs[0][1]+locs[1][1])/2]
    canvas = visual.Rect(
        win=win, width=w, height=h, pos=canvas_loc,
        fillColor='darkGray', lineColor='black')
    canvas.setAutoDraw(True)
    objs.append(canvas)
    
    # create the answer objects
    for i in range(2):
        # draw the response:
        resp = responses[i]
        loc = locs[i]
        to_report = to_report_codes[i]
        cue_color = colors[i]

        if to_report:
            if mode == 'click':
                resp_obj = ClickAnswerObject(
                    win, resp, loc, radius, 0.01)
            elif mode == 'draw':
                resp_obj= DrawAnswerObject(
                    win, resp, is_gt, loc, radius)
            else:
                raise ValueError(f'Unknown mode {mode}')
            objs.append(resp_obj)
            
            # add the color code
            color_cue = SingleCueObject(
                win, loc, 99, radius*0.6, cue_color)
            objs.append(color_cue)
        else:
            if mode == 'click':
                resp_obj = createPatch(
                    win, pos=loc, radius=radius,
                    fill_color='gray', line_color='black')
            elif mode == 'draw':
                dark_gray = [-0.3, -0.3, -0.3]
                resp_obj= createPatch(
                    win, pos=loc, radius=radius,
                    fill_color=dark_gray, line_color=dark_gray)
            else:
                raise ValueError(f'Unknown mode {mode}')
            resp_obj.setAutoDraw(True)
            objs.append(resp_obj)
    
    # return list of objects
    return objs

def CreateWholePracticeFeedback(
        oris, responses, resp_locs,
        colors, cue_codes, report_codes, mode):
    
    objs_display = []

    # layout settings
    group_xs = [-0.3, 0.3]
    display_ys = [0.14, -0.3]
    display_xs = [-0.15, 0.15]
    stim_xs = [-0.2, 0.2]
    display_r = 0.13
    little_display_r = 0.08
    title_ys = [display_ys[0]+display_r+0.02, display_ys[1]+little_display_r*2.75]
    
    little_canvas_h = 4.5 * little_display_r
    little_canvas_w = 6 * little_display_r
    
    # create stimuli
    locs = [
        [stim_xs[0], display_ys[0]],
        [stim_xs[1], display_ys[0]],
    ]
    stim_objs = DrawStimsDiagram(
        oris, locs, [99, 99], 
        colors, display_r)
    objs_display += stim_objs
    # draw the title for stimuli display
    all_stim_title = FeedbackText(
        win, f'Stimuli', 
        [0, title_ys[0]+0.05], display_r*1.5)
    objs_display.append(all_stim_title)
    
    # create participant response review
    # convert resp loc to offset
    loc_offsets = []
    for i in range(2):
        x_offset = resp_locs[i][0] * little_canvas_w
        y_offset = resp_locs[i][1] * little_canvas_h
        loc_offsets.append([x_offset, y_offset])
        
    # create partcipants' response
    locs = [
        [loc_offsets[0][0]+group_xs[0], loc_offsets[0][1]+display_ys[1]],
        [loc_offsets[1][0]+group_xs[0], loc_offsets[1][1]+display_ys[1]],
    ]
    past_resp_objs = DrawResponseCanvasDiagram(
        responses, locs, report_codes, 
        colors, little_display_r, 
        little_canvas_h, little_canvas_w, mode, False)
    objs_display += past_resp_objs
    # and the title
    all_resp_title = FeedbackText(
        win, f'Your Response', 
        [group_xs[0], title_ys[1]], display_r*1.2)
    objs_display.append(all_resp_title)

    # create correct answers
    locs = [
        [loc_offsets[0][0]+group_xs[1], loc_offsets[0][1]+display_ys[1]],
        [loc_offsets[1][0]+group_xs[1], loc_offsets[1][1]+display_ys[1]],
    ]
    ans_objs = DrawResponseCanvasDiagram(
        oris, locs, report_codes, 
        colors, little_display_r, 
        little_canvas_h, little_canvas_w, mode, True)
    objs_display += ans_objs
    # and the title
    all_ans_title = FeedbackText(
        win, f'Correct Answer', 
        [group_xs[1], title_ys[1]], display_r*1.2)
    objs_display.append(all_ans_title)
    
    return objs_display

# Run 'Before Experiment' code from codeTextInstructionText
text_practice_instruction_precue = "In this type of trials, " \
    + "you don't need to remember all stimuli shown. " \
    + "Instead, you only need to remember the ones marked with a filled star, " \
    + "and you can forget about those with an empty star. " \
    + "During the response phase, " \
    + "report only the stimuli you were instructed to remember."
text_practice_instruction_postcue = "In this type of trial, " \
    + "you always need to remember all the stimuli presented. " \
    + "However, during the response phase, " \
    + "sometimes only one of the two stimuli will be asked about." \
    + "More specifically, you only need to report stimuli whose corresponding " \
    + "report regions are valid."

text_practice_instructions = [
    text_practice_instruction_precue,
    text_practice_instruction_postcue,
]

# Run 'Before Experiment' code from connectTracker
import platform
from PIL import Image  # for preparing the Host backdrop image
from string import ascii_letters
import time

# import eyelink libs
import pylink
from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy


# Run 'Before Experiment' code from codePrepareSingleTrial
class ResponseLocationSampler:
    def __init__(self, xs, ys, allow_diagonal, allow_off_axis, allow_row_align):
        self.xs = [0, ] + xs[:]
        self.ys = [0, ] + ys[:]
        
        # pre-register possible locations
        resp_loc_id_candidates = []
        
        # register locations on axis
        if allow_diagonal:
            for loc_flip in [0, 1]:
                resp_loc_ids = [
                    [1, 1+loc_flip],
                    [2, 2-loc_flip]]
                resp_loc_id_candidates.append(resp_loc_ids)

        # register row- or column-align positions
        fix_pos_ids = [0, 1, 2] if allow_off_axis else [0, ]
        for fix_pos_id in fix_pos_ids:
            resp_loc_ids = [[fix_pos_id, 1], [fix_pos_id, 2]]
            resp_loc_id_candidates.append(resp_loc_ids)
            if allow_row_align:
                resp_loc_ids = [[1, fix_pos_id], [2, fix_pos_id]]
                resp_loc_id_candidates.append(resp_loc_ids)
                
        self.resp_loc_id_candidates = resp_loc_id_candidates
        
    def sample_location(self, allow_swap):
        loc_id_pos = random.randint(0, len(self.resp_loc_id_candidates)-1)
        loc_ids = self.resp_loc_id_candidates[loc_id_pos]
        resp_locs = []
        if allow_swap and random.random() < 0.5:
            resp_locs = [
                [self.xs[loc_ids[1][0]], self.ys[loc_ids[1][1]]],
                [self.xs[loc_ids[0][0]], self.ys[loc_ids[0][1]]]]
        else:
            resp_locs = [
                [self.xs[loc_ids[0][0]], self.ys[loc_ids[0][1]]],
                [self.xs[loc_ids[1][0]], self.ys[loc_ids[1][1]]]]

        return resp_locs
# Run 'Before Experiment' code from codeDelay
def sample_n_from_trange(n, duration, min_space, tmin, tmax):
    result = []
    trange_start = tmin
    for i in range(n):
        trange_end = tmax- (n-i) * (min_space + duration)
        if trange_end <= trange_start:
            # no more points available
            continue
        sample = random.uniform(trange_start, trange_end)
        result.append(sample)
        trange_start = sample + min_space + duration
    return result

# Run 'Before Experiment' code from codeBlockFeedback
def check_block_quality(errors):
    ERR_THRESHOLD = 30 # 20
    ACC_THRESHOLD = 0.5 # 0.55
    MISS_THRESHOLD = 0.2 # 0.15
    
    n_total = len(errors)
    n_hit = 0
    n_miss = 0
    for i in range(n_total):
        err = errors[i]
        if err is None:
            n_miss += 1
        elif abs(err) <= ERR_THRESHOLD:
            n_hit += 1
    pass_hit_check = (n_hit / n_total) >= ACC_THRESHOLD
    pass_miss_check = (n_miss / n_total) <= MISS_THRESHOLD
    return pass_hit_check and pass_miss_check

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'ModeEyeTrackInstr'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'test mode': ["online", "lab"],
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1080]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\siy009\\Desktop\\mode_compare_eyetrack_pilot_25spring\\ModeEyeTrackComplete_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('error')


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('keyPracticeFreeDraw') is None:
        # initialise keyPracticeFreeDraw
        keyPracticeFreeDraw = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='keyPracticeFreeDraw',
        )
    if deviceManager.getDevice('keyRespInput') is None:
        # initialise keyRespInput
        keyRespInput = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='keyRespInput',
        )
    if deviceManager.getDevice('keyBreakTime') is None:
        # initialise keyBreakTime
        keyBreakTime = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='keyBreakTime',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "prepareComponents" ---
    # Run 'Begin Experiment' code from codePrepareComponents
    # show the cursor
    win.mouseVisible = True
    
    # --- Initialize components for Routine "prepareQuiz" ---
    
    # --- Initialize components for Routine "prepareExperiment" ---
    # Run 'Begin Experiment' code from codePrepareExperiment
    import random
    
    random.seed()
    
    # split the space into 4 regions
    n_ori_regions = 4
    
    # TODO: read the allowed trial code
    trial_types_allowed = [1, 0]
    n_trial_types_allowed = len(trial_types_allowed)
    
    # TODO: read the allowed sampling strategies
    sampling_strategies_allowed = ['mixed',]
    n_sampling_strategies_allowed = len(sampling_strategies_allowed)
    
    # TODO: figure out if we have cases where people only do one mode?
    response_modes_allowed = ['draw', 'click']
    n_response_modes_allowed = len(response_modes_allowed)
    
    # TODO: define n blocks each sub type?
    n_blocks_each_subtype = 4
    
    # THE PATH TO BACKUP RECORDS
    BACKUP_IDX = 0
    N_BACKUPS = 2
    BACKUP_FOLDER_PATH = thisExp.dataFileName
    if not os.path.exists(BACKUP_FOLDER_PATH):
        os.makedirs(BACKUP_FOLDER_PATH)
    
    
    # --- Initialize components for Routine "prepareTrialConfig" ---
    # Run 'Begin Experiment' code from codePrepareTrialConfig
    # configuration for time
    pre_trial_time = 1
    display_time = 0.75
    display_cue_onset_time = 0.25
    click_response_time = 4
    draw_response_time = 4
    enforce_response_time_limit = False # TO TUNE
    delay_time_min = 2.5
    delay_time_max = 8.5
    post_trial_time_min = 1
    post_trial_time_max = 3
    
    # set delay after one stim
    delay_post_stim = 1.5
    
    # configurations for display
    patch_radius = 0.16
    click_radius = 0.01
    wheel_width = 5
    button_width = 0.1
    button_height = 0.05
    
    # params control noise patch
    noise_patch_display_time = 0.5
    noise_patch_display_chance = 1.0
    
    
    # --- Initialize components for Routine "practiceFreeDrawing" ---
    textPracticeFreeDraw = visual.TextStim(win=win, name='textPracticeFreeDraw',
        text='Let’s first get familiar with the tablet and pen...\n(press [space] to continue)',
        font='Open Sans',
        pos=(0, 0.4), draggable=False, height=0.04, wrapWidth=1.0, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    brushPractice = visual.Brush(win=win, name='brushPractice',
       lineWidth=1.5,
       lineColor=[0,0,0],
       lineColorSpace='rgb',
       opacity=None,
       buttonRequired=True,
       depth=-1
    )
    keyPracticeFreeDraw = keyboard.Keyboard(deviceName='keyPracticeFreeDraw')
    
    # --- Initialize components for Routine "fullInstruction" ---
    # Run 'Begin Experiment' code from codeFullInstruction
    continue_button = None
    textInstruction = visual.TextStim(win=win, name='textInstruction',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=1.2, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    mouseFullInstruction = event.Mouse(win=win)
    x, y = [None, None]
    mouseFullInstruction.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "startModeInstr" ---
    # Run 'Begin Experiment' code from codeStartModeInstr
    continue_button = None
    textStartModeInstr = visual.TextStim(win=win, name='textStartModeInstr',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=1.2, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    mouseStartModeInstr = event.Mouse(win=win)
    x, y = [None, None]
    mouseStartModeInstr.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "responseInstruction" ---
    # Run 'Begin Experiment' code from codeResponseInstruction
    continue_button = None
    textResponseInstruction = visual.TextStim(win=win, name='textResponseInstruction',
        text='',
        font='Open Sans',
        pos=(0, 0.4), draggable=False, height=0.05, wrapWidth=1.4, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    imageResponseInstruction = visual.ImageStim(
        win=win,
        name='imageResponseInstruction', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1.12, 0.63),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    mouseResponseInstruction = event.Mouse(win=win)
    x, y = [None, None]
    mouseResponseInstruction.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "prepareModePractice" ---
    
    # --- Initialize components for Routine "displayOneStim" ---
    mouseOneStim = event.Mouse(win=win)
    x, y = [None, None]
    mouseOneStim.mouseClock = core.Clock()
    textDisplayOneStim = visual.TextStim(win=win, name='textDisplayOneStim',
        text='',
        font='Open Sans',
        pos=(0, 0.4), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "feedbackOneStim" ---
    # Run 'Begin Experiment' code from codeFeedbackOneStim
    
    
    
    textFeedbackOneStim = visual.TextStim(win=win, name='textFeedbackOneStim',
        text='Feedback',
        font='Open Sans',
        pos=(0, 0.4), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    mouseFeedbackOneStim = event.Mouse(win=win)
    x, y = [None, None]
    mouseFeedbackOneStim.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "prepareAllPractice" ---
    
    # --- Initialize components for Routine "startColorRulePractice" ---
    # Run 'Begin Experiment' code from codeStartColorRule
    continue_button = None
    mouseStartColorRule = event.Mouse(win=win)
    x, y = [None, None]
    mouseStartColorRule.mouseClock = core.Clock()
    textStartColorRule = visual.TextStim(win=win, name='textStartColorRule',
        text='In the next section, we will explain how the stimuli will be presented and how to respond correctly when there are multiple items to report.',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "overviewColorRule" ---
    # Run 'Begin Experiment' code from codeOverviewInstruction
    continue_button = None
    imageOverviewInstruction = visual.ImageStim(
        win=win,
        name='imageOverviewInstruction', 
        image='resources/overview.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0.05), draggable=False, size=(1.34, 0.75),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    mouseOverviewInstruction = event.Mouse(win=win)
    x, y = [None, None]
    mouseOverviewInstruction.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "prepareColorRulePractice" ---
    # Run 'Begin Experiment' code from codePrepareColorRulePractice
    n_color_rule_practice = 4
    textPrepareColorRulePractice = visual.TextStim(win=win, name='textPrepareColorRulePractice',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    mousePrepareColorRulePractice = event.Mouse(win=win)
    x, y = [None, None]
    mousePrepareColorRulePractice.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "colorRuleStim" ---
    textColorRuleStim = visual.TextStim(win=win, name='textColorRuleStim',
        text=None,
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "colorRuleDelay" ---
    textColorRuleDelay = visual.TextStim(win=win, name='textColorRuleDelay',
        text=None,
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "colorRuleResponse" ---
    mouseColorRuleResponse = event.Mouse(win=win)
    x, y = [None, None]
    mouseColorRuleResponse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "colorRuleFeedback" ---
    # Run 'Begin Experiment' code from codeColorRuleFeedback
    
    
    
    textColorRuleFeedback = visual.TextStim(win=win, name='textColorRuleFeedback',
        text='Feedback',
        font='Open Sans',
        pos=(0, 0.42), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    mouseColorRuleFeedback = event.Mouse(win=win)
    x, y = [None, None]
    mouseColorRuleFeedback.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "taskStart" ---
    textTaskPhase = visual.TextStim(win=win, name='textTaskPhase',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    mouseTaskPhase = event.Mouse(win=win)
    x, y = [None, None]
    mouseTaskPhase.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "taskInstructionText" ---
    textTaskInstruction = visual.TextStim(win=win, name='textTaskInstruction',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    mouseTaskInstruction = event.Mouse(win=win)
    x, y = [None, None]
    mouseTaskInstruction.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "taskInstructionDiagram" ---
    # Run 'Begin Experiment' code from codeTaskInstrDiagram
    task_diagram_text_to_display_list = [
            'Trials with a delay and an informatic cue',
            'Trials with a delay but no informatic cue'
    ]
    textTaskInstrDiagram = visual.TextStim(win=win, name='textTaskInstrDiagram',
        text='',
        font='Open Sans',
        pos=(0, 0.4), draggable=False, height=0.05, wrapWidth=1.3, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    mouseTaskInstrDiagram = event.Mouse(win=win)
    x, y = [None, None]
    mouseTaskInstrDiagram.mouseClock = core.Clock()
    imgTaskInstrDiagram = visual.ImageStim(
        win=win,
        name='imgTaskInstrDiagram', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0.0), draggable=False, size=(1.12, 0.63),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    
    # --- Initialize components for Routine "prepareTaskInstrPractice" ---
    
    # --- Initialize components for Routine "startPracticeBlock" ---
    mouseStartPracticeBlock = event.Mouse(win=win)
    x, y = [None, None]
    mouseStartPracticeBlock.mouseClock = core.Clock()
    textStartPracticeBlock = visual.TextStim(win=win, name='textStartPracticeBlock',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "prepareSinglePracticeTrial" ---
    textPracticePreTrial = visual.TextStim(win=win, name='textPracticePreTrial',
        text=None,
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "taskPracticeStim" ---
    textTaskPracticeStim = visual.TextStim(win=win, name='textTaskPracticeStim',
        text=None,
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "taskPracticeStimDelay" ---
    textColorRuleDelay_2 = visual.TextStim(win=win, name='textColorRuleDelay_2',
        text=None,
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "taskPracticeResponse" ---
    mouseTaskPracticeResponse = event.Mouse(win=win)
    x, y = [None, None]
    mouseTaskPracticeResponse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "taskPracticeTrialFeedback" ---
    # Run 'Begin Experiment' code from codeTaskPracticeTrialFeedback
    
    
    
    textTaskPracticeTrialFeedback = visual.TextStim(win=win, name='textTaskPracticeTrialFeedback',
        text='Feedback',
        font='Open Sans',
        pos=(0, 0.42), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    mouseTaskPracticeTrialFeedback = event.Mouse(win=win)
    x, y = [None, None]
    mouseTaskPracticeTrialFeedback.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "summarizeTrialBlock" ---
    
    # --- Initialize components for Routine "connectEL" ---
    # Run 'Begin Experiment' code from connectTracker
    # this is adapted from the SR research's eyelink tutorial code
    #parameters to change
    calib_style = 13 #9 for head fixed, 13 for remote 
    samprate = 1000 #250, 500, 1000, or 2000
    calib_tar_size = 24 #size for the calibration target
    
    """change this"""
    width_param = 53.0 
    distance_param =70.0
    
    
    #create a folder to store all edf files, call this folder 'results'
    edf_folder = 'C:/Users/siy009/Documents/elRaw_sihanYueying'
    if not os.path.exists(edf_folder):
        os.makedirs(edf_folder)
    
    # We download EDF data file from the EyeLink Host PC to the local hard
    # drive at the end of each testing session, here we rename the EDF to
    # include session start date/time
    time_str = time.strftime("_%Y_%m_%d_%H_%M", time.localtime())
    session_identifier = str(expInfo['participant']) + time_str
    
    # create a folder for the current testing session in the "results" folder
    session_folder = os.path.join(edf_folder, session_identifier)
    if not os.path.exists(session_folder):
        os.makedirs(session_folder)
    
    
    #helper function for displaying text
    def clear_screen(win):
        """ clear up the PsychoPy window"""
    
        win.fillColor = genv.getBackgroundColor()
        win.flip()
    def show_msg(win, text, wait_for_keypress=True):
        """ Show task instructions on screen"""
    
        msg = visual.TextStim(win, text,
                              color=genv.getForegroundColor(),
                              wrapWidth=scn_width/2)
        clear_screen(win)
        msg.draw()
        win.flip()
    
        # wait indefinitely, terminates upon any key press
        if wait_for_keypress:
            event.waitKeys(keyList = ['space','escape'],maxWait = 60)
            clear_screen(win)
         
    #function to terminate task and retrieve the EDF data file from the host PC and 
    #download to the display pc
    def terminate_task():
        el_tracker = pylink.getEYELINK()
    
        if el_tracker.isConnected():
    
            # Put tracker in Offline mode
            el_tracker.setOfflineMode()
    
            # Clear the Host PC screen and wait for 500 ms
            el_tracker.sendCommand('clear_screen 0')
            pylink.msecDelay(500)
    
            # Close the edf data file on the Host
            el_tracker.closeDataFile()
    
            # Show a file transfer message on the screen
            msg = 'EDF data is transferring from EyeLink Host PC...'
            show_msg(win, msg, wait_for_keypress=False)
    
            # Download the EDF data file from the Host PC to a local data folder
            # parameters: source_file_on_the_host, destination_file_on_local_drive
            local_edf = os.path.join(session_folder, session_identifier + '.EDF')
            try:
                el_tracker.receiveDataFile(edf_file, local_edf)
            except RuntimeError as error:
                print('ERROR:', error)
    
            # Close the link to the tracker.
            el_tracker.close()
    
        # close the PsychoPy window
        win.close()
    
        # quit PsychoPy
        core.quit()
        sys.exit()
        
    def abort_trial():
        """Ends recording """
    
        el_tracker = pylink.getEYELINK()
    
        # Stop recording
        if el_tracker.isRecording():
            # add 100 ms to catch final trial events
            pylink.pumpDelay(100)
            el_tracker.stopRecording()
    
        # clear the screen
        clear_screen(win)
        # Send a message to clear the Data Viewer screen
        bgcolor_RGB = (116, 116, 116)
        el_tracker.sendMessage('!V CLEAR %d %d %d' % bgcolor_RGB)
    
        # send a message to mark trial end
        el_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_ERROR)
    
    
    # --- Initialize components for Routine "prepareTaskPhase" ---
    mousePrepareTaskPhase = event.Mouse(win=win)
    x, y = [None, None]
    mousePrepareTaskPhase.mouseClock = core.Clock()
    textPrepareTaskPhase = visual.TextStim(win=win, name='textPrepareTaskPhase',
        text='',
        font='Open Sans',
        pos=(0, 0.05), draggable=False, height=0.05, wrapWidth=1.3, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "prepareSubSampleStage" ---
    
    # --- Initialize components for Routine "prepare_block" ---
    # Run 'Begin Experiment' code from codePrepareBlock
    block_instruction = ''
    continue_button = None
    
    textPrepareBlock = visual.TextStim(win=win, name='textPrepareBlock',
        text=None,
        font='Open Sans',
        pos=(0, 0.05), draggable=False, height=0.05, wrapWidth=1.3, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    mousePrepareBlock = event.Mouse(win=win)
    x, y = [None, None]
    mousePrepareBlock.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "drift_check" ---
    
    # --- Initialize components for Routine "prepareSingleTrial" ---
    # Run 'Begin Experiment' code from codePrepareSingleTrial
    # define the 4 quadrants to display/response
    quadrant_x_pos = [-0.25, 0.25]
    quadrant_y_pos = [0.3, -0.24]
    
    # pre-register possible locations
    allow_diagonal = False # TO TUNE
    allow_off_axis = False # TO TUNE
    allow_row_align = True # TO TUNE   
    allow_location_swap = True # TO TUNE
    resp_location_sampler = ResponseLocationSampler(
        xs=quadrant_x_pos, 
        ys=quadrant_y_pos, 
        allow_diagonal=allow_diagonal, 
        allow_off_axis=allow_off_axis, 
        allow_row_align=allow_location_swap)
    
    
    # --- Initialize components for Routine "pre_trial" ---
    textPreTrial = visual.TextStim(win=win, name='textPreTrial',
        text=None,
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "display_stimuli" ---
    mouseStim = event.Mouse(win=win)
    x, y = [None, None]
    mouseStim.mouseClock = core.Clock()
    textDisplayStim = visual.TextStim(win=win, name='textDisplayStim',
        text=None,
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "delayAfterStim" ---
    mousePostStim = event.Mouse(win=win)
    x, y = [None, None]
    mousePostStim.mouseClock = core.Clock()
    textDelayAfterStim = visual.TextStim(win=win, name='textDelayAfterStim',
        text=None,
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "delay" ---
    mouseDelay = event.Mouse(win=win)
    x, y = [None, None]
    mouseDelay.mouseClock = core.Clock()
    # Run 'Begin Experiment' code from codeDelay
    n_max_cues = 0 # TO TUNE
    delay_cue_duration = 0.5 
    delay_cue_space = 1 # TO TUNE
    textDelay = visual.TextStim(win=win, name='textDelay',
        text=None,
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "response" ---
    responseMouse = event.Mouse(win=win)
    x, y = [None, None]
    responseMouse.mouseClock = core.Clock()
    keyRespInput = keyboard.Keyboard(deviceName='keyRespInput')
    
    # --- Initialize components for Routine "post_trial" ---
    textPostTrial = visual.TextStim(win=win, name='textPostTrial',
        text=None,
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "backupSave" ---
    
    # --- Initialize components for Routine "block_feedback" ---
    # Run 'Begin Experiment' code from codeBlockFeedback
    # create functions for stats of errors
    # within 10 degrees
    # within 30 degrees
    # no response
    def count_errors(errors):
        n_datapoint = len(errors)
        
        # for empty set
        if n_datapoint == 0:
            return 0, 0, 0
        n_err_5 = 0
        n_err_10 = 0
        n_err_30 = 0
        n_err_none = 0
        for i in range(n_datapoint):
            err = errors[i]
            if err is None:
                n_err_none += 1
            else:
                abs_err = abs(err)
                if abs_err <= 5:
                    n_err_5 += 1
                if abs_err <= 10:
                    n_err_10 += 1
                if abs_err <= 30:
                    n_err_30 += 1
        return n_err_5 / n_datapoint, n_err_10 / n_datapoint, \
            n_err_30 / n_datapoint, n_err_none / n_datapoint
    funcCountErrrors = count_errors
        
    def create_freq_bar(freq_data, window, pos, size, color, text):
        bar_x = pos[0]
        bar_y = pos[1]
        bar_width = size[0]
        bar_height = size[1]
        objs = []
        
        left_bar = None
        if freq_data > 0:
            left_bar_width = bar_width * freq_data
            left_bar_x = bar_x - (bar_width - left_bar_width) / 2
            left_bar = visual.Rect(window,
                size=(left_bar_width, bar_height),
                pos=(left_bar_x, bar_y),
                fillColor=color, lineColor='white')
            left_bar.setAutoDraw(True)
            objs.append(left_bar)
            
        # right bar
        right_bar = None
        if freq_data < 1:
            right_bar_width = bar_width * (1 - freq_data)
            right_bar_x = bar_x + (bar_width - right_bar_width) / 2
            right_bar = visual.Rect(window,
                size=(right_bar_width, bar_height),
                pos=(right_bar_x, bar_y),
                fillColor='white', lineColor='white')
            right_bar.setAutoDraw(True)
            objs.append(right_bar)
            
        # add annotation
        text_x = -0.4
        text_y = bar_y
        annot_text = visual.TextStim(
            window, text, pos=(text_x, text_y),
            height=bar_height*0.5,
            color='black', colorSpace='rgb')
        annot_text.setAutoDraw(True)
        objs.append(annot_text)
        return objs
    funcCreateFreqBar = create_freq_bar
    
    # to draw all stats
    def plot_stats(error_stats, window):
        bar_x = 0.3
        bar_width = 0.6
        bar_height = 0.08
        feedback_display_objects = []
        
        # plot error < 5
        freq5 = error_stats[0]
        freq5_y = 0.2
        freq5_objects = funcCreateFreqBar(
            freq5, window, (bar_x, freq5_y), 
            (bar_width, bar_height), 'blue'
            , f"Minor discrepancies: {freq5:.0%}"
        )
        feedback_display_objects += freq5_objects
    
        # plot error < 10
        freq10 = error_stats[1]
        freq10_y = 0.05
        freq10_objects = funcCreateFreqBar(
            freq10, window, 
            [bar_x, freq10_y], 
            [bar_width, bar_height], 'blue'
            , f"Errors within a moderate range: {freq10:.0%}"
        )
        feedback_display_objects += freq10_objects
        
        # plot error < 30
        freq30 = error_stats[2]
        freq30_y = - 0.1
        freq30_objects = funcCreateFreqBar(
            freq30, window, 
            [bar_x, freq30_y], 
            [bar_width, bar_height], 'blue'
            , f"Acceptable level of error: {freq30:.0%}"
        )
        feedback_display_objects += freq30_objects
            
        # plot has response
        freq_noresp = error_stats[3]
        freq_noresp_y = - 0.25
        freq_noresp_objects = funcCreateFreqBar(
            freq_noresp, window, 
            [bar_x, freq_noresp_y], 
            [bar_width, bar_height], 'red'
            , f'no response: {freq_noresp:.0%}'
        )
        feedback_display_objects += freq_noresp_objects
        return feedback_display_objects
    funcPlotStats = plot_stats
    
    textBlockFeedback = visual.TextStim(win=win, name='textBlockFeedback',
        text=None,
        font='Open Sans',
        pos=(0, 0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    blockFeedbackMouse = event.Mouse(win=win)
    x, y = [None, None]
    blockFeedbackMouse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "breakTime" ---
    textBreakTime = visual.TextStim(win=win, name='textBreakTime',
        text='Now we will pause to take a longer break.\nPlease inform the experimenter first.',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    keyBreakTime = keyboard.Keyboard(deviceName='keyBreakTime')
    
    # --- Initialize components for Routine "terminateExp" ---
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "prepareComponents" ---
    # create an object to store info about Routine prepareComponents
    prepareComponents = data.Routine(
        name='prepareComponents',
        components=[],
    )
    prepareComponents.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for prepareComponents
    prepareComponents.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    prepareComponents.tStart = globalClock.getTime(format='float')
    prepareComponents.status = STARTED
    thisExp.addData('prepareComponents.started', prepareComponents.tStart)
    prepareComponents.maxDuration = None
    # keep track of which components have finished
    prepareComponentsComponents = prepareComponents.components
    for thisComponent in prepareComponents.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "prepareComponents" ---
    prepareComponents.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            prepareComponents.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in prepareComponents.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "prepareComponents" ---
    for thisComponent in prepareComponents.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for prepareComponents
    prepareComponents.tStop = globalClock.getTime(format='float')
    prepareComponents.tStopRefresh = tThisFlipGlobal
    thisExp.addData('prepareComponents.stopped', prepareComponents.tStop)
    thisExp.nextEntry()
    # the Routine "prepareComponents" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "prepareQuiz" ---
    # create an object to store info about Routine prepareQuiz
    prepareQuiz = data.Routine(
        name='prepareQuiz',
        components=[],
    )
    prepareQuiz.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for prepareQuiz
    prepareQuiz.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    prepareQuiz.tStart = globalClock.getTime(format='float')
    prepareQuiz.status = STARTED
    thisExp.addData('prepareQuiz.started', prepareQuiz.tStart)
    prepareQuiz.maxDuration = None
    # keep track of which components have finished
    prepareQuizComponents = prepareQuiz.components
    for thisComponent in prepareQuiz.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "prepareQuiz" ---
    prepareQuiz.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            prepareQuiz.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in prepareQuiz.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "prepareQuiz" ---
    for thisComponent in prepareQuiz.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for prepareQuiz
    prepareQuiz.tStop = globalClock.getTime(format='float')
    prepareQuiz.tStopRefresh = tThisFlipGlobal
    thisExp.addData('prepareQuiz.stopped', prepareQuiz.tStop)
    thisExp.nextEntry()
    # the Routine "prepareQuiz" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "prepareExperiment" ---
    # create an object to store info about Routine prepareExperiment
    prepareExperiment = data.Routine(
        name='prepareExperiment',
        components=[],
    )
    prepareExperiment.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from codePrepareExperiment
    # keep track of errors
    experiment_error_records = []
    
    # initialize the list of trials to run
    sampling_type_to_sample = len(sampling_strategies_allowed)
    trial_type_list = random.sample(trial_types_allowed, n_trial_types_allowed)
    mode_type_list = random.sample(response_modes_allowed, n_response_modes_allowed)
    sampling_type_list = random.sample(sampling_strategies_allowed, n_sampling_strategies_allowed)
    
    # compute the stats
    n_kinds_of_tasks = n_trial_types_allowed
    n_blocks_each_subtask = n_sampling_strategies_allowed * n_blocks_each_subtype
    n_blocks_each_task = n_blocks_each_subtask * n_response_modes_allowed
    n_trials_each_block = 10
    if expInfo["test mode"] == "online":
        n_trials_each_block = 10
    # Run 'Begin Routine' code from el_ctrl
    eyetracking = 1 #0 is off tracker; 1 is on tracker
    # store start times for prepareExperiment
    prepareExperiment.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    prepareExperiment.tStart = globalClock.getTime(format='float')
    prepareExperiment.status = STARTED
    thisExp.addData('prepareExperiment.started', prepareExperiment.tStart)
    prepareExperiment.maxDuration = None
    # keep track of which components have finished
    prepareExperimentComponents = prepareExperiment.components
    for thisComponent in prepareExperiment.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "prepareExperiment" ---
    prepareExperiment.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            prepareExperiment.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in prepareExperiment.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "prepareExperiment" ---
    for thisComponent in prepareExperiment.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for prepareExperiment
    prepareExperiment.tStop = globalClock.getTime(format='float')
    prepareExperiment.tStopRefresh = tThisFlipGlobal
    thisExp.addData('prepareExperiment.stopped', prepareExperiment.tStop)
    thisExp.nextEntry()
    # the Routine "prepareExperiment" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "prepareTrialConfig" ---
    # create an object to store info about Routine prepareTrialConfig
    prepareTrialConfig = data.Routine(
        name='prepareTrialConfig',
        components=[],
    )
    prepareTrialConfig.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for prepareTrialConfig
    prepareTrialConfig.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    prepareTrialConfig.tStart = globalClock.getTime(format='float')
    prepareTrialConfig.status = STARTED
    thisExp.addData('prepareTrialConfig.started', prepareTrialConfig.tStart)
    prepareTrialConfig.maxDuration = None
    # keep track of which components have finished
    prepareTrialConfigComponents = prepareTrialConfig.components
    for thisComponent in prepareTrialConfig.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "prepareTrialConfig" ---
    prepareTrialConfig.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            prepareTrialConfig.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in prepareTrialConfig.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "prepareTrialConfig" ---
    for thisComponent in prepareTrialConfig.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for prepareTrialConfig
    prepareTrialConfig.tStop = globalClock.getTime(format='float')
    prepareTrialConfig.tStopRefresh = tThisFlipGlobal
    thisExp.addData('prepareTrialConfig.stopped', prepareTrialConfig.tStop)
    thisExp.nextEntry()
    # the Routine "prepareTrialConfig" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    DEBUG = data.TrialHandler2(
        name='DEBUG',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(DEBUG)  # add the loop to the experiment
    thisDEBUG = DEBUG.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisDEBUG.rgb)
    if thisDEBUG != None:
        for paramName in thisDEBUG:
            globals()[paramName] = thisDEBUG[paramName]
    
    for thisDEBUG in DEBUG:
        currentLoop = DEBUG
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisDEBUG.rgb)
        if thisDEBUG != None:
            for paramName in thisDEBUG:
                globals()[paramName] = thisDEBUG[paramName]
        
        # --- Prepare to start Routine "practiceFreeDrawing" ---
        # create an object to store info about Routine practiceFreeDrawing
        practiceFreeDrawing = data.Routine(
            name='practiceFreeDrawing',
            components=[textPracticeFreeDraw, brushPractice, keyPracticeFreeDraw],
        )
        practiceFreeDrawing.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        brushPractice.reset()
        # create starting attributes for keyPracticeFreeDraw
        keyPracticeFreeDraw.keys = []
        keyPracticeFreeDraw.rt = []
        _keyPracticeFreeDraw_allKeys = []
        # store start times for practiceFreeDrawing
        practiceFreeDrawing.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        practiceFreeDrawing.tStart = globalClock.getTime(format='float')
        practiceFreeDrawing.status = STARTED
        thisExp.addData('practiceFreeDrawing.started', practiceFreeDrawing.tStart)
        practiceFreeDrawing.maxDuration = None
        # keep track of which components have finished
        practiceFreeDrawingComponents = practiceFreeDrawing.components
        for thisComponent in practiceFreeDrawing.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "practiceFreeDrawing" ---
        # if trial has changed, end Routine now
        if isinstance(DEBUG, data.TrialHandler2) and thisDEBUG.thisN != DEBUG.thisTrial.thisN:
            continueRoutine = False
        practiceFreeDrawing.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *textPracticeFreeDraw* updates
            
            # if textPracticeFreeDraw is starting this frame...
            if textPracticeFreeDraw.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textPracticeFreeDraw.frameNStart = frameN  # exact frame index
                textPracticeFreeDraw.tStart = t  # local t and not account for scr refresh
                textPracticeFreeDraw.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textPracticeFreeDraw, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textPracticeFreeDraw.started')
                # update status
                textPracticeFreeDraw.status = STARTED
                textPracticeFreeDraw.setAutoDraw(True)
            
            # if textPracticeFreeDraw is active this frame...
            if textPracticeFreeDraw.status == STARTED:
                # update params
                pass
            
            # *brushPractice* updates
            
            # if brushPractice is starting this frame...
            if brushPractice.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                brushPractice.frameNStart = frameN  # exact frame index
                brushPractice.tStart = t  # local t and not account for scr refresh
                brushPractice.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(brushPractice, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'brushPractice.started')
                # update status
                brushPractice.status = STARTED
                brushPractice.setAutoDraw(True)
            
            # if brushPractice is active this frame...
            if brushPractice.status == STARTED:
                # update params
                pass
            
            # *keyPracticeFreeDraw* updates
            waitOnFlip = False
            
            # if keyPracticeFreeDraw is starting this frame...
            if keyPracticeFreeDraw.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                keyPracticeFreeDraw.frameNStart = frameN  # exact frame index
                keyPracticeFreeDraw.tStart = t  # local t and not account for scr refresh
                keyPracticeFreeDraw.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(keyPracticeFreeDraw, 'tStartRefresh')  # time at next scr refresh
                # update status
                keyPracticeFreeDraw.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(keyPracticeFreeDraw.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(keyPracticeFreeDraw.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if keyPracticeFreeDraw.status == STARTED and not waitOnFlip:
                theseKeys = keyPracticeFreeDraw.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _keyPracticeFreeDraw_allKeys.extend(theseKeys)
                if len(_keyPracticeFreeDraw_allKeys):
                    keyPracticeFreeDraw.keys = _keyPracticeFreeDraw_allKeys[-1].name  # just the last key pressed
                    keyPracticeFreeDraw.rt = _keyPracticeFreeDraw_allKeys[-1].rt
                    keyPracticeFreeDraw.duration = _keyPracticeFreeDraw_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                practiceFreeDrawing.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in practiceFreeDrawing.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "practiceFreeDrawing" ---
        for thisComponent in practiceFreeDrawing.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for practiceFreeDrawing
        practiceFreeDrawing.tStop = globalClock.getTime(format='float')
        practiceFreeDrawing.tStopRefresh = tThisFlipGlobal
        thisExp.addData('practiceFreeDrawing.stopped', practiceFreeDrawing.tStop)
        # the Routine "practiceFreeDrawing" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "fullInstruction" ---
        # create an object to store info about Routine fullInstruction
        fullInstruction = data.Routine(
            name='fullInstruction',
            components=[textInstruction, mouseFullInstruction],
        )
        fullInstruction.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from codeFullInstruction
        continue_button = funcCreateButton(
            win, 'continue', 
            [0, -0.4], [0.25, 0.1])
        textInstruction.setText('In this experiment, you will complete tasks that involve remembering and reproducing visual patterns.\n\n In each trial, you will see two oriented patches. Your task is to remember them for a few seconds and reproduce them afterward. \n\n The experiment consists of '+ str(n_kinds_of_tasks) + ' types of tasks. For each type, you will first see instructions, followed by a few practice trials before starting the actual task.')
        # setup some python lists for storing info about the mouseFullInstruction
        gotValidClick = False  # until a click is received
        # store start times for fullInstruction
        fullInstruction.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        fullInstruction.tStart = globalClock.getTime(format='float')
        fullInstruction.status = STARTED
        thisExp.addData('fullInstruction.started', fullInstruction.tStart)
        fullInstruction.maxDuration = None
        # keep track of which components have finished
        fullInstructionComponents = fullInstruction.components
        for thisComponent in fullInstruction.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fullInstruction" ---
        # if trial has changed, end Routine now
        if isinstance(DEBUG, data.TrialHandler2) and thisDEBUG.thisN != DEBUG.thisTrial.thisN:
            continueRoutine = False
        fullInstruction.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from codeFullInstruction
            left_button = mouseFullInstruction.getPressed()[0]
            mouse_position = mouseFullInstruction.getPos()
            if left_button and (t > 0.5) and continue_button.contains(mouse_position):
                continueRoutine = False
            
            
            # *textInstruction* updates
            
            # if textInstruction is starting this frame...
            if textInstruction.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textInstruction.frameNStart = frameN  # exact frame index
                textInstruction.tStart = t  # local t and not account for scr refresh
                textInstruction.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textInstruction, 'tStartRefresh')  # time at next scr refresh
                # update status
                textInstruction.status = STARTED
                textInstruction.setAutoDraw(True)
            
            # if textInstruction is active this frame...
            if textInstruction.status == STARTED:
                # update params
                pass
            # *mouseFullInstruction* updates
            
            # if mouseFullInstruction is starting this frame...
            if mouseFullInstruction.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouseFullInstruction.frameNStart = frameN  # exact frame index
                mouseFullInstruction.tStart = t  # local t and not account for scr refresh
                mouseFullInstruction.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouseFullInstruction, 'tStartRefresh')  # time at next scr refresh
                # update status
                mouseFullInstruction.status = STARTED
                mouseFullInstruction.mouseClock.reset()
                prevButtonState = mouseFullInstruction.getPressed()  # if button is down already this ISN'T a new click
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                fullInstruction.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fullInstruction.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fullInstruction" ---
        for thisComponent in fullInstruction.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for fullInstruction
        fullInstruction.tStop = globalClock.getTime(format='float')
        fullInstruction.tStopRefresh = tThisFlipGlobal
        thisExp.addData('fullInstruction.stopped', fullInstruction.tStop)
        # Run 'End Routine' code from codeFullInstruction
        continue_button.setAutoDraw(False);
        continue_button = None
        # store data for DEBUG (TrialHandler)
        # the Routine "fullInstruction" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "startModeInstr" ---
        # create an object to store info about Routine startModeInstr
        startModeInstr = data.Routine(
            name='startModeInstr',
            components=[textStartModeInstr, mouseStartModeInstr],
        )
        startModeInstr.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from codeStartModeInstr
        continue_button = funcCreateButton(
            win, 'continue', 
            [0, -0.4], [0.25, 0.1])
        textStartModeInstr.setText('To report an orientation, you will either draw a line or adjust the position of a dot on a wheel. (More details will follow.)')
        # setup some python lists for storing info about the mouseStartModeInstr
        gotValidClick = False  # until a click is received
        # store start times for startModeInstr
        startModeInstr.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        startModeInstr.tStart = globalClock.getTime(format='float')
        startModeInstr.status = STARTED
        thisExp.addData('startModeInstr.started', startModeInstr.tStart)
        startModeInstr.maxDuration = None
        # keep track of which components have finished
        startModeInstrComponents = startModeInstr.components
        for thisComponent in startModeInstr.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "startModeInstr" ---
        # if trial has changed, end Routine now
        if isinstance(DEBUG, data.TrialHandler2) and thisDEBUG.thisN != DEBUG.thisTrial.thisN:
            continueRoutine = False
        startModeInstr.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from codeStartModeInstr
            left_button = mouseStartModeInstr.getPressed()[0]
            mouse_position = mouseStartModeInstr.getPos()
            if left_button and (t > 0.5) and continue_button.contains(mouse_position):
                continueRoutine = False
            
            
            # *textStartModeInstr* updates
            
            # if textStartModeInstr is starting this frame...
            if textStartModeInstr.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textStartModeInstr.frameNStart = frameN  # exact frame index
                textStartModeInstr.tStart = t  # local t and not account for scr refresh
                textStartModeInstr.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textStartModeInstr, 'tStartRefresh')  # time at next scr refresh
                # update status
                textStartModeInstr.status = STARTED
                textStartModeInstr.setAutoDraw(True)
            
            # if textStartModeInstr is active this frame...
            if textStartModeInstr.status == STARTED:
                # update params
                pass
            # *mouseStartModeInstr* updates
            
            # if mouseStartModeInstr is starting this frame...
            if mouseStartModeInstr.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouseStartModeInstr.frameNStart = frameN  # exact frame index
                mouseStartModeInstr.tStart = t  # local t and not account for scr refresh
                mouseStartModeInstr.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouseStartModeInstr, 'tStartRefresh')  # time at next scr refresh
                # update status
                mouseStartModeInstr.status = STARTED
                mouseStartModeInstr.mouseClock.reset()
                prevButtonState = mouseStartModeInstr.getPressed()  # if button is down already this ISN'T a new click
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                startModeInstr.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in startModeInstr.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "startModeInstr" ---
        for thisComponent in startModeInstr.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for startModeInstr
        startModeInstr.tStop = globalClock.getTime(format='float')
        startModeInstr.tStopRefresh = tThisFlipGlobal
        thisExp.addData('startModeInstr.stopped', startModeInstr.tStop)
        # Run 'End Routine' code from codeStartModeInstr
        continue_button.setAutoDraw(False);
        continue_button = None
        # store data for DEBUG (TrialHandler)
        # the Routine "startModeInstr" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        response_instruction_loop = data.TrialHandler2(
            name='response_instruction_loop',
            nReps=n_response_modes_allowed, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(response_instruction_loop)  # add the loop to the experiment
        thisResponse_instruction_loop = response_instruction_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisResponse_instruction_loop.rgb)
        if thisResponse_instruction_loop != None:
            for paramName in thisResponse_instruction_loop:
                globals()[paramName] = thisResponse_instruction_loop[paramName]
        
        for thisResponse_instruction_loop in response_instruction_loop:
            currentLoop = response_instruction_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # abbreviate parameter names if possible (e.g. rgb = thisResponse_instruction_loop.rgb)
            if thisResponse_instruction_loop != None:
                for paramName in thisResponse_instruction_loop:
                    globals()[paramName] = thisResponse_instruction_loop[paramName]
            
            # --- Prepare to start Routine "responseInstruction" ---
            # create an object to store info about Routine responseInstruction
            responseInstruction = data.Routine(
                name='responseInstruction',
                components=[textResponseInstruction, imageResponseInstruction, mouseResponseInstruction],
            )
            responseInstruction.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from codeResponseInstruction
            instruction_mode = response_modes_allowed[response_instruction_loop.thisN]
            continue_button = funcCreateButton(
                win, 'continue', 
                [0, -0.4], [0.25, 0.1])
            textResponseInstruction.setText('For reporting orientation by ' + instruction_mode + 'ing, here is an example of a grating stimulus and the correct response')
            imageResponseInstruction.setImage('resources/'+instruction_mode+'ing.png')
            # setup some python lists for storing info about the mouseResponseInstruction
            gotValidClick = False  # until a click is received
            # store start times for responseInstruction
            responseInstruction.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            responseInstruction.tStart = globalClock.getTime(format='float')
            responseInstruction.status = STARTED
            thisExp.addData('responseInstruction.started', responseInstruction.tStart)
            responseInstruction.maxDuration = None
            # keep track of which components have finished
            responseInstructionComponents = responseInstruction.components
            for thisComponent in responseInstruction.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "responseInstruction" ---
            # if trial has changed, end Routine now
            if isinstance(response_instruction_loop, data.TrialHandler2) and thisResponse_instruction_loop.thisN != response_instruction_loop.thisTrial.thisN:
                continueRoutine = False
            responseInstruction.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from codeResponseInstruction
                left_button = mouseResponseInstruction.getPressed()[0]
                mouse_position = mouseResponseInstruction.getPos()
                if left_button and (t > 0.5) and continue_button.contains(mouse_position):
                    continueRoutine = False
                
                
                # *textResponseInstruction* updates
                
                # if textResponseInstruction is starting this frame...
                if textResponseInstruction.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textResponseInstruction.frameNStart = frameN  # exact frame index
                    textResponseInstruction.tStart = t  # local t and not account for scr refresh
                    textResponseInstruction.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textResponseInstruction, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    textResponseInstruction.status = STARTED
                    textResponseInstruction.setAutoDraw(True)
                
                # if textResponseInstruction is active this frame...
                if textResponseInstruction.status == STARTED:
                    # update params
                    pass
                
                # *imageResponseInstruction* updates
                
                # if imageResponseInstruction is starting this frame...
                if imageResponseInstruction.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    imageResponseInstruction.frameNStart = frameN  # exact frame index
                    imageResponseInstruction.tStart = t  # local t and not account for scr refresh
                    imageResponseInstruction.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(imageResponseInstruction, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'imageResponseInstruction.started')
                    # update status
                    imageResponseInstruction.status = STARTED
                    imageResponseInstruction.setAutoDraw(True)
                
                # if imageResponseInstruction is active this frame...
                if imageResponseInstruction.status == STARTED:
                    # update params
                    pass
                # *mouseResponseInstruction* updates
                
                # if mouseResponseInstruction is starting this frame...
                if mouseResponseInstruction.status == NOT_STARTED and t >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    mouseResponseInstruction.frameNStart = frameN  # exact frame index
                    mouseResponseInstruction.tStart = t  # local t and not account for scr refresh
                    mouseResponseInstruction.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(mouseResponseInstruction, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    mouseResponseInstruction.status = STARTED
                    mouseResponseInstruction.mouseClock.reset()
                    prevButtonState = mouseResponseInstruction.getPressed()  # if button is down already this ISN'T a new click
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    responseInstruction.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in responseInstruction.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "responseInstruction" ---
            for thisComponent in responseInstruction.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for responseInstruction
            responseInstruction.tStop = globalClock.getTime(format='float')
            responseInstruction.tStopRefresh = tThisFlipGlobal
            thisExp.addData('responseInstruction.stopped', responseInstruction.tStop)
            # Run 'End Routine' code from codeResponseInstruction
            continue_button.setAutoDraw(False);
            continue_button = None
            # store data for response_instruction_loop (TrialHandler)
            # the Routine "responseInstruction" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "prepareModePractice" ---
            # create an object to store info about Routine prepareModePractice
            prepareModePractice = data.Routine(
                name='prepareModePractice',
                components=[],
            )
            prepareModePractice.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from codePrepareModePractice
            n_one_stim_practice = 3
            
            # prepare list of stimuli
            practice_stims = [10, 70, 130]
            
            # prepare timing
            
            # store start times for prepareModePractice
            prepareModePractice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            prepareModePractice.tStart = globalClock.getTime(format='float')
            prepareModePractice.status = STARTED
            thisExp.addData('prepareModePractice.started', prepareModePractice.tStart)
            prepareModePractice.maxDuration = None
            # keep track of which components have finished
            prepareModePracticeComponents = prepareModePractice.components
            for thisComponent in prepareModePractice.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "prepareModePractice" ---
            # if trial has changed, end Routine now
            if isinstance(response_instruction_loop, data.TrialHandler2) and thisResponse_instruction_loop.thisN != response_instruction_loop.thisTrial.thisN:
                continueRoutine = False
            prepareModePractice.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    prepareModePractice.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in prepareModePractice.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "prepareModePractice" ---
            for thisComponent in prepareModePractice.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for prepareModePractice
            prepareModePractice.tStop = globalClock.getTime(format='float')
            prepareModePractice.tStopRefresh = tThisFlipGlobal
            thisExp.addData('prepareModePractice.stopped', prepareModePractice.tStop)
            # the Routine "prepareModePractice" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # set up handler to look after randomisation of conditions etc
            practiceOneStim = data.TrialHandler2(
                name='practiceOneStim',
                nReps=n_one_stim_practice, 
                method='sequential', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=[None], 
                seed=None, 
            )
            thisExp.addLoop(practiceOneStim)  # add the loop to the experiment
            thisPracticeOneStim = practiceOneStim.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisPracticeOneStim.rgb)
            if thisPracticeOneStim != None:
                for paramName in thisPracticeOneStim:
                    globals()[paramName] = thisPracticeOneStim[paramName]
            
            for thisPracticeOneStim in practiceOneStim:
                currentLoop = practiceOneStim
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                # abbreviate parameter names if possible (e.g. rgb = thisPracticeOneStim.rgb)
                if thisPracticeOneStim != None:
                    for paramName in thisPracticeOneStim:
                        globals()[paramName] = thisPracticeOneStim[paramName]
                
                # --- Prepare to start Routine "displayOneStim" ---
                # create an object to store info about Routine displayOneStim
                displayOneStim = data.Routine(
                    name='displayOneStim',
                    components=[mouseOneStim, textDisplayOneStim],
                )
                displayOneStim.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from codeDisplayOneStim
                objs_display = []
                resp_obj = None
                
                # fetch the id of the stimulus
                i = practiceOneStim.thisN
                
                # now display the stimulus
                ori = practice_stims[i]
                loc = [-0.25, 0]
                stim_obj = funcDrawStim(win, ori, loc, patch_radius)
                objs_display.append(stim_obj)
                
                # now display the response
                loc = [0.25, 0]
                if instruction_mode == 'click':
                    resp_obj = funcDrawAdjustResponse(
                        win, loc, patch_radius, click_radius, True)
                elif instruction_mode == 'draw':
                    resp_obj = funcDrawDrawResponse(
                        win, loc, patch_radius, True)
                else:
                    raise NotImplementedError(f'Unknown response mode: {instruction_mode}')
                objs_display.append(resp_obj)
                
                # to add the confirm button
                continue_button = funcCreateButton(
                    win, 'continue', 
                    [0, -0.4], [0.25, 0.1])
                objs_display.append(continue_button)
                
                # to detect whether a pressed has released
                has_released = True
                
                # record mouse
                display_mouse_xs = []
                display_mouse_ys = []
                # setup some python lists for storing info about the mouseOneStim
                gotValidClick = False  # until a click is received
                textDisplayOneStim.setText('Practice: report orientation by ' + instruction_mode + 'ing')
                # store start times for displayOneStim
                displayOneStim.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                displayOneStim.tStart = globalClock.getTime(format='float')
                displayOneStim.status = STARTED
                thisExp.addData('displayOneStim.started', displayOneStim.tStart)
                displayOneStim.maxDuration = None
                # keep track of which components have finished
                displayOneStimComponents = displayOneStim.components
                for thisComponent in displayOneStim.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "displayOneStim" ---
                # if trial has changed, end Routine now
                if isinstance(practiceOneStim, data.TrialHandler2) and thisPracticeOneStim.thisN != practiceOneStim.thisTrial.thisN:
                    continueRoutine = False
                displayOneStim.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # Run 'Each Frame' code from codeDisplayOneStim
                    # read mouse click
                    # if there are objects clicked
                    left_button = mouseOneStim.getPressed()[0]
                    mouse_position = mouseOneStim.getPos()
                    mouse_time = mouseOneStim.mouseClock.getTime()
                    if left_button:
                        if continue_button.contains(mouse_position):
                            # first check if it's over the continue button
                            no_pending = resp_obj.ever_have_response
                            if no_pending:
                                continueRoutine = False
                        else:
                            if instruction_mode == 'click':
                                # only register the first click
                                if has_released:
                                    has_released = False
                                    resp_obj.register_click(mouse_position, mouse_time)
                            elif instruction_mode == 'draw':
                                # register all mouse movements
                                has_released = False
                                has_redo = resp_obj.register_mouse(
                                    mouse_position, mouse_time)
                    else:
                        # check if we should update has_released
                        has_released = True
                        if instruction_mode == 'draw':
                            resp_obj.deregister_mouse(mouse_position, mouse_time)
                    
                    # *mouseOneStim* updates
                    
                    # if mouseOneStim is starting this frame...
                    if mouseOneStim.status == NOT_STARTED and t >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        mouseOneStim.frameNStart = frameN  # exact frame index
                        mouseOneStim.tStart = t  # local t and not account for scr refresh
                        mouseOneStim.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(mouseOneStim, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        mouseOneStim.status = STARTED
                        mouseOneStim.mouseClock.reset()
                        prevButtonState = [0, 0, 0]  # if now button is down we will treat as 'new' click
                    
                    # *textDisplayOneStim* updates
                    
                    # if textDisplayOneStim is starting this frame...
                    if textDisplayOneStim.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        textDisplayOneStim.frameNStart = frameN  # exact frame index
                        textDisplayOneStim.tStart = t  # local t and not account for scr refresh
                        textDisplayOneStim.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(textDisplayOneStim, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'textDisplayOneStim.started')
                        # update status
                        textDisplayOneStim.status = STARTED
                        textDisplayOneStim.setAutoDraw(True)
                    
                    # if textDisplayOneStim is active this frame...
                    if textDisplayOneStim.status == STARTED:
                        # update params
                        pass
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        displayOneStim.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in displayOneStim.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "displayOneStim" ---
                for thisComponent in displayOneStim.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for displayOneStim
                displayOneStim.tStop = globalClock.getTime(format='float')
                displayOneStim.tStopRefresh = tThisFlipGlobal
                thisExp.addData('displayOneStim.stopped', displayOneStim.tStop)
                # Run 'End Routine' code from codeDisplayOneStim
                # reset click recording
                has_released = True
                
                # save unsaved
                if instruction_mode == 'draw':
                    # save the stroke
                    if resp_obj.has_unsaved_stroke:
                        resp_obj.save_stroke()
                    # saving drawings (if there are unsaved)
                    if resp_obj.has_unsaved_drawing:
                        resp_obj.save_drawing()
                
                # record the practice response for later use
                last_practice_response = resp_obj.compute_response()
                if (instruction_mode == 'draw') and (last_practice_response is not None):
                    # for drawing: just copy everything from last drawing
                    original_drawing = (
                        resp_obj.all_drawings_x[-1],
                        resp_obj.all_drawings_y[-1])
                    last_practice_response = original_drawing
                
                # clear all objects
                for i in range(len(objs_display)):
                    display_obj = objs_display[i]
                    display_obj.setAutoDraw(False)
                
                objs_display = []
                response_obj = None
                
                # store data for practiceOneStim (TrialHandler)
                # the Routine "displayOneStim" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                
                # --- Prepare to start Routine "feedbackOneStim" ---
                # create an object to store info about Routine feedbackOneStim
                feedbackOneStim = data.Routine(
                    name='feedbackOneStim',
                    components=[textFeedbackOneStim, mouseFeedbackOneStim],
                )
                feedbackOneStim.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from codeFeedbackOneStim
                objs_display = []
                
                display_ys = [0.16, -0.2]
                display_xs = [-0.25, 0.25]
                display_r = 0.13
                title_ys = [display_ys[0]+display_r+0.02, display_ys[1]+display_r+0.02]
                
                # draw the stimulus
                i = practiceOneStim.thisN
                ori = practice_stims[i]
                stim_loc = [0, display_ys[0]]
                stim_obj = funcDrawStim(win, ori, stim_loc, display_r)
                objs_display.append(stim_obj)
                stim_title = FeedbackText(
                    win, f'stimulus', 
                    [stim_loc[0], title_ys[0]], display_r)
                objs_display.append(stim_title)
                    
                # draw the participant's response
                resp_loc = [display_xs[0], display_ys[1]]
                resp_obj = None
                if instruction_mode == 'click':
                    resp_obj = ClickAnswerObject(
                        win, last_practice_response, resp_loc, display_r, 0.01)
                elif instruction_mode == 'draw':
                    resp_obj= DrawAnswerObject(
                        win, last_practice_response, False, resp_loc, display_r)
                objs_display.append(resp_obj)
                resp_title = FeedbackText(
                    win, 'your response', 
                    [resp_loc[0], title_ys[1]], display_r)
                objs_display.append(resp_title)
                
                # draw the correct response
                ans_loc = [display_xs[1], display_ys[1]]
                ans_obj = None
                if instruction_mode == 'click':
                    ans_obj = ClickAnswerObject(
                        win, ori, ans_loc, display_r, 0.01)
                elif instruction_mode == 'draw':
                    ans_obj = DrawAnswerObject(
                        win, ori, True, ans_loc, display_r)
                objs_display.append(ans_obj)
                ans_title = FeedbackText(
                    win, 'answer', 
                    [ans_loc[0], title_ys[1]], display_r)
                objs_display.append(ans_title)
                
                # add the continue button
                continue_button = funcCreateButton(
                    win, 'continue', pos=(0.52,0.4),
                    size=(0.25, 0.1))
                objs_display.append(continue_button)
                
                # setup some python lists for storing info about the mouseFeedbackOneStim
                gotValidClick = False  # until a click is received
                # store start times for feedbackOneStim
                feedbackOneStim.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                feedbackOneStim.tStart = globalClock.getTime(format='float')
                feedbackOneStim.status = STARTED
                thisExp.addData('feedbackOneStim.started', feedbackOneStim.tStart)
                feedbackOneStim.maxDuration = None
                # keep track of which components have finished
                feedbackOneStimComponents = feedbackOneStim.components
                for thisComponent in feedbackOneStim.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "feedbackOneStim" ---
                # if trial has changed, end Routine now
                if isinstance(practiceOneStim, data.TrialHandler2) and thisPracticeOneStim.thisN != practiceOneStim.thisTrial.thisN:
                    continueRoutine = False
                feedbackOneStim.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # Run 'Each Frame' code from codeFeedbackOneStim
                    left_button = mouseFeedbackOneStim.getPressed()[0]
                    mouse_position = mouseFeedbackOneStim.getPos()
                    if left_button and (t > 0.5) and continue_button.contains(mouse_position):
                        continueRoutine = False
                    
                    # *textFeedbackOneStim* updates
                    
                    # if textFeedbackOneStim is starting this frame...
                    if textFeedbackOneStim.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        textFeedbackOneStim.frameNStart = frameN  # exact frame index
                        textFeedbackOneStim.tStart = t  # local t and not account for scr refresh
                        textFeedbackOneStim.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(textFeedbackOneStim, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'textFeedbackOneStim.started')
                        # update status
                        textFeedbackOneStim.status = STARTED
                        textFeedbackOneStim.setAutoDraw(True)
                    
                    # if textFeedbackOneStim is active this frame...
                    if textFeedbackOneStim.status == STARTED:
                        # update params
                        pass
                    # *mouseFeedbackOneStim* updates
                    
                    # if mouseFeedbackOneStim is starting this frame...
                    if mouseFeedbackOneStim.status == NOT_STARTED and t >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        mouseFeedbackOneStim.frameNStart = frameN  # exact frame index
                        mouseFeedbackOneStim.tStart = t  # local t and not account for scr refresh
                        mouseFeedbackOneStim.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(mouseFeedbackOneStim, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        mouseFeedbackOneStim.status = STARTED
                        mouseFeedbackOneStim.mouseClock.reset()
                        prevButtonState = mouseFeedbackOneStim.getPressed()  # if button is down already this ISN'T a new click
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        feedbackOneStim.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in feedbackOneStim.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "feedbackOneStim" ---
                for thisComponent in feedbackOneStim.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for feedbackOneStim
                feedbackOneStim.tStop = globalClock.getTime(format='float')
                feedbackOneStim.tStopRefresh = tThisFlipGlobal
                thisExp.addData('feedbackOneStim.stopped', feedbackOneStim.tStop)
                # Run 'End Routine' code from codeFeedbackOneStim
                # clear all objects
                for i in range(len(objs_display)):
                    display_obj = objs_display[i]
                    display_obj.setAutoDraw(False)
                    
                objs_display = []
                # store data for practiceOneStim (TrialHandler)
                # the Routine "feedbackOneStim" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
            # completed n_one_stim_practice repeats of 'practiceOneStim'
            
        # completed n_response_modes_allowed repeats of 'response_instruction_loop'
        
        
        # --- Prepare to start Routine "prepareAllPractice" ---
        # create an object to store info about Routine prepareAllPractice
        prepareAllPractice = data.Routine(
            name='prepareAllPractice',
            components=[],
        )
        prepareAllPractice.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for prepareAllPractice
        prepareAllPractice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prepareAllPractice.tStart = globalClock.getTime(format='float')
        prepareAllPractice.status = STARTED
        thisExp.addData('prepareAllPractice.started', prepareAllPractice.tStart)
        prepareAllPractice.maxDuration = None
        # keep track of which components have finished
        prepareAllPracticeComponents = prepareAllPractice.components
        for thisComponent in prepareAllPractice.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "prepareAllPractice" ---
        # if trial has changed, end Routine now
        if isinstance(DEBUG, data.TrialHandler2) and thisDEBUG.thisN != DEBUG.thisTrial.thisN:
            continueRoutine = False
        prepareAllPractice.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                prepareAllPractice.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prepareAllPractice.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prepareAllPractice" ---
        for thisComponent in prepareAllPractice.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prepareAllPractice
        prepareAllPractice.tStop = globalClock.getTime(format='float')
        prepareAllPractice.tStopRefresh = tThisFlipGlobal
        thisExp.addData('prepareAllPractice.stopped', prepareAllPractice.tStop)
        # the Routine "prepareAllPractice" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "startColorRulePractice" ---
        # create an object to store info about Routine startColorRulePractice
        startColorRulePractice = data.Routine(
            name='startColorRulePractice',
            components=[mouseStartColorRule, textStartColorRule],
        )
        startColorRulePractice.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from codeStartColorRule
        continue_button = funcCreateButton(
            win, 'continue', 
            [0, -0.4], [0.25, 0.1])
        # setup some python lists for storing info about the mouseStartColorRule
        gotValidClick = False  # until a click is received
        # store start times for startColorRulePractice
        startColorRulePractice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        startColorRulePractice.tStart = globalClock.getTime(format='float')
        startColorRulePractice.status = STARTED
        thisExp.addData('startColorRulePractice.started', startColorRulePractice.tStart)
        startColorRulePractice.maxDuration = None
        # keep track of which components have finished
        startColorRulePracticeComponents = startColorRulePractice.components
        for thisComponent in startColorRulePractice.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "startColorRulePractice" ---
        # if trial has changed, end Routine now
        if isinstance(DEBUG, data.TrialHandler2) and thisDEBUG.thisN != DEBUG.thisTrial.thisN:
            continueRoutine = False
        startColorRulePractice.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from codeStartColorRule
            left_button = mouseStartColorRule.getPressed()[0]
            mouse_position = mouseStartColorRule.getPos()
            if left_button and (t > 0.5) and continue_button.contains(mouse_position):
                continueRoutine = False
            
            # *mouseStartColorRule* updates
            
            # if mouseStartColorRule is starting this frame...
            if mouseStartColorRule.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouseStartColorRule.frameNStart = frameN  # exact frame index
                mouseStartColorRule.tStart = t  # local t and not account for scr refresh
                mouseStartColorRule.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouseStartColorRule, 'tStartRefresh')  # time at next scr refresh
                # update status
                mouseStartColorRule.status = STARTED
                mouseStartColorRule.mouseClock.reset()
                prevButtonState = mouseStartColorRule.getPressed()  # if button is down already this ISN'T a new click
            
            # *textStartColorRule* updates
            
            # if textStartColorRule is starting this frame...
            if textStartColorRule.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textStartColorRule.frameNStart = frameN  # exact frame index
                textStartColorRule.tStart = t  # local t and not account for scr refresh
                textStartColorRule.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textStartColorRule, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textStartColorRule.started')
                # update status
                textStartColorRule.status = STARTED
                textStartColorRule.setAutoDraw(True)
            
            # if textStartColorRule is active this frame...
            if textStartColorRule.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                startColorRulePractice.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in startColorRulePractice.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "startColorRulePractice" ---
        for thisComponent in startColorRulePractice.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for startColorRulePractice
        startColorRulePractice.tStop = globalClock.getTime(format='float')
        startColorRulePractice.tStopRefresh = tThisFlipGlobal
        thisExp.addData('startColorRulePractice.stopped', startColorRulePractice.tStop)
        # Run 'End Routine' code from codeStartColorRule
        continue_button.setAutoDraw(False);
        continue_button = None
        # store data for DEBUG (TrialHandler)
        # the Routine "startColorRulePractice" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "overviewColorRule" ---
        # create an object to store info about Routine overviewColorRule
        overviewColorRule = data.Routine(
            name='overviewColorRule',
            components=[imageOverviewInstruction, mouseOverviewInstruction],
        )
        overviewColorRule.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from codeOverviewInstruction
        continue_button = funcCreateButton(
            win, 'continue', 
            [0, -0.4], [0.25, 0.1])
        # setup some python lists for storing info about the mouseOverviewInstruction
        gotValidClick = False  # until a click is received
        # store start times for overviewColorRule
        overviewColorRule.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        overviewColorRule.tStart = globalClock.getTime(format='float')
        overviewColorRule.status = STARTED
        thisExp.addData('overviewColorRule.started', overviewColorRule.tStart)
        overviewColorRule.maxDuration = None
        # keep track of which components have finished
        overviewColorRuleComponents = overviewColorRule.components
        for thisComponent in overviewColorRule.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "overviewColorRule" ---
        # if trial has changed, end Routine now
        if isinstance(DEBUG, data.TrialHandler2) and thisDEBUG.thisN != DEBUG.thisTrial.thisN:
            continueRoutine = False
        overviewColorRule.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from codeOverviewInstruction
            left_button = mouseOverviewInstruction.getPressed()[0]
            mouse_position = mouseOverviewInstruction.getPos()
            if left_button and (t > 0.5) and continue_button.contains(mouse_position):
                continueRoutine = False
            
            
            # *imageOverviewInstruction* updates
            
            # if imageOverviewInstruction is starting this frame...
            if imageOverviewInstruction.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                imageOverviewInstruction.frameNStart = frameN  # exact frame index
                imageOverviewInstruction.tStart = t  # local t and not account for scr refresh
                imageOverviewInstruction.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(imageOverviewInstruction, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'imageOverviewInstruction.started')
                # update status
                imageOverviewInstruction.status = STARTED
                imageOverviewInstruction.setAutoDraw(True)
            
            # if imageOverviewInstruction is active this frame...
            if imageOverviewInstruction.status == STARTED:
                # update params
                pass
            # *mouseOverviewInstruction* updates
            
            # if mouseOverviewInstruction is starting this frame...
            if mouseOverviewInstruction.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouseOverviewInstruction.frameNStart = frameN  # exact frame index
                mouseOverviewInstruction.tStart = t  # local t and not account for scr refresh
                mouseOverviewInstruction.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouseOverviewInstruction, 'tStartRefresh')  # time at next scr refresh
                # update status
                mouseOverviewInstruction.status = STARTED
                mouseOverviewInstruction.mouseClock.reset()
                prevButtonState = mouseOverviewInstruction.getPressed()  # if button is down already this ISN'T a new click
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                overviewColorRule.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in overviewColorRule.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "overviewColorRule" ---
        for thisComponent in overviewColorRule.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for overviewColorRule
        overviewColorRule.tStop = globalClock.getTime(format='float')
        overviewColorRule.tStopRefresh = tThisFlipGlobal
        thisExp.addData('overviewColorRule.stopped', overviewColorRule.tStop)
        # Run 'End Routine' code from codeOverviewInstruction
        continue_button.setAutoDraw(False);
        continue_button = None
        # store data for DEBUG (TrialHandler)
        # the Routine "overviewColorRule" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        practiceColorRuleLoop = data.TrialHandler2(
            name='practiceColorRuleLoop',
            nReps=n_response_modes_allowed, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(practiceColorRuleLoop)  # add the loop to the experiment
        thisPracticeColorRuleLoop = practiceColorRuleLoop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisPracticeColorRuleLoop.rgb)
        if thisPracticeColorRuleLoop != None:
            for paramName in thisPracticeColorRuleLoop:
                globals()[paramName] = thisPracticeColorRuleLoop[paramName]
        
        for thisPracticeColorRuleLoop in practiceColorRuleLoop:
            currentLoop = practiceColorRuleLoop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # abbreviate parameter names if possible (e.g. rgb = thisPracticeColorRuleLoop.rgb)
            if thisPracticeColorRuleLoop != None:
                for paramName in thisPracticeColorRuleLoop:
                    globals()[paramName] = thisPracticeColorRuleLoop[paramName]
            
            # --- Prepare to start Routine "prepareColorRulePractice" ---
            # create an object to store info about Routine prepareColorRulePractice
            prepareColorRulePractice = data.Routine(
                name='prepareColorRulePractice',
                components=[textPrepareColorRulePractice, mousePrepareColorRulePractice],
            )
            prepareColorRulePractice.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from codePrepareColorRulePractice
            # response mode
            response_mode = response_modes_allowed[practiceColorRuleLoop.thisN]
            
            # initialize stims
            all_practice_stims = [
                [20, 110],
                [60, 150],
                [100, 10],
                [140, 50],
            ]
            # initialize color codes
            all_color_codes = [
                [1, 2],
                [2, 1],
                [2, 1],
                [1, 2],
            ]
            
            # initialize response regions
            all_report_xs = [
                [-0.25, 0.25],
                [0, 0],
                [0.25, -0.25],
                [0, 0],
            ]
            all_report_ys = [
                [0.03, 0.03],
                [0.3, -0.24],
                [0.03, 0.03],
                [-0.24, 0.3],
            ]
            
            # initialize timing
            delay_lengths = [1.5, 5.5]
            
            # make it slower for practice
            slow_down_factor = 1.5
            practice_display_time = display_time * slow_down_factor
            practice_display_cue_onset_time = display_cue_onset_time * slow_down_factor
            
            # provide the button
            continue_button = funcCreateButton(
                win, 'continue', 
                [0, -0.4], [0.25, 0.1])
            
            textPrepareColorRulePractice.setText('In the following trials, we will practice reporting two sequentially presented stimuli (by ' + response_mode + 'ing)')
            # setup some python lists for storing info about the mousePrepareColorRulePractice
            gotValidClick = False  # until a click is received
            # store start times for prepareColorRulePractice
            prepareColorRulePractice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            prepareColorRulePractice.tStart = globalClock.getTime(format='float')
            prepareColorRulePractice.status = STARTED
            thisExp.addData('prepareColorRulePractice.started', prepareColorRulePractice.tStart)
            prepareColorRulePractice.maxDuration = None
            # keep track of which components have finished
            prepareColorRulePracticeComponents = prepareColorRulePractice.components
            for thisComponent in prepareColorRulePractice.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "prepareColorRulePractice" ---
            # if trial has changed, end Routine now
            if isinstance(practiceColorRuleLoop, data.TrialHandler2) and thisPracticeColorRuleLoop.thisN != practiceColorRuleLoop.thisTrial.thisN:
                continueRoutine = False
            prepareColorRulePractice.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from codePrepareColorRulePractice
                left_button = mousePrepareColorRulePractice.getPressed()[0]
                mouse_position = mousePrepareColorRulePractice.getPos()
                if left_button and (t > 0.5) and continue_button.contains(mouse_position):
                    continueRoutine = False
                
                
                # *textPrepareColorRulePractice* updates
                
                # if textPrepareColorRulePractice is starting this frame...
                if textPrepareColorRulePractice.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textPrepareColorRulePractice.frameNStart = frameN  # exact frame index
                    textPrepareColorRulePractice.tStart = t  # local t and not account for scr refresh
                    textPrepareColorRulePractice.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textPrepareColorRulePractice, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textPrepareColorRulePractice.started')
                    # update status
                    textPrepareColorRulePractice.status = STARTED
                    textPrepareColorRulePractice.setAutoDraw(True)
                
                # if textPrepareColorRulePractice is active this frame...
                if textPrepareColorRulePractice.status == STARTED:
                    # update params
                    pass
                # *mousePrepareColorRulePractice* updates
                
                # if mousePrepareColorRulePractice is starting this frame...
                if mousePrepareColorRulePractice.status == NOT_STARTED and t >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    mousePrepareColorRulePractice.frameNStart = frameN  # exact frame index
                    mousePrepareColorRulePractice.tStart = t  # local t and not account for scr refresh
                    mousePrepareColorRulePractice.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(mousePrepareColorRulePractice, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    mousePrepareColorRulePractice.status = STARTED
                    mousePrepareColorRulePractice.mouseClock.reset()
                    prevButtonState = mousePrepareColorRulePractice.getPressed()  # if button is down already this ISN'T a new click
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    prepareColorRulePractice.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in prepareColorRulePractice.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "prepareColorRulePractice" ---
            for thisComponent in prepareColorRulePractice.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for prepareColorRulePractice
            prepareColorRulePractice.tStop = globalClock.getTime(format='float')
            prepareColorRulePractice.tStopRefresh = tThisFlipGlobal
            thisExp.addData('prepareColorRulePractice.stopped', prepareColorRulePractice.tStop)
            # Run 'End Routine' code from codePrepareColorRulePractice
            continue_button.setAutoDraw(False)
            continue_button = None
            # store data for practiceColorRuleLoop (TrialHandler)
            # the Routine "prepareColorRulePractice" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # set up handler to look after randomisation of conditions etc
            ColorRulePracticeTrials = data.TrialHandler2(
                name='ColorRulePracticeTrials',
                nReps=n_color_rule_practice , 
                method='sequential', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=[None], 
                seed=None, 
            )
            thisExp.addLoop(ColorRulePracticeTrials)  # add the loop to the experiment
            thisColorRulePracticeTrial = ColorRulePracticeTrials.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisColorRulePracticeTrial.rgb)
            if thisColorRulePracticeTrial != None:
                for paramName in thisColorRulePracticeTrial:
                    globals()[paramName] = thisColorRulePracticeTrial[paramName]
            
            for thisColorRulePracticeTrial in ColorRulePracticeTrials:
                currentLoop = ColorRulePracticeTrials
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                # abbreviate parameter names if possible (e.g. rgb = thisColorRulePracticeTrial.rgb)
                if thisColorRulePracticeTrial != None:
                    for paramName in thisColorRulePracticeTrial:
                        globals()[paramName] = thisColorRulePracticeTrial[paramName]
                
                # set up handler to look after randomisation of conditions etc
                colorRuleStimDisplayLoop = data.TrialHandler2(
                    name='colorRuleStimDisplayLoop',
                    nReps=2.0, 
                    method='sequential', 
                    extraInfo=expInfo, 
                    originPath=-1, 
                    trialList=[None], 
                    seed=None, 
                )
                thisExp.addLoop(colorRuleStimDisplayLoop)  # add the loop to the experiment
                thisColorRuleStimDisplayLoop = colorRuleStimDisplayLoop.trialList[0]  # so we can initialise stimuli with some values
                # abbreviate parameter names if possible (e.g. rgb = thisColorRuleStimDisplayLoop.rgb)
                if thisColorRuleStimDisplayLoop != None:
                    for paramName in thisColorRuleStimDisplayLoop:
                        globals()[paramName] = thisColorRuleStimDisplayLoop[paramName]
                
                for thisColorRuleStimDisplayLoop in colorRuleStimDisplayLoop:
                    currentLoop = colorRuleStimDisplayLoop
                    thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                    # abbreviate parameter names if possible (e.g. rgb = thisColorRuleStimDisplayLoop.rgb)
                    if thisColorRuleStimDisplayLoop != None:
                        for paramName in thisColorRuleStimDisplayLoop:
                            globals()[paramName] = thisColorRuleStimDisplayLoop[paramName]
                    
                    # --- Prepare to start Routine "colorRuleStim" ---
                    # create an object to store info about Routine colorRuleStim
                    colorRuleStim = data.Routine(
                        name='colorRuleStim',
                        components=[textColorRuleStim],
                    )
                    colorRuleStim.status = NOT_STARTED
                    continueRoutine = True
                    # update component parameters for each repeat
                    # Run 'Begin Routine' code from codeColorRuleStim
                    objs_display = []
                    
                    # fetch the id of the stimulus
                    color_rule_i = ColorRulePracticeTrials.thisN
                    color_rule_j = colorRuleStimDisplayLoop.thisN
                    
                    # now display the stimulus
                    ori = all_practice_stims[color_rule_i][color_rule_j]
                    loc = [0, 0]
                    stim_obj = funcDrawStim(win, ori, loc, patch_radius)
                    objs_display.append(stim_obj)
                    
                    # to detect whether a pressed has released
                    has_released = True
                    
                    # to mark if the cue has displayed
                    has_cue_shown = False
                    
                    textColorRuleStim.setText('')
                    # store start times for colorRuleStim
                    colorRuleStim.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                    colorRuleStim.tStart = globalClock.getTime(format='float')
                    colorRuleStim.status = STARTED
                    thisExp.addData('colorRuleStim.started', colorRuleStim.tStart)
                    colorRuleStim.maxDuration = None
                    # keep track of which components have finished
                    colorRuleStimComponents = colorRuleStim.components
                    for thisComponent in colorRuleStim.components:
                        thisComponent.tStart = None
                        thisComponent.tStop = None
                        thisComponent.tStartRefresh = None
                        thisComponent.tStopRefresh = None
                        if hasattr(thisComponent, 'status'):
                            thisComponent.status = NOT_STARTED
                    # reset timers
                    t = 0
                    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                    frameN = -1
                    
                    # --- Run Routine "colorRuleStim" ---
                    # if trial has changed, end Routine now
                    if isinstance(colorRuleStimDisplayLoop, data.TrialHandler2) and thisColorRuleStimDisplayLoop.thisN != colorRuleStimDisplayLoop.thisTrial.thisN:
                        continueRoutine = False
                    colorRuleStim.forceEnded = routineForceEnded = not continueRoutine
                    while continueRoutine:
                        # get current time
                        t = routineTimer.getTime()
                        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                        # update/draw components on each frame
                        # Run 'Each Frame' code from codeColorRuleStim
                        all_color_codes
                        
                        # display the cue when onset
                        if (not has_cue_shown) and t >= practice_display_cue_onset_time:
                            cue_loc = [0, 0]
                            cue_color = all_color_codes[color_rule_i][color_rule_j]
                            cue_code = 99 # default
                            report_cue = SingleCueObject(
                                win, cue_loc, cue_code, patch_radius*0.6, cue_color)
                            objs_display.append(report_cue)
                            has_cue_shown = True
                        
                        # end routine when time up
                        if t > practice_display_time:
                            continueRoutine = False
                        
                        
                        # *textColorRuleStim* updates
                        
                        # if textColorRuleStim is starting this frame...
                        if textColorRuleStim.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            textColorRuleStim.frameNStart = frameN  # exact frame index
                            textColorRuleStim.tStart = t  # local t and not account for scr refresh
                            textColorRuleStim.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(textColorRuleStim, 'tStartRefresh')  # time at next scr refresh
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'textColorRuleStim.started')
                            # update status
                            textColorRuleStim.status = STARTED
                            textColorRuleStim.setAutoDraw(True)
                        
                        # if textColorRuleStim is active this frame...
                        if textColorRuleStim.status == STARTED:
                            # update params
                            pass
                        
                        # check for quit (typically the Esc key)
                        if defaultKeyboard.getKeys(keyList=["escape"]):
                            thisExp.status = FINISHED
                        if thisExp.status == FINISHED or endExpNow:
                            endExperiment(thisExp, win=win)
                            return
                        # pause experiment here if requested
                        if thisExp.status == PAUSED:
                            pauseExperiment(
                                thisExp=thisExp, 
                                win=win, 
                                timers=[routineTimer], 
                                playbackComponents=[]
                            )
                            # skip the frame we paused on
                            continue
                        
                        # check if all components have finished
                        if not continueRoutine:  # a component has requested a forced-end of Routine
                            colorRuleStim.forceEnded = routineForceEnded = True
                            break
                        continueRoutine = False  # will revert to True if at least one component still running
                        for thisComponent in colorRuleStim.components:
                            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                continueRoutine = True
                                break  # at least one component has not yet finished
                        
                        # refresh the screen
                        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                            win.flip()
                    
                    # --- Ending Routine "colorRuleStim" ---
                    for thisComponent in colorRuleStim.components:
                        if hasattr(thisComponent, "setAutoDraw"):
                            thisComponent.setAutoDraw(False)
                    # store stop times for colorRuleStim
                    colorRuleStim.tStop = globalClock.getTime(format='float')
                    colorRuleStim.tStopRefresh = tThisFlipGlobal
                    thisExp.addData('colorRuleStim.stopped', colorRuleStim.tStop)
                    # Run 'End Routine' code from codeColorRuleStim
                    # reset click recording
                    has_released = True
                    
                    # don't record response
                    # thisExp.addData(f'mouseStim.x_{seq_display_loop.thisN+1}', display_mouse_xs)
                    # thisExp.addData(f'mouseStim.y_{seq_display_loop.thisN+1}', display_mouse_ys)
                    
                    # clear all objects
                    for i in range(len(objs_display)):
                        display_obj = objs_display[i]
                        display_obj.setAutoDraw(False)
                    
                    objs_display = []
                    
                    # the Routine "colorRuleStim" was not non-slip safe, so reset the non-slip timer
                    routineTimer.reset()
                    
                    # --- Prepare to start Routine "colorRuleDelay" ---
                    # create an object to store info about Routine colorRuleDelay
                    colorRuleDelay = data.Routine(
                        name='colorRuleDelay',
                        components=[textColorRuleDelay],
                    )
                    colorRuleDelay.status = NOT_STARTED
                    continueRoutine = True
                    # update component parameters for each repeat
                    # Run 'Begin Routine' code from codeColorRuleDelay
                    objs_display = []
                    has_noise_patch_displaying = False
                    
                    # decide whether to display noise patch
                    to_display_noise_patch =  random.random() < noise_patch_display_chance
                    if to_display_noise_patch:
                        loc = [0, 0]
                        stim_obj = NoisePatch(win, loc, patch_radius)
                        objs_display.append(stim_obj)
                        has_noise_patch_displaying = True
                    
                    textColorRuleDelay.setText('')
                    # store start times for colorRuleDelay
                    colorRuleDelay.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                    colorRuleDelay.tStart = globalClock.getTime(format='float')
                    colorRuleDelay.status = STARTED
                    thisExp.addData('colorRuleDelay.started', colorRuleDelay.tStart)
                    colorRuleDelay.maxDuration = None
                    # keep track of which components have finished
                    colorRuleDelayComponents = colorRuleDelay.components
                    for thisComponent in colorRuleDelay.components:
                        thisComponent.tStart = None
                        thisComponent.tStop = None
                        thisComponent.tStartRefresh = None
                        thisComponent.tStopRefresh = None
                        if hasattr(thisComponent, 'status'):
                            thisComponent.status = NOT_STARTED
                    # reset timers
                    t = 0
                    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                    frameN = -1
                    
                    # --- Run Routine "colorRuleDelay" ---
                    # if trial has changed, end Routine now
                    if isinstance(colorRuleStimDisplayLoop, data.TrialHandler2) and thisColorRuleStimDisplayLoop.thisN != colorRuleStimDisplayLoop.thisTrial.thisN:
                        continueRoutine = False
                    colorRuleDelay.forceEnded = routineForceEnded = not continueRoutine
                    while continueRoutine:
                        # get current time
                        t = routineTimer.getTime()
                        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                        # update/draw components on each frame
                        # Run 'Each Frame' code from codeColorRuleDelay
                        # end routine when time up
                        if t > delay_post_stim:
                            continueRoutine = False
                        
                        # stop displaying all noise patches
                        if has_noise_patch_displaying:
                            if t > noise_patch_display_time:
                                for i in range(len(objs_display)):
                                    display_obj = objs_display[i]
                                    display_obj.setAutoDraw(False)
                                has_noise_patch_displaying = False
                                
                                # display the fixation
                                objs_display = []
                                fixation = DefaultFixation(win)
                                objs_display.append(fixation)
                        
                        
                        # *textColorRuleDelay* updates
                        
                        # if textColorRuleDelay is starting this frame...
                        if textColorRuleDelay.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            textColorRuleDelay.frameNStart = frameN  # exact frame index
                            textColorRuleDelay.tStart = t  # local t and not account for scr refresh
                            textColorRuleDelay.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(textColorRuleDelay, 'tStartRefresh')  # time at next scr refresh
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'textColorRuleDelay.started')
                            # update status
                            textColorRuleDelay.status = STARTED
                            textColorRuleDelay.setAutoDraw(True)
                        
                        # if textColorRuleDelay is active this frame...
                        if textColorRuleDelay.status == STARTED:
                            # update params
                            pass
                        
                        # check for quit (typically the Esc key)
                        if defaultKeyboard.getKeys(keyList=["escape"]):
                            thisExp.status = FINISHED
                        if thisExp.status == FINISHED or endExpNow:
                            endExperiment(thisExp, win=win)
                            return
                        # pause experiment here if requested
                        if thisExp.status == PAUSED:
                            pauseExperiment(
                                thisExp=thisExp, 
                                win=win, 
                                timers=[routineTimer], 
                                playbackComponents=[]
                            )
                            # skip the frame we paused on
                            continue
                        
                        # check if all components have finished
                        if not continueRoutine:  # a component has requested a forced-end of Routine
                            colorRuleDelay.forceEnded = routineForceEnded = True
                            break
                        continueRoutine = False  # will revert to True if at least one component still running
                        for thisComponent in colorRuleDelay.components:
                            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                continueRoutine = True
                                break  # at least one component has not yet finished
                        
                        # refresh the screen
                        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                            win.flip()
                    
                    # --- Ending Routine "colorRuleDelay" ---
                    for thisComponent in colorRuleDelay.components:
                        if hasattr(thisComponent, "setAutoDraw"):
                            thisComponent.setAutoDraw(False)
                    # store stop times for colorRuleDelay
                    colorRuleDelay.tStop = globalClock.getTime(format='float')
                    colorRuleDelay.tStopRefresh = tThisFlipGlobal
                    thisExp.addData('colorRuleDelay.stopped', colorRuleDelay.tStop)
                    # Run 'End Routine' code from codeColorRuleDelay
                    # no recording
                    # thisExp.addData(f'mousePostStim.x_{seq_display_loop.thisN+1}', post_display_mouse_xs)
                    # thisExp.addData(f'mousePostStim.y_{seq_display_loop.thisN+1}', post_display_mouse_ys)
                    
                    # clear all objects
                    for i in range(len(objs_display)):
                        display_obj = objs_display[i]
                        display_obj.setAutoDraw(False)
                    
                    objs_display = []
                    
                    # the Routine "colorRuleDelay" was not non-slip safe, so reset the non-slip timer
                    routineTimer.reset()
                # completed 2.0 repeats of 'colorRuleStimDisplayLoop'
                
                
                # --- Prepare to start Routine "colorRuleResponse" ---
                # create an object to store info about Routine colorRuleResponse
                colorRuleResponse = data.Routine(
                    name='colorRuleResponse',
                    components=[mouseColorRuleResponse],
                )
                colorRuleResponse.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from codeColorRuleResp
                objs_display = []
                response_objects = []
                for i in range(2):
                    loc = [
                        all_report_xs[color_rule_i][i],
                        all_report_ys[color_rule_i][i],
                    ]
                    to_report = True # default; always report it
                    response_obj = None
                    if response_mode == 'click':
                        response_obj = funcDrawAdjustResponse(
                            win, loc, patch_radius, click_radius, to_report)
                    elif response_mode == 'draw':
                        response_obj = funcDrawDrawResponse(
                            win, loc, patch_radius, to_report)
                    else:
                        raise NotImplementedError(f'Unknown response mode: {response_mode}')
                    objs_display.append(response_obj)
                    response_objects.append(response_obj)
                    
                    # give order color cue
                    color_code = all_color_codes[color_rule_i][i]
                    report_cue = SingleCueObject(
                        win, loc, 99, patch_radius*0.6, color_code)
                    objs_display.append(report_cue)
                
                # to add the confirm button
                continue_button = None
                if len(response_objects) > 0:
                    continue_button = funcCreateButton(
                        win, 'continue', pos=(0, 0),
                        size=(button_width*1.3, button_height*1.3))
                    objs_display.append(continue_button)
                
                # to detect whether a pressed has released
                has_released = True
                # setup some python lists for storing info about the mouseColorRuleResponse
                gotValidClick = False  # until a click is received
                # store start times for colorRuleResponse
                colorRuleResponse.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                colorRuleResponse.tStart = globalClock.getTime(format='float')
                colorRuleResponse.status = STARTED
                thisExp.addData('colorRuleResponse.started', colorRuleResponse.tStart)
                colorRuleResponse.maxDuration = None
                # keep track of which components have finished
                colorRuleResponseComponents = colorRuleResponse.components
                for thisComponent in colorRuleResponse.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "colorRuleResponse" ---
                # if trial has changed, end Routine now
                if isinstance(ColorRulePracticeTrials, data.TrialHandler2) and thisColorRulePracticeTrial.thisN != ColorRulePracticeTrials.thisTrial.thisN:
                    continueRoutine = False
                colorRuleResponse.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # Run 'Each Frame' code from codeColorRuleResp
                    # if there are objects clicked
                    left_button = mouseColorRuleResponse.getPressed()[0]
                    mouse_position = mouseColorRuleResponse.getPos()
                    mouse_time = t
                    if left_button:
                        if continue_button.contains(mouse_position):
                            # first check if it's over the continue button
                            no_pending = funcCheckNoPending(response_objects)
                            if no_pending:
                                continueRoutine = False
                        else:
                            if response_mode == 'click':
                                # only register the first click
                                if has_released:
                                    # this is a new press, register it
                                    has_released = False
                                    # otherwise, ignore it
                                    for i in range(len(response_objects)):
                                        # update each position with the click
                                        resp_obj = response_objects[i]
                                        resp_obj.register_click(mouse_position, mouse_time)
                            elif response_mode == 'draw':
                                # register all mouse movements
                                has_released = False
                                for i in range(len(response_objects)):
                                    # update each position with the click
                                    resp_obj = response_objects[i]
                                    has_redo = resp_obj.register_mouse(
                                        mouse_position, mouse_time)
                                        
                    else:
                        # check if we should update has_released
                        has_released = True
                        if response_mode == 'draw':
                            for i in range(len(response_objects)):
                                # update each position with the click
                                resp_obj = response_objects[i]
                                resp_obj.deregister_mouse(mouse_position, mouse_time)
                    
                    # *mouseColorRuleResponse* updates
                    
                    # if mouseColorRuleResponse is starting this frame...
                    if mouseColorRuleResponse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        mouseColorRuleResponse.frameNStart = frameN  # exact frame index
                        mouseColorRuleResponse.tStart = t  # local t and not account for scr refresh
                        mouseColorRuleResponse.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(mouseColorRuleResponse, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        mouseColorRuleResponse.status = STARTED
                        mouseColorRuleResponse.mouseClock.reset()
                        prevButtonState = mouseColorRuleResponse.getPressed()  # if button is down already this ISN'T a new click
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        colorRuleResponse.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in colorRuleResponse.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "colorRuleResponse" ---
                for thisComponent in colorRuleResponse.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for colorRuleResponse
                colorRuleResponse.tStop = globalClock.getTime(format='float')
                colorRuleResponse.tStopRefresh = tThisFlipGlobal
                thisExp.addData('colorRuleResponse.stopped', colorRuleResponse.tStop)
                # Run 'End Routine' code from codeColorRuleResp
                # reset click recording
                has_released = True
                
                practice_responses = []
                
                # save unsaved
                for i in range(2):
                    resp_obj = response_objects[i]
                    if response_mode == 'draw':
                        # save the stroke
                        if resp_obj.has_unsaved_stroke:
                            resp_obj.save_stroke()
                        # saving drawings (if there are unsaved)
                        if resp_obj.has_unsaved_drawing:
                            resp_obj.save_drawing()
                
                    # record the practice response for later use
                    last_practice_response = resp_obj.compute_response()
                    if (response_mode == 'draw') and (last_practice_response is not None):
                        # for drawing: just copy everything from last drawing
                        original_drawing = (
                            resp_obj.all_drawings_x[-1],
                            resp_obj.all_drawings_y[-1])
                        last_practice_response = original_drawing
                        
                    practice_responses.append(last_practice_response)
                
                # clear all objects
                for i in range(len(objs_display)):
                    display_obj = objs_display[i]
                    display_obj.setAutoDraw(False)
                
                objs_display = []
                response_objects = []
                ####################
                
                # store data for ColorRulePracticeTrials (TrialHandler)
                # the Routine "colorRuleResponse" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                
                # --- Prepare to start Routine "colorRuleFeedback" ---
                # create an object to store info about Routine colorRuleFeedback
                colorRuleFeedback = data.Routine(
                    name='colorRuleFeedback',
                    components=[textColorRuleFeedback, mouseColorRuleFeedback],
                )
                colorRuleFeedback.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from codeColorRuleFeedback
                objs_display = []
                
                resp_locs = [
                    [all_report_xs[color_rule_i][0], all_report_ys[color_rule_i][0]],
                    [all_report_xs[color_rule_i][1], all_report_ys[color_rule_i][1]],
                ]
                feeaback_objects = CreateWholePracticeFeedback(
                        oris=all_practice_stims[color_rule_i], 
                        responses=practice_responses, 
                        resp_locs=resp_locs,
                        colors=all_color_codes[color_rule_i], 
                        cue_codes=[99, 99], 
                        report_codes=[True, True],
                        mode=response_mode)
                objs_display += feeaback_objects
                
                # add the continue button
                continue_button = funcCreateButton(
                    win, 'continue', pos=(0.52,0.42),
                    size=(0.25, 0.1))
                objs_display.append(continue_button)
                
                # setup some python lists for storing info about the mouseColorRuleFeedback
                gotValidClick = False  # until a click is received
                # store start times for colorRuleFeedback
                colorRuleFeedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                colorRuleFeedback.tStart = globalClock.getTime(format='float')
                colorRuleFeedback.status = STARTED
                thisExp.addData('colorRuleFeedback.started', colorRuleFeedback.tStart)
                colorRuleFeedback.maxDuration = None
                # keep track of which components have finished
                colorRuleFeedbackComponents = colorRuleFeedback.components
                for thisComponent in colorRuleFeedback.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "colorRuleFeedback" ---
                # if trial has changed, end Routine now
                if isinstance(ColorRulePracticeTrials, data.TrialHandler2) and thisColorRulePracticeTrial.thisN != ColorRulePracticeTrials.thisTrial.thisN:
                    continueRoutine = False
                colorRuleFeedback.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # Run 'Each Frame' code from codeColorRuleFeedback
                    left_button = mouseColorRuleFeedback.getPressed()[0]
                    mouse_position = mouseColorRuleFeedback.getPos()
                    if left_button and (t > 0.5) and continue_button.contains(mouse_position):
                        continueRoutine = False
                    
                    # *textColorRuleFeedback* updates
                    
                    # if textColorRuleFeedback is starting this frame...
                    if textColorRuleFeedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        textColorRuleFeedback.frameNStart = frameN  # exact frame index
                        textColorRuleFeedback.tStart = t  # local t and not account for scr refresh
                        textColorRuleFeedback.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(textColorRuleFeedback, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'textColorRuleFeedback.started')
                        # update status
                        textColorRuleFeedback.status = STARTED
                        textColorRuleFeedback.setAutoDraw(True)
                    
                    # if textColorRuleFeedback is active this frame...
                    if textColorRuleFeedback.status == STARTED:
                        # update params
                        pass
                    # *mouseColorRuleFeedback* updates
                    
                    # if mouseColorRuleFeedback is starting this frame...
                    if mouseColorRuleFeedback.status == NOT_STARTED and t >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        mouseColorRuleFeedback.frameNStart = frameN  # exact frame index
                        mouseColorRuleFeedback.tStart = t  # local t and not account for scr refresh
                        mouseColorRuleFeedback.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(mouseColorRuleFeedback, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        mouseColorRuleFeedback.status = STARTED
                        mouseColorRuleFeedback.mouseClock.reset()
                        prevButtonState = mouseColorRuleFeedback.getPressed()  # if button is down already this ISN'T a new click
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        colorRuleFeedback.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in colorRuleFeedback.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "colorRuleFeedback" ---
                for thisComponent in colorRuleFeedback.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for colorRuleFeedback
                colorRuleFeedback.tStop = globalClock.getTime(format='float')
                colorRuleFeedback.tStopRefresh = tThisFlipGlobal
                thisExp.addData('colorRuleFeedback.stopped', colorRuleFeedback.tStop)
                # Run 'End Routine' code from codeColorRuleFeedback
                # clear all objects
                for i in range(len(objs_display)):
                    display_obj = objs_display[i]
                    display_obj.setAutoDraw(False)
                    
                objs_display = []
                # store data for ColorRulePracticeTrials (TrialHandler)
                # the Routine "colorRuleFeedback" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
            # completed n_color_rule_practice  repeats of 'ColorRulePracticeTrials'
            
        # completed n_response_modes_allowed repeats of 'practiceColorRuleLoop'
        
    # completed 1.0 repeats of 'DEBUG'
    
    
    # set up handler to look after randomisation of conditions etc
    TasktypeLoop = data.TrialHandler2(
        name='TasktypeLoop',
        nReps=max(1, n_trial_types_allowed), 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(TasktypeLoop)  # add the loop to the experiment
    thisTasktypeLoop = TasktypeLoop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTasktypeLoop.rgb)
    if thisTasktypeLoop != None:
        for paramName in thisTasktypeLoop:
            globals()[paramName] = thisTasktypeLoop[paramName]
    
    for thisTasktypeLoop in TasktypeLoop:
        currentLoop = TasktypeLoop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisTasktypeLoop.rgb)
        if thisTasktypeLoop != None:
            for paramName in thisTasktypeLoop:
                globals()[paramName] = thisTasktypeLoop[paramName]
        
        # --- Prepare to start Routine "taskStart" ---
        # create an object to store info about Routine taskStart
        taskStart = data.Routine(
            name='taskStart',
            components=[textTaskPhase, mouseTaskPhase],
        )
        taskStart.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from codeTaskPhase
        # specify the task id
        # note that order of task could have been shuffled
        trial_type = trial_type_list[TasktypeLoop.thisN]
        
        trial_phase_id_text = ['FIRST', 'SECOND'][TasktypeLoop.thisN]
        
        # specify the order or response modes (draw or click first?)
        mode_type_list = random.sample(response_modes_allowed, n_response_modes_allowed)
        
        # create the button
        continue_button = funcCreateButton(
            win, 'continue', 
            [0, -0.4], [0.25, 0.1])
        textTaskPhase.setText('We will now begin the ' + trial_phase_id_text + ' of two parts of the experiment. First, we will explain the task and prepare you with two practice blocks. After that, the experimenter will calibrate the eye tracker. The actual trials will then follow.')
        # setup some python lists for storing info about the mouseTaskPhase
        gotValidClick = False  # until a click is received
        # store start times for taskStart
        taskStart.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        taskStart.tStart = globalClock.getTime(format='float')
        taskStart.status = STARTED
        thisExp.addData('taskStart.started', taskStart.tStart)
        taskStart.maxDuration = None
        # keep track of which components have finished
        taskStartComponents = taskStart.components
        for thisComponent in taskStart.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "taskStart" ---
        # if trial has changed, end Routine now
        if isinstance(TasktypeLoop, data.TrialHandler2) and thisTasktypeLoop.thisN != TasktypeLoop.thisTrial.thisN:
            continueRoutine = False
        taskStart.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from codeTaskPhase
            left_button = mouseTaskPhase.getPressed()[0]
            mouse_position = mouseTaskPhase.getPos()
            if left_button and (t > 0.5) and continue_button.contains(mouse_position):
                continueRoutine = False
            
            # *textTaskPhase* updates
            
            # if textTaskPhase is starting this frame...
            if textTaskPhase.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textTaskPhase.frameNStart = frameN  # exact frame index
                textTaskPhase.tStart = t  # local t and not account for scr refresh
                textTaskPhase.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textTaskPhase, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textTaskPhase.started')
                # update status
                textTaskPhase.status = STARTED
                textTaskPhase.setAutoDraw(True)
            
            # if textTaskPhase is active this frame...
            if textTaskPhase.status == STARTED:
                # update params
                pass
            # *mouseTaskPhase* updates
            
            # if mouseTaskPhase is starting this frame...
            if mouseTaskPhase.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouseTaskPhase.frameNStart = frameN  # exact frame index
                mouseTaskPhase.tStart = t  # local t and not account for scr refresh
                mouseTaskPhase.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouseTaskPhase, 'tStartRefresh')  # time at next scr refresh
                # update status
                mouseTaskPhase.status = STARTED
                mouseTaskPhase.mouseClock.reset()
                prevButtonState = mouseTaskPhase.getPressed()  # if button is down already this ISN'T a new click
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                taskStart.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in taskStart.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "taskStart" ---
        for thisComponent in taskStart.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for taskStart
        taskStart.tStop = globalClock.getTime(format='float')
        taskStart.tStopRefresh = tThisFlipGlobal
        thisExp.addData('taskStart.stopped', taskStart.tStop)
        # Run 'End Routine' code from codeTaskPhase
        continue_button.setAutoDraw(False)
        continue_button = None
        # store data for TasktypeLoop (TrialHandler)
        # the Routine "taskStart" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "taskInstructionText" ---
        # create an object to store info about Routine taskInstructionText
        taskInstructionText = data.Routine(
            name='taskInstructionText',
            components=[textTaskInstruction, mouseTaskInstruction],
        )
        taskInstructionText.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from codeTextInstructionText
        continue_button = funcCreateButton(
            win, 'continue', 
            [0, -0.4], [0.25, 0.1])
        textTaskInstruction.setText(text_practice_instructions[trial_type] + '\n You will perform this task in both drawing and clicking mode')
        # setup some python lists for storing info about the mouseTaskInstruction
        gotValidClick = False  # until a click is received
        # store start times for taskInstructionText
        taskInstructionText.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        taskInstructionText.tStart = globalClock.getTime(format='float')
        taskInstructionText.status = STARTED
        thisExp.addData('taskInstructionText.started', taskInstructionText.tStart)
        taskInstructionText.maxDuration = None
        # keep track of which components have finished
        taskInstructionTextComponents = taskInstructionText.components
        for thisComponent in taskInstructionText.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "taskInstructionText" ---
        # if trial has changed, end Routine now
        if isinstance(TasktypeLoop, data.TrialHandler2) and thisTasktypeLoop.thisN != TasktypeLoop.thisTrial.thisN:
            continueRoutine = False
        taskInstructionText.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from codeTextInstructionText
            left_button = mouseTaskInstruction.getPressed()[0]
            mouse_position = mouseTaskInstruction.getPos()
            if left_button and (t > 0.5) and continue_button.contains(mouse_position):
                continueRoutine = False
            
            # *textTaskInstruction* updates
            
            # if textTaskInstruction is starting this frame...
            if textTaskInstruction.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textTaskInstruction.frameNStart = frameN  # exact frame index
                textTaskInstruction.tStart = t  # local t and not account for scr refresh
                textTaskInstruction.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textTaskInstruction, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textTaskInstruction.started')
                # update status
                textTaskInstruction.status = STARTED
                textTaskInstruction.setAutoDraw(True)
            
            # if textTaskInstruction is active this frame...
            if textTaskInstruction.status == STARTED:
                # update params
                pass
            # *mouseTaskInstruction* updates
            
            # if mouseTaskInstruction is starting this frame...
            if mouseTaskInstruction.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouseTaskInstruction.frameNStart = frameN  # exact frame index
                mouseTaskInstruction.tStart = t  # local t and not account for scr refresh
                mouseTaskInstruction.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouseTaskInstruction, 'tStartRefresh')  # time at next scr refresh
                # update status
                mouseTaskInstruction.status = STARTED
                mouseTaskInstruction.mouseClock.reset()
                prevButtonState = mouseTaskInstruction.getPressed()  # if button is down already this ISN'T a new click
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                taskInstructionText.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in taskInstructionText.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "taskInstructionText" ---
        for thisComponent in taskInstructionText.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for taskInstructionText
        taskInstructionText.tStop = globalClock.getTime(format='float')
        taskInstructionText.tStopRefresh = tThisFlipGlobal
        thisExp.addData('taskInstructionText.stopped', taskInstructionText.tStop)
        # Run 'End Routine' code from codeTextInstructionText
        continue_button.setAutoDraw(False)
        continue_button = None
        # store data for TasktypeLoop (TrialHandler)
        # the Routine "taskInstructionText" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        TaskInstrLoop = data.TrialHandler2(
            name='TaskInstrLoop',
            nReps=max(0, n_response_modes_allowed), 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(TaskInstrLoop)  # add the loop to the experiment
        thisTaskInstrLoop = TaskInstrLoop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTaskInstrLoop.rgb)
        if thisTaskInstrLoop != None:
            for paramName in thisTaskInstrLoop:
                globals()[paramName] = thisTaskInstrLoop[paramName]
        
        for thisTaskInstrLoop in TaskInstrLoop:
            currentLoop = TaskInstrLoop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # abbreviate parameter names if possible (e.g. rgb = thisTaskInstrLoop.rgb)
            if thisTaskInstrLoop != None:
                for paramName in thisTaskInstrLoop:
                    globals()[paramName] = thisTaskInstrLoop[paramName]
            
            # --- Prepare to start Routine "taskInstructionDiagram" ---
            # create an object to store info about Routine taskInstructionDiagram
            taskInstructionDiagram = data.Routine(
                name='taskInstructionDiagram',
                components=[textTaskInstrDiagram, mouseTaskInstrDiagram, imgTaskInstrDiagram],
            )
            taskInstructionDiagram.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from codeTaskInstrDiagram
            # specify the response mode
            response_mode = response_modes_allowed[TaskInstrLoop.thisN]
            # create the title
            task_diagram_text_to_display = 'Now we will practice ' + response_mode + 'ing to report answer \n'
            task_diagram_text_to_display += 'for this type of trials.'
            # create the continue button
            continue_button = funcCreateButton(
                win, 'continue', 
                [0, -0.4], [0.25, 0.1])
            textTaskInstrDiagram.setText(task_diagram_text_to_display)
            # setup some python lists for storing info about the mouseTaskInstrDiagram
            gotValidClick = False  # until a click is received
            imgTaskInstrDiagram.setImage('resources/trial_type_' + str(trial_type) + '_' + response_mode + '.png')
            # store start times for taskInstructionDiagram
            taskInstructionDiagram.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            taskInstructionDiagram.tStart = globalClock.getTime(format='float')
            taskInstructionDiagram.status = STARTED
            thisExp.addData('taskInstructionDiagram.started', taskInstructionDiagram.tStart)
            taskInstructionDiagram.maxDuration = None
            # keep track of which components have finished
            taskInstructionDiagramComponents = taskInstructionDiagram.components
            for thisComponent in taskInstructionDiagram.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "taskInstructionDiagram" ---
            # if trial has changed, end Routine now
            if isinstance(TaskInstrLoop, data.TrialHandler2) and thisTaskInstrLoop.thisN != TaskInstrLoop.thisTrial.thisN:
                continueRoutine = False
            taskInstructionDiagram.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from codeTaskInstrDiagram
                left_button = mouseTaskInstrDiagram.getPressed()[0]
                mouse_position = mouseTaskInstrDiagram.getPos()
                if left_button and (t > 0.5) and continue_button.contains(mouse_position):
                    continueRoutine = False
                
                # *textTaskInstrDiagram* updates
                
                # if textTaskInstrDiagram is starting this frame...
                if textTaskInstrDiagram.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textTaskInstrDiagram.frameNStart = frameN  # exact frame index
                    textTaskInstrDiagram.tStart = t  # local t and not account for scr refresh
                    textTaskInstrDiagram.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textTaskInstrDiagram, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textTaskInstrDiagram.started')
                    # update status
                    textTaskInstrDiagram.status = STARTED
                    textTaskInstrDiagram.setAutoDraw(True)
                
                # if textTaskInstrDiagram is active this frame...
                if textTaskInstrDiagram.status == STARTED:
                    # update params
                    pass
                # *mouseTaskInstrDiagram* updates
                
                # if mouseTaskInstrDiagram is starting this frame...
                if mouseTaskInstrDiagram.status == NOT_STARTED and t >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    mouseTaskInstrDiagram.frameNStart = frameN  # exact frame index
                    mouseTaskInstrDiagram.tStart = t  # local t and not account for scr refresh
                    mouseTaskInstrDiagram.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(mouseTaskInstrDiagram, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    mouseTaskInstrDiagram.status = STARTED
                    mouseTaskInstrDiagram.mouseClock.reset()
                    prevButtonState = mouseTaskInstrDiagram.getPressed()  # if button is down already this ISN'T a new click
                
                # *imgTaskInstrDiagram* updates
                
                # if imgTaskInstrDiagram is starting this frame...
                if imgTaskInstrDiagram.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    imgTaskInstrDiagram.frameNStart = frameN  # exact frame index
                    imgTaskInstrDiagram.tStart = t  # local t and not account for scr refresh
                    imgTaskInstrDiagram.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(imgTaskInstrDiagram, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'imgTaskInstrDiagram.started')
                    # update status
                    imgTaskInstrDiagram.status = STARTED
                    imgTaskInstrDiagram.setAutoDraw(True)
                
                # if imgTaskInstrDiagram is active this frame...
                if imgTaskInstrDiagram.status == STARTED:
                    # update params
                    pass
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    taskInstructionDiagram.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in taskInstructionDiagram.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "taskInstructionDiagram" ---
            for thisComponent in taskInstructionDiagram.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for taskInstructionDiagram
            taskInstructionDiagram.tStop = globalClock.getTime(format='float')
            taskInstructionDiagram.tStopRefresh = tThisFlipGlobal
            thisExp.addData('taskInstructionDiagram.stopped', taskInstructionDiagram.tStop)
            # Run 'End Routine' code from codeTaskInstrDiagram
            continue_button.setAutoDraw(False)
            continue_button = None
            # store data for TaskInstrLoop (TrialHandler)
            # the Routine "taskInstructionDiagram" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "prepareTaskInstrPractice" ---
            # create an object to store info about Routine prepareTaskInstrPractice
            prepareTaskInstrPractice = data.Routine(
                name='prepareTaskInstrPractice',
                components=[],
            )
            prepareTaskInstrPractice.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from codePrepareTaskInstrPractice
            # start the instruction + practice
            has_finished_instruction = False
            max_instruction_practice_loop = 2
            
            # create a sampler for instruction
            instruction_sampler = StimSamplingScheduler(
                n_ori_regions, ['orthogonal'])
            instruction_sampler.initialize_stage()
            
            # store start times for prepareTaskInstrPractice
            prepareTaskInstrPractice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            prepareTaskInstrPractice.tStart = globalClock.getTime(format='float')
            prepareTaskInstrPractice.status = STARTED
            thisExp.addData('prepareTaskInstrPractice.started', prepareTaskInstrPractice.tStart)
            prepareTaskInstrPractice.maxDuration = None
            # keep track of which components have finished
            prepareTaskInstrPracticeComponents = prepareTaskInstrPractice.components
            for thisComponent in prepareTaskInstrPractice.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "prepareTaskInstrPractice" ---
            # if trial has changed, end Routine now
            if isinstance(TaskInstrLoop, data.TrialHandler2) and thisTaskInstrLoop.thisN != TaskInstrLoop.thisTrial.thisN:
                continueRoutine = False
            prepareTaskInstrPractice.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    prepareTaskInstrPractice.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in prepareTaskInstrPractice.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "prepareTaskInstrPractice" ---
            for thisComponent in prepareTaskInstrPractice.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for prepareTaskInstrPractice
            prepareTaskInstrPractice.tStop = globalClock.getTime(format='float')
            prepareTaskInstrPractice.tStopRefresh = tThisFlipGlobal
            thisExp.addData('prepareTaskInstrPractice.stopped', prepareTaskInstrPractice.tStop)
            # the Routine "prepareTaskInstrPractice" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # set up handler to look after randomisation of conditions etc
            taskPracticeBlockLoop = data.TrialHandler2(
                name='taskPracticeBlockLoop',
                nReps=max_instruction_practice_loop, 
                method='sequential', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=[None], 
                seed=None, 
            )
            thisExp.addLoop(taskPracticeBlockLoop)  # add the loop to the experiment
            thisTaskPracticeBlockLoop = taskPracticeBlockLoop.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisTaskPracticeBlockLoop.rgb)
            if thisTaskPracticeBlockLoop != None:
                for paramName in thisTaskPracticeBlockLoop:
                    globals()[paramName] = thisTaskPracticeBlockLoop[paramName]
            
            for thisTaskPracticeBlockLoop in taskPracticeBlockLoop:
                currentLoop = taskPracticeBlockLoop
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                # abbreviate parameter names if possible (e.g. rgb = thisTaskPracticeBlockLoop.rgb)
                if thisTaskPracticeBlockLoop != None:
                    for paramName in thisTaskPracticeBlockLoop:
                        globals()[paramName] = thisTaskPracticeBlockLoop[paramName]
                
                # --- Prepare to start Routine "startPracticeBlock" ---
                # create an object to store info about Routine startPracticeBlock
                startPracticeBlock = data.Routine(
                    name='startPracticeBlock',
                    components=[mouseStartPracticeBlock, textStartPracticeBlock],
                )
                startPracticeBlock.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from codeStartPraticeBlock
                # specify number of trials
                n_trials_to_practice = 4 if has_finished_instruction else 3
                
                # renew the sampler
                instruction_sampler.initialize_block()
                
                # initialize response regions
                report_xs_options = [
                    [-0.25, 0.25],
                    [0, 0],
                    [0.25, -0.25],
                    [0, 0],
                ]
                report_ys_options = [
                    [0.03, 0.03],
                    [0.3, -0.24],
                    [0.03, 0.03],
                    [-0.24, 0.3],
                ]
                
                # initialize timing
                delay_lengths = [1.5, 5.5]
                
                # make it slower for practice
                slow_down_factor = 1 if has_finished_instruction else 1.5
                practice_display_time = display_time * slow_down_factor
                practice_display_cue_onset_time = display_cue_onset_time * slow_down_factor
                
                # provide the button
                continue_button = funcCreateButton(
                    win, 'continue', 
                    [0, -0.4], [0.25, 0.1])
                
                # create a little piece of prompt
                text_start_practice_block = ''
                if has_finished_instruction:
                    text_start_practice_block = 'Now we will start a block of PRACTICE trials, ' \
                        + 'but at the normal speed, just as what you will see in actual trials.'
                else:
                    text_start_practice_block = 'Now we will try some PRACTICE trials, ' \
                        + 'but slower than actual trials. ' \
                        + 'Please check the feedback at the end of each trial carefully before moving on.'
                
                # setup some python lists for storing info about the mouseStartPracticeBlock
                gotValidClick = False  # until a click is received
                textStartPracticeBlock.setText(text_start_practice_block)
                # store start times for startPracticeBlock
                startPracticeBlock.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                startPracticeBlock.tStart = globalClock.getTime(format='float')
                startPracticeBlock.status = STARTED
                thisExp.addData('startPracticeBlock.started', startPracticeBlock.tStart)
                startPracticeBlock.maxDuration = None
                # keep track of which components have finished
                startPracticeBlockComponents = startPracticeBlock.components
                for thisComponent in startPracticeBlock.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "startPracticeBlock" ---
                # if trial has changed, end Routine now
                if isinstance(taskPracticeBlockLoop, data.TrialHandler2) and thisTaskPracticeBlockLoop.thisN != taskPracticeBlockLoop.thisTrial.thisN:
                    continueRoutine = False
                startPracticeBlock.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # Run 'Each Frame' code from codeStartPraticeBlock
                    left_button = mouseStartPracticeBlock.getPressed()[0]
                    mouse_position = mouseStartPracticeBlock.getPos()
                    if left_button and (t > 0.5) and continue_button.contains(mouse_position):
                        continueRoutine = False
                    
                    # *mouseStartPracticeBlock* updates
                    
                    # if mouseStartPracticeBlock is starting this frame...
                    if mouseStartPracticeBlock.status == NOT_STARTED and t >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        mouseStartPracticeBlock.frameNStart = frameN  # exact frame index
                        mouseStartPracticeBlock.tStart = t  # local t and not account for scr refresh
                        mouseStartPracticeBlock.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(mouseStartPracticeBlock, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        mouseStartPracticeBlock.status = STARTED
                        mouseStartPracticeBlock.mouseClock.reset()
                        prevButtonState = mouseStartPracticeBlock.getPressed()  # if button is down already this ISN'T a new click
                    
                    # *textStartPracticeBlock* updates
                    
                    # if textStartPracticeBlock is starting this frame...
                    if textStartPracticeBlock.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        textStartPracticeBlock.frameNStart = frameN  # exact frame index
                        textStartPracticeBlock.tStart = t  # local t and not account for scr refresh
                        textStartPracticeBlock.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(textStartPracticeBlock, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'textStartPracticeBlock.started')
                        # update status
                        textStartPracticeBlock.status = STARTED
                        textStartPracticeBlock.setAutoDraw(True)
                    
                    # if textStartPracticeBlock is active this frame...
                    if textStartPracticeBlock.status == STARTED:
                        # update params
                        pass
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        startPracticeBlock.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in startPracticeBlock.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "startPracticeBlock" ---
                for thisComponent in startPracticeBlock.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for startPracticeBlock
                startPracticeBlock.tStop = globalClock.getTime(format='float')
                startPracticeBlock.tStopRefresh = tThisFlipGlobal
                thisExp.addData('startPracticeBlock.stopped', startPracticeBlock.tStop)
                # Run 'End Routine' code from codeStartPraticeBlock
                continue_button.setAutoDraw(False)
                continue_button = None
                # store data for taskPracticeBlockLoop (TrialHandler)
                # the Routine "startPracticeBlock" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                
                # set up handler to look after randomisation of conditions etc
                taskPracticeTrialLoop = data.TrialHandler2(
                    name='taskPracticeTrialLoop',
                    nReps=n_trials_to_practice, 
                    method='sequential', 
                    extraInfo=expInfo, 
                    originPath=-1, 
                    trialList=[None], 
                    seed=None, 
                )
                thisExp.addLoop(taskPracticeTrialLoop)  # add the loop to the experiment
                thisTaskPracticeTrialLoop = taskPracticeTrialLoop.trialList[0]  # so we can initialise stimuli with some values
                # abbreviate parameter names if possible (e.g. rgb = thisTaskPracticeTrialLoop.rgb)
                if thisTaskPracticeTrialLoop != None:
                    for paramName in thisTaskPracticeTrialLoop:
                        globals()[paramName] = thisTaskPracticeTrialLoop[paramName]
                
                for thisTaskPracticeTrialLoop in taskPracticeTrialLoop:
                    currentLoop = taskPracticeTrialLoop
                    thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                    # abbreviate parameter names if possible (e.g. rgb = thisTaskPracticeTrialLoop.rgb)
                    if thisTaskPracticeTrialLoop != None:
                        for paramName in thisTaskPracticeTrialLoop:
                            globals()[paramName] = thisTaskPracticeTrialLoop[paramName]
                    
                    # --- Prepare to start Routine "prepareSinglePracticeTrial" ---
                    # create an object to store info about Routine prepareSinglePracticeTrial
                    prepareSinglePracticeTrial = data.Routine(
                        name='prepareSinglePracticeTrial',
                        components=[textPracticePreTrial],
                    )
                    prepareSinglePracticeTrial.status = NOT_STARTED
                    continueRoutine = True
                    # update component parameters for each repeat
                    # Run 'Begin Routine' code from codeSinglePracticeTrial
                    practice_trial_id = taskPracticeTrialLoop.thisN
                    
                    # sample stimuli
                    _, cur_stims = instruction_sampler.sample_for_trial(
                        allow_same_region=False)
                    
                    # sample color code
                    cur_colors = (1, 2) if random.random() < 0.5 else (2, 1)
                    
                    # determine cue code or report code
                    cur_cue_codes = [False, False]
                    cur_report_codes = [False, False]
                    if trial_type == 0:
                        cur_cue_codes[0] = (practice_trial_id % 3) != 1 
                        cur_cue_codes[1] = (practice_trial_id % 3) != 0
                        cur_report_codes = cur_cue_codes[:]
                    elif trial_type == 1:
                        cur_cue_codes = [True, True]
                        cur_report_codes[0] = (practice_trial_id % 3) != 1 
                        cur_report_codes[1] = (practice_trial_id % 3) != 0
                    else:
                        raise ValueError(f'Unknown trial type {trial_type}')
                    
                    # determine response region
                    cur_resp_region_id = random.randint(0, len(report_xs_options)-1)
                    cur_resp_regions = [
                        [report_xs_options[cur_resp_region_id][0], report_ys_options[cur_resp_region_id][0]],
                        [report_xs_options[cur_resp_region_id][1], report_ys_options[cur_resp_region_id][1]],
                    ]
                    
                    # finally, create the fixation
                    fixation = DefaultFixation(win)
                    
                    textPracticePreTrial.setText('')
                    # store start times for prepareSinglePracticeTrial
                    prepareSinglePracticeTrial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                    prepareSinglePracticeTrial.tStart = globalClock.getTime(format='float')
                    prepareSinglePracticeTrial.status = STARTED
                    thisExp.addData('prepareSinglePracticeTrial.started', prepareSinglePracticeTrial.tStart)
                    prepareSinglePracticeTrial.maxDuration = None
                    # keep track of which components have finished
                    prepareSinglePracticeTrialComponents = prepareSinglePracticeTrial.components
                    for thisComponent in prepareSinglePracticeTrial.components:
                        thisComponent.tStart = None
                        thisComponent.tStop = None
                        thisComponent.tStartRefresh = None
                        thisComponent.tStopRefresh = None
                        if hasattr(thisComponent, 'status'):
                            thisComponent.status = NOT_STARTED
                    # reset timers
                    t = 0
                    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                    frameN = -1
                    
                    # --- Run Routine "prepareSinglePracticeTrial" ---
                    # if trial has changed, end Routine now
                    if isinstance(taskPracticeTrialLoop, data.TrialHandler2) and thisTaskPracticeTrialLoop.thisN != taskPracticeTrialLoop.thisTrial.thisN:
                        continueRoutine = False
                    prepareSinglePracticeTrial.forceEnded = routineForceEnded = not continueRoutine
                    while continueRoutine:
                        # get current time
                        t = routineTimer.getTime()
                        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                        # update/draw components on each frame
                        # Run 'Each Frame' code from codeSinglePracticeTrial
                        # end routine when time up
                        if t > pre_trial_time:
                            continueRoutine = False
                        
                        # *textPracticePreTrial* updates
                        
                        # if textPracticePreTrial is starting this frame...
                        if textPracticePreTrial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            textPracticePreTrial.frameNStart = frameN  # exact frame index
                            textPracticePreTrial.tStart = t  # local t and not account for scr refresh
                            textPracticePreTrial.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(textPracticePreTrial, 'tStartRefresh')  # time at next scr refresh
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'textPracticePreTrial.started')
                            # update status
                            textPracticePreTrial.status = STARTED
                            textPracticePreTrial.setAutoDraw(True)
                        
                        # if textPracticePreTrial is active this frame...
                        if textPracticePreTrial.status == STARTED:
                            # update params
                            pass
                        
                        # check for quit (typically the Esc key)
                        if defaultKeyboard.getKeys(keyList=["escape"]):
                            thisExp.status = FINISHED
                        if thisExp.status == FINISHED or endExpNow:
                            endExperiment(thisExp, win=win)
                            return
                        # pause experiment here if requested
                        if thisExp.status == PAUSED:
                            pauseExperiment(
                                thisExp=thisExp, 
                                win=win, 
                                timers=[routineTimer], 
                                playbackComponents=[]
                            )
                            # skip the frame we paused on
                            continue
                        
                        # check if all components have finished
                        if not continueRoutine:  # a component has requested a forced-end of Routine
                            prepareSinglePracticeTrial.forceEnded = routineForceEnded = True
                            break
                        continueRoutine = False  # will revert to True if at least one component still running
                        for thisComponent in prepareSinglePracticeTrial.components:
                            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                continueRoutine = True
                                break  # at least one component has not yet finished
                        
                        # refresh the screen
                        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                            win.flip()
                    
                    # --- Ending Routine "prepareSinglePracticeTrial" ---
                    for thisComponent in prepareSinglePracticeTrial.components:
                        if hasattr(thisComponent, "setAutoDraw"):
                            thisComponent.setAutoDraw(False)
                    # store stop times for prepareSinglePracticeTrial
                    prepareSinglePracticeTrial.tStop = globalClock.getTime(format='float')
                    prepareSinglePracticeTrial.tStopRefresh = tThisFlipGlobal
                    thisExp.addData('prepareSinglePracticeTrial.stopped', prepareSinglePracticeTrial.tStop)
                    # Run 'End Routine' code from codeSinglePracticeTrial
                    fixation.setAutoDraw(False)
                    fixation = None
                    # the Routine "prepareSinglePracticeTrial" was not non-slip safe, so reset the non-slip timer
                    routineTimer.reset()
                    
                    # set up handler to look after randomisation of conditions etc
                    taskPracticeStimLoop = data.TrialHandler2(
                        name='taskPracticeStimLoop',
                        nReps=2.0, 
                        method='sequential', 
                        extraInfo=expInfo, 
                        originPath=-1, 
                        trialList=[None], 
                        seed=None, 
                    )
                    thisExp.addLoop(taskPracticeStimLoop)  # add the loop to the experiment
                    thisTaskPracticeStimLoop = taskPracticeStimLoop.trialList[0]  # so we can initialise stimuli with some values
                    # abbreviate parameter names if possible (e.g. rgb = thisTaskPracticeStimLoop.rgb)
                    if thisTaskPracticeStimLoop != None:
                        for paramName in thisTaskPracticeStimLoop:
                            globals()[paramName] = thisTaskPracticeStimLoop[paramName]
                    
                    for thisTaskPracticeStimLoop in taskPracticeStimLoop:
                        currentLoop = taskPracticeStimLoop
                        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                        # abbreviate parameter names if possible (e.g. rgb = thisTaskPracticeStimLoop.rgb)
                        if thisTaskPracticeStimLoop != None:
                            for paramName in thisTaskPracticeStimLoop:
                                globals()[paramName] = thisTaskPracticeStimLoop[paramName]
                        
                        # --- Prepare to start Routine "taskPracticeStim" ---
                        # create an object to store info about Routine taskPracticeStim
                        taskPracticeStim = data.Routine(
                            name='taskPracticeStim',
                            components=[textTaskPracticeStim],
                        )
                        taskPracticeStim.status = NOT_STARTED
                        continueRoutine = True
                        # update component parameters for each repeat
                        # Run 'Begin Routine' code from codeTaskPracticeStim
                        objs_display = []
                        
                        # now display the stimulus
                        stim_id = taskPracticeStimLoop.thisN
                        ori = cur_stims[stim_id]
                        loc = [0, 0]
                        stim_obj = funcDrawStim(win, ori, loc, patch_radius)
                        objs_display.append(stim_obj)
                        
                        # to detect whether a pressed has released
                        has_released = True
                        
                        # to mark if the cue has displayed
                        has_cue_shown = False
                        
                        textTaskPracticeStim.setText('')
                        # store start times for taskPracticeStim
                        taskPracticeStim.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                        taskPracticeStim.tStart = globalClock.getTime(format='float')
                        taskPracticeStim.status = STARTED
                        thisExp.addData('taskPracticeStim.started', taskPracticeStim.tStart)
                        taskPracticeStim.maxDuration = None
                        # keep track of which components have finished
                        taskPracticeStimComponents = taskPracticeStim.components
                        for thisComponent in taskPracticeStim.components:
                            thisComponent.tStart = None
                            thisComponent.tStop = None
                            thisComponent.tStartRefresh = None
                            thisComponent.tStopRefresh = None
                            if hasattr(thisComponent, 'status'):
                                thisComponent.status = NOT_STARTED
                        # reset timers
                        t = 0
                        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                        frameN = -1
                        
                        # --- Run Routine "taskPracticeStim" ---
                        # if trial has changed, end Routine now
                        if isinstance(taskPracticeStimLoop, data.TrialHandler2) and thisTaskPracticeStimLoop.thisN != taskPracticeStimLoop.thisTrial.thisN:
                            continueRoutine = False
                        taskPracticeStim.forceEnded = routineForceEnded = not continueRoutine
                        while continueRoutine:
                            # get current time
                            t = routineTimer.getTime()
                            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                            # update/draw components on each frame
                            # Run 'Each Frame' code from codeTaskPracticeStim
                            # display the cue when onset
                            if (not has_cue_shown) and t >= practice_display_cue_onset_time:
                                cue_loc = [0, 0]
                                cue_color = cur_colors[stim_id]
                                cue_code = cur_cue_codes[stim_id]
                                report_cue = SingleCueObject(
                                    win, cue_loc, cue_code, patch_radius, cue_color)
                                objs_display.append(report_cue)
                                has_cue_shown = True
                                
                                # additionally, for instruction purpose
                                # show the verbal cue
                                if not has_finished_instruction:
                                    # if this is to report
                                    s_to_cue = False
                                    if trial_type == 1:
                                        # post cue: remember everything
                                        s_to_cue = True
                                    elif trial_type == 0:
                                        # otherwise only remember the cued
                                        s_to_cue = cue_code
                                    if s_to_cue:
                                        additional_cue = TrainingCue(
                                            win, 'remember this', 
                                            cue_loc, patch_radius)
                                        objs_display.append(additional_cue)
                            
                            # end routine when time up
                            if t > practice_display_time:
                                continueRoutine = False
                            
                            
                            # *textTaskPracticeStim* updates
                            
                            # if textTaskPracticeStim is starting this frame...
                            if textTaskPracticeStim.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                                # keep track of start time/frame for later
                                textTaskPracticeStim.frameNStart = frameN  # exact frame index
                                textTaskPracticeStim.tStart = t  # local t and not account for scr refresh
                                textTaskPracticeStim.tStartRefresh = tThisFlipGlobal  # on global time
                                win.timeOnFlip(textTaskPracticeStim, 'tStartRefresh')  # time at next scr refresh
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'textTaskPracticeStim.started')
                                # update status
                                textTaskPracticeStim.status = STARTED
                                textTaskPracticeStim.setAutoDraw(True)
                            
                            # if textTaskPracticeStim is active this frame...
                            if textTaskPracticeStim.status == STARTED:
                                # update params
                                pass
                            
                            # check for quit (typically the Esc key)
                            if defaultKeyboard.getKeys(keyList=["escape"]):
                                thisExp.status = FINISHED
                            if thisExp.status == FINISHED or endExpNow:
                                endExperiment(thisExp, win=win)
                                return
                            # pause experiment here if requested
                            if thisExp.status == PAUSED:
                                pauseExperiment(
                                    thisExp=thisExp, 
                                    win=win, 
                                    timers=[routineTimer], 
                                    playbackComponents=[]
                                )
                                # skip the frame we paused on
                                continue
                            
                            # check if all components have finished
                            if not continueRoutine:  # a component has requested a forced-end of Routine
                                taskPracticeStim.forceEnded = routineForceEnded = True
                                break
                            continueRoutine = False  # will revert to True if at least one component still running
                            for thisComponent in taskPracticeStim.components:
                                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                    continueRoutine = True
                                    break  # at least one component has not yet finished
                            
                            # refresh the screen
                            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                                win.flip()
                        
                        # --- Ending Routine "taskPracticeStim" ---
                        for thisComponent in taskPracticeStim.components:
                            if hasattr(thisComponent, "setAutoDraw"):
                                thisComponent.setAutoDraw(False)
                        # store stop times for taskPracticeStim
                        taskPracticeStim.tStop = globalClock.getTime(format='float')
                        taskPracticeStim.tStopRefresh = tThisFlipGlobal
                        thisExp.addData('taskPracticeStim.stopped', taskPracticeStim.tStop)
                        # Run 'End Routine' code from codeTaskPracticeStim
                        # reset click recording
                        has_released = True
                        
                        # don't record response
                        # thisExp.addData(f'mouseStim.x_{seq_display_loop.thisN+1}', display_mouse_xs)
                        # thisExp.addData(f'mouseStim.y_{seq_display_loop.thisN+1}', display_mouse_ys)
                        
                        # clear all objects
                        for i in range(len(objs_display)):
                            display_obj = objs_display[i]
                            display_obj.setAutoDraw(False)
                        
                        objs_display = []
                        
                        # the Routine "taskPracticeStim" was not non-slip safe, so reset the non-slip timer
                        routineTimer.reset()
                        
                        # --- Prepare to start Routine "taskPracticeStimDelay" ---
                        # create an object to store info about Routine taskPracticeStimDelay
                        taskPracticeStimDelay = data.Routine(
                            name='taskPracticeStimDelay',
                            components=[textColorRuleDelay_2],
                        )
                        taskPracticeStimDelay.status = NOT_STARTED
                        continueRoutine = True
                        # update component parameters for each repeat
                        # Run 'Begin Routine' code from codeTaskPracticeStimDelay
                        objs_display = []
                        has_noise_patch_displaying = True
                        
                        textColorRuleDelay_2.setText('')
                        # store start times for taskPracticeStimDelay
                        taskPracticeStimDelay.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                        taskPracticeStimDelay.tStart = globalClock.getTime(format='float')
                        taskPracticeStimDelay.status = STARTED
                        thisExp.addData('taskPracticeStimDelay.started', taskPracticeStimDelay.tStart)
                        taskPracticeStimDelay.maxDuration = None
                        # keep track of which components have finished
                        taskPracticeStimDelayComponents = taskPracticeStimDelay.components
                        for thisComponent in taskPracticeStimDelay.components:
                            thisComponent.tStart = None
                            thisComponent.tStop = None
                            thisComponent.tStartRefresh = None
                            thisComponent.tStopRefresh = None
                            if hasattr(thisComponent, 'status'):
                                thisComponent.status = NOT_STARTED
                        # reset timers
                        t = 0
                        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                        frameN = -1
                        
                        # --- Run Routine "taskPracticeStimDelay" ---
                        # if trial has changed, end Routine now
                        if isinstance(taskPracticeStimLoop, data.TrialHandler2) and thisTaskPracticeStimLoop.thisN != taskPracticeStimLoop.thisTrial.thisN:
                            continueRoutine = False
                        taskPracticeStimDelay.forceEnded = routineForceEnded = not continueRoutine
                        while continueRoutine:
                            # get current time
                            t = routineTimer.getTime()
                            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                            # update/draw components on each frame
                            # Run 'Each Frame' code from codeTaskPracticeStimDelay
                            # end routine when time up
                            if t > delay_post_stim:
                                continueRoutine = False
                            
                            # stop displaying all noise patches
                            if has_noise_patch_displaying:
                                if t > noise_patch_display_time:
                                    for i in range(len(objs_display)):
                                        display_obj = objs_display[i]
                                        display_obj.setAutoDraw(False)
                                    has_noise_patch_displaying = False
                                    
                                    # display the fixation
                                    objs_display = []
                                    fixation = DefaultFixation(win)
                                    objs_display.append(fixation)
                            
                            
                            # *textColorRuleDelay_2* updates
                            
                            # if textColorRuleDelay_2 is starting this frame...
                            if textColorRuleDelay_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                                # keep track of start time/frame for later
                                textColorRuleDelay_2.frameNStart = frameN  # exact frame index
                                textColorRuleDelay_2.tStart = t  # local t and not account for scr refresh
                                textColorRuleDelay_2.tStartRefresh = tThisFlipGlobal  # on global time
                                win.timeOnFlip(textColorRuleDelay_2, 'tStartRefresh')  # time at next scr refresh
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'textColorRuleDelay_2.started')
                                # update status
                                textColorRuleDelay_2.status = STARTED
                                textColorRuleDelay_2.setAutoDraw(True)
                            
                            # if textColorRuleDelay_2 is active this frame...
                            if textColorRuleDelay_2.status == STARTED:
                                # update params
                                pass
                            
                            # check for quit (typically the Esc key)
                            if defaultKeyboard.getKeys(keyList=["escape"]):
                                thisExp.status = FINISHED
                            if thisExp.status == FINISHED or endExpNow:
                                endExperiment(thisExp, win=win)
                                return
                            # pause experiment here if requested
                            if thisExp.status == PAUSED:
                                pauseExperiment(
                                    thisExp=thisExp, 
                                    win=win, 
                                    timers=[routineTimer], 
                                    playbackComponents=[]
                                )
                                # skip the frame we paused on
                                continue
                            
                            # check if all components have finished
                            if not continueRoutine:  # a component has requested a forced-end of Routine
                                taskPracticeStimDelay.forceEnded = routineForceEnded = True
                                break
                            continueRoutine = False  # will revert to True if at least one component still running
                            for thisComponent in taskPracticeStimDelay.components:
                                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                    continueRoutine = True
                                    break  # at least one component has not yet finished
                            
                            # refresh the screen
                            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                                win.flip()
                        
                        # --- Ending Routine "taskPracticeStimDelay" ---
                        for thisComponent in taskPracticeStimDelay.components:
                            if hasattr(thisComponent, "setAutoDraw"):
                                thisComponent.setAutoDraw(False)
                        # store stop times for taskPracticeStimDelay
                        taskPracticeStimDelay.tStop = globalClock.getTime(format='float')
                        taskPracticeStimDelay.tStopRefresh = tThisFlipGlobal
                        thisExp.addData('taskPracticeStimDelay.stopped', taskPracticeStimDelay.tStop)
                        # Run 'End Routine' code from codeTaskPracticeStimDelay
                        # no recording
                        # thisExp.addData(f'mousePostStim.x_{seq_display_loop.thisN+1}', post_display_mouse_xs)
                        # thisExp.addData(f'mousePostStim.y_{seq_display_loop.thisN+1}', post_display_mouse_ys)
                        
                        # clear all objects
                        for i in range(len(objs_display)):
                            display_obj = objs_display[i]
                            display_obj.setAutoDraw(False)
                        
                        objs_display = []
                        
                        # the Routine "taskPracticeStimDelay" was not non-slip safe, so reset the non-slip timer
                        routineTimer.reset()
                    # completed 2.0 repeats of 'taskPracticeStimLoop'
                    
                    
                    # --- Prepare to start Routine "taskPracticeResponse" ---
                    # create an object to store info about Routine taskPracticeResponse
                    taskPracticeResponse = data.Routine(
                        name='taskPracticeResponse',
                        components=[mouseTaskPracticeResponse],
                    )
                    taskPracticeResponse.status = NOT_STARTED
                    continueRoutine = True
                    # update component parameters for each repeat
                    # Run 'Begin Routine' code from codeTaskPracticeResponse
                    objs_display = []
                    response_objects = []
                    for i in range(2):
                        loc = cur_resp_regions[i]
                        to_report = cur_report_codes[i] # default; always report it
                        response_obj = None
                        if response_mode == 'click':
                            response_obj = funcDrawAdjustResponse(
                                win, loc, patch_radius, click_radius, to_report)
                        elif response_mode == 'draw':
                            response_obj = funcDrawDrawResponse(
                                win, loc, patch_radius, to_report)
                        else:
                            raise NotImplementedError(f'Unknown response mode: {response_mode}')
                        objs_display.append(response_obj)
                        response_objects.append(response_obj)
                        
                        # give order color cue
                        if to_report:
                            color_code = cur_colors[i]
                            report_cue = SingleCueObject(
                                win, loc, to_report, patch_radius, color_code)
                            objs_display.append(report_cue)
                    
                    # to add the confirm button
                    continue_button = None
                    if len(response_objects) > 0:
                        continue_button = funcCreateButton(
                            win, 'continue', pos=(0, 0),
                            size=(button_width*1.3, button_height*1.3))
                        objs_display.append(continue_button)
                    
                    # to detect whether a pressed has released
                    has_released = True
                    # setup some python lists for storing info about the mouseTaskPracticeResponse
                    gotValidClick = False  # until a click is received
                    # store start times for taskPracticeResponse
                    taskPracticeResponse.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                    taskPracticeResponse.tStart = globalClock.getTime(format='float')
                    taskPracticeResponse.status = STARTED
                    thisExp.addData('taskPracticeResponse.started', taskPracticeResponse.tStart)
                    taskPracticeResponse.maxDuration = None
                    # keep track of which components have finished
                    taskPracticeResponseComponents = taskPracticeResponse.components
                    for thisComponent in taskPracticeResponse.components:
                        thisComponent.tStart = None
                        thisComponent.tStop = None
                        thisComponent.tStartRefresh = None
                        thisComponent.tStopRefresh = None
                        if hasattr(thisComponent, 'status'):
                            thisComponent.status = NOT_STARTED
                    # reset timers
                    t = 0
                    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                    frameN = -1
                    
                    # --- Run Routine "taskPracticeResponse" ---
                    # if trial has changed, end Routine now
                    if isinstance(taskPracticeTrialLoop, data.TrialHandler2) and thisTaskPracticeTrialLoop.thisN != taskPracticeTrialLoop.thisTrial.thisN:
                        continueRoutine = False
                    taskPracticeResponse.forceEnded = routineForceEnded = not continueRoutine
                    while continueRoutine:
                        # get current time
                        t = routineTimer.getTime()
                        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                        # update/draw components on each frame
                        # Run 'Each Frame' code from codeTaskPracticeResponse
                        # if there are objects clicked
                        left_button = mouseTaskPracticeResponse.getPressed()[0]
                        mouse_position = mouseTaskPracticeResponse.getPos()
                        mouse_time = t
                        if left_button:
                            if continue_button.contains(mouse_position):
                                # first check if it's over the continue button
                                no_pending = funcCheckNoPending(response_objects)
                                if no_pending:
                                    continueRoutine = False
                            else:
                                if response_mode == 'click':
                                    # only register the first click
                                    if has_released:
                                        # this is a new press, register it
                                        has_released = False
                                        # otherwise, ignore it
                                        for i in range(len(response_objects)):
                                            # update each position with the click
                                            resp_obj = response_objects[i]
                                            resp_obj.register_click(mouse_position, mouse_time)
                                elif response_mode == 'draw':
                                    # register all mouse movements
                                    has_released = False
                                    for i in range(len(response_objects)):
                                        # update each position with the click
                                        resp_obj = response_objects[i]
                                        has_redo = resp_obj.register_mouse(
                                            mouse_position, mouse_time)
                                            
                        else:
                            # check if we should update has_released
                            has_released = True
                            if response_mode == 'draw':
                                for i in range(len(response_objects)):
                                    # update each position with the click
                                    resp_obj = response_objects[i]
                                    resp_obj.deregister_mouse(mouse_position, mouse_time)
                        
                        # *mouseTaskPracticeResponse* updates
                        
                        # if mouseTaskPracticeResponse is starting this frame...
                        if mouseTaskPracticeResponse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            mouseTaskPracticeResponse.frameNStart = frameN  # exact frame index
                            mouseTaskPracticeResponse.tStart = t  # local t and not account for scr refresh
                            mouseTaskPracticeResponse.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(mouseTaskPracticeResponse, 'tStartRefresh')  # time at next scr refresh
                            # update status
                            mouseTaskPracticeResponse.status = STARTED
                            mouseTaskPracticeResponse.mouseClock.reset()
                            prevButtonState = mouseTaskPracticeResponse.getPressed()  # if button is down already this ISN'T a new click
                        
                        # check for quit (typically the Esc key)
                        if defaultKeyboard.getKeys(keyList=["escape"]):
                            thisExp.status = FINISHED
                        if thisExp.status == FINISHED or endExpNow:
                            endExperiment(thisExp, win=win)
                            return
                        # pause experiment here if requested
                        if thisExp.status == PAUSED:
                            pauseExperiment(
                                thisExp=thisExp, 
                                win=win, 
                                timers=[routineTimer], 
                                playbackComponents=[]
                            )
                            # skip the frame we paused on
                            continue
                        
                        # check if all components have finished
                        if not continueRoutine:  # a component has requested a forced-end of Routine
                            taskPracticeResponse.forceEnded = routineForceEnded = True
                            break
                        continueRoutine = False  # will revert to True if at least one component still running
                        for thisComponent in taskPracticeResponse.components:
                            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                continueRoutine = True
                                break  # at least one component has not yet finished
                        
                        # refresh the screen
                        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                            win.flip()
                    
                    # --- Ending Routine "taskPracticeResponse" ---
                    for thisComponent in taskPracticeResponse.components:
                        if hasattr(thisComponent, "setAutoDraw"):
                            thisComponent.setAutoDraw(False)
                    # store stop times for taskPracticeResponse
                    taskPracticeResponse.tStop = globalClock.getTime(format='float')
                    taskPracticeResponse.tStopRefresh = tThisFlipGlobal
                    thisExp.addData('taskPracticeResponse.stopped', taskPracticeResponse.tStop)
                    # Run 'End Routine' code from codeTaskPracticeResponse
                    # reset click recording
                    has_released = True
                    
                    practice_responses = []
                    
                    # save unsaved
                    for i in range(2):
                        to_report = cur_report_codes[i]
                        if to_report:
                            resp_obj = response_objects[i]
                            if response_mode == 'draw':
                                # save the stroke
                                if resp_obj.has_unsaved_stroke:
                                    resp_obj.save_stroke()
                                # saving drawings (if there are unsaved)
                                if resp_obj.has_unsaved_drawing:
                                    resp_obj.save_drawing()
                    
                            # record the practice response for later use
                            last_practice_response = resp_obj.compute_response()
                            if (response_mode == 'draw') and (last_practice_response is not None):
                                # for drawing: just copy everything from last drawing
                                original_drawing = (
                                    resp_obj.all_drawings_x[-1],
                                    resp_obj.all_drawings_y[-1])
                                last_practice_response = original_drawing
                        else:
                            last_practice_response = None
                        practice_responses.append(last_practice_response)
                            
                    
                    # clear all objects
                    for i in range(len(objs_display)):
                        display_obj = objs_display[i]
                        display_obj.setAutoDraw(False)
                    
                    objs_display = []
                    response_objects = []
                    ####################
                    
                    # store data for taskPracticeTrialLoop (TrialHandler)
                    # the Routine "taskPracticeResponse" was not non-slip safe, so reset the non-slip timer
                    routineTimer.reset()
                    
                    # --- Prepare to start Routine "taskPracticeTrialFeedback" ---
                    # create an object to store info about Routine taskPracticeTrialFeedback
                    taskPracticeTrialFeedback = data.Routine(
                        name='taskPracticeTrialFeedback',
                        components=[textTaskPracticeTrialFeedback, mouseTaskPracticeTrialFeedback],
                    )
                    taskPracticeTrialFeedback.status = NOT_STARTED
                    continueRoutine = True
                    # update component parameters for each repeat
                    # Run 'Begin Routine' code from codeTaskPracticeTrialFeedback
                    objs_display = []
                    feeaback_objects = CreateWholePracticeFeedback(
                            oris=cur_stims, 
                            responses=practice_responses, 
                            resp_locs=cur_resp_regions,
                            colors=cur_colors, 
                            cue_codes=cur_cue_codes, 
                            report_codes=cur_report_codes,
                            mode=response_mode)
                    objs_display += feeaback_objects
                    
                    # add the continue button
                    continue_button = funcCreateButton(
                        win, 'continue', pos=(0.52,0.42),
                        size=(0.25, 0.1))
                    objs_display.append(continue_button)
                    
                    # setup some python lists for storing info about the mouseTaskPracticeTrialFeedback
                    gotValidClick = False  # until a click is received
                    # store start times for taskPracticeTrialFeedback
                    taskPracticeTrialFeedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                    taskPracticeTrialFeedback.tStart = globalClock.getTime(format='float')
                    taskPracticeTrialFeedback.status = STARTED
                    thisExp.addData('taskPracticeTrialFeedback.started', taskPracticeTrialFeedback.tStart)
                    taskPracticeTrialFeedback.maxDuration = None
                    # keep track of which components have finished
                    taskPracticeTrialFeedbackComponents = taskPracticeTrialFeedback.components
                    for thisComponent in taskPracticeTrialFeedback.components:
                        thisComponent.tStart = None
                        thisComponent.tStop = None
                        thisComponent.tStartRefresh = None
                        thisComponent.tStopRefresh = None
                        if hasattr(thisComponent, 'status'):
                            thisComponent.status = NOT_STARTED
                    # reset timers
                    t = 0
                    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                    frameN = -1
                    
                    # --- Run Routine "taskPracticeTrialFeedback" ---
                    # if trial has changed, end Routine now
                    if isinstance(taskPracticeTrialLoop, data.TrialHandler2) and thisTaskPracticeTrialLoop.thisN != taskPracticeTrialLoop.thisTrial.thisN:
                        continueRoutine = False
                    taskPracticeTrialFeedback.forceEnded = routineForceEnded = not continueRoutine
                    while continueRoutine:
                        # get current time
                        t = routineTimer.getTime()
                        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                        # update/draw components on each frame
                        # Run 'Each Frame' code from codeTaskPracticeTrialFeedback
                        left_button = mouseTaskPracticeTrialFeedback.getPressed()[0]
                        mouse_position = mouseTaskPracticeTrialFeedback.getPos()
                        if left_button and (t > 0.5) and continue_button.contains(mouse_position):
                            continueRoutine = False
                        
                        # *textTaskPracticeTrialFeedback* updates
                        
                        # if textTaskPracticeTrialFeedback is starting this frame...
                        if textTaskPracticeTrialFeedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            textTaskPracticeTrialFeedback.frameNStart = frameN  # exact frame index
                            textTaskPracticeTrialFeedback.tStart = t  # local t and not account for scr refresh
                            textTaskPracticeTrialFeedback.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(textTaskPracticeTrialFeedback, 'tStartRefresh')  # time at next scr refresh
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'textTaskPracticeTrialFeedback.started')
                            # update status
                            textTaskPracticeTrialFeedback.status = STARTED
                            textTaskPracticeTrialFeedback.setAutoDraw(True)
                        
                        # if textTaskPracticeTrialFeedback is active this frame...
                        if textTaskPracticeTrialFeedback.status == STARTED:
                            # update params
                            pass
                        # *mouseTaskPracticeTrialFeedback* updates
                        
                        # if mouseTaskPracticeTrialFeedback is starting this frame...
                        if mouseTaskPracticeTrialFeedback.status == NOT_STARTED and t >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            mouseTaskPracticeTrialFeedback.frameNStart = frameN  # exact frame index
                            mouseTaskPracticeTrialFeedback.tStart = t  # local t and not account for scr refresh
                            mouseTaskPracticeTrialFeedback.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(mouseTaskPracticeTrialFeedback, 'tStartRefresh')  # time at next scr refresh
                            # update status
                            mouseTaskPracticeTrialFeedback.status = STARTED
                            mouseTaskPracticeTrialFeedback.mouseClock.reset()
                            prevButtonState = mouseTaskPracticeTrialFeedback.getPressed()  # if button is down already this ISN'T a new click
                        
                        # check for quit (typically the Esc key)
                        if defaultKeyboard.getKeys(keyList=["escape"]):
                            thisExp.status = FINISHED
                        if thisExp.status == FINISHED or endExpNow:
                            endExperiment(thisExp, win=win)
                            return
                        # pause experiment here if requested
                        if thisExp.status == PAUSED:
                            pauseExperiment(
                                thisExp=thisExp, 
                                win=win, 
                                timers=[routineTimer], 
                                playbackComponents=[]
                            )
                            # skip the frame we paused on
                            continue
                        
                        # check if all components have finished
                        if not continueRoutine:  # a component has requested a forced-end of Routine
                            taskPracticeTrialFeedback.forceEnded = routineForceEnded = True
                            break
                        continueRoutine = False  # will revert to True if at least one component still running
                        for thisComponent in taskPracticeTrialFeedback.components:
                            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                continueRoutine = True
                                break  # at least one component has not yet finished
                        
                        # refresh the screen
                        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                            win.flip()
                    
                    # --- Ending Routine "taskPracticeTrialFeedback" ---
                    for thisComponent in taskPracticeTrialFeedback.components:
                        if hasattr(thisComponent, "setAutoDraw"):
                            thisComponent.setAutoDraw(False)
                    # store stop times for taskPracticeTrialFeedback
                    taskPracticeTrialFeedback.tStop = globalClock.getTime(format='float')
                    taskPracticeTrialFeedback.tStopRefresh = tThisFlipGlobal
                    thisExp.addData('taskPracticeTrialFeedback.stopped', taskPracticeTrialFeedback.tStop)
                    # Run 'End Routine' code from codeTaskPracticeTrialFeedback
                    # clear all objects
                    for i in range(len(objs_display)):
                        display_obj = objs_display[i]
                        display_obj.setAutoDraw(False)
                        
                    objs_display = []
                    # store data for taskPracticeTrialLoop (TrialHandler)
                    # the Routine "taskPracticeTrialFeedback" was not non-slip safe, so reset the non-slip timer
                    routineTimer.reset()
                # completed n_trials_to_practice repeats of 'taskPracticeTrialLoop'
                
                
                # --- Prepare to start Routine "summarizeTrialBlock" ---
                # create an object to store info about Routine summarizeTrialBlock
                summarizeTrialBlock = data.Routine(
                    name='summarizeTrialBlock',
                    components=[],
                )
                summarizeTrialBlock.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from codeSummarizeTrialBlock
                if not has_finished_instruction:
                    has_finished_instruction = True
                # store start times for summarizeTrialBlock
                summarizeTrialBlock.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                summarizeTrialBlock.tStart = globalClock.getTime(format='float')
                summarizeTrialBlock.status = STARTED
                thisExp.addData('summarizeTrialBlock.started', summarizeTrialBlock.tStart)
                summarizeTrialBlock.maxDuration = None
                # keep track of which components have finished
                summarizeTrialBlockComponents = summarizeTrialBlock.components
                for thisComponent in summarizeTrialBlock.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "summarizeTrialBlock" ---
                # if trial has changed, end Routine now
                if isinstance(taskPracticeBlockLoop, data.TrialHandler2) and thisTaskPracticeBlockLoop.thisN != taskPracticeBlockLoop.thisTrial.thisN:
                    continueRoutine = False
                summarizeTrialBlock.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        summarizeTrialBlock.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in summarizeTrialBlock.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "summarizeTrialBlock" ---
                for thisComponent in summarizeTrialBlock.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for summarizeTrialBlock
                summarizeTrialBlock.tStop = globalClock.getTime(format='float')
                summarizeTrialBlock.tStopRefresh = tThisFlipGlobal
                thisExp.addData('summarizeTrialBlock.stopped', summarizeTrialBlock.tStop)
                # the Routine "summarizeTrialBlock" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
            # completed max_instruction_practice_loop repeats of 'taskPracticeBlockLoop'
            
        # completed max(0, n_response_modes_allowed) repeats of 'TaskInstrLoop'
        
        
        # --- Prepare to start Routine "connectEL" ---
        # create an object to store info about Routine connectEL
        connectEL = data.Routine(
            name='connectEL',
            components=[],
        )
        connectEL.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from connectTracker
        # Step 1: Connect to the EyeLink Host PC
        host_ip = "100.1.1.1"
        if eyetracking == 1:
            try:
                el_tracker = pylink.EyeLink(host_ip)
            except RuntimeError as error:
                print('ERROR:', error)
                core.quit()
                sys.exit()
                
            #if already opened edf file, dont do this again
            if TasktypeLoop.thisN == 0 : 
                # Step 2: Open an EDF data file on the Host PC
                edf_file = str(expInfo['participant']) + ".EDF"
                try:
                    el_tracker.openDataFile(edf_file)
                except RuntimeError as err:
                    print('ERROR:', err)
                    # close the link if we have one open
                    if el_tracker.isConnected():
                        el_tracker.close()
                    core.quit()
                    sys.exit()
        
            # Step 3: Configure the tracker
            # Put the tracker in offline mode before we change tracking parameters
            el_tracker.setOfflineMode()
            # File and Link data control
            # what eye events to save in the EDF file, include everything by default
            file_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT'
            file_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,HREF,RAW,PUPIL,AREA,HTARGET,STATUS,INPUT'
            # what eye events to make available over the link, include everything by default
            link_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON,FIXUPDATE,INPUT'
            link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,HTARGET,STATUS,INPUT'
            el_tracker.sendCommand("file_event_filter = %s" % file_event_flags)
            el_tracker.sendCommand("file_sample_data = %s" % file_sample_flags)
            el_tracker.sendCommand("link_event_filter = %s" % link_event_flags)
            el_tracker.sendCommand("link_sample_data = %s" % link_sample_flags)
            #set calibration style
            el_tracker.sendCommand("calibration_type = HV%s" % str(calib_style))
            #set sampling rate
            el_tracker.sendCommand("sample_rate %s" % str(samprate))
         
            # Step 4: set up a graphics environment for calibration
            # get the native screen resolution used by PsychoPy
            scn_width, scn_height = win.size
            # resolution fix for Mac retina displays
            if 'Darwin' in platform.system():
                if use_retina:
                    scn_width = int(scn_width/2.0)
                    scn_height = int(scn_height/2.0)
            # Pass the display pixel coordinates (left, top, right, bottom) to the tracker
            el_coords = "screen_pixel_coords = 0 0 %d %d" % (scn_width - 1, scn_height - 1)
            el_tracker.sendCommand(el_coords)
            # Write a DISPLAY_COORDS message to the EDF file
            # Data Viewer needs this piece of info for proper visualization
            dv_coords = "DISPLAY_COORDS  0 0 %d %d" % (scn_width - 1, scn_height - 1)
            el_tracker.sendMessage(dv_coords)
            
            if TasktypeLoop.thisN == 0:### added to handle 2 calibs
                # Configure a graphics environment (genv) for tracker calibration
                genv = EyeLinkCoreGraphicsPsychoPy(el_tracker, win)
                # Set background and foreground colors for the calibration target
                # in PsychoPy, (-1, -1, -1)=black, (1, 1, 1)=white, (0, 0, 0)=mid-gray
                foreground_color = (-1, -1, -1)
                background_color = win.color
                genv.setCalibrationColors(foreground_color, background_color)
                genv.setTargetSize(calib_tar_size)
                # Request Pylink to use the PsychoPy window we opened above for calibration
                pylink.openGraphicsEx(genv)
            task_msg = 'Press <space>, then <enter> to start calibration'
            show_msg(win, task_msg)
            print('line69')
            
            try:
                el_tracker.doTrackerSetup()
            except RuntimeError as err:
                print('ERROR:', err)
                el_tracker.exitCalibration()
            
        else:
            continueRoutine = False
        
        # store start times for connectEL
        connectEL.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        connectEL.tStart = globalClock.getTime(format='float')
        connectEL.status = STARTED
        thisExp.addData('connectEL.started', connectEL.tStart)
        connectEL.maxDuration = None
        # keep track of which components have finished
        connectELComponents = connectEL.components
        for thisComponent in connectEL.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "connectEL" ---
        # if trial has changed, end Routine now
        if isinstance(TasktypeLoop, data.TrialHandler2) and thisTasktypeLoop.thisN != TasktypeLoop.thisTrial.thisN:
            continueRoutine = False
        connectEL.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                connectEL.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in connectEL.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "connectEL" ---
        for thisComponent in connectEL.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for connectEL
        connectEL.tStop = globalClock.getTime(format='float')
        connectEL.tStopRefresh = tThisFlipGlobal
        thisExp.addData('connectEL.stopped', connectEL.tStop)
        # Run 'End Routine' code from connectTracker
        #dont record yet
        if eyetracking == 1:
            el_tracker.setOfflineMode()
        win.mouseVisible = True
        
        # the Routine "connectEL" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        ActualTaskLoop = data.TrialHandler2(
            name='ActualTaskLoop',
            nReps=max(0, n_response_modes_allowed), 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(ActualTaskLoop)  # add the loop to the experiment
        thisActualTaskLoop = ActualTaskLoop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisActualTaskLoop.rgb)
        if thisActualTaskLoop != None:
            for paramName in thisActualTaskLoop:
                globals()[paramName] = thisActualTaskLoop[paramName]
        
        for thisActualTaskLoop in ActualTaskLoop:
            currentLoop = ActualTaskLoop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # abbreviate parameter names if possible (e.g. rgb = thisActualTaskLoop.rgb)
            if thisActualTaskLoop != None:
                for paramName in thisActualTaskLoop:
                    globals()[paramName] = thisActualTaskLoop[paramName]
            
            # --- Prepare to start Routine "prepareTaskPhase" ---
            # create an object to store info about Routine prepareTaskPhase
            prepareTaskPhase = data.Routine(
                name='prepareTaskPhase',
                components=[mousePrepareTaskPhase, textPrepareTaskPhase],
            )
            prepareTaskPhase.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from codePrepareTaskPhase
            # set the trial to run
            has_finished_instruction = True # old-ver compatible
            has_finished_rehearsal = True # old-ver compatible
            
            # trial_type already defined
            # response mode need to be updated
            response_mode = mode_type_list[ActualTaskLoop.thisN]
            
            # check sampling methods
            # the order of sampling strategies
            sampling_type_list = random.sample(sampling_strategies_allowed, n_sampling_strategies_allowed)
            stim_sampler = StimSamplingScheduler(n_ori_regions, sampling_type_list)
            
            # create the continue button
            continue_button = funcCreateButton(
                win, 'continue', 
                [0, -0.4], [0.25, 0.1])
                
            # record phase id for eyetracker
            EYETRACK_task_mode_phase_id = TasktypeLoop.thisN * n_response_modes_allowed + ActualTaskLoop.thisN
            
            # setup some python lists for storing info about the mousePrepareTaskPhase
            gotValidClick = False  # until a click is received
            textPrepareTaskPhase.setText('Now the following are ACTUAL trials,\n where you need to ' + response_mode + ' to report your answer.\n For this part, you need to complete ' + str(n_blocks_each_subtask) + ' blocks,\n each consists of ' + str(n_trials_each_block) + ' trials.'   )
            # store start times for prepareTaskPhase
            prepareTaskPhase.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            prepareTaskPhase.tStart = globalClock.getTime(format='float')
            prepareTaskPhase.status = STARTED
            thisExp.addData('prepareTaskPhase.started', prepareTaskPhase.tStart)
            prepareTaskPhase.maxDuration = None
            # keep track of which components have finished
            prepareTaskPhaseComponents = prepareTaskPhase.components
            for thisComponent in prepareTaskPhase.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "prepareTaskPhase" ---
            # if trial has changed, end Routine now
            if isinstance(ActualTaskLoop, data.TrialHandler2) and thisActualTaskLoop.thisN != ActualTaskLoop.thisTrial.thisN:
                continueRoutine = False
            prepareTaskPhase.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from codePrepareTaskPhase
                left_button = mousePrepareTaskPhase.getPressed()[0]
                mouse_position = mousePrepareTaskPhase.getPos()
                if left_button and (t > 0.5) and continue_button.contains(mouse_position):
                    continueRoutine = False
                # *mousePrepareTaskPhase* updates
                
                # if mousePrepareTaskPhase is starting this frame...
                if mousePrepareTaskPhase.status == NOT_STARTED and t >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    mousePrepareTaskPhase.frameNStart = frameN  # exact frame index
                    mousePrepareTaskPhase.tStart = t  # local t and not account for scr refresh
                    mousePrepareTaskPhase.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(mousePrepareTaskPhase, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    mousePrepareTaskPhase.status = STARTED
                    mousePrepareTaskPhase.mouseClock.reset()
                    prevButtonState = mousePrepareTaskPhase.getPressed()  # if button is down already this ISN'T a new click
                
                # *textPrepareTaskPhase* updates
                
                # if textPrepareTaskPhase is starting this frame...
                if textPrepareTaskPhase.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textPrepareTaskPhase.frameNStart = frameN  # exact frame index
                    textPrepareTaskPhase.tStart = t  # local t and not account for scr refresh
                    textPrepareTaskPhase.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textPrepareTaskPhase, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textPrepareTaskPhase.started')
                    # update status
                    textPrepareTaskPhase.status = STARTED
                    textPrepareTaskPhase.setAutoDraw(True)
                
                # if textPrepareTaskPhase is active this frame...
                if textPrepareTaskPhase.status == STARTED:
                    # update params
                    pass
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    prepareTaskPhase.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in prepareTaskPhase.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "prepareTaskPhase" ---
            for thisComponent in prepareTaskPhase.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for prepareTaskPhase
            prepareTaskPhase.tStop = globalClock.getTime(format='float')
            prepareTaskPhase.tStopRefresh = tThisFlipGlobal
            thisExp.addData('prepareTaskPhase.stopped', prepareTaskPhase.tStop)
            # Run 'End Routine' code from codePrepareTaskPhase
            continue_button.setAutoDraw(False)
            continue_button = None
            # store data for ActualTaskLoop (TrialHandler)
            # the Routine "prepareTaskPhase" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # set up handler to look after randomisation of conditions etc
            sub_sample_stage_loop = data.TrialHandler2(
                name='sub_sample_stage_loop',
                nReps=n_sampling_strategies_allowed, 
                method='sequential', 
                extraInfo=expInfo, 
                originPath=-1, 
                trialList=[None], 
                seed=None, 
            )
            thisExp.addLoop(sub_sample_stage_loop)  # add the loop to the experiment
            thisSub_sample_stage_loop = sub_sample_stage_loop.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisSub_sample_stage_loop.rgb)
            if thisSub_sample_stage_loop != None:
                for paramName in thisSub_sample_stage_loop:
                    globals()[paramName] = thisSub_sample_stage_loop[paramName]
            
            for thisSub_sample_stage_loop in sub_sample_stage_loop:
                currentLoop = sub_sample_stage_loop
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                # abbreviate parameter names if possible (e.g. rgb = thisSub_sample_stage_loop.rgb)
                if thisSub_sample_stage_loop != None:
                    for paramName in thisSub_sample_stage_loop:
                        globals()[paramName] = thisSub_sample_stage_loop[paramName]
                
                # --- Prepare to start Routine "prepareSubSampleStage" ---
                # create an object to store info about Routine prepareSubSampleStage
                prepareSubSampleStage = data.Routine(
                    name='prepareSubSampleStage',
                    components=[],
                )
                prepareSubSampleStage.status = NOT_STARTED
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from codePrepareSubSampleStage
                stim_sampler.initialize_stage()
                
                # for eye tracker
                EYETRACK_real_stage_id = EYETRACK_task_mode_phase_id * n_sampling_strategies_allowed + sub_sample_stage_loop.thisN
                
                # store start times for prepareSubSampleStage
                prepareSubSampleStage.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                prepareSubSampleStage.tStart = globalClock.getTime(format='float')
                prepareSubSampleStage.status = STARTED
                thisExp.addData('prepareSubSampleStage.started', prepareSubSampleStage.tStart)
                prepareSubSampleStage.maxDuration = None
                # keep track of which components have finished
                prepareSubSampleStageComponents = prepareSubSampleStage.components
                for thisComponent in prepareSubSampleStage.components:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "prepareSubSampleStage" ---
                # if trial has changed, end Routine now
                if isinstance(sub_sample_stage_loop, data.TrialHandler2) and thisSub_sample_stage_loop.thisN != sub_sample_stage_loop.thisTrial.thisN:
                    continueRoutine = False
                prepareSubSampleStage.forceEnded = routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                        )
                        # skip the frame we paused on
                        continue
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        prepareSubSampleStage.forceEnded = routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in prepareSubSampleStage.components:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "prepareSubSampleStage" ---
                for thisComponent in prepareSubSampleStage.components:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # store stop times for prepareSubSampleStage
                prepareSubSampleStage.tStop = globalClock.getTime(format='float')
                prepareSubSampleStage.tStopRefresh = tThisFlipGlobal
                thisExp.addData('prepareSubSampleStage.stopped', prepareSubSampleStage.tStop)
                # the Routine "prepareSubSampleStage" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                
                # set up handler to look after randomisation of conditions etc
                block_stage = data.TrialHandler2(
                    name='block_stage',
                    nReps=n_blocks_each_subtype, 
                    method='sequential', 
                    extraInfo=expInfo, 
                    originPath=-1, 
                    trialList=[None], 
                    seed=None, 
                )
                thisExp.addLoop(block_stage)  # add the loop to the experiment
                thisBlock_stage = block_stage.trialList[0]  # so we can initialise stimuli with some values
                # abbreviate parameter names if possible (e.g. rgb = thisBlock_stage.rgb)
                if thisBlock_stage != None:
                    for paramName in thisBlock_stage:
                        globals()[paramName] = thisBlock_stage[paramName]
                
                for thisBlock_stage in block_stage:
                    currentLoop = block_stage
                    thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                    # abbreviate parameter names if possible (e.g. rgb = thisBlock_stage.rgb)
                    if thisBlock_stage != None:
                        for paramName in thisBlock_stage:
                            globals()[paramName] = thisBlock_stage[paramName]
                    
                    # --- Prepare to start Routine "prepare_block" ---
                    # create an object to store info about Routine prepare_block
                    prepare_block = data.Routine(
                        name='prepare_block',
                        components=[textPrepareBlock, mousePrepareBlock],
                    )
                    prepare_block.status = NOT_STARTED
                    continueRoutine = True
                    # update component parameters for each repeat
                    # Run 'Begin Routine' code from codePrepareBlock
                    block_instruction = ''
                    block_idx = block_stage.thisN
                    real_block_id = sub_sample_stage_loop.thisN * n_blocks_each_subtype + block_idx
                    display_block_idx = real_block_id + 1
                    
                    block_instruction = "Block " + str(display_block_idx) + "/" + str(n_blocks_each_subtask) + ".\n\n" 
                    block_instruction += "This block consists of " + str(n_trials_each_block) + " trials. " 
                    block_instruction += "You can get feedback about your performance and take a short break after completing this block.\n\n"
                    
                    # initialize sampler
                    stim_sampler.initialize_block()
                        
                    # set the text
                    textPrepareBlock.setText(block_instruction)
                    
                    # add the button to continue
                    continue_button = funcCreateButton(
                        win, 'continue', 
                        [0, -0.4], [0.25, 0.1])
                    
                    # keep track of errors
                    block_error_records = []
                    
                    # block id for eye tracker
                    EYETRACK_real_block_id = EYETRACK_real_stage_id * n_blocks_each_subtype + block_idx
                    
                    # setup some python lists for storing info about the mousePrepareBlock
                    gotValidClick = False  # until a click is received
                    # store start times for prepare_block
                    prepare_block.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                    prepare_block.tStart = globalClock.getTime(format='float')
                    prepare_block.status = STARTED
                    thisExp.addData('prepare_block.started', prepare_block.tStart)
                    prepare_block.maxDuration = None
                    # keep track of which components have finished
                    prepare_blockComponents = prepare_block.components
                    for thisComponent in prepare_block.components:
                        thisComponent.tStart = None
                        thisComponent.tStop = None
                        thisComponent.tStartRefresh = None
                        thisComponent.tStopRefresh = None
                        if hasattr(thisComponent, 'status'):
                            thisComponent.status = NOT_STARTED
                    # reset timers
                    t = 0
                    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                    frameN = -1
                    
                    # --- Run Routine "prepare_block" ---
                    # if trial has changed, end Routine now
                    if isinstance(block_stage, data.TrialHandler2) and thisBlock_stage.thisN != block_stage.thisTrial.thisN:
                        continueRoutine = False
                    prepare_block.forceEnded = routineForceEnded = not continueRoutine
                    while continueRoutine:
                        # get current time
                        t = routineTimer.getTime()
                        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                        # update/draw components on each frame
                        # Run 'Each Frame' code from codePrepareBlock
                        left_button = mousePrepareBlock.getPressed()[0]
                        mouse_position = mousePrepareBlock.getPos()
                        if left_button and (t > 0.5) and continue_button.contains(mouse_position):
                            continueRoutine = False
                        
                        
                        # *textPrepareBlock* updates
                        
                        # if textPrepareBlock is starting this frame...
                        if textPrepareBlock.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            textPrepareBlock.frameNStart = frameN  # exact frame index
                            textPrepareBlock.tStart = t  # local t and not account for scr refresh
                            textPrepareBlock.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(textPrepareBlock, 'tStartRefresh')  # time at next scr refresh
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'textPrepareBlock.started')
                            # update status
                            textPrepareBlock.status = STARTED
                            textPrepareBlock.setAutoDraw(True)
                        
                        # if textPrepareBlock is active this frame...
                        if textPrepareBlock.status == STARTED:
                            # update params
                            pass
                        # *mousePrepareBlock* updates
                        
                        # if mousePrepareBlock is starting this frame...
                        if mousePrepareBlock.status == NOT_STARTED and t >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            mousePrepareBlock.frameNStart = frameN  # exact frame index
                            mousePrepareBlock.tStart = t  # local t and not account for scr refresh
                            mousePrepareBlock.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(mousePrepareBlock, 'tStartRefresh')  # time at next scr refresh
                            # update status
                            mousePrepareBlock.status = STARTED
                            mousePrepareBlock.mouseClock.reset()
                            prevButtonState = mousePrepareBlock.getPressed()  # if button is down already this ISN'T a new click
                        
                        # check for quit (typically the Esc key)
                        if defaultKeyboard.getKeys(keyList=["escape"]):
                            thisExp.status = FINISHED
                        if thisExp.status == FINISHED or endExpNow:
                            endExperiment(thisExp, win=win)
                            return
                        # pause experiment here if requested
                        if thisExp.status == PAUSED:
                            pauseExperiment(
                                thisExp=thisExp, 
                                win=win, 
                                timers=[routineTimer], 
                                playbackComponents=[]
                            )
                            # skip the frame we paused on
                            continue
                        
                        # check if all components have finished
                        if not continueRoutine:  # a component has requested a forced-end of Routine
                            prepare_block.forceEnded = routineForceEnded = True
                            break
                        continueRoutine = False  # will revert to True if at least one component still running
                        for thisComponent in prepare_block.components:
                            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                continueRoutine = True
                                break  # at least one component has not yet finished
                        
                        # refresh the screen
                        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                            win.flip()
                    
                    # --- Ending Routine "prepare_block" ---
                    for thisComponent in prepare_block.components:
                        if hasattr(thisComponent, "setAutoDraw"):
                            thisComponent.setAutoDraw(False)
                    # store stop times for prepare_block
                    prepare_block.tStop = globalClock.getTime(format='float')
                    prepare_block.tStopRefresh = tThisFlipGlobal
                    thisExp.addData('prepare_block.stopped', prepare_block.tStop)
                    # Run 'End Routine' code from codePrepareBlock
                    continue_button.setAutoDraw(False)
                    continue_button = None
                    
                    # hide the cursor
                    win.mouseVisible = False
                    # store data for block_stage (TrialHandler)
                    # the Routine "prepare_block" was not non-slip safe, so reset the non-slip timer
                    routineTimer.reset()
                    
                    # --- Prepare to start Routine "drift_check" ---
                    # create an object to store info about Routine drift_check
                    drift_check = data.Routine(
                        name='drift_check',
                        components=[],
                    )
                    drift_check.status = NOT_STARTED
                    continueRoutine = True
                    # update component parameters for each repeat
                    # Run 'Begin Routine' code from drift_check_code
                    if eyetracking == 1:
                        # drift-check and re-do camera setup if ESCAPE is pressed
                        try:
                            error = el_tracker.doDriftCorrect(int(scn_width/2.0),
                                                              int(scn_height/2.0), 1, 1)
                            # break following a success drift-check
                        except:
                            continue
                        # put tracker in idle/offline mode before recording
                        el_tracker.setOfflineMode()
                    # store start times for drift_check
                    drift_check.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                    drift_check.tStart = globalClock.getTime(format='float')
                    drift_check.status = STARTED
                    thisExp.addData('drift_check.started', drift_check.tStart)
                    drift_check.maxDuration = None
                    # keep track of which components have finished
                    drift_checkComponents = drift_check.components
                    for thisComponent in drift_check.components:
                        thisComponent.tStart = None
                        thisComponent.tStop = None
                        thisComponent.tStartRefresh = None
                        thisComponent.tStopRefresh = None
                        if hasattr(thisComponent, 'status'):
                            thisComponent.status = NOT_STARTED
                    # reset timers
                    t = 0
                    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                    frameN = -1
                    
                    # --- Run Routine "drift_check" ---
                    # if trial has changed, end Routine now
                    if isinstance(block_stage, data.TrialHandler2) and thisBlock_stage.thisN != block_stage.thisTrial.thisN:
                        continueRoutine = False
                    drift_check.forceEnded = routineForceEnded = not continueRoutine
                    while continueRoutine:
                        # get current time
                        t = routineTimer.getTime()
                        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                        # update/draw components on each frame
                        
                        # check for quit (typically the Esc key)
                        if defaultKeyboard.getKeys(keyList=["escape"]):
                            thisExp.status = FINISHED
                        if thisExp.status == FINISHED or endExpNow:
                            endExperiment(thisExp, win=win)
                            return
                        # pause experiment here if requested
                        if thisExp.status == PAUSED:
                            pauseExperiment(
                                thisExp=thisExp, 
                                win=win, 
                                timers=[routineTimer], 
                                playbackComponents=[]
                            )
                            # skip the frame we paused on
                            continue
                        
                        # check if all components have finished
                        if not continueRoutine:  # a component has requested a forced-end of Routine
                            drift_check.forceEnded = routineForceEnded = True
                            break
                        continueRoutine = False  # will revert to True if at least one component still running
                        for thisComponent in drift_check.components:
                            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                continueRoutine = True
                                break  # at least one component has not yet finished
                        
                        # refresh the screen
                        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                            win.flip()
                    
                    # --- Ending Routine "drift_check" ---
                    for thisComponent in drift_check.components:
                        if hasattr(thisComponent, "setAutoDraw"):
                            thisComponent.setAutoDraw(False)
                    # store stop times for drift_check
                    drift_check.tStop = globalClock.getTime(format='float')
                    drift_check.tStopRefresh = tThisFlipGlobal
                    thisExp.addData('drift_check.stopped', drift_check.tStop)
                    # the Routine "drift_check" was not non-slip safe, so reset the non-slip timer
                    routineTimer.reset()
                    
                    # set up handler to look after randomisation of conditions etc
                    block = data.TrialHandler2(
                        name='block',
                        nReps=n_trials_each_block, 
                        method='sequential', 
                        extraInfo=expInfo, 
                        originPath=-1, 
                        trialList=[None], 
                        seed=None, 
                    )
                    thisExp.addLoop(block)  # add the loop to the experiment
                    thisBlock = block.trialList[0]  # so we can initialise stimuli with some values
                    # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
                    if thisBlock != None:
                        for paramName in thisBlock:
                            globals()[paramName] = thisBlock[paramName]
                    if thisSession is not None:
                        # if running in a Session with a Liaison client, send data up to now
                        thisSession.sendExperimentData()
                    
                    for thisBlock in block:
                        currentLoop = block
                        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                        if thisSession is not None:
                            # if running in a Session with a Liaison client, send data up to now
                            thisSession.sendExperimentData()
                        # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
                        if thisBlock != None:
                            for paramName in thisBlock:
                                globals()[paramName] = thisBlock[paramName]
                        
                        # --- Prepare to start Routine "prepareSingleTrial" ---
                        # create an object to store info about Routine prepareSingleTrial
                        prepareSingleTrial = data.Routine(
                            name='prepareSingleTrial',
                            components=[],
                        )
                        prepareSingleTrial.status = NOT_STARTED
                        continueRoutine = True
                        # update component parameters for each repeat
                        # Run 'Begin Routine' code from codePrepareSingleTrial
                        # initialize stimuli
                        # during instruction, two stims are forced to be from different regions
                        allow_same_region = False
                        if has_finished_instruction:
                            # in real trials, 2/3 allow same region
                            # allow_same_region = random.random() < 2/3
                            allow_same_region = False
                        stim_region_ids_selected, stims_selected = stim_sampler.sample_for_trial(
                            allow_same_region=allow_same_region)
                            
                        # initialize which to report
                        # 0: to report only the first
                        # 1: to report only the second
                        # 2 or 3: to report both
                        report_type_id = random.randint(0, 3)
                        if not has_finished_instruction:
                            report_type_id = block.thisN
                        to_report_code = [True, True]
                        if report_type_id == 0:
                            to_report_code[1] = False
                        elif report_type_id == 1:
                            to_report_code[0] = False
                            
                        # define the color of cues
                        color_codes = (1, 2)
                        if random.random() < 0.5:
                            color_codes = (2, 1)
                        
                        # place holder: to record the future response
                        participant_responses = [None, None]
                        
                        # set where to display stimuli
                        # eyetracking: all at the center
                        stim_locs = [(0, 0), (0, 0)]
                        # select response locations
                        resp_locs = resp_location_sampler.sample_location(allow_location_swap)
                        
                        # set time limits
                        ## total max time for response
                        response_time = draw_response_time if response_mode == "draw" else click_response_time
                        ## for display
                        trial_display_time = display_time
                        ## for delay
                        delay_time = 4 # random.uniform(delay_time_min, delay_time_max)
                        ## for response
                        trial_resp_time = response_time * sum(to_report_code)
                        # generate a reasonable range of ITI total duration
                        total_duration_min = delay_time_min \
                            + display_time + response_time * sum(to_report_code) * 0.5 + post_trial_time_min
                        total_duration_max = delay_time_max \
                            + display_time + response_time * sum(to_report_code) * 0.5 + post_trial_time_max * 0.5
                        
                        # make things slower for instruction
                        slow_down_factor = 1.5
                        if not has_finished_instruction:
                            trial_display_time *= slow_down_factor
                            display_cue_onset_time *= slow_down_factor
                            trial_resp_time *= slow_down_factor
                        
                        # start the timer for the trial
                        trial_timer = core.Clock()
                        
                        # Run 'Begin Routine' code from trlIDSetup
                        trlId = (expInfo['participant'],block.thisN, EYETRACK_real_block_id)
                        thisExp.addData('TRIALID',trlId)
                        # store start times for prepareSingleTrial
                        prepareSingleTrial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                        prepareSingleTrial.tStart = globalClock.getTime(format='float')
                        prepareSingleTrial.status = STARTED
                        thisExp.addData('prepareSingleTrial.started', prepareSingleTrial.tStart)
                        prepareSingleTrial.maxDuration = None
                        # keep track of which components have finished
                        prepareSingleTrialComponents = prepareSingleTrial.components
                        for thisComponent in prepareSingleTrial.components:
                            thisComponent.tStart = None
                            thisComponent.tStop = None
                            thisComponent.tStartRefresh = None
                            thisComponent.tStopRefresh = None
                            if hasattr(thisComponent, 'status'):
                                thisComponent.status = NOT_STARTED
                        # reset timers
                        t = 0
                        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                        frameN = -1
                        
                        # --- Run Routine "prepareSingleTrial" ---
                        # if trial has changed, end Routine now
                        if isinstance(block, data.TrialHandler2) and thisBlock.thisN != block.thisTrial.thisN:
                            continueRoutine = False
                        prepareSingleTrial.forceEnded = routineForceEnded = not continueRoutine
                        while continueRoutine:
                            # get current time
                            t = routineTimer.getTime()
                            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                            # update/draw components on each frame
                            
                            # check for quit (typically the Esc key)
                            if defaultKeyboard.getKeys(keyList=["escape"]):
                                thisExp.status = FINISHED
                            if thisExp.status == FINISHED or endExpNow:
                                endExperiment(thisExp, win=win)
                                return
                            # pause experiment here if requested
                            if thisExp.status == PAUSED:
                                pauseExperiment(
                                    thisExp=thisExp, 
                                    win=win, 
                                    timers=[routineTimer], 
                                    playbackComponents=[]
                                )
                                # skip the frame we paused on
                                continue
                            
                            # check if all components have finished
                            if not continueRoutine:  # a component has requested a forced-end of Routine
                                prepareSingleTrial.forceEnded = routineForceEnded = True
                                break
                            continueRoutine = False  # will revert to True if at least one component still running
                            for thisComponent in prepareSingleTrial.components:
                                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                    continueRoutine = True
                                    break  # at least one component has not yet finished
                            
                            # refresh the screen
                            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                                win.flip()
                        
                        # --- Ending Routine "prepareSingleTrial" ---
                        for thisComponent in prepareSingleTrial.components:
                            if hasattr(thisComponent, "setAutoDraw"):
                                thisComponent.setAutoDraw(False)
                        # store stop times for prepareSingleTrial
                        prepareSingleTrial.tStop = globalClock.getTime(format='float')
                        prepareSingleTrial.tStopRefresh = tThisFlipGlobal
                        thisExp.addData('prepareSingleTrial.stopped', prepareSingleTrial.tStop)
                        # Run 'End Routine' code from codePrepareSingleTrial
                        # record data
                        # record whether this is a practice trial and other misc
                        thisExp.addData('is_real_trial', has_finished_rehearsal)
                        thisExp.addData('sample_stage', stim_sampler.stage_idx)
                        thisExp.addData('block', stim_sampler.block_idx)
                        
                        ## record the stimuli
                        thisExp.addData('stim_1', stims_selected[0])
                        thisExp.addData('stim_2', stims_selected[1])
                        thisExp.addData('stim_region_1', stim_region_ids_selected[0])
                        thisExp.addData('stim_region_2', stim_region_ids_selected[1])
                        thisExp.addData('stim_1_loc_x', stim_locs[0][0])
                        thisExp.addData('stim_1_loc_y', stim_locs[0][1])
                        thisExp.addData('stim_2_loc_x', stim_locs[1][0])
                        thisExp.addData('stim_2_loc_y', stim_locs[1][1])
                        thisExp.addData('stim_1_color', color_codes[0])
                        thisExp.addData('stim_2_color', color_codes[1])
                        
                        ## record the trial type
                        thisExp.addData('mode', response_mode)
                        thisExp.addData('trial_code', trial_type)
                        ## record which is asked to report
                        thisExp.addData('stim_1_to_report', to_report_code[0])
                        thisExp.addData('stim_2_to_report', to_report_code[1])
                        ## record delay
                        thisExp.addData('delay_post_stim', delay_post_stim)
                        thisExp.addData('delay', delay_time)
                        ## record response position
                        thisExp.addData('resp_1_loc_x', resp_locs[0][0])
                        thisExp.addData('resp_1_loc_y', resp_locs[0][1])
                        thisExp.addData('resp_2_loc_x', resp_locs[1][0])
                        thisExp.addData('resp_2_loc_y', resp_locs[1][1])
                        ## record the strategy for stim generation
                        thisExp.addData('stim_sample_method', stim_sampler.strategy)
                        
                        # the Routine "prepareSingleTrial" was not non-slip safe, so reset the non-slip timer
                        routineTimer.reset()
                        
                        # --- Prepare to start Routine "pre_trial" ---
                        # create an object to store info about Routine pre_trial
                        pre_trial = data.Routine(
                            name='pre_trial',
                            components=[textPreTrial],
                        )
                        pre_trial.status = NOT_STARTED
                        continueRoutine = True
                        # update component parameters for each repeat
                        # Run 'Begin Routine' code from codePreTrial
                        # create the fixation
                        pre_fixation = DefaultFixation(win)
                        textPreTrial.setText('')
                        # Run 'Begin Routine' code from elRecord_fixation
                        this_epoch = 'fixation'
                        aaa = core.monotonicClock.getTime
                        thisExp.addData('fixationStart',str(aaa()))
                        if eyetracking == 1:
                            # get a reference to the currently active EyeLink connection
                            el_tracker = pylink.getEYELINK()
                        
                            try:
                                #start recording
                                el_tracker.startRecording(1, 1, 1, 1) 
                                #send message to tracker to count trial number
                                el_tracker.sendMessage('TRIALID %s' % str(trlId))
                                el_tracker.sendMessage('fixationRest')
                            except RuntimeError as error:
                                print("ERROR:", error)
                                abort_trial()
                        
                            
                        
                        # store start times for pre_trial
                        pre_trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                        pre_trial.tStart = globalClock.getTime(format='float')
                        pre_trial.status = STARTED
                        thisExp.addData('pre_trial.started', pre_trial.tStart)
                        pre_trial.maxDuration = None
                        # keep track of which components have finished
                        pre_trialComponents = pre_trial.components
                        for thisComponent in pre_trial.components:
                            thisComponent.tStart = None
                            thisComponent.tStop = None
                            thisComponent.tStartRefresh = None
                            thisComponent.tStopRefresh = None
                            if hasattr(thisComponent, 'status'):
                                thisComponent.status = NOT_STARTED
                        # reset timers
                        t = 0
                        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                        frameN = -1
                        
                        # --- Run Routine "pre_trial" ---
                        # if trial has changed, end Routine now
                        if isinstance(block, data.TrialHandler2) and thisBlock.thisN != block.thisTrial.thisN:
                            continueRoutine = False
                        pre_trial.forceEnded = routineForceEnded = not continueRoutine
                        while continueRoutine:
                            # get current time
                            t = routineTimer.getTime()
                            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                            # update/draw components on each frame
                            # Run 'Each Frame' code from codePreTrial
                            # end routine when time up
                            if t > pre_trial_time:
                                continueRoutine = False
                            
                            # *textPreTrial* updates
                            
                            # if textPreTrial is starting this frame...
                            if textPreTrial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                                # keep track of start time/frame for later
                                textPreTrial.frameNStart = frameN  # exact frame index
                                textPreTrial.tStart = t  # local t and not account for scr refresh
                                textPreTrial.tStartRefresh = tThisFlipGlobal  # on global time
                                win.timeOnFlip(textPreTrial, 'tStartRefresh')  # time at next scr refresh
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'textPreTrial.started')
                                # update status
                                textPreTrial.status = STARTED
                                textPreTrial.setAutoDraw(True)
                            
                            # if textPreTrial is active this frame...
                            if textPreTrial.status == STARTED:
                                # update params
                                pass
                            
                            # check for quit (typically the Esc key)
                            if defaultKeyboard.getKeys(keyList=["escape"]):
                                thisExp.status = FINISHED
                            if thisExp.status == FINISHED or endExpNow:
                                endExperiment(thisExp, win=win)
                                return
                            # pause experiment here if requested
                            if thisExp.status == PAUSED:
                                pauseExperiment(
                                    thisExp=thisExp, 
                                    win=win, 
                                    timers=[routineTimer], 
                                    playbackComponents=[]
                                )
                                # skip the frame we paused on
                                continue
                            
                            # check if all components have finished
                            if not continueRoutine:  # a component has requested a forced-end of Routine
                                pre_trial.forceEnded = routineForceEnded = True
                                break
                            continueRoutine = False  # will revert to True if at least one component still running
                            for thisComponent in pre_trial.components:
                                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                    continueRoutine = True
                                    break  # at least one component has not yet finished
                            
                            # refresh the screen
                            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                                win.flip()
                        
                        # --- Ending Routine "pre_trial" ---
                        for thisComponent in pre_trial.components:
                            if hasattr(thisComponent, "setAutoDraw"):
                                thisComponent.setAutoDraw(False)
                        # store stop times for pre_trial
                        pre_trial.tStop = globalClock.getTime(format='float')
                        pre_trial.tStopRefresh = tThisFlipGlobal
                        thisExp.addData('pre_trial.stopped', pre_trial.tStop)
                        # Run 'End Routine' code from codePreTrial
                        pre_fixation.setAutoDraw(False)
                        pre_fixation = None
                        # Run 'End Routine' code from elRecord_fixation
                        aaa = core.monotonicClock.getTime
                        thisExp.addData('fixationEnd',str(aaa()))
                        
                        # the Routine "pre_trial" was not non-slip safe, so reset the non-slip timer
                        routineTimer.reset()
                        
                        # set up handler to look after randomisation of conditions etc
                        seq_display_loop = data.TrialHandler2(
                            name='seq_display_loop',
                            nReps=2.0, 
                            method='sequential', 
                            extraInfo=expInfo, 
                            originPath=-1, 
                            trialList=[None], 
                            seed=None, 
                        )
                        thisExp.addLoop(seq_display_loop)  # add the loop to the experiment
                        thisSeq_display_loop = seq_display_loop.trialList[0]  # so we can initialise stimuli with some values
                        # abbreviate parameter names if possible (e.g. rgb = thisSeq_display_loop.rgb)
                        if thisSeq_display_loop != None:
                            for paramName in thisSeq_display_loop:
                                globals()[paramName] = thisSeq_display_loop[paramName]
                        
                        for thisSeq_display_loop in seq_display_loop:
                            currentLoop = seq_display_loop
                            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                            # abbreviate parameter names if possible (e.g. rgb = thisSeq_display_loop.rgb)
                            if thisSeq_display_loop != None:
                                for paramName in thisSeq_display_loop:
                                    globals()[paramName] = thisSeq_display_loop[paramName]
                            
                            # --- Prepare to start Routine "display_stimuli" ---
                            # create an object to store info about Routine display_stimuli
                            display_stimuli = data.Routine(
                                name='display_stimuli',
                                components=[mouseStim, textDisplayStim],
                            )
                            display_stimuli.status = NOT_STARTED
                            continueRoutine = True
                            # update component parameters for each repeat
                            # setup some python lists for storing info about the mouseStim
                            gotValidClick = False  # until a click is received
                            # Run 'Begin Routine' code from codeDisplayStim
                            objs_display = []
                            
                            # fetch the id of the stimulus
                            i = seq_display_loop.thisN
                            
                            # now display the stimulus
                            ori = stims_selected[i]
                            loc = stim_locs[i]
                            stim_obj = funcDrawStim(win, ori, loc, patch_radius)
                            objs_display.append(stim_obj)
                            
                            # dump the no delay condition
                            
                            # to detect whether a pressed has released
                            has_released = True
                            
                            # to mark if the cue has displayed
                            has_cue_shown = False
                            
                            # record mouse
                            display_mouse_xs = []
                            display_mouse_ys = []
                            textDisplayStim.setText('')
                            # Run 'Begin Routine' code from elRecord_stim
                            this_epoch = 'stim'+str(seq_display_loop.thisN)
                            
                            aaa = core.monotonicClock.getTime
                            thisExp.addData(this_epoch+'Start',str(aaa()))
                            
                            if eyetracking == 1:
                                #el_tracker.startRecording(1, 1, 1, 1)
                                el_tracker.sendMessage(this_epoch)
                            
                            
                                
                            
                            # store start times for display_stimuli
                            display_stimuli.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                            display_stimuli.tStart = globalClock.getTime(format='float')
                            display_stimuli.status = STARTED
                            thisExp.addData('display_stimuli.started', display_stimuli.tStart)
                            display_stimuli.maxDuration = None
                            # keep track of which components have finished
                            display_stimuliComponents = display_stimuli.components
                            for thisComponent in display_stimuli.components:
                                thisComponent.tStart = None
                                thisComponent.tStop = None
                                thisComponent.tStartRefresh = None
                                thisComponent.tStopRefresh = None
                                if hasattr(thisComponent, 'status'):
                                    thisComponent.status = NOT_STARTED
                            # reset timers
                            t = 0
                            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                            frameN = -1
                            
                            # --- Run Routine "display_stimuli" ---
                            # if trial has changed, end Routine now
                            if isinstance(seq_display_loop, data.TrialHandler2) and thisSeq_display_loop.thisN != seq_display_loop.thisTrial.thisN:
                                continueRoutine = False
                            display_stimuli.forceEnded = routineForceEnded = not continueRoutine
                            while continueRoutine:
                                # get current time
                                t = routineTimer.getTime()
                                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                                # update/draw components on each frame
                                # *mouseStim* updates
                                
                                # if mouseStim is starting this frame...
                                if mouseStim.status == NOT_STARTED and t >= 0.0-frameTolerance:
                                    # keep track of start time/frame for later
                                    mouseStim.frameNStart = frameN  # exact frame index
                                    mouseStim.tStart = t  # local t and not account for scr refresh
                                    mouseStim.tStartRefresh = tThisFlipGlobal  # on global time
                                    win.timeOnFlip(mouseStim, 'tStartRefresh')  # time at next scr refresh
                                    # update status
                                    mouseStim.status = STARTED
                                    mouseStim.mouseClock.reset()
                                    prevButtonState = [0, 0, 0]  # if now button is down we will treat as 'new' click
                                
                                # if mouseStim is stopping this frame...
                                if mouseStim.status == STARTED:
                                    # is it time to stop? (based on global clock, using actual start)
                                    if tThisFlipGlobal > mouseStim.tStartRefresh + trial_display_time-frameTolerance:
                                        # keep track of stop time/frame for later
                                        mouseStim.tStop = t  # not accounting for scr refresh
                                        mouseStim.tStopRefresh = tThisFlipGlobal  # on global time
                                        mouseStim.frameNStop = frameN  # exact frame index
                                        # update status
                                        mouseStim.status = FINISHED
                                # Run 'Each Frame' code from codeDisplayStim
                                # display the cue when onset
                                if (not has_cue_shown) and t >= display_cue_onset_time:
                                    i = seq_display_loop.thisN
                                    cue_loc = stim_locs[i]
                                    cue_color = color_codes[i]
                                    report_cue = None
                                    cue_code = None
                                    if trial_type == 0 :
                                        # pre-cue
                                        cue_code = to_report_code[seq_display_loop.thisN]
                                    elif trial_type == 1:
                                        cue_code = True # remind them to remember it
                                    else:
                                        raise ValueError('Unknown trial type {trial_type}') 
                                    report_cue = SingleCueObject(
                                        win, cue_loc, cue_code, patch_radius, cue_color)
                                    objs_display.append(report_cue)
                                    has_cue_shown = True
                                
                                # monitor mouse behavior
                                mouse_position = mouseStim.getPos()
                                display_mouse_xs.append(mouse_position[0])
                                display_mouse_ys.append(mouse_position[1])
                                            
                                # end routine when time up
                                if t > trial_display_time:
                                    continueRoutine = False
                                
                                
                                # *textDisplayStim* updates
                                
                                # if textDisplayStim is starting this frame...
                                if textDisplayStim.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                                    # keep track of start time/frame for later
                                    textDisplayStim.frameNStart = frameN  # exact frame index
                                    textDisplayStim.tStart = t  # local t and not account for scr refresh
                                    textDisplayStim.tStartRefresh = tThisFlipGlobal  # on global time
                                    win.timeOnFlip(textDisplayStim, 'tStartRefresh')  # time at next scr refresh
                                    # add timestamp to datafile
                                    thisExp.timestampOnFlip(win, 'textDisplayStim.started')
                                    # update status
                                    textDisplayStim.status = STARTED
                                    textDisplayStim.setAutoDraw(True)
                                
                                # if textDisplayStim is active this frame...
                                if textDisplayStim.status == STARTED:
                                    # update params
                                    pass
                                
                                # if textDisplayStim is stopping this frame...
                                if textDisplayStim.status == STARTED:
                                    # is it time to stop? (based on global clock, using actual start)
                                    if tThisFlipGlobal > textDisplayStim.tStartRefresh + trial_display_time-frameTolerance:
                                        # keep track of stop time/frame for later
                                        textDisplayStim.tStop = t  # not accounting for scr refresh
                                        textDisplayStim.tStopRefresh = tThisFlipGlobal  # on global time
                                        textDisplayStim.frameNStop = frameN  # exact frame index
                                        # add timestamp to datafile
                                        thisExp.timestampOnFlip(win, 'textDisplayStim.stopped')
                                        # update status
                                        textDisplayStim.status = FINISHED
                                        textDisplayStim.setAutoDraw(False)
                                
                                # check for quit (typically the Esc key)
                                if defaultKeyboard.getKeys(keyList=["escape"]):
                                    thisExp.status = FINISHED
                                if thisExp.status == FINISHED or endExpNow:
                                    endExperiment(thisExp, win=win)
                                    return
                                # pause experiment here if requested
                                if thisExp.status == PAUSED:
                                    pauseExperiment(
                                        thisExp=thisExp, 
                                        win=win, 
                                        timers=[routineTimer], 
                                        playbackComponents=[]
                                    )
                                    # skip the frame we paused on
                                    continue
                                
                                # check if all components have finished
                                if not continueRoutine:  # a component has requested a forced-end of Routine
                                    display_stimuli.forceEnded = routineForceEnded = True
                                    break
                                continueRoutine = False  # will revert to True if at least one component still running
                                for thisComponent in display_stimuli.components:
                                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                        continueRoutine = True
                                        break  # at least one component has not yet finished
                                
                                # refresh the screen
                                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                                    win.flip()
                            
                            # --- Ending Routine "display_stimuli" ---
                            for thisComponent in display_stimuli.components:
                                if hasattr(thisComponent, "setAutoDraw"):
                                    thisComponent.setAutoDraw(False)
                            # store stop times for display_stimuli
                            display_stimuli.tStop = globalClock.getTime(format='float')
                            display_stimuli.tStopRefresh = tThisFlipGlobal
                            thisExp.addData('display_stimuli.stopped', display_stimuli.tStop)
                            # store data for seq_display_loop (TrialHandler)
                            # Run 'End Routine' code from codeDisplayStim
                            # reset click recording
                            has_released = True
                            
                            # record mouse position
                            thisExp.addData(f'mouseStim.x_{seq_display_loop.thisN+1}', display_mouse_xs)
                            thisExp.addData(f'mouseStim.y_{seq_display_loop.thisN+1}', display_mouse_ys)
                            
                            # clear all objects
                            for i in range(len(objs_display)):
                                display_obj = objs_display[i]
                                display_obj.setAutoDraw(False)
                            
                            objs_display = []
                            response_objects = []
                            
                            # Run 'End Routine' code from elRecord_stim
                            aaa = core.monotonicClock.getTime
                            thisExp.addData(this_epoch+'End',str(aaa()))
                            
                            # the Routine "display_stimuli" was not non-slip safe, so reset the non-slip timer
                            routineTimer.reset()
                            
                            # --- Prepare to start Routine "delayAfterStim" ---
                            # create an object to store info about Routine delayAfterStim
                            delayAfterStim = data.Routine(
                                name='delayAfterStim',
                                components=[mousePostStim, textDelayAfterStim],
                            )
                            delayAfterStim.status = NOT_STARTED
                            continueRoutine = True
                            # update component parameters for each repeat
                            # setup some python lists for storing info about the mousePostStim
                            gotValidClick = False  # until a click is received
                            # Run 'Begin Routine' code from codeDelayAfterStim
                            objs_display = []
                            has_noise_patch_displaying = False
                            # decide whether to display noise patch
                            to_display_noise_patch =  random.random() < noise_patch_display_chance
                            if to_display_noise_patch:
                                loc = stim_locs[seq_display_loop.thisN]
                                stim_obj = NoisePatch(win, loc, patch_radius)
                                objs_display.append(stim_obj)
                                has_noise_patch_displaying = True
                            
                            # record mouse
                            post_display_mouse_xs = []
                            post_display_mouse_ys = []
                            textDelayAfterStim.setText('')
                            # Run 'Begin Routine' code from elRecord_maskShortDelay
                            this_epoch = 'maskShortDelay'+str(seq_display_loop.thisN)
                            
                            aaa = core.monotonicClock.getTime
                            thisExp.addData(this_epoch+'Start',str(aaa()))
                            
                            if eyetracking == 1:
                                #el_tracker.startRecording(1, 1, 1, 1)
                                el_tracker.sendMessage(this_epoch)
                            
                            
                                
                            
                            # store start times for delayAfterStim
                            delayAfterStim.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                            delayAfterStim.tStart = globalClock.getTime(format='float')
                            delayAfterStim.status = STARTED
                            thisExp.addData('delayAfterStim.started', delayAfterStim.tStart)
                            delayAfterStim.maxDuration = None
                            # keep track of which components have finished
                            delayAfterStimComponents = delayAfterStim.components
                            for thisComponent in delayAfterStim.components:
                                thisComponent.tStart = None
                                thisComponent.tStop = None
                                thisComponent.tStartRefresh = None
                                thisComponent.tStopRefresh = None
                                if hasattr(thisComponent, 'status'):
                                    thisComponent.status = NOT_STARTED
                            # reset timers
                            t = 0
                            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                            frameN = -1
                            
                            # --- Run Routine "delayAfterStim" ---
                            # if trial has changed, end Routine now
                            if isinstance(seq_display_loop, data.TrialHandler2) and thisSeq_display_loop.thisN != seq_display_loop.thisTrial.thisN:
                                continueRoutine = False
                            delayAfterStim.forceEnded = routineForceEnded = not continueRoutine
                            while continueRoutine:
                                # get current time
                                t = routineTimer.getTime()
                                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                                # update/draw components on each frame
                                # *mousePostStim* updates
                                
                                # if mousePostStim is starting this frame...
                                if mousePostStim.status == NOT_STARTED and t >= 0.0-frameTolerance:
                                    # keep track of start time/frame for later
                                    mousePostStim.frameNStart = frameN  # exact frame index
                                    mousePostStim.tStart = t  # local t and not account for scr refresh
                                    mousePostStim.tStartRefresh = tThisFlipGlobal  # on global time
                                    win.timeOnFlip(mousePostStim, 'tStartRefresh')  # time at next scr refresh
                                    # update status
                                    mousePostStim.status = STARTED
                                    mousePostStim.mouseClock.reset()
                                    prevButtonState = mousePostStim.getPressed()  # if button is down already this ISN'T a new click
                                
                                # if mousePostStim is stopping this frame...
                                if mousePostStim.status == STARTED:
                                    # is it time to stop? (based on global clock, using actual start)
                                    if tThisFlipGlobal > mousePostStim.tStartRefresh + delay_post_stim-frameTolerance:
                                        # keep track of stop time/frame for later
                                        mousePostStim.tStop = t  # not accounting for scr refresh
                                        mousePostStim.tStopRefresh = tThisFlipGlobal  # on global time
                                        mousePostStim.frameNStop = frameN  # exact frame index
                                        # update status
                                        mousePostStim.status = FINISHED
                                # Run 'Each Frame' code from codeDelayAfterStim
                                # end routine when time up
                                if t > delay_post_stim:
                                    continueRoutine = False
                                
                                # stop displaying all noise patches
                                if has_noise_patch_displaying:
                                    if t > noise_patch_display_time:
                                        for i in range(len(objs_display)):
                                            display_obj = objs_display[i]
                                            display_obj.setAutoDraw(False)
                                        has_noise_patch_displaying = False
                                        
                                        # display the fixation
                                        # objs_display = []
                                        # fixation = DefaultFixation(win)
                                        # objs_display.append(fixation)
                                
                                # monitor mouse behavior
                                mouse_position = mousePostStim.getPos()
                                post_display_mouse_xs.append(mouse_position[0])
                                post_display_mouse_ys.append(mouse_position[1])
                                
                                # *textDelayAfterStim* updates
                                
                                # if textDelayAfterStim is starting this frame...
                                if textDelayAfterStim.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                                    # keep track of start time/frame for later
                                    textDelayAfterStim.frameNStart = frameN  # exact frame index
                                    textDelayAfterStim.tStart = t  # local t and not account for scr refresh
                                    textDelayAfterStim.tStartRefresh = tThisFlipGlobal  # on global time
                                    win.timeOnFlip(textDelayAfterStim, 'tStartRefresh')  # time at next scr refresh
                                    # add timestamp to datafile
                                    thisExp.timestampOnFlip(win, 'textDelayAfterStim.started')
                                    # update status
                                    textDelayAfterStim.status = STARTED
                                    textDelayAfterStim.setAutoDraw(True)
                                
                                # if textDelayAfterStim is active this frame...
                                if textDelayAfterStim.status == STARTED:
                                    # update params
                                    pass
                                
                                # if textDelayAfterStim is stopping this frame...
                                if textDelayAfterStim.status == STARTED:
                                    # is it time to stop? (based on global clock, using actual start)
                                    if tThisFlipGlobal > textDelayAfterStim.tStartRefresh + delay_post_stim-frameTolerance:
                                        # keep track of stop time/frame for later
                                        textDelayAfterStim.tStop = t  # not accounting for scr refresh
                                        textDelayAfterStim.tStopRefresh = tThisFlipGlobal  # on global time
                                        textDelayAfterStim.frameNStop = frameN  # exact frame index
                                        # add timestamp to datafile
                                        thisExp.timestampOnFlip(win, 'textDelayAfterStim.stopped')
                                        # update status
                                        textDelayAfterStim.status = FINISHED
                                        textDelayAfterStim.setAutoDraw(False)
                                
                                # check for quit (typically the Esc key)
                                if defaultKeyboard.getKeys(keyList=["escape"]):
                                    thisExp.status = FINISHED
                                if thisExp.status == FINISHED or endExpNow:
                                    endExperiment(thisExp, win=win)
                                    return
                                # pause experiment here if requested
                                if thisExp.status == PAUSED:
                                    pauseExperiment(
                                        thisExp=thisExp, 
                                        win=win, 
                                        timers=[routineTimer], 
                                        playbackComponents=[]
                                    )
                                    # skip the frame we paused on
                                    continue
                                
                                # check if all components have finished
                                if not continueRoutine:  # a component has requested a forced-end of Routine
                                    delayAfterStim.forceEnded = routineForceEnded = True
                                    break
                                continueRoutine = False  # will revert to True if at least one component still running
                                for thisComponent in delayAfterStim.components:
                                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                        continueRoutine = True
                                        break  # at least one component has not yet finished
                                
                                # refresh the screen
                                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                                    win.flip()
                            
                            # --- Ending Routine "delayAfterStim" ---
                            for thisComponent in delayAfterStim.components:
                                if hasattr(thisComponent, "setAutoDraw"):
                                    thisComponent.setAutoDraw(False)
                            # store stop times for delayAfterStim
                            delayAfterStim.tStop = globalClock.getTime(format='float')
                            delayAfterStim.tStopRefresh = tThisFlipGlobal
                            thisExp.addData('delayAfterStim.stopped', delayAfterStim.tStop)
                            # store data for seq_display_loop (TrialHandler)
                            # Run 'End Routine' code from codeDelayAfterStim
                            # record mouse position
                            thisExp.addData(f'mousePostStim.x_{seq_display_loop.thisN+1}', post_display_mouse_xs)
                            thisExp.addData(f'mousePostStim.y_{seq_display_loop.thisN+1}', post_display_mouse_ys)
                            
                            # clear all objects
                            for i in range(len(objs_display)):
                                display_obj = objs_display[i]
                                display_obj.setAutoDraw(False)
                            
                            objs_display = []
                            
                            # Run 'End Routine' code from elRecord_maskShortDelay
                            aaa = core.monotonicClock.getTime
                            thisExp.addData(this_epoch+'End',str(aaa()))
                            
                            # the Routine "delayAfterStim" was not non-slip safe, so reset the non-slip timer
                            routineTimer.reset()
                        # completed 2.0 repeats of 'seq_display_loop'
                        
                        
                        # --- Prepare to start Routine "delay" ---
                        # create an object to store info about Routine delay
                        delay = data.Routine(
                            name='delay',
                            components=[mouseDelay, textDelay],
                        )
                        delay.status = NOT_STARTED
                        continueRoutine = True
                        # update component parameters for each repeat
                        # setup some python lists for storing info about the mouseDelay
                        mouseDelay.x = []
                        mouseDelay.y = []
                        mouseDelay.leftButton = []
                        mouseDelay.midButton = []
                        mouseDelay.rightButton = []
                        mouseDelay.time = []
                        gotValidClick = False  # until a click is received
                        # Run 'Begin Routine' code from codeDelay
                        incidental_cue = None
                        t_start_delay_cue = delay_time
                        t_end_delay_cue = delay_time
                        
                        # initialize the start of cues
                        n_cues = random.randint(0, n_max_cues)
                        tstarts = sample_n_from_trange(
                            n=n_cues, duration=delay_cue_duration, 
                            min_space=delay_cue_space, 
                            tmin=0, tmax=delay_time)
                        # logging.log(level=logging.WARN, msg=f'delay: {delay_time}, TSTARTS: {tstarts}')
                        n_cue_to_show = len(tstarts)
                        tindex = 0
                        
                        # initialize the stimulus they map to
                        stim_ids = [0, 1] if random.random() < 0.5 else [1, 0]
                        stim_ids = stim_ids[0:n_cue_to_show]
                        # logging.log(level=logging.WARN, msg=f'stim ids {stim_ids}')
                        
                        # create fixation
                        # fixation = DefaultFixation(win)
                        
                        # record mouse
                        long_delay_mouse_xs = []
                        long_delay_mouse_ys = []
                        
                        textDelay.setText('')
                        # Run 'Begin Routine' code from elRecord_delay
                        this_epoch = 'delay'
                        
                        aaa = core.monotonicClock.getTime
                        thisExp.addData(this_epoch+'Start',str(aaa()))
                        
                        if eyetracking == 1:
                            #el_tracker.startRecording(1, 1, 1, 1)
                            el_tracker.sendMessage(this_epoch)
                        
                        
                            
                        
                        # store start times for delay
                        delay.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                        delay.tStart = globalClock.getTime(format='float')
                        delay.status = STARTED
                        thisExp.addData('delay.started', delay.tStart)
                        delay.maxDuration = None
                        # keep track of which components have finished
                        delayComponents = delay.components
                        for thisComponent in delay.components:
                            thisComponent.tStart = None
                            thisComponent.tStop = None
                            thisComponent.tStartRefresh = None
                            thisComponent.tStopRefresh = None
                            if hasattr(thisComponent, 'status'):
                                thisComponent.status = NOT_STARTED
                        # reset timers
                        t = 0
                        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                        frameN = -1
                        
                        # --- Run Routine "delay" ---
                        # if trial has changed, end Routine now
                        if isinstance(block, data.TrialHandler2) and thisBlock.thisN != block.thisTrial.thisN:
                            continueRoutine = False
                        delay.forceEnded = routineForceEnded = not continueRoutine
                        while continueRoutine:
                            # get current time
                            t = routineTimer.getTime()
                            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                            # update/draw components on each frame
                            # *mouseDelay* updates
                            
                            # if mouseDelay is starting this frame...
                            if mouseDelay.status == NOT_STARTED and t >= 0.0-frameTolerance:
                                # keep track of start time/frame for later
                                mouseDelay.frameNStart = frameN  # exact frame index
                                mouseDelay.tStart = t  # local t and not account for scr refresh
                                mouseDelay.tStartRefresh = tThisFlipGlobal  # on global time
                                win.timeOnFlip(mouseDelay, 'tStartRefresh')  # time at next scr refresh
                                # add timestamp to datafile
                                thisExp.addData('mouseDelay.started', t)
                                # update status
                                mouseDelay.status = STARTED
                                mouseDelay.mouseClock.reset()
                                prevButtonState = mouseDelay.getPressed()  # if button is down already this ISN'T a new click
                            
                            # if mouseDelay is stopping this frame...
                            if mouseDelay.status == STARTED:
                                # is it time to stop? (based on global clock, using actual start)
                                if tThisFlipGlobal > mouseDelay.tStartRefresh + delay_time-frameTolerance:
                                    # keep track of stop time/frame for later
                                    mouseDelay.tStop = t  # not accounting for scr refresh
                                    mouseDelay.tStopRefresh = tThisFlipGlobal  # on global time
                                    mouseDelay.frameNStop = frameN  # exact frame index
                                    # add timestamp to datafile
                                    thisExp.addData('mouseDelay.stopped', t)
                                    # update status
                                    mouseDelay.status = FINISHED
                            if mouseDelay.status == STARTED:  # only update if started and not finished!
                                x, y = mouseDelay.getPos()
                                mouseDelay.x.append(x)
                                mouseDelay.y.append(y)
                                buttons = mouseDelay.getPressed()
                                mouseDelay.leftButton.append(buttons[0])
                                mouseDelay.midButton.append(buttons[1])
                                mouseDelay.rightButton.append(buttons[2])
                                mouseDelay.time.append(mouseDelay.mouseClock.getTime())
                            # Run 'Each Frame' code from codeDelay
                            # end routine when time up
                            if t > delay_time:
                                continueRoutine = False
                            
                            if (incidental_cue is not None) and (t > t_end_delay_cue):
                                # time to kill the old cue
                                incidental_cue.setAutoDraw(False)
                                incidental_cue = None
                                
                            if n_cue_to_show > 0:
                                # update time for the next
                                t_start_delay_cue = tstarts[tindex]
                                if t > t_start_delay_cue:
                                    # update the end time
                                    t_end_delay_cue = t_start_delay_cue + delay_cue_duration
                                    # logging.log(level=logging.WARN, msg=f'time: {t_start_delay_cue} -- {t_end_delay_cue}')
                                    
                                    # fetch stim information
                                    stim_id = stim_ids[tindex]
                                    # logging.log(level=logging.WARN, msg=f'stim id: {stim_id}')
                                    cue_code= to_report_code[stim_id] if trial_type == 0 else False
                                    cue_loc = stim_locs[stim_id]
                                    cue_color = color_codes[stim_id]
                                    # logging.log(level=logging.WARN, msg=f'stim info:  code {cue_code}, loc {cue_loc}, color {cue_color}')
                                    
                                    # create the cue
                                    incidental_cue = SingleCueObject(
                                        win, cue_loc, cue_code, patch_radius, cue_color)
                            
                                    # update stats
                                    tindex += 1
                                    n_cue_to_show -= 1
                                
                            # monitor mouse behavior
                            mouse_position = mouseDelay.getPos()
                            long_delay_mouse_xs.append(mouse_position[0])
                            long_delay_mouse_ys.append(mouse_position[1])
                            
                            
                            # *textDelay* updates
                            
                            # if textDelay is starting this frame...
                            if textDelay.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                                # keep track of start time/frame for later
                                textDelay.frameNStart = frameN  # exact frame index
                                textDelay.tStart = t  # local t and not account for scr refresh
                                textDelay.tStartRefresh = tThisFlipGlobal  # on global time
                                win.timeOnFlip(textDelay, 'tStartRefresh')  # time at next scr refresh
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'textDelay.started')
                                # update status
                                textDelay.status = STARTED
                                textDelay.setAutoDraw(True)
                            
                            # if textDelay is active this frame...
                            if textDelay.status == STARTED:
                                # update params
                                pass
                            
                            # check for quit (typically the Esc key)
                            if defaultKeyboard.getKeys(keyList=["escape"]):
                                thisExp.status = FINISHED
                            if thisExp.status == FINISHED or endExpNow:
                                endExperiment(thisExp, win=win)
                                return
                            # pause experiment here if requested
                            if thisExp.status == PAUSED:
                                pauseExperiment(
                                    thisExp=thisExp, 
                                    win=win, 
                                    timers=[routineTimer], 
                                    playbackComponents=[]
                                )
                                # skip the frame we paused on
                                continue
                            
                            # check if all components have finished
                            if not continueRoutine:  # a component has requested a forced-end of Routine
                                delay.forceEnded = routineForceEnded = True
                                break
                            continueRoutine = False  # will revert to True if at least one component still running
                            for thisComponent in delay.components:
                                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                    continueRoutine = True
                                    break  # at least one component has not yet finished
                            
                            # refresh the screen
                            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                                win.flip()
                        
                        # --- Ending Routine "delay" ---
                        for thisComponent in delay.components:
                            if hasattr(thisComponent, "setAutoDraw"):
                                thisComponent.setAutoDraw(False)
                        # store stop times for delay
                        delay.tStop = globalClock.getTime(format='float')
                        delay.tStopRefresh = tThisFlipGlobal
                        thisExp.addData('delay.stopped', delay.tStop)
                        # store data for block (TrialHandler)
                        block.addData('mouseDelay.x', mouseDelay.x)
                        block.addData('mouseDelay.y', mouseDelay.y)
                        block.addData('mouseDelay.leftButton', mouseDelay.leftButton)
                        block.addData('mouseDelay.midButton', mouseDelay.midButton)
                        block.addData('mouseDelay.rightButton', mouseDelay.rightButton)
                        block.addData('mouseDelay.time', mouseDelay.time)
                        # Run 'End Routine' code from codeDelay
                        if incidental_cue is not None:
                            incidental_cue.setAutoDraw(False)
                            incidental_cue = None
                            
                        # remove the fixation
                        # fixation.setAutoDraw(False)
                            
                        thisExp.addData('inci_cue_time', tstarts)
                        thisExp.addData('inci_cue_ids', stim_ids)
                        
                        # record mouse position
                        thisExp.addData(f'mouseDelay.x_{seq_display_loop.thisN+1}', long_delay_mouse_xs)
                        thisExp.addData(f'mouseDelay.y_{seq_display_loop.thisN+1}', long_delay_mouse_ys)
                        
                        # show the cursor
                        win.mouseVisible = True
                        
                        # Run 'End Routine' code from elRecord_delay
                        aaa = core.monotonicClock.getTime
                        thisExp.addData(this_epoch+'End',str(aaa()))
                        
                        # the Routine "delay" was not non-slip safe, so reset the non-slip timer
                        routineTimer.reset()
                        
                        # --- Prepare to start Routine "response" ---
                        # create an object to store info about Routine response
                        response = data.Routine(
                            name='response',
                            components=[responseMouse, keyRespInput],
                        )
                        response.status = NOT_STARTED
                        continueRoutine = True
                        # update component parameters for each repeat
                        # Run 'Begin Routine' code from codeDelayResp
                        objs_display = []
                        response_objects = []
                        for i in range(2):
                            loc = resp_locs[i]
                            to_report = to_report_code[i]
                            response_obj = None
                            if response_mode == 'click':
                                response_obj = funcDrawAdjustResponse(
                                    win, loc, patch_radius, click_radius, to_report)
                            elif response_mode == 'draw':
                                response_obj = funcDrawDrawResponse(
                                    win, loc, patch_radius, to_report)
                            else:
                                raise NotImplementedError(f'Unknown response mode: {response_mode}')
                            objs_display.append(response_obj)
                            response_objects.append(response_obj)
                            
                            # give order color cue
                            if to_report: # only if its valid region
                                report_cue = SingleCueObject(
                                    win, loc, 99, patch_radius*0.6, color_codes[i])
                                objs_display.append(report_cue)
                            
                            # give cues for instruction
                            if not has_finished_instruction:
                                if to_report:
                                    additional_cue = TrainingCue(
                                        win, 'report here', 
                                        loc, patch_radius)
                                    objs_display.append(additional_cue)
                        
                        # to add the confirm button
                        continue_button = None
                        if len(response_objects) > 0:
                            # loc_x = (resp_locs[0][0] + resp_locs[1][0]) / 2
                            # loc_y = (resp_locs[0][1] + resp_locs[1][1]) / 2
                            continue_button = funcCreateButton(
                                win, 'continue', pos=(0, 0),
                                size=(button_width*1.3, button_height*1.3))
                            objs_display.append(continue_button)
                        
                        # to detect whether a pressed has released
                        has_released = True
                        # setup some python lists for storing info about the responseMouse
                        gotValidClick = False  # until a click is received
                        # create starting attributes for keyRespInput
                        keyRespInput.keys = []
                        keyRespInput.rt = []
                        _keyRespInput_allKeys = []
                        # Run 'Begin Routine' code from elRecord_rsp
                        this_epoch = 'rsp'
                        
                        aaa = core.monotonicClock.getTime
                        thisExp.addData(this_epoch+'Start',str(aaa()))
                        
                        if eyetracking == 1:
                            #el_tracker.startRecording(1, 1, 1, 1)
                            el_tracker.sendMessage(this_epoch)
                        
                        
                            
                        
                        # store start times for response
                        response.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                        response.tStart = globalClock.getTime(format='float')
                        response.status = STARTED
                        thisExp.addData('response.started', response.tStart)
                        response.maxDuration = None
                        # keep track of which components have finished
                        responseComponents = response.components
                        for thisComponent in response.components:
                            thisComponent.tStart = None
                            thisComponent.tStop = None
                            thisComponent.tStartRefresh = None
                            thisComponent.tStopRefresh = None
                            if hasattr(thisComponent, 'status'):
                                thisComponent.status = NOT_STARTED
                        # reset timers
                        t = 0
                        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                        frameN = -1
                        
                        # --- Run Routine "response" ---
                        # if trial has changed, end Routine now
                        if isinstance(block, data.TrialHandler2) and thisBlock.thisN != block.thisTrial.thisN:
                            continueRoutine = False
                        response.forceEnded = routineForceEnded = not continueRoutine
                        while continueRoutine:
                            # get current time
                            t = routineTimer.getTime()
                            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                            # update/draw components on each frame
                            # Run 'Each Frame' code from codeDelayResp
                            # read mouse click
                            if len(response_objects) > 0:
                                # if there are objects clicked
                                left_button = responseMouse.getPressed()[0]
                                mouse_position = responseMouse.getPos()
                                mouse_time = t
                                if left_button:
                                    if continue_button.contains(mouse_position):
                                        # first check if it's over the continue button
                                        no_pending = funcCheckNoPending(response_objects)
                                        if no_pending:
                                            continueRoutine = False
                                    else:
                                        if response_mode == 'click':
                                            # only register the first click
                                            if has_released:
                                                # this is a new press, register it
                                                has_released = False
                                                # otherwise, ignore it
                                                for i in range(len(response_objects)):
                                                    # update each position with the click
                                                    resp_obj = response_objects[i]
                                                    resp_obj.register_click(mouse_position, mouse_time)
                                        elif response_mode == 'draw':
                                            # register all mouse movements
                                            has_released = False
                                            for i in range(len(response_objects)):
                                                # update each position with the click
                                                resp_obj = response_objects[i]
                                                has_redo = resp_obj.register_mouse(
                                                    mouse_position, mouse_time)
                                                # update the time limit
                                                if has_redo:
                                                    trial_resp_time += response_time
                                                    # trial_display_time = t + response_time 
                                else:
                                    # check if we should update has_released
                                    has_released = True
                                    if response_mode == 'draw':
                                        for i in range(len(response_objects)):
                                            # update each position with the click
                                            resp_obj = response_objects[i]
                                            resp_obj.deregister_mouse(mouse_position, mouse_time)
                            
                                # register click on space
                                key_inputs = keyRespInput.getKeys(clear=True)
                                if 'space' in key_inputs:
                                    # check if all required response is there
                                    no_pending = funcCheckNoPending(response_objects)
                                    if no_pending:
                                        continueRoutine = False
                                        
                            if enforce_response_time_limit and (t > trial_resp_time):
                                continueRoutine = False
                            # *responseMouse* updates
                            
                            # if responseMouse is starting this frame...
                            if responseMouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                                # keep track of start time/frame for later
                                responseMouse.frameNStart = frameN  # exact frame index
                                responseMouse.tStart = t  # local t and not account for scr refresh
                                responseMouse.tStartRefresh = tThisFlipGlobal  # on global time
                                win.timeOnFlip(responseMouse, 'tStartRefresh')  # time at next scr refresh
                                # update status
                                responseMouse.status = STARTED
                                responseMouse.mouseClock.reset()
                                prevButtonState = responseMouse.getPressed()  # if button is down already this ISN'T a new click
                            
                            # *keyRespInput* updates
                            
                            # if keyRespInput is starting this frame...
                            if keyRespInput.status == NOT_STARTED and t >= 0.0-frameTolerance:
                                # keep track of start time/frame for later
                                keyRespInput.frameNStart = frameN  # exact frame index
                                keyRespInput.tStart = t  # local t and not account for scr refresh
                                keyRespInput.tStartRefresh = tThisFlipGlobal  # on global time
                                win.timeOnFlip(keyRespInput, 'tStartRefresh')  # time at next scr refresh
                                # update status
                                keyRespInput.status = STARTED
                                # keyboard checking is just starting
                                keyRespInput.clock.reset()  # now t=0
                            if keyRespInput.status == STARTED:
                                theseKeys = keyRespInput.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                                _keyRespInput_allKeys.extend(theseKeys)
                                if len(_keyRespInput_allKeys):
                                    keyRespInput.keys = _keyRespInput_allKeys[-1].name  # just the last key pressed
                                    keyRespInput.rt = _keyRespInput_allKeys[-1].rt
                                    keyRespInput.duration = _keyRespInput_allKeys[-1].duration
                            
                            # check for quit (typically the Esc key)
                            if defaultKeyboard.getKeys(keyList=["escape"]):
                                thisExp.status = FINISHED
                            if thisExp.status == FINISHED or endExpNow:
                                endExperiment(thisExp, win=win)
                                return
                            # pause experiment here if requested
                            if thisExp.status == PAUSED:
                                pauseExperiment(
                                    thisExp=thisExp, 
                                    win=win, 
                                    timers=[routineTimer], 
                                    playbackComponents=[]
                                )
                                # skip the frame we paused on
                                continue
                            
                            # check if all components have finished
                            if not continueRoutine:  # a component has requested a forced-end of Routine
                                response.forceEnded = routineForceEnded = True
                                break
                            continueRoutine = False  # will revert to True if at least one component still running
                            for thisComponent in response.components:
                                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                    continueRoutine = True
                                    break  # at least one component has not yet finished
                            
                            # refresh the screen
                            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                                win.flip()
                        
                        # --- Ending Routine "response" ---
                        for thisComponent in response.components:
                            if hasattr(thisComponent, "setAutoDraw"):
                                thisComponent.setAutoDraw(False)
                        # store stop times for response
                        response.tStop = globalClock.getTime(format='float')
                        response.tStopRefresh = tThisFlipGlobal
                        thisExp.addData('response.stopped', response.tStop)
                        # Run 'End Routine' code from codeDelayResp
                        # reset click recording
                        has_released = True
                        
                        # record responses, if no-delay
                        if len(response_objects) > 0:
                            resp_obj_id = 0
                            for i in range(2):
                                to_report = to_report_code[i]
                                resp_x_to_record = []
                                resp_y_to_record = []
                                resp_time_to_record = []
                                resp_has_invalid = False
                                if to_report:
                                    # resp_obj = response_objects[resp_obj_id]
                                    resp_obj = response_objects[i]
                                    if response_mode == 'click':
                                        resp_x_to_record = resp_obj.clicks_x
                                        resp_y_to_record = resp_obj.clicks_y
                                        resp_time_to_record = resp_obj.clicks_time
                                    elif response_mode == 'draw':
                                        # save the stroke
                                        if resp_obj.has_unsaved_stroke:
                                            resp_obj.save_stroke()
                                        # saving drawings (if there are unsaved)
                                        if resp_obj.has_unsaved_drawing:
                                            resp_obj.save_drawing()
                                        resp_x_to_record = resp_obj.all_drawings_x
                                        resp_y_to_record = resp_obj.all_drawings_y
                                        resp_time_to_record = resp_obj.all_drawings_time
                                    
                                    # approximate the response
                                    # if it's not none, compute the error and add to records
                                    resp_approximated = resp_obj.compute_response()
                                    if resp_approximated is not None:
                                        ground_truth = stims_selected[i]
                                        error_approximated = funcComputeOriDiff(resp_approximated, ground_truth)
                                        block_error_records.append(error_approximated)
                                    else:
                                        # record those no response
                                        block_error_records.append(None)
                                        
                                    # record the response for feedback
                                    participant_responses[i] = resp_approximated
                                    if not has_finished_rehearsal and response_mode == 'draw':
                                        # save the last drawings to display later
                                        original_drawing = None
                                        if resp_approximated is not None:
                                            original_drawing = (
                                                resp_obj.all_drawings_x[-1],
                                                resp_obj.all_drawings_y[-1])
                                        participant_responses[i] = original_drawing
                                    
                                    resp_obj_id += 1
                                    
                                else:
                                    resp_obj = response_objects[i]
                                    if resp_obj.has_response:
                                        resp_has_invalid = True
                                    
                                thisExp.addData(f'resp_{i+1}_x', resp_x_to_record)
                                thisExp.addData(f'resp_{i+1}_y', resp_y_to_record)
                                thisExp.addData(f'resp_{i+1}_time', resp_time_to_record)
                                thisExp.addData(f'resp_{i+1}_invalid', resp_has_invalid)
                        
                        # clear all objects
                        for i in range(len(objs_display)):
                            display_obj = objs_display[i]
                            display_obj.setAutoDraw(False)
                        
                        objs_display = []
                        response_objects = []
                        
                        # hide the cursor
                        win.mouseVisible = False
                        # store data for block (TrialHandler)
                        # Run 'End Routine' code from elRecord_rsp
                        aaa = core.monotonicClock.getTime
                        thisExp.addData(this_epoch+'End',str(aaa()))
                        
                        # the Routine "response" was not non-slip safe, so reset the non-slip timer
                        routineTimer.reset()
                        
                        # --- Prepare to start Routine "post_trial" ---
                        # create an object to store info about Routine post_trial
                        post_trial = data.Routine(
                            name='post_trial',
                            components=[textPostTrial],
                        )
                        post_trial.status = NOT_STARTED
                        continueRoutine = True
                        # update component parameters for each repeat
                        # Run 'Begin Routine' code from codePostTrial
                        # get time elapsed
                        trial_time_elapsed = trial_timer.getTime()
                        trial_timer = None
                        
                        # set ITI
                        post_trial_time = random.uniform(
                            post_trial_time_min, post_trial_time_max)
                        
                        textPostTrial.setText('')
                        # Run 'Begin Routine' code from elRecord_iti
                        aaa = core.monotonicClock.getTime
                        thisExp.addData('itiStart',str(aaa()))
                        this_epoch = 'ITI'
                        
                        if eyetracking == 1:
                            el_tracker.sendMessage('ITI')
                        # store start times for post_trial
                        post_trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                        post_trial.tStart = globalClock.getTime(format='float')
                        post_trial.status = STARTED
                        thisExp.addData('post_trial.started', post_trial.tStart)
                        post_trial.maxDuration = None
                        # keep track of which components have finished
                        post_trialComponents = post_trial.components
                        for thisComponent in post_trial.components:
                            thisComponent.tStart = None
                            thisComponent.tStop = None
                            thisComponent.tStartRefresh = None
                            thisComponent.tStopRefresh = None
                            if hasattr(thisComponent, 'status'):
                                thisComponent.status = NOT_STARTED
                        # reset timers
                        t = 0
                        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                        frameN = -1
                        
                        # --- Run Routine "post_trial" ---
                        # if trial has changed, end Routine now
                        if isinstance(block, data.TrialHandler2) and thisBlock.thisN != block.thisTrial.thisN:
                            continueRoutine = False
                        post_trial.forceEnded = routineForceEnded = not continueRoutine
                        while continueRoutine:
                            # get current time
                            t = routineTimer.getTime()
                            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                            # update/draw components on each frame
                            
                            # *textPostTrial* updates
                            
                            # if textPostTrial is starting this frame...
                            if textPostTrial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                                # keep track of start time/frame for later
                                textPostTrial.frameNStart = frameN  # exact frame index
                                textPostTrial.tStart = t  # local t and not account for scr refresh
                                textPostTrial.tStartRefresh = tThisFlipGlobal  # on global time
                                win.timeOnFlip(textPostTrial, 'tStartRefresh')  # time at next scr refresh
                                # update status
                                textPostTrial.status = STARTED
                                textPostTrial.setAutoDraw(True)
                            
                            # if textPostTrial is active this frame...
                            if textPostTrial.status == STARTED:
                                # update params
                                pass
                            
                            # if textPostTrial is stopping this frame...
                            if textPostTrial.status == STARTED:
                                # is it time to stop? (based on global clock, using actual start)
                                if tThisFlipGlobal > textPostTrial.tStartRefresh + post_trial_time-frameTolerance:
                                    # keep track of stop time/frame for later
                                    textPostTrial.tStop = t  # not accounting for scr refresh
                                    textPostTrial.tStopRefresh = tThisFlipGlobal  # on global time
                                    textPostTrial.frameNStop = frameN  # exact frame index
                                    # update status
                                    textPostTrial.status = FINISHED
                                    textPostTrial.setAutoDraw(False)
                            
                            # check for quit (typically the Esc key)
                            if defaultKeyboard.getKeys(keyList=["escape"]):
                                thisExp.status = FINISHED
                            if thisExp.status == FINISHED or endExpNow:
                                endExperiment(thisExp, win=win)
                                return
                            # pause experiment here if requested
                            if thisExp.status == PAUSED:
                                pauseExperiment(
                                    thisExp=thisExp, 
                                    win=win, 
                                    timers=[routineTimer], 
                                    playbackComponents=[]
                                )
                                # skip the frame we paused on
                                continue
                            
                            # check if all components have finished
                            if not continueRoutine:  # a component has requested a forced-end of Routine
                                post_trial.forceEnded = routineForceEnded = True
                                break
                            continueRoutine = False  # will revert to True if at least one component still running
                            for thisComponent in post_trial.components:
                                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                    continueRoutine = True
                                    break  # at least one component has not yet finished
                            
                            # refresh the screen
                            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                                win.flip()
                        
                        # --- Ending Routine "post_trial" ---
                        for thisComponent in post_trial.components:
                            if hasattr(thisComponent, "setAutoDraw"):
                                thisComponent.setAutoDraw(False)
                        # store stop times for post_trial
                        post_trial.tStop = globalClock.getTime(format='float')
                        post_trial.tStopRefresh = tThisFlipGlobal
                        thisExp.addData('post_trial.stopped', post_trial.tStop)
                        # Run 'End Routine' code from codePostTrial
                        ## record ITI
                        thisExp.addData('ITI', post_trial_time)
                        # Run 'End Routine' code from elRecord_iti
                        aaa = core.monotonicClock.getTime
                        thisExp.addData('itiEnd',str(aaa()))
                        if eyetracking == 1:
                           
                            el_tracker.sendMessage('trialEnd')
                            el_tracker.sendMessage('!V TRIAL_VAR TRIALID %s'% str(trlId))
                            el_tracker.stopRecording()
                        
                        # the Routine "post_trial" was not non-slip safe, so reset the non-slip timer
                        routineTimer.reset()
                        
                        # --- Prepare to start Routine "backupSave" ---
                        # create an object to store info about Routine backupSave
                        backupSave = data.Routine(
                            name='backupSave',
                            components=[],
                        )
                        backupSave.status = NOT_STARTED
                        continueRoutine = True
                        # update component parameters for each repeat
                        # Run 'Begin Routine' code from codeBackupSave
                        BACKUP_PATH = os.path.join(BACKUP_FOLDER_PATH, f'{BACKUP_IDX+1}.csv')
                        BACKUP_IDX = (BACKUP_IDX + 1) % N_BACKUPS
                        thisExp.saveAsWideText(
                            fileName=BACKUP_PATH, 
                            appendFile=False, 
                            fileCollisionMethod='overwrite')
                        
                        # store start times for backupSave
                        backupSave.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                        backupSave.tStart = globalClock.getTime(format='float')
                        backupSave.status = STARTED
                        thisExp.addData('backupSave.started', backupSave.tStart)
                        backupSave.maxDuration = None
                        # keep track of which components have finished
                        backupSaveComponents = backupSave.components
                        for thisComponent in backupSave.components:
                            thisComponent.tStart = None
                            thisComponent.tStop = None
                            thisComponent.tStartRefresh = None
                            thisComponent.tStopRefresh = None
                            if hasattr(thisComponent, 'status'):
                                thisComponent.status = NOT_STARTED
                        # reset timers
                        t = 0
                        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                        frameN = -1
                        
                        # --- Run Routine "backupSave" ---
                        # if trial has changed, end Routine now
                        if isinstance(block, data.TrialHandler2) and thisBlock.thisN != block.thisTrial.thisN:
                            continueRoutine = False
                        backupSave.forceEnded = routineForceEnded = not continueRoutine
                        while continueRoutine:
                            # get current time
                            t = routineTimer.getTime()
                            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                            # update/draw components on each frame
                            
                            # check for quit (typically the Esc key)
                            if defaultKeyboard.getKeys(keyList=["escape"]):
                                thisExp.status = FINISHED
                            if thisExp.status == FINISHED or endExpNow:
                                endExperiment(thisExp, win=win)
                                return
                            # pause experiment here if requested
                            if thisExp.status == PAUSED:
                                pauseExperiment(
                                    thisExp=thisExp, 
                                    win=win, 
                                    timers=[routineTimer], 
                                    playbackComponents=[]
                                )
                                # skip the frame we paused on
                                continue
                            
                            # check if all components have finished
                            if not continueRoutine:  # a component has requested a forced-end of Routine
                                backupSave.forceEnded = routineForceEnded = True
                                break
                            continueRoutine = False  # will revert to True if at least one component still running
                            for thisComponent in backupSave.components:
                                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                    continueRoutine = True
                                    break  # at least one component has not yet finished
                            
                            # refresh the screen
                            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                                win.flip()
                        
                        # --- Ending Routine "backupSave" ---
                        for thisComponent in backupSave.components:
                            if hasattr(thisComponent, "setAutoDraw"):
                                thisComponent.setAutoDraw(False)
                        # store stop times for backupSave
                        backupSave.tStop = globalClock.getTime(format='float')
                        backupSave.tStopRefresh = tThisFlipGlobal
                        thisExp.addData('backupSave.stopped', backupSave.tStop)
                        # the Routine "backupSave" was not non-slip safe, so reset the non-slip timer
                        routineTimer.reset()
                        thisExp.nextEntry()
                        
                    # completed n_trials_each_block repeats of 'block'
                    
                    if thisSession is not None:
                        # if running in a Session with a Liaison client, send data up to now
                        thisSession.sendExperimentData()
                    
                    # --- Prepare to start Routine "block_feedback" ---
                    # create an object to store info about Routine block_feedback
                    block_feedback = data.Routine(
                        name='block_feedback',
                        components=[textBlockFeedback, blockFeedbackMouse],
                    )
                    block_feedback.status = NOT_STARTED
                    continueRoutine = True
                    # update component parameters for each repeat
                    # Run 'Begin Routine' code from codeBlockFeedback
                    # show the cursor
                    win.mouseVisible = True
                    
                    block_error_stats = funcCountErrrors(block_error_records)
                    display_objects = funcPlotStats(block_error_stats, win)
                    
                    # flag
                    to_terminate_stage = False
                    
                    # add the buttons
                    continue_button = None
                    repeat_button = None
                    has_passed_qc = False
                    
                    if has_finished_instruction and not has_finished_rehearsal:
                        # still practicing
                        has_passed_qc = check_block_quality(block_error_records)
                        has_ever_passed_qc = has_ever_passed_qc or has_passed_qc
                        if block_stage.thisN < block_to_run_each_stage:
                            # if haven't hit max trial numbers
                            if has_passed_qc:
                                # if one pass the qc, allowed to move on to real trials 
                                repeat_button = funcCreateButton(
                                    win, 'practice again', pos=(-0.2,-0.42),
                                    size=(0.35, 0.1))
                                display_objects.append(repeat_button)
                                
                                continue_button = funcCreateButton(
                                    win, 'continue', pos=(0.2,-0.42),
                                    size=(0.35, 0.1))
                                display_objects.append(continue_button)
                                
                                # set the text
                                textBlockFeedback.setText(
                                    'Good! Press [continue] to start real trials \n'+
                                    'or [practice again] to start another practice')
                            else:
                                repeat_button = funcCreateButton(
                                    win, 'practice again', pos=(0,-0.42),
                                    size=(0.35, 0.1))
                                display_objects.append(repeat_button)
                                textBlockFeedback.setText(
                                    'You need more practice before starting the real trials.\n'+
                                    'Press [practice again] to start another practice')
                    
                    if repeat_button is None and continue_button is None:
                        # other cases
                        continue_button = funcCreateButton(
                            win, 'continue', pos=(0, -0.42),
                            size=(0.35, 0.1))
                        display_objects.append(continue_button)
                        textBlockFeedback.setText(
                            'Your performance in this block.\n'+
                            'Press [continue] to continue')
                    
                    
                    # setup some python lists for storing info about the blockFeedbackMouse
                    gotValidClick = False  # until a click is received
                    # store start times for block_feedback
                    block_feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
                    block_feedback.tStart = globalClock.getTime(format='float')
                    block_feedback.status = STARTED
                    thisExp.addData('block_feedback.started', block_feedback.tStart)
                    block_feedback.maxDuration = None
                    # keep track of which components have finished
                    block_feedbackComponents = block_feedback.components
                    for thisComponent in block_feedback.components:
                        thisComponent.tStart = None
                        thisComponent.tStop = None
                        thisComponent.tStartRefresh = None
                        thisComponent.tStopRefresh = None
                        if hasattr(thisComponent, 'status'):
                            thisComponent.status = NOT_STARTED
                    # reset timers
                    t = 0
                    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                    frameN = -1
                    
                    # --- Run Routine "block_feedback" ---
                    # if trial has changed, end Routine now
                    if isinstance(block_stage, data.TrialHandler2) and thisBlock_stage.thisN != block_stage.thisTrial.thisN:
                        continueRoutine = False
                    block_feedback.forceEnded = routineForceEnded = not continueRoutine
                    while continueRoutine:
                        # get current time
                        t = routineTimer.getTime()
                        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                        # update/draw components on each frame
                        # Run 'Each Frame' code from codeBlockFeedback
                        left_button = blockFeedbackMouse.getPressed()[0]
                        mouse_position = blockFeedbackMouse.getPos()
                        
                        if left_button and (t > 0.5):
                            if repeat_button is not None:
                                if repeat_button.contains(mouse_position):
                                    # to repeat is just to continue
                                    continueRoutine = False
                                elif continue_button is not None:
                                    if continue_button.contains(mouse_position):
                                        # confirm stop practicing
                                        if has_finished_instruction and not has_finished_rehearsal:
                                            to_terminate_stage = True
                                        continueRoutine = False
                            else: # there must exist a continue button
                                if continue_button.contains(mouse_position):
                                    # to repeat is just to continue
                                    continueRoutine = False
                        
                        # *textBlockFeedback* updates
                        
                        # if textBlockFeedback is starting this frame...
                        if textBlockFeedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            textBlockFeedback.frameNStart = frameN  # exact frame index
                            textBlockFeedback.tStart = t  # local t and not account for scr refresh
                            textBlockFeedback.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(textBlockFeedback, 'tStartRefresh')  # time at next scr refresh
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'textBlockFeedback.started')
                            # update status
                            textBlockFeedback.status = STARTED
                            textBlockFeedback.setAutoDraw(True)
                        
                        # if textBlockFeedback is active this frame...
                        if textBlockFeedback.status == STARTED:
                            # update params
                            pass
                        # *blockFeedbackMouse* updates
                        
                        # if blockFeedbackMouse is starting this frame...
                        if blockFeedbackMouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            blockFeedbackMouse.frameNStart = frameN  # exact frame index
                            blockFeedbackMouse.tStart = t  # local t and not account for scr refresh
                            blockFeedbackMouse.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(blockFeedbackMouse, 'tStartRefresh')  # time at next scr refresh
                            # update status
                            blockFeedbackMouse.status = STARTED
                            blockFeedbackMouse.mouseClock.reset()
                            prevButtonState = blockFeedbackMouse.getPressed()  # if button is down already this ISN'T a new click
                        
                        # check for quit (typically the Esc key)
                        if defaultKeyboard.getKeys(keyList=["escape"]):
                            thisExp.status = FINISHED
                        if thisExp.status == FINISHED or endExpNow:
                            endExperiment(thisExp, win=win)
                            return
                        # pause experiment here if requested
                        if thisExp.status == PAUSED:
                            pauseExperiment(
                                thisExp=thisExp, 
                                win=win, 
                                timers=[routineTimer], 
                                playbackComponents=[]
                            )
                            # skip the frame we paused on
                            continue
                        
                        # check if all components have finished
                        if not continueRoutine:  # a component has requested a forced-end of Routine
                            block_feedback.forceEnded = routineForceEnded = True
                            break
                        continueRoutine = False  # will revert to True if at least one component still running
                        for thisComponent in block_feedback.components:
                            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                continueRoutine = True
                                break  # at least one component has not yet finished
                        
                        # refresh the screen
                        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                            win.flip()
                    
                    # --- Ending Routine "block_feedback" ---
                    for thisComponent in block_feedback.components:
                        if hasattr(thisComponent, "setAutoDraw"):
                            thisComponent.setAutoDraw(False)
                    # store stop times for block_feedback
                    block_feedback.tStop = globalClock.getTime(format='float')
                    block_feedback.tStopRefresh = tThisFlipGlobal
                    thisExp.addData('block_feedback.stopped', block_feedback.tStop)
                    # Run 'End Routine' code from codeBlockFeedback
                    # remove all objects
                    n_display_objects = len(display_objects)
                    for i in range(n_display_objects):
                        disp_obj = display_objects[i]
                        disp_obj.setAutoDraw(False)
                    display_objects = []
                    
                    if to_terminate_stage:
                        block_stage.finished = True
                    # store data for block_stage (TrialHandler)
                    # the Routine "block_feedback" was not non-slip safe, so reset the non-slip timer
                    routineTimer.reset()
                # completed n_blocks_each_subtype repeats of 'block_stage'
                
            # completed n_sampling_strategies_allowed repeats of 'sub_sample_stage_loop'
            
        # completed max(0, n_response_modes_allowed) repeats of 'ActualTaskLoop'
        
        
        # set up handler to look after randomisation of conditions etc
        breakTimeLoop = data.TrialHandler2(
            name='breakTimeLoop',
            nReps=TasktypeLoop.nTotal > (TasktypeLoop.thisN+1), 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(breakTimeLoop)  # add the loop to the experiment
        thisBreakTimeLoop = breakTimeLoop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisBreakTimeLoop.rgb)
        if thisBreakTimeLoop != None:
            for paramName in thisBreakTimeLoop:
                globals()[paramName] = thisBreakTimeLoop[paramName]
        
        for thisBreakTimeLoop in breakTimeLoop:
            currentLoop = breakTimeLoop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # abbreviate parameter names if possible (e.g. rgb = thisBreakTimeLoop.rgb)
            if thisBreakTimeLoop != None:
                for paramName in thisBreakTimeLoop:
                    globals()[paramName] = thisBreakTimeLoop[paramName]
            
            # --- Prepare to start Routine "breakTime" ---
            # create an object to store info about Routine breakTime
            breakTime = data.Routine(
                name='breakTime',
                components=[textBreakTime, keyBreakTime],
            )
            breakTime.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # create starting attributes for keyBreakTime
            keyBreakTime.keys = []
            keyBreakTime.rt = []
            _keyBreakTime_allKeys = []
            # store start times for breakTime
            breakTime.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            breakTime.tStart = globalClock.getTime(format='float')
            breakTime.status = STARTED
            thisExp.addData('breakTime.started', breakTime.tStart)
            breakTime.maxDuration = None
            # keep track of which components have finished
            breakTimeComponents = breakTime.components
            for thisComponent in breakTime.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "breakTime" ---
            # if trial has changed, end Routine now
            if isinstance(breakTimeLoop, data.TrialHandler2) and thisBreakTimeLoop.thisN != breakTimeLoop.thisTrial.thisN:
                continueRoutine = False
            breakTime.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *textBreakTime* updates
                
                # if textBreakTime is starting this frame...
                if textBreakTime.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textBreakTime.frameNStart = frameN  # exact frame index
                    textBreakTime.tStart = t  # local t and not account for scr refresh
                    textBreakTime.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textBreakTime, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textBreakTime.started')
                    # update status
                    textBreakTime.status = STARTED
                    textBreakTime.setAutoDraw(True)
                
                # if textBreakTime is active this frame...
                if textBreakTime.status == STARTED:
                    # update params
                    pass
                
                # *keyBreakTime* updates
                waitOnFlip = False
                
                # if keyBreakTime is starting this frame...
                if keyBreakTime.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    keyBreakTime.frameNStart = frameN  # exact frame index
                    keyBreakTime.tStart = t  # local t and not account for scr refresh
                    keyBreakTime.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(keyBreakTime, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    keyBreakTime.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(keyBreakTime.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(keyBreakTime.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if keyBreakTime.status == STARTED and not waitOnFlip:
                    theseKeys = keyBreakTime.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                    _keyBreakTime_allKeys.extend(theseKeys)
                    if len(_keyBreakTime_allKeys):
                        keyBreakTime.keys = _keyBreakTime_allKeys[-1].name  # just the last key pressed
                        keyBreakTime.rt = _keyBreakTime_allKeys[-1].rt
                        keyBreakTime.duration = _keyBreakTime_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    breakTime.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in breakTime.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "breakTime" ---
            for thisComponent in breakTime.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for breakTime
            breakTime.tStop = globalClock.getTime(format='float')
            breakTime.tStopRefresh = tThisFlipGlobal
            thisExp.addData('breakTime.stopped', breakTime.tStop)
            # the Routine "breakTime" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
        # completed TasktypeLoop.nTotal > (TasktypeLoop.thisN+1) repeats of 'breakTimeLoop'
        
    # completed max(1, n_trial_types_allowed) repeats of 'TasktypeLoop'
    
    
    # --- Prepare to start Routine "terminateExp" ---
    # create an object to store info about Routine terminateExp
    terminateExp = data.Routine(
        name='terminateExp',
        components=[],
    )
    terminateExp.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from closeEl
    if eyetracking == 1:
        
        # Step 7: disconnect, download the EDF file, then terminate the task
        terminate_task()
        
    
    # store start times for terminateExp
    terminateExp.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    terminateExp.tStart = globalClock.getTime(format='float')
    terminateExp.status = STARTED
    thisExp.addData('terminateExp.started', terminateExp.tStart)
    terminateExp.maxDuration = None
    # keep track of which components have finished
    terminateExpComponents = terminateExp.components
    for thisComponent in terminateExp.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "terminateExp" ---
    terminateExp.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            terminateExp.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in terminateExp.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "terminateExp" ---
    for thisComponent in terminateExp.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for terminateExp
    terminateExp.tStop = globalClock.getTime(format='float')
    terminateExp.tStopRefresh = tThisFlipGlobal
    thisExp.addData('terminateExp.stopped', terminateExp.tStop)
    thisExp.nextEntry()
    # the Routine "terminateExp" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
