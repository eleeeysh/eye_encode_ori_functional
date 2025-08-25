#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.1.1),
    on Mon Aug 25 15:00:28 2025
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
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from codePrepareExp
import random
import numpy as np
import itertools

""" For sampling """
class SimplePerturbationLoader:
    def __init__(self):
        # create stimulus samples
        self.stim_ranges = [
            [12, 36],
            [54, 78],
            [102, 126],
            [144, 168],
        ]
        self.n_sample_per_range = 5
        self.stim_samples = [
            np.linspace(s[0], s[1], self.n_sample_per_range) 
            for s in self.stim_ranges]
        
        # create perturbation sampling range
        self.perturbations = [
            (1, 1), # quadrant 1
            (-1, 1), # quadrant 2
            (-1, -1), # quadrant 3
            (1, -1), # quadrant 4
        ]
        
        # set other configurations
        self.jittering = 2
        
    def stim_add_noise(self, xs):
        noise = np.random.uniform(low=-1, high=1, size=xs.shape) * self.jittering
        return xs + noise
        
    def generate_one_batch_stims(self):
        stim_list = np.concatenate(self.stim_samples)
        # add some noise
        sampled_stims = self.stim_add_noise(stim_list)
        # combination
        combinations = list(itertools.product(sampled_stims, self.perturbations))
        random.shuffle(combinations)
        return combinations
        
""" For calibration """
class DefaultFixation:
    def __init__(self, window):
        self.fixation = visual.ShapeStim(
            win=window, vertices='cross',
            size=(0.06, 0.06), lineWidth=1,
            lineColor='white', fillColor='white')
        
        self.setAutoDraw(True)
        
    def setAutoDraw(self, autoDraw):
        self.fixation.setAutoDraw(autoDraw)

# Run 'Before Experiment' code from codeDisplay
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
# Run 'Before Experiment' code from codeAfterMask
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
# Run 'Before Experiment' code from codeDelay
""" For generating perturbations """
class SinglePerturbItemManager:
    def __init__(self, win, item_type):
        self.perturb_scale_range = [0.1, 0.25]
        self.item_size = 0.05
        
        # load resources
        if item_type == 'landolt':
            self.has_ori = True
            self.img = visual.ImageStim(
                win, image='resources/landolt_white.png', 
                pos=(0, 0),
                ori=0,
                size=(self.item_size, self.item_size))
        elif item_type == 'circle':
            self.has_ori = False
            self.img = visual.Circle(
                win, radius=self.item_size/2,
                pos=(0, 0),
                lineColor='white',
                fillColor='white'
            )
        else:
            raise NotImplementedError
        
    def generate_perturb_position(self, perturb_code):
        perturb_center = np.random.uniform(
            self.perturb_scale_range[0],
            self.perturb_scale_range[1],
            size=2
        ) * np.array(perturb_code) # center of perturbation
        return perturb_center
        
    def move(self, perturb_code):
        ## position
        center = self.generate_perturb_position(perturb_code)
        perturb_info = {
            'pos': center,
            'radius': self.item_size/2,
        }
        self.img.pos = center
        
        ## rotation
        if self.has_ori:
            rotation = np.random.randint(10, 170) * (np.random.randint(2)+1) # avoid ambiguous
            self.img.ori = rotation
            perturb_info['ori'] = rotation
        
        return self.img, perturb_info
        
class PerturbItemManager:
    def __init__(self, win):
        # a few things needed
        self.landolt = SinglePerturbItemManager(win, 'landolt')
        self.circle = SinglePerturbItemManager(win, 'circle')
        self.center = SinglePerturbItemManager(win, 'circle')
        self.reset()
    
    def reset(self):
        self.center.move([0, 0])
        self.active_items = [self.center,]
        
    def update(self, is_task, perturb_code):
        draw_item, move_info = None, None
        self.reset()
        if perturb_code is not None:
            if is_task: # overt movement
                draw_item, move_info = self.landolt.move(perturb_code)
            else: # covert movement
                draw_item, move_info = self.circle.move(perturb_code)
        else:
            if is_task: # judgement at center
                draw_item, move_info = self.landolt.move([0, 0])
            else:
                draw_item, move_info = self.center.move([0, 0])
        self.active_items = [draw_item,]
        return move_info
        
    def draw(self):
        for item in self.active_items:
            item.draw()


class PerturbSchedueler:
    def __init__(self, win, timing_settings, delay_length):
        self.perturb_item_manager = PerturbItemManager(win)
        self.n_perturb = timing_settings['n_perturb']
        self.n_judge = timing_settings['n_judge']
        self.per_t_min, self.per_t_max = timing_settings['duration']
        self.earliest, self.latest = timing_settings['timing']
        self.min_gap = timing_settings['min_gap']
        self.delay_length = delay_length
        
        self.clear_schedule()
        
    def clear_schedule(self):
        # reset schedule
        self.schedule = None
        self.records = None
        self.current_id = -1
        self.last_record_id = -1
        self.on = False
        self.require_input = False
        self.perturb_type = None

    def reset_schedule(self, perturb_type):
        self.clear_schedule()
        self.perturb_item_manager.reset()
        self.schedule = []
        self.records = {'judge': [], 'perturb': []}
        
        # determine duration of each epochs
        assert self.delay_length >= self.latest
        self.perturb_type = perturb_type
        if self.perturb_type == 'covert':
            self.n_events = self.n_perturb + self.n_judge
        elif self.perturb_type == 'overt':
            assert self.n_perturb == self.n_judge
            self.n_events = self.n_perturb
        assert (
            self.n_events * self.per_t_max + 
            (self.n_events-1) * self.min_gap +
            self.earliest) <= self.latest
        durations = np.random.uniform(
            self.per_t_min, self.per_t_max, 
            size=self.n_events
        )
        
        # sample the start points
        reduced_duration = self.latest - self.earliest - np.sum(
            durations) - (self.n_events-1) * self.min_gap
        starts = np.random.uniform(0, reduced_duration, size=self.n_events)
        starts = np.sort(starts)
        
        # get time start end for each
        t_overhead = self.earliest
        for i in range(self.n_events):
            start_t = t_overhead + starts[i]
            end_t = start_t + durations[i]
            self.schedule.append([start_t, end_t])
            t_overhead += self.min_gap 
            t_overhead += end_t - start_t
            
        # finally, decided which is task and which is perturb
        self.perturb_type_codes = [] # each pair: is_task + is_perturb
        if self.perturb_type == 'overt':
            self.perturb_type_codes += [
                {'is_task': True, 'is_perturb': True} for _ in range(self.n_events)]
        elif self.perturb_type == 'covert':
            self.perturb_type_codes += [
                {'is_task': False, 'is_perturb': True} for _ in range(self.n_perturb)]
            self.perturb_type_codes += [
                {'is_task': True, 'is_perturb': False} for _ in range(self.n_judge)]
            random.shuffle(self.perturb_type_codes)
            
        # other stuffs
        self.current_id = 0
        
    def check_schedule(self, t, perturb_code):
        # return the event the current t is in
        # check whether it's on or off
        n_total = len(self.schedule)
        if self.current_id < n_total:
            epoch = self.schedule[self.current_id]
            if t < epoch[0]:
                self.on = False
            elif t > epoch[1]:
                self.on = False
                while self.current_id < n_total-1 and t > epoch[1]:
                    # find the next match
                    self.current_id += 1
                    epoch = self.schedule[self.current_id]
            else:
                self.on = True
        else:
            self.on = False
        
        if self.on:
            # currently something is happening
            current_perturb_type_codes = self.perturb_type_codes[
                    self.current_id]
            if self.current_id > self.last_record_id:
                # there is something not plotted or recorded yet --> update it
                is_task = current_perturb_type_codes['is_task']
                is_perturb = current_perturb_type_codes['is_perturb']
                perturb_code = perturb_code if is_perturb else None
                update_info = self.perturb_item_manager.update(
                    is_task = is_task,
                    perturb_code = perturb_code,
                )
                # append time info
                update_info['tstart'] = t # use actual start time
                update_info['tend'] = self.schedule[self.current_id][1]
                # send to records
                if is_task:
                    self.records['judge'].append(update_info.copy())
                if is_perturb:
                    self.records['perturb'].append(update_info.copy())
                # update all states
                self.last_record_id = self.current_id
                self.require_input = is_task
            
        else: # just give the center fixation
            self.perturb_item_manager.update(
                is_task = False,
                perturb_code = None,
            )
            self.require_input = False

        self.perturb_item_manager.draw()
        

    def check_input(self, t, key_input, key_map):
        feedback = None
        if self.require_input:
            if key_input: # there is some input...
                # only care if input is required
                ans_is_left = self.records['judge'][-1]['ori'] > 180
                ## check the key
                last_key = key_input[-1].name
                key_is_left = last_key.lower() == key_map[0]
                key_is_correct = key_is_left == ans_is_left
                ## update the result
                self.records['judge'][-1]['ans_correct'] = key_is_correct
                self.records['judge'][-1]['rt'] = t - self.records['judge'][-1]['tstart']
                ## end it
                self.on = False
                self.require_input = False
                self.current_id += 1
                ## prepare feedback
                feedback = key_is_correct
            else:
                # TODO: ignore it?
                pass
        return feedback
                
         
# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.1.1'
expName = 'ClickOnly'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'dominant_hand': ['right','left'],
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
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
_winSize = [2560, 1440]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

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
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/siy009/Desktop/experiments/psychopy/mode_eyetrack_functional/ClickOnly_lastrun.py',
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
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


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
            winType='pyglet', allowGUI=False, allowStencil=True,
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
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
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
    if deviceManager.getDevice('keyDelay') is None:
        # initialise keyDelay
        keyDelay = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='keyDelay',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
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
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
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
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
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
    
    # --- Initialize components for Routine "prepareExp" ---
    # Run 'Begin Experiment' code from codePrepareExp
    # set configrations
    N_REPEATS = 4
    N_REPEAT_BLOCK = 40 # FOR DEBUGGING, 4 will be decent?
    STIM_LENGTH = 0.75
    DELAY_LENGTH = 5
    
    # read exp settings
    dominant_hand = expInfo['dominant_hand']
    is_right_handed = dominant_hand == 'right'
    
    # add managers
    stim_perturb_loader = SimplePerturbationLoader()
    
    # some configuration for display
    PATCH_RADIUS = 0.16
    
    
    
    # --- Initialize components for Routine "PreBlock" ---
    
    # --- Initialize components for Routine "PreTrial" ---
    
    # --- Initialize components for Routine "Display" ---
    textDisplayPlaceholder = visual.TextBox2(
         win, text=None, placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0, 0), draggable=False,      letterHeight=0.05,
         size=(0.5, 0.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='textDisplayPlaceholder',
         depth=-1, autoLog=True,
    )
    
    # --- Initialize components for Routine "AfterMask" ---
    
    # --- Initialize components for Routine "Delay" ---
    keyDelay = keyboard.Keyboard(deviceName='keyDelay')
    # Run 'Begin Experiment' code from codeDelay
    # a few configurations
    ## number of perturbations during the delay
    PERTURB_TIMING_SETTINGS = {
        'n_perturb': 1,
        'n_judge': 1,
        'duration': (1.0, 1.5),
        'timing': (0.5, 4.5),
        'min_gap': 0.1,
    }
    perturb_scheduler = PerturbSchedueler(
        win=win,
        timing_settings=PERTURB_TIMING_SETTINGS, 
        delay_length=DELAY_LENGTH)
    
    ## left / right hand mapping
    left_hand_keys = ['j', 'k'] # for left-handed
    right_hand_keys = ['a', 's'] # for right-handed
    assert len(left_hand_keys) == len(right_hand_keys)
    resp_key_map = right_hand_keys if is_right_handed else left_hand_keys
    
    ## load sound feedback
    correct_sound = sound.Sound('bleep.wav')
    wrong_sound   = sound.Sound('buzz.wav')
    
    textDelayPlaceholder = visual.TextBox2(
         win, text=None, placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0, 0), draggable=False,      letterHeight=0.05,
         size=(0.5, 0.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='textDelayPlaceholder',
         depth=-2, autoLog=True,
    )
    
    # --- Initialize components for Routine "OriTest" ---
    
    # --- Initialize components for Routine "postTrial" ---
    
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
    
    # --- Prepare to start Routine "prepareExp" ---
    # create an object to store info about Routine prepareExp
    prepareExp = data.Routine(
        name='prepareExp',
        components=[],
    )
    prepareExp.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from codePrepareExp
    # now generate the stimuli and perturbations
    all_repeats_stims = []
    BLOCK_SIZE = 0
    for _ in range(N_REPEATS):
        # generate stims and perturbations
        stim_perturb_list = stim_perturb_loader.generate_one_batch_stims()
        # cut into blocks
        assert len(stim_perturb_list) % N_REPEAT_BLOCK == 0
        BLOCK_SIZE = len(stim_perturb_list) // N_REPEAT_BLOCK
        for i in range(N_REPEAT_BLOCK):
            all_repeats_stims.append(stim_perturb_list[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE])
    
    N_BLOCKS = len(all_repeats_stims)
    
    # also determine the type of perturbation (overt vs covert)
    PERTURB_TYPE_CANDIDATES = ['overt', 'covert']
    N_PERTURB_TYPES = len(PERTURB_TYPE_CANDIDATES)
    all_perturb_types = (PERTURB_TYPE_CANDIDATES * 
        ((N_BLOCKS + N_PERTURB_TYPES - 1) // N_PERTURB_TYPES))[:N_BLOCKS]
    
    logging.info(N_BLOCKS)
    
            
    
    
    # store start times for prepareExp
    prepareExp.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    prepareExp.tStart = globalClock.getTime(format='float')
    prepareExp.status = STARTED
    thisExp.addData('prepareExp.started', prepareExp.tStart)
    prepareExp.maxDuration = None
    # keep track of which components have finished
    prepareExpComponents = prepareExp.components
    for thisComponent in prepareExp.components:
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
    
    # --- Run Routine "prepareExp" ---
    prepareExp.forceEnded = routineForceEnded = not continueRoutine
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
                timers=[routineTimer, globalClock], 
                currentRoutine=prepareExp,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            prepareExp.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in prepareExp.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "prepareExp" ---
    for thisComponent in prepareExp.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for prepareExp
    prepareExp.tStop = globalClock.getTime(format='float')
    prepareExp.tStopRefresh = tThisFlipGlobal
    thisExp.addData('prepareExp.stopped', prepareExp.tStop)
    thisExp.nextEntry()
    # the Routine "prepareExp" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    blockLoop = data.TrialHandler2(
        name='blockLoop',
        nReps=N_BLOCKS, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(blockLoop)  # add the loop to the experiment
    thisBlockLoop = blockLoop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlockLoop.rgb)
    if thisBlockLoop != None:
        for paramName in thisBlockLoop:
            globals()[paramName] = thisBlockLoop[paramName]
    
    for thisBlockLoop in blockLoop:
        blockLoop.status = STARTED
        if hasattr(thisBlockLoop, 'status'):
            thisBlockLoop.status = STARTED
        currentLoop = blockLoop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisBlockLoop.rgb)
        if thisBlockLoop != None:
            for paramName in thisBlockLoop:
                globals()[paramName] = thisBlockLoop[paramName]
        
        # --- Prepare to start Routine "PreBlock" ---
        # create an object to store info about Routine PreBlock
        PreBlock = data.Routine(
            name='PreBlock',
            components=[],
        )
        PreBlock.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from codePreBlock
        block_id = blockLoop.thisN
        cur_block_stim_perturbs = all_repeats_stims[blockLoop.thisN]
        cur_block_perturb_type = all_perturb_types[blockLoop.thisN]
        # store start times for PreBlock
        PreBlock.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        PreBlock.tStart = globalClock.getTime(format='float')
        PreBlock.status = STARTED
        thisExp.addData('PreBlock.started', PreBlock.tStart)
        PreBlock.maxDuration = None
        # keep track of which components have finished
        PreBlockComponents = PreBlock.components
        for thisComponent in PreBlock.components:
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
        
        # --- Run Routine "PreBlock" ---
        PreBlock.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisBlockLoop, 'status') and thisBlockLoop.status == STOPPING:
                continueRoutine = False
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
                    timers=[routineTimer, globalClock], 
                    currentRoutine=PreBlock,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                PreBlock.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in PreBlock.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "PreBlock" ---
        for thisComponent in PreBlock.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for PreBlock
        PreBlock.tStop = globalClock.getTime(format='float')
        PreBlock.tStopRefresh = tThisFlipGlobal
        thisExp.addData('PreBlock.stopped', PreBlock.tStop)
        # the Routine "PreBlock" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trialLoop = data.TrialHandler2(
            name='trialLoop',
            nReps=BLOCK_SIZE, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(trialLoop)  # add the loop to the experiment
        thisTrialLoop = trialLoop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrialLoop.rgb)
        if thisTrialLoop != None:
            for paramName in thisTrialLoop:
                globals()[paramName] = thisTrialLoop[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisTrialLoop in trialLoop:
            trialLoop.status = STARTED
            if hasattr(thisTrialLoop, 'status'):
                thisTrialLoop.status = STARTED
            currentLoop = trialLoop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisTrialLoop.rgb)
            if thisTrialLoop != None:
                for paramName in thisTrialLoop:
                    globals()[paramName] = thisTrialLoop[paramName]
            
            # --- Prepare to start Routine "PreTrial" ---
            # create an object to store info about Routine PreTrial
            PreTrial = data.Routine(
                name='PreTrial',
                components=[],
            )
            PreTrial.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from codePretrial
            cur_trial_i = trialLoop.thisN
            cur_trial_stim_perturbs = cur_block_stim_perturbs[cur_trial_i]
            # store start times for PreTrial
            PreTrial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            PreTrial.tStart = globalClock.getTime(format='float')
            PreTrial.status = STARTED
            thisExp.addData('PreTrial.started', PreTrial.tStart)
            PreTrial.maxDuration = None
            # keep track of which components have finished
            PreTrialComponents = PreTrial.components
            for thisComponent in PreTrial.components:
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
            
            # --- Run Routine "PreTrial" ---
            PreTrial.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisTrialLoop, 'status') and thisTrialLoop.status == STOPPING:
                    continueRoutine = False
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
                        timers=[routineTimer, globalClock], 
                        currentRoutine=PreTrial,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    PreTrial.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in PreTrial.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "PreTrial" ---
            for thisComponent in PreTrial.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for PreTrial
            PreTrial.tStop = globalClock.getTime(format='float')
            PreTrial.tStopRefresh = tThisFlipGlobal
            thisExp.addData('PreTrial.stopped', PreTrial.tStop)
            # the Routine "PreTrial" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "Display" ---
            # create an object to store info about Routine Display
            Display = data.Routine(
                name='Display',
                components=[textDisplayPlaceholder],
            )
            Display.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from codeDisplay
            stim_deg = cur_block_stim_perturbs[cur_trial_i][0]
            stim_obj = funcDrawStim(win, stim_deg, [0, 0], PATCH_RADIUS)
            thisExp.addData('stim', stim_deg)
            
            textDisplayPlaceholder.reset()
            # store start times for Display
            Display.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            Display.tStart = globalClock.getTime(format='float')
            Display.status = STARTED
            thisExp.addData('Display.started', Display.tStart)
            Display.maxDuration = None
            # keep track of which components have finished
            DisplayComponents = Display.components
            for thisComponent in Display.components:
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
            
            # --- Run Routine "Display" ---
            Display.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisTrialLoop, 'status') and thisTrialLoop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *textDisplayPlaceholder* updates
                
                # if textDisplayPlaceholder is starting this frame...
                if textDisplayPlaceholder.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textDisplayPlaceholder.frameNStart = frameN  # exact frame index
                    textDisplayPlaceholder.tStart = t  # local t and not account for scr refresh
                    textDisplayPlaceholder.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textDisplayPlaceholder, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textDisplayPlaceholder.started')
                    # update status
                    textDisplayPlaceholder.status = STARTED
                    textDisplayPlaceholder.setAutoDraw(True)
                
                # if textDisplayPlaceholder is active this frame...
                if textDisplayPlaceholder.status == STARTED:
                    # update params
                    pass
                
                # if textDisplayPlaceholder is stopping this frame...
                if textDisplayPlaceholder.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > textDisplayPlaceholder.tStartRefresh + STIM_LENGTH-frameTolerance:
                        # keep track of stop time/frame for later
                        textDisplayPlaceholder.tStop = t  # not accounting for scr refresh
                        textDisplayPlaceholder.tStopRefresh = tThisFlipGlobal  # on global time
                        textDisplayPlaceholder.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'textDisplayPlaceholder.stopped')
                        # update status
                        textDisplayPlaceholder.status = FINISHED
                        textDisplayPlaceholder.setAutoDraw(False)
                
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
                        timers=[routineTimer, globalClock], 
                        currentRoutine=Display,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    Display.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in Display.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Display" ---
            for thisComponent in Display.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for Display
            Display.tStop = globalClock.getTime(format='float')
            Display.tStopRefresh = tThisFlipGlobal
            thisExp.addData('Display.stopped', Display.tStop)
            # Run 'End Routine' code from codeDisplay
            stim_obj.setAutoDraw(False)
            # the Routine "Display" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "AfterMask" ---
            # create an object to store info about Routine AfterMask
            AfterMask = data.Routine(
                name='AfterMask',
                components=[],
            )
            AfterMask.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for AfterMask
            AfterMask.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            AfterMask.tStart = globalClock.getTime(format='float')
            AfterMask.status = STARTED
            thisExp.addData('AfterMask.started', AfterMask.tStart)
            AfterMask.maxDuration = None
            # keep track of which components have finished
            AfterMaskComponents = AfterMask.components
            for thisComponent in AfterMask.components:
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
            
            # --- Run Routine "AfterMask" ---
            AfterMask.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisTrialLoop, 'status') and thisTrialLoop.status == STOPPING:
                    continueRoutine = False
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
                        timers=[routineTimer, globalClock], 
                        currentRoutine=AfterMask,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    AfterMask.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in AfterMask.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "AfterMask" ---
            for thisComponent in AfterMask.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for AfterMask
            AfterMask.tStop = globalClock.getTime(format='float')
            AfterMask.tStopRefresh = tThisFlipGlobal
            thisExp.addData('AfterMask.stopped', AfterMask.tStop)
            # the Routine "AfterMask" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "Delay" ---
            # create an object to store info about Routine Delay
            Delay = data.Routine(
                name='Delay',
                components=[keyDelay, textDelayPlaceholder],
            )
            Delay.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # create starting attributes for keyDelay
            keyDelay.keys = []
            keyDelay.rt = []
            _keyDelay_allKeys = []
            # Run 'Begin Routine' code from codeDelay
            # set up the timing of displaying perturbations
            perturb_scheduler.reset_schedule(cur_block_perturb_type)
            
            # get stim perturb codes
            perturb_code = cur_block_stim_perturbs[cur_trial_i][1]
            
            # save info
            thisExp.addData("perturb_code_x", perturb_code[0])
            thisExp.addData("perturb_code_y", perturb_code[1])
            thisExp.addData("perturb_type", cur_block_perturb_type)
            
            textDelayPlaceholder.reset()
            # store start times for Delay
            Delay.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            Delay.tStart = globalClock.getTime(format='float')
            Delay.status = STARTED
            thisExp.addData('Delay.started', Delay.tStart)
            Delay.maxDuration = None
            # keep track of which components have finished
            DelayComponents = Delay.components
            for thisComponent in Delay.components:
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
            
            # --- Run Routine "Delay" ---
            Delay.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisTrialLoop, 'status') and thisTrialLoop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *keyDelay* updates
                waitOnFlip = False
                
                # if keyDelay is starting this frame...
                if keyDelay.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    keyDelay.frameNStart = frameN  # exact frame index
                    keyDelay.tStart = t  # local t and not account for scr refresh
                    keyDelay.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(keyDelay, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    keyDelay.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(keyDelay.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(keyDelay.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if keyDelay is stopping this frame...
                if keyDelay.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > keyDelay.tStartRefresh + DELAY_LENGTH-frameTolerance:
                        # keep track of stop time/frame for later
                        keyDelay.tStop = t  # not accounting for scr refresh
                        keyDelay.tStopRefresh = tThisFlipGlobal  # on global time
                        keyDelay.frameNStop = frameN  # exact frame index
                        # update status
                        keyDelay.status = FINISHED
                        keyDelay.status = FINISHED
                if keyDelay.status == STARTED and not waitOnFlip:
                    theseKeys = keyDelay.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
                    _keyDelay_allKeys.extend(theseKeys)
                    if len(_keyDelay_allKeys):
                        keyDelay.keys = _keyDelay_allKeys[-1].name  # just the last key pressed
                        keyDelay.rt = _keyDelay_allKeys[-1].rt
                        keyDelay.duration = _keyDelay_allKeys[-1].duration
                # Run 'Each Frame' code from codeDelay
                # check schedule
                perturb_scheduler.check_schedule(t, perturb_code)
                
                if perturb_scheduler.require_input:
                    # read the keyboard input
                    keys = keyDelay.getKeys(resp_key_map)
                    perturb_feedback = perturb_scheduler.check_input(t, keys, resp_key_map)
                    if perturb_feedback is not None:
                        # there are some correct/incorrect feedback
                        correct_sound.stop()
                        wrong_sound.stop()
                        if perturb_feedback:
                            correct_sound.play()
                        else:
                            wrong_sound.play()
                
                
                # *textDelayPlaceholder* updates
                
                # if textDelayPlaceholder is starting this frame...
                if textDelayPlaceholder.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textDelayPlaceholder.frameNStart = frameN  # exact frame index
                    textDelayPlaceholder.tStart = t  # local t and not account for scr refresh
                    textDelayPlaceholder.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textDelayPlaceholder, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textDelayPlaceholder.started')
                    # update status
                    textDelayPlaceholder.status = STARTED
                    textDelayPlaceholder.setAutoDraw(True)
                
                # if textDelayPlaceholder is active this frame...
                if textDelayPlaceholder.status == STARTED:
                    # update params
                    pass
                
                # if textDelayPlaceholder is stopping this frame...
                if textDelayPlaceholder.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > textDelayPlaceholder.tStartRefresh + DELAY_LENGTH -frameTolerance:
                        # keep track of stop time/frame for later
                        textDelayPlaceholder.tStop = t  # not accounting for scr refresh
                        textDelayPlaceholder.tStopRefresh = tThisFlipGlobal  # on global time
                        textDelayPlaceholder.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'textDelayPlaceholder.stopped')
                        # update status
                        textDelayPlaceholder.status = FINISHED
                        textDelayPlaceholder.setAutoDraw(False)
                
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
                        timers=[routineTimer, globalClock], 
                        currentRoutine=Delay,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    Delay.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in Delay.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Delay" ---
            for thisComponent in Delay.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for Delay
            Delay.tStop = globalClock.getTime(format='float')
            Delay.tStopRefresh = tThisFlipGlobal
            thisExp.addData('Delay.stopped', Delay.tStop)
            # Run 'End Routine' code from codeDelay
            # save all records
            for record_type in ['judge', 'perturb']:
                records = perturb_scheduler.records[record_type]
                for i, record in enumerate(records):
                    for k,v in record.items():
                        thisExp.addData(f"{record_type}_{i}_{k}", v)
            
            # the Routine "Delay" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "OriTest" ---
            # create an object to store info about Routine OriTest
            OriTest = data.Routine(
                name='OriTest',
                components=[],
            )
            OriTest.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for OriTest
            OriTest.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            OriTest.tStart = globalClock.getTime(format='float')
            OriTest.status = STARTED
            thisExp.addData('OriTest.started', OriTest.tStart)
            OriTest.maxDuration = None
            # keep track of which components have finished
            OriTestComponents = OriTest.components
            for thisComponent in OriTest.components:
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
            
            # --- Run Routine "OriTest" ---
            OriTest.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisTrialLoop, 'status') and thisTrialLoop.status == STOPPING:
                    continueRoutine = False
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
                        timers=[routineTimer, globalClock], 
                        currentRoutine=OriTest,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    OriTest.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in OriTest.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "OriTest" ---
            for thisComponent in OriTest.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for OriTest
            OriTest.tStop = globalClock.getTime(format='float')
            OriTest.tStopRefresh = tThisFlipGlobal
            thisExp.addData('OriTest.stopped', OriTest.tStop)
            # the Routine "OriTest" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "postTrial" ---
            # create an object to store info about Routine postTrial
            postTrial = data.Routine(
                name='postTrial',
                components=[],
            )
            postTrial.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for postTrial
            postTrial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            postTrial.tStart = globalClock.getTime(format='float')
            postTrial.status = STARTED
            thisExp.addData('postTrial.started', postTrial.tStart)
            postTrial.maxDuration = None
            # keep track of which components have finished
            postTrialComponents = postTrial.components
            for thisComponent in postTrial.components:
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
            
            # --- Run Routine "postTrial" ---
            postTrial.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisTrialLoop, 'status') and thisTrialLoop.status == STOPPING:
                    continueRoutine = False
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
                        timers=[routineTimer, globalClock], 
                        currentRoutine=postTrial,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    postTrial.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in postTrial.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "postTrial" ---
            for thisComponent in postTrial.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for postTrial
            postTrial.tStop = globalClock.getTime(format='float')
            postTrial.tStopRefresh = tThisFlipGlobal
            thisExp.addData('postTrial.stopped', postTrial.tStop)
            # the Routine "postTrial" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            # mark thisTrialLoop as finished
            if hasattr(thisTrialLoop, 'status'):
                thisTrialLoop.status = FINISHED
            # if awaiting a pause, pause now
            if trialLoop.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                trialLoop.status = STARTED
            thisExp.nextEntry()
            
        # completed BLOCK_SIZE repeats of 'trialLoop'
        trialLoop.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # mark thisBlockLoop as finished
        if hasattr(thisBlockLoop, 'status'):
            thisBlockLoop.status = FINISHED
        # if awaiting a pause, pause now
        if blockLoop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            blockLoop.status = STARTED
    # completed N_BLOCKS repeats of 'blockLoop'
    blockLoop.status = FINISHED
    
    
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
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


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
    logging.flush()
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
