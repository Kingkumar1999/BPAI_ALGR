from __future__ import unicode_literals, print_function, absolute_import, division, generators, nested_scopes
from robobo.base import Robobo
import time
from robobo.MLPbrain import MLPbrain
import vrep
import cv2
import numpy as np
import random

class VREPCommunicationError(Exception):
    pass

class Robot():
    def __init__(self, number):
        self.ID = number
        self.gender = 0

class SimulationRobobo(Robobo):
    def __init__(self, lifespan, ips, number=""):
        self._clientID = None
        self._value_number = number

        # Variables
        self.lifespan = lifespan
        self.init_pop_size = ips

        self.max_pop_size = 25
        self.mentalStack = 2
        self.dataPointsPer = 3
        self.range = 0.3
        self.mutation_rate = 0.2
        self.speed = 30

        # Not used atm
        self.move_time = 150

        # Setup
        self.population = []
        self.dead = set(np.arange(self.max_pop_size))
        self.alive = set()
        for i in range(self.max_pop_size):
            self.population.append(Robot(i))

    def get_name(self):
        return "L{}I{}".format(self.lifespan, self.init_pop_size)


    def connect(self, address='192.168.56.1', port=19999):
        vrep.simxFinish(-1)  # just in case, close all opened connections
        self._clientID = vrep.simxStart(address, port, True, True, 5000, 5)  # Connect to V-REP
        if self._clientID >= 0: #  and clientID_0 != -1:
            self.wait_for_ping()
            print('Connected to remote API server: client id {}'.format(self._clientID))
        else:
            raise VREPCommunicationError('Failed connecting to remote API server')

        get_handles_timeout = 3000.0

        startTime = time.time()
        while time.time() - startTime < get_handles_timeout:
            try:
                self.initialize_handles()
                return self
            except vrep.VrepApiError as _e:
                print("Handle initialization failed, retrying.")
                time.sleep(0.05)

        return False
    
    # # New func ---- don't think well use this
    # def duplicate(self, handles):
    #     newOb =vrep.simxCopyPasteObjects(self._clientID, handles, vrep.simx_opmode_blocking)
    #     self

    def disconnect(self):
        vrep.unwrap_vrep(
            vrep.simxFinish(self._clientID)
        )

    def initialize_handles(self):
        
        for i in range(self.max_pop_size):
            self.population[i]._RightMotor = self._vrep_get_object_handle('Right_Motor#{}{}'.format(i, self._value_number), vrep.simx_opmode_blocking)
            self.population[i]._LeftMotor = self._vrep_get_object_handle('Left_Motor#{}{}'.format(i, self._value_number), vrep.simx_opmode_blocking)
            self.population[i]._Robobo = self._vrep_get_object_handle('Robobo#{}{}'.format(i, self._value_number), vrep.simx_opmode_blocking)

            # self.population[i]._IrBackC = self._vrep_get_object_handle('Ir_Back_C#{}{}'.format(i, self._value_number), vrep.simx_opmode_blocking)
            # self.population[i]._IrFrontC = self._vrep_get_object_handle('Ir_Front_C#{}{}'.format(i, self._value_number), vrep.simx_opmode_blocking)
            # self.population[i]._IrFrontLL = self._vrep_get_object_handle('Ir_Front_LL#{}{}'.format(i, self._value_number), vrep.simx_opmode_blocking)
            # self.population[i]._IrFrontRR = self._vrep_get_object_handle('Ir_Front_RR#{}{}'.format(i, self._value_number), vrep.simx_opmode_blocking)
            # self.population[i]._IrBackL = self._vrep_get_object_handle('Ir_Back_L#{}{}'.format(i, self._value_number), vrep.simx_opmode_blocking)
            # self.population[i]._IrBackLFloor = self._vrep_get_object_handle('Ir_Back_L_Floor#{}{}'.format(i, self._value_number),
            #                                                     vrep.simx_opmode_blocking)
            # self.population[i]._IrBackR = self._vrep_get_object_handle('Ir_Back_R#{}{}'.format(i, self._value_number), vrep.simx_opmode_blocking)
            # self.population[i]._IrBackRFloor = self._vrep_get_object_handle('Ir_Back_R_Floor#{}{}'.format(i, self._value_number),
            #                                                     vrep.simx_opmode_blocking)
            # self.population[i]._IrFrontL = self._vrep_get_object_handle('Ir_Front_L#{}{}'.format(i, self._value_number), vrep.simx_opmode_blocking)
            # self.population[i]._IrFrontLFloor = self._vrep_get_object_handle('Ir_Front_L_Floor#{}{}'.format(i, self._value_number),
            #                                                     vrep.simx_opmode_blocking)
            # self.population[i]._IrFrontR = self._vrep_get_object_handle('Ir_Front_R#{}{}'.format(i, self._value_number), vrep.simx_opmode_blocking)
            # self.population[i]._IrFrontRFloor = self._vrep_get_object_handle('Ir_Front_R_Floor#{}{}'.format(i, self._value_number),
                                                                # vrep.simx_opmode_blocking)
            self.population[i]._TiltMotor = self._vrep_get_object_handle('Tilt_Motor#{}{}'.format(i, self._value_number), vrep.simx_opmode_blocking)
            self.population[i]._PanMotor = self._vrep_get_object_handle('Pan_Motor#{}{}'.format(i, self._value_number), vrep.simx_opmode_blocking)
            self.population[i]._FrontalCamera = self._vrep_get_object_handle('Frontal_Camera#{}{}'.format(i, self._value_number), vrep.simx_opmode_blocking)

            # self.population[i]._Phone = self._vrep_get_object_handle('Tilt_Smartphone_Visual#{}{}'.format(i, self._value_number), vrep.simx_opmode_blocking)

            # try:
            #     self.population[i]._base = self._vrep_get_object_handle("Base_Proximity_sensor#{}".format(i), vrep.simx_opmode_blocking)
            # except vrep.VrepApiError as _e:
            #     self.population[i]._base = None

            # read a first value in streaming mode
            # self._vrep_read_proximity_sensor_ignore_error(self.population[i]._IrFrontC)
            # self._vrep_read_proximity_sensor_ignore_error(self.population[i]._IrBackC)
            # self._vrep_read_proximity_sensor_ignore_error(self.population[i]._IrFrontLL)
            # self._vrep_read_proximity_sensor_ignore_error(self.population[i]._IrFrontRR)
            # self._vrep_read_proximity_sensor_ignore_error(self.population[i]._IrBackL)
            # self._vrep_read_proximity_sensor_ignore_error(self.population[i]._IrBackLFloor)
            # self._vrep_read_proximity_sensor_ignore_error(self.population[i]._IrBackR)
            # self._vrep_read_proximity_sensor_ignore_error(self.population[i]._IrBackRFloor)
            # self._vrep_read_proximity_sensor_ignore_error(self.population[i]._IrFrontR)
            # self._vrep_read_proximity_sensor_ignore_error(self.population[i]._IrFrontRFloor)
            # self._vrep_read_proximity_sensor_ignore_error(self.population[i]._IrFrontL)
            # self._vrep_read_proximity_sensor_ignore_error(self.population[i]._IrFrontLFloor)
            # if self.population[i]._base is not None:
            #     self._vrep_read_proximity_sensor_ignore_error(self.population[i]._base)

            # setup join positions
            vrep.simxGetJointPosition(self._clientID, self.population[i]._RightMotor, vrep.simx_opmode_buffer)
            vrep.simxGetJointPosition(self._clientID, self.population[i]._LeftMotor, vrep.simx_opmode_buffer)
            vrep.simxGetObjectPosition(self._clientID, self.population[i]._Robobo, -1, vrep.simx_opmode_buffer)
            # if self.population[i]._base is not None:
            #     vrep.simxGetObjectPosition(self._clientID, self.population[i]._base, -1, vrep.simx_opmode_buffer)

            # read a first value in buffer mode
            self._vrep_get_vision_sensor_image_ignore_error(self.population[i]._FrontalCamera, vrep.simx_opmode_streaming)

        self.wait_for_ping()

    def sleep(self, seconds):
        duration = seconds * 1000
        start_time = self.get_sim_time()
        while self.get_sim_time() - start_time < duration:
            pass

    def wait_for_ping(self, timeout_seconds=120.0):
        startTime = time.time()
        while time.time() - startTime < timeout_seconds:
            try:
                self._vrep_get_ping_time()
                # print("check success")
                return True
            except vrep.VrepApiError as _e:
                # print("check failed")
                time.sleep(0.05)
        
        print("{} seconds passed with ping not coming online, you may expericence problems with the connection".format(timeout_seconds))
        return False

    def _vrep_get_ping_time(self):
        return vrep.unwrap_vrep(vrep.simxGetPingTime(self._clientID))

    def _vrep_get_object_handle(self, name, opmode):
        return vrep.unwrap_vrep(vrep.simxGetObjectHandle(self._clientID, name, opmode))

    def _vrep_read_proximity_sensor(self, handle, opmode=vrep.simx_opmode_streaming):
        return vrep.unwrap_vrep(vrep.simxReadProximitySensor(self._clientID, handle, opmode))

    def _vrep_read_proximity_sensor_ignore_error(self, handle, opmode=vrep.simx_opmode_streaming):
        try:
            self._vrep_read_proximity_sensor(handle, opmode)
        except vrep.error.VrepApiError as error:
            if error.ret_code is not vrep.simx_return_novalue_flag:
                raise
        
    def _vrep_get_vision_sensor_image(self, camera_handle, opmode=vrep.simx_opmode_buffer, a=0):
        return vrep.unwrap_vrep(vrep.simxGetVisionSensorImage(self._clientID, camera_handle, a, opmode))

    def _vrep_get_vision_sensor_image_ignore_error(self, camera_handle, opmode=vrep.simx_opmode_buffer, a=0):
        try:
            self._vrep_get_vision_sensor_image(camera_handle, opmode, a)
        except vrep.error.VrepApiError as error:
            if error.ret_code is not vrep.simx_return_novalue_flag:
                raise

    def _vrep_set_joint_target_velocity(self, handle, speed, opmode):
        return vrep.unwrap_vrep(vrep.simxSetJointTargetVelocity(self._clientID, handle, speed, opmode))

    def _vrep_set_joint_target_position(self, handle, position, opmode=vrep.simx_opmode_oneshot):
        return vrep.unwrap_vrep(vrep.simxSetJointTargetPosition(self._clientID, handle, position, opmode))
    
    def _vrep_set_object_position(self, handle, position, opmode=vrep.simx_opmode_blocking):
        print("yes")
        vrep.simxPauseCommunication(self._clientID, True)
        vrep.unwrap_vrep(vrep.simxSetObjectPosition(self._clientID, self.population[0]._Robobo, -1, [1,1,0.001], vrep.simx_opmode_oneshot))
        vrep.simxPauseCommunication(self._clientID, False)

    def spin(self):
        raise NotImplementedError("Not implemeted yet")

    def set_emotion(self, emotion):
        print("ROBOT EMOTION: {}".format(emotion))

    def move(self, indiv, left, right, millis=500):
        normalizer = 10.0
        left = left/normalizer
        right = right/normalizer

        self._vrep_set_joint_target_velocity(self.population[indiv]._LeftMotor, left, vrep.simx_opmode_oneshot)
        self._vrep_set_joint_target_velocity(self.population[indiv]._RightMotor, right, vrep.simx_opmode_oneshot)
        # self.wait_for_ping() #get_sim_time is already waiting for ping

        # duration = millis #/ 1000.0
        # startTime = time.time()
        # while time.time() - startTime < duration:
        #     # rightMotorAngPos = vrep.unwrap_vrep(vrep.simxGetJointPosition(self._clientID, self._RightMotor, vrep.simx_opmode_blocking))
        #     # leftMotorAngPos  = vrep.unwrap_vrep(vrep.simxGetJointPosition(self._clientID, self._LeftMotor, vrep.simx_opmode_blocking))
        #     # RoboAbsPos       = vrep.unwrap_vrep(vrep.simxGetObjectPosition(self._clientID, self._Robobo, -1, vrep.simx_opmode_blocking))
        #     time.sleep(0.005)
        # print("sleeping for {}".format(duration))

        # busy waiting
        # start_time = self.get_sim_time()
        # while self.get_sim_time() - start_time < duration:
        #     time.sleep(0.1)
        #     pass
        
        # # Stop to move the wheels motor. Angular velocity.
        # stopRightVelocity = stopLeftVelocity = 0
        # self._vrep_set_joint_target_velocity(self.population[indiv]._LeftMotor, stopLeftVelocity,
        #                                           vrep.simx_opmode_oneshot)
        # self._vrep_set_joint_target_velocity(self.population[indiv]._RightMotor, stopRightVelocity,
        #                                           vrep.simx_opmode_oneshot)
        self.wait_for_ping()

    def set_position(self, indiv, x, y, z):
        
        print(self._vrep_set_object_position(self.population[indiv]._RightMotor, [x, y, z],  vrep.simx_opmode_oneshot))

    def talk(self, message):
        print("ROBOT SAYS: {}".format(message))

    def set_led(self, selector, color):
        raise NotImplementedError("Not implemeted yet")
    
    def read_irs(self, indiv):
        """
        returns sensor readings: [backR, backC, backL, frontRR, frontR, frontC, frontL, frontLL]
        """      
        detectionStateIrFrontC, detectedPointIrFrontC, detectedObjectHandleIrFrontC, \
        detectedSurfaceNormalVectorIrFrontC = self._vrep_read_proximity_sensor(
            self.population[indiv]._IrFrontC, vrep.simx_opmode_buffer)
        detectionStateIrBackC, detectedPointIrIrBackC, detectedObjectHandleIrBackC, \
        detectedSurfaceNormalVectorIrBackC = self._vrep_read_proximity_sensor(
            self.population[indiv]._IrBackC, vrep.simx_opmode_buffer)
        detectionStateIrFrontLL, detectedPointIrFrontLL, detectedObjectHandleIrFrontLL, \
        detectedSurfaceNormalVectorIrFrontLL = self._vrep_read_proximity_sensor(
            self.population[indiv]._IrFrontLL, vrep.simx_opmode_buffer)
        detectionStateIrFrontRR, detectedPointIrFrontRR, detectedObjectHandleIrFrontRR, \
        detectedSurfaceNormalVectorIrFrontRR = self._vrep_read_proximity_sensor(
            self.population[indiv]._IrFrontRR, vrep.simx_opmode_buffer)
        detectionStateIrBackL, detectedPointIrBackL, detectedObjectHandleIrBackL, \
        detectedSurfaceNormalVectorIrBackL = self._vrep_read_proximity_sensor(
            self.population[indiv]._IrBackL, vrep.simx_opmode_buffer)

        detectionStateIrBackR, detectedPointIrBackR, detectedObjectHandleIrBackR, \
        detectedSurfaceNormalVectorIrBackR = self._vrep_read_proximity_sensor(
            self.population[indiv]._IrBackR, vrep.simx_opmode_buffer)

        detectionStateIrFrontR, detectedPointIrFrontR, detectedObjectHandleIrFrontR, \
        detectedSurfaceNormalVectorIrFrontR = self._vrep_read_proximity_sensor(
            self.population[indiv]._IrFrontR, vrep.simx_opmode_buffer)

        detectionStateIrFrontL, detectedPointIrFrontL, detectedObjectHandleIrFrontL, \
        detectedSurfaceNormalVectorIrFrontL = self._vrep_read_proximity_sensor(
            self.population[indiv]._IrFrontL, vrep.simx_opmode_buffer)

        vect = [np.sqrt(detectedPointIrBackR[0]   ** 2 + detectedPointIrBackR[1]   ** 2 + detectedPointIrBackR[2]   ** 2)
                if detectionStateIrBackR   else False,
                np.sqrt(detectedPointIrIrBackC[0] ** 2 + detectedPointIrIrBackC[1] ** 2 + detectedPointIrIrBackC[2] ** 2)
                if detectionStateIrBackC   else False,
                np.sqrt(detectedPointIrBackL[0] ** 2   + detectedPointIrBackL[1]   ** 2 + detectedPointIrBackL[2]   ** 2)
                if detectionStateIrBackL   else False,
                np.sqrt(detectedPointIrFrontRR[0] ** 2 + detectedPointIrFrontRR[1] ** 2 + detectedPointIrFrontRR[2] ** 2)
                if detectionStateIrFrontRR else False,
                np.sqrt(detectedPointIrFrontR[0] ** 2  + detectedPointIrFrontR[1]  ** 2 + detectedPointIrFrontR[2]  ** 2)
                if detectionStateIrFrontR  else False,
                np.sqrt(detectedPointIrFrontC[0] ** 2  + detectedPointIrFrontC[1]  ** 2 + detectedPointIrFrontC[2]  ** 2)
                if detectionStateIrFrontC   else False,
                np.sqrt(detectedPointIrFrontL[0] ** 2  + detectedPointIrFrontL[1]  ** 2 + detectedPointIrFrontL[2]  ** 2)
                if detectionStateIrFrontL  else False,
                np.sqrt(detectedPointIrFrontLL[0] ** 2 + detectedPointIrFrontLL[1] ** 2 + detectedPointIrFrontLL[2] ** 2)
                if detectionStateIrFrontLL else False]

        # old_min = 0
        # old_max = 0.20
        # new_min = 18000
        # new_max = 0
        # return [(((old_value - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min for old_value in vect]
        return vect

    def get_image_front(self,indiv):
        return self._get_image(self.population[indiv]._FrontalCamera)

    def _get_image(self, camera):
        self.wait_for_ping()

        # get image
        resolution, image = self._vrep_get_vision_sensor_image(camera)

        # reshape image
        image = image[::-1]
        im_cv2 = np.array(image, dtype=np.uint8)
        im_cv2.resize([resolution[0], resolution[1], 3])
        im_cv2 = cv2.flip(im_cv2, 1)

        return im_cv2

    def set_phone_pan(self, pan_position, pan_speed):
        """
        Command the robot to move the smartphone holder in the horizontal (pan) axis.

        Arguments

        pan_position: Angle to position the pan at.
        pan_speed: Movement speed for the pan mechanism.
        """
        # tilt_position = np.pi / 4.0
        self._vrep_set_joint_target_position(self._PanMotor, pan_position)
        self.wait_for_ping()

    def set_phone_tilt(self, tilt_position, tilt_speed):
        """
        Command the robot to move the smartphone holder in the vertical (tilt) axis.

        Arguments

        tilt_position: Angle to position the tilt at.
        tilt_speed: Movement speed for the tilt mechanism.
        """
        # tilt_position = np.pi / 4.0
        self._vrep_set_joint_target_position(self._TiltMotor, tilt_position)
        self.wait_for_ping()
    
    def pause_simulation(self):
        vrep.unwrap_vrep(
            vrep.simxPauseSimulation(self._clientID, vrep.simx_opmode_blocking)
        )
    
    def play_simulation(self):
        vrep.unwrap_vrep(
            vrep.simxStartSimulation(self._clientID, vrep.simx_opmode_blocking)
        )
        self.wait_for_ping()

    def stop_world(self):
        vrep.unwrap_vrep(
            vrep.simxStopSimulation(self._clientID, vrep.simx_opmode_blocking)
        )
        self.wait_for_ping()

    def check_simulation_state(self):
        self.wait_for_ping()
        return vrep.unwrap_vrep(
            vrep.simxGetInMessageInfo(self._clientID, vrep.simx_headeroffset_server_state),
            ignore_novalue_error=True
        )

    def is_simulation_stopped(self):
        return not self.is_simulation_running()

    def is_simulation_running(self):
        info = self.check_simulation_state()
        return info & 1

    def wait_for_stop(self):
        """
        This function busy waits until the simulation is stopped
        """
        while self.is_simulation_running():
            pass

    def get_sim_time(self):
        """
        Gets the simulation time. Returns zero if the simulation is stopped.
        :return: simulation time in milliseconds.
        """
        self.wait_for_ping()
        return vrep.simxGetLastCmdTime(self._clientID)

    def position(self, indiv):
        return vrep.simxGetObjectPosition(self._clientID, self.population[indiv]._Robobo, -1, vrep.simx_opmode_blocking)
    
    def getPositions(self):
        return vrep.simxGetObjectGroupData(self._clientID, self.object_camera_type, 3, vrep.simx_opmode_blocking)
        

    def collected_food(self):
        ints, floats, strings, buffer = vrep.unwrap_vrep(
            vrep.simxCallScriptFunction(self._clientID, "Food", vrep.sim_scripttype_childscript, "remote_get_collected_food",
                                        [],[],[],bytearray(),vrep.simx_opmode_blocking)
        )
        return ints[0]

    def change_colour(self, indiv, colour):

        # handle = self.population[indiv]._Phone
        # print(handle)
        return vrep.simxCallScriptFunction(self._clientID, "Smartphone_Respondable#" + str(indiv), vrep.sim_scripttype_childscript, "changeColor",
                                        [indiv, colour],[],[],bytearray(), vrep.simx_opmode_blocking)
        

    def base_position(self):
        return vrep.unwrap_vrep(
            vrep.simxGetObjectPosition(self._clientID, self._base, -1, vrep.simx_opmode_blocking)
        )
    
    def base_detects_food(self):
        detection, _detection_point, _detected_handle, _detected_normal \
            = self._vrep_read_proximity_sensor(self._base, vrep.simx_opmode_buffer)
        return bool(detection)
    
    def load_image(self, indiv):

        # handle = self.population[indiv]._Phone
        # print(handle)
        return vrep.simxCallScriptFunction(self._clientID, "Frontal_Camera#" + str(indiv), vrep.sim_scripttype_childscript, "loadImage",
                                        [indiv],[],[],bytearray(), vrep.simx_opmode_blocking)
    
    def useMask(self, mask, gender):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # blobs = len(cnts)
        output = []
        # cv2.imwrite("test_pictures{}.png".format(gender), mask)
        for x in cnts:
            area = cv2.contourArea(x)
            if area > 1: # If there is a single pixel the code below gives a 'divide by 0' error
                M = cv2.moments(x)
                # cX = int(M["m10"] / M["m00"])
                # cY = int(M["m01"] / M["m00"])
                direction = (int(M["m10"] / M["m00"])/64)-1 # transformed the y coord to a range of -1 to 1
                output.append([area, direction, gender])

        return output

    # Following code gets features from camera
    def useCamera(self, indiv, gender):
        # IMPORTANT! `image` returned by the simulator is BGR, not RGB
        self.load_image(indiv)
        image = self.get_image_front(indiv)
        # cv2.imwrite("test_pictures.png", image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value= 0)

        greenMask = cv2.inRange(image, (45,70, 70), (85, 255, 255))
        redMask = cv2.inRange(image, (0,70, 70), (45, 255, 255))

        return self.useMask(redMask, gender != 0) + self.useMask(greenMask, gender != 1)

    # Initializes a neural network brain which has a set architecture but random weights
    def createBrain(self):
        return MLPbrain.randomBrain(size_inputs=self.mentalStack*self.dataPointsPer, size_layer1=6, size_out=2)
    
    def prepInputs(self, inputs):
        inputs = np.array(inputs)
        if inputs.size > 0:
            inputs = inputs[inputs[:,0].argsort()[::-1]].flatten()
            if inputs.size > self.mentalStack*self.dataPointsPer:
                inputs = inputs[0:self.mentalStack*self.dataPointsPer]
            elif inputs.size < self.mentalStack*self.dataPointsPer:
                inputs = np.concatenate([inputs, np.zeros([self.mentalStack*self.dataPointsPer-inputs.size])])
        else: 
            inputs = np.zeros(self.mentalStack*self.dataPointsPer)
        return np.concatenate([inputs])

    # Returns the ID of the robot that had just been born and moves them to the alive set
    def birth(self):
        # indiv = self.dead.pop()
        indiv = random.choice(list(self.dead))
        self.dead.remove(indiv)
        self.alive.add(indiv)
        return indiv
    
    # Moves the ID of the robot given to the dead set
    def death(self, indiv):
        self.alive.remove(indiv)
        self.dead.add(indiv)
        print("death of {}".format(indiv))


    
