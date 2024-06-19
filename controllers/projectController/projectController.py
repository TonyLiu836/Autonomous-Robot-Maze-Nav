"""projectController controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Supervisor, Robot, Motor, DistanceSensor#, Gyro
from controller import Camera, CameraRecognitionObject
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import scipy.io
import scipy.stats

# create the Robot instance.
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
#robot = Supervisor()
#print("here!")
camera = robot.getDevice('camera')
camera.enable(1)

lidar = robot.getDevice('lidar')
lidar.enable(timestep)
lidar.enablePointCloud()

accel = robot.getDevice('accelerometer')
accel.enable(1)

if camera.hasRecognition():
    camera.recognitionEnable(1)
    camera.enableRecognitionSegmentation()
else:
    print("Your camera does not have recognition")

robotNode = robot.getFromDef("e-puck")

x,y = robotNode.getPosition()[0:2]     #used to get initial robot position & orientation 
ori_rotMat = robotNode.getOrientation()
theta = math.atan2(ori_rotMat[3], ori_rotMat[0])

wheelRadius = 0.0205
axleLength = 0.0568 # Data from Webots website seems wrong. The real axle length should be about 56-57mm
updateFreq = 200 # update every 200 timesteps

leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))

# init variables
dt = timestep / 2500.0
x_hat_t = np.array([x,y,theta])
Sigma_x_t = np.zeros((3,3))
Sigma_x_t[0,0], Sigma_x_t[1,1], Sigma_x_t[2,2] = 0.01, 0.01, np.pi/90
Sigma_n = np.zeros((2,2))
std_n_v = 0.01
std_n_omega = np.pi/60
Sigma_n[0,0] = std_n_v * std_n_v
Sigma_n[1,1] = std_n_omega * std_n_omega

counter = 0
timer = 0

#helper functions
def omegaToWheelSpeeds(omega, v):
    wd = omega * axleLength * 0.5
    return v - wd, v + wd

def rotMat(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])

def wallFollow(frontLeft, front, frontRight):
    thresh2 = 0.45
    #no obstacles
    if (front > thresh2 and frontRight > thresh2 and frontLeft > thresh2):
        v = 0.03                                #EKFpropagate doesnt behave properly when v and omega both > 0 for some reason.
        omega = -12*v

    #left
    elif (front > thresh2 and frontLeft < thresh2 and  frontRight > thresh2):
        v = 0.03
        omega = -5*v

    #front
    elif (front < thresh2 and frontLeft > thresh2 and  frontRight > thresh2):
        omega = 1
        v =  0
            
    #right
    elif (front > thresh2 and frontLeft > thresh2 and  (frontRight < thresh2 or right < thresh2)):
        if frontRight < 0.15:
            v = 0
            omega = 1
        else:
            v = 0.06
            omega = 0
        
    #front, frontLeft
    elif (front < thresh2 and frontLeft < thresh2 and  frontRight > thresh2):
        omega = 1
        v = 0
            
    #front, frontRight
    elif (front < thresh2 and frontLeft > thresh2 and  frontRight < thresh2):
        omega = 1
        v = 0

    #frontLeft, front, frontRight 
    elif (front < thresh2 and frontLeft < thresh2 and  frontRight < thresh2):
        omega = 1
        v = 0
        
    #frontLeft, frontRight 
    elif (front > thresh2 and frontLeft < thresh2 and  frontRight < thresh2):
        v = 0.06
        omega = -v
    
    return v, omega 

def EKFPropagate(x_hat_t, # robot position and orientation
                 Sigma_x_t, # estimation uncertainty
                 u, # control signals
                 Sigma_n, # uncertainty in control signals
                 dt # timestep
    ):
    v = u[0]
    theta = x_hat_t[-1]
    Phi = np.array([[1, 0, -dt*v*math.sin(theta)],
                    [0, 1, dt*v*math.cos(theta)],
                    [0, 0, 1]])
     
    
    B = np.array([[dt*math.cos(theta), 0],
                  [dt*math.sin(theta), 0],
                  [0, dt]])
    
    x_hat_t = x_hat_t + np.transpose(B @ np.transpose([u]))[0]
    #x_hat_t[-1] = fixHeading(x_hat_t[-1])
    Sigma_x_t = Phi @ Sigma_x_t @ np.transpose(Phi) + B @ Sigma_n @ np.transpose(B)

    return x_hat_t, Sigma_x_t

def EKFRelPosUpdate(x_hat_t, # robot position and orientation
                    Sigma_x_t, # estimation uncertainty
                    z, # measurements
                    Sigma_m, # measurements' uncertainty
                    G_p_L, # landmarks' global positions
                    dt # timestep
                   ):
    # TODO: Update the robot state estimation and variance based on the received measurement

    Sigma_m = np.asarray(Sigma_m)
    z = np.transpose([np.array(z)])
    rotmat = rotMat(x_hat_t[-1])
    J = np.array([[0, -1], [1, 0]])

    z_hat = np.transpose(rotmat)@np.transpose([np.array(G_p_L - x_hat_t)[0:2]])
    r = z - z_hat      #2x1
    
    H = np.hstack((-1*np.transpose(rotmat), -1*np.transpose(rotmat)@J@np.transpose([np.array(G_p_L - x_hat_t)[0:2]])))
    
    R = Sigma_m
    S = H @ Sigma_x_t @ np.transpose(H) + R
    K = Sigma_x_t @ np.transpose(H) @ np.linalg.inv(S)

    x_hat_t += np.transpose(K@r)[0]

    Sigma_x_t -= Sigma_x_t @ np.transpose(H) @ np.linalg.inv(S) @ H @ Sigma_x_t
    #x_hat_t[-1] = fixHeading(x_hat_t[-1])
    return x_hat_t, Sigma_x_t
    
numObjsRecog = 0
firstObj = [0,0,0]

robotState = None               #2 = move to goal; 1 = follow wall/obstacle
robotTask = 1                #1 = adjust bearing; 2= move fwd; 3 = at goal

hitPt = np.array([0.0,0.0])              #coordinates where obstacle is 1st encountered
leavePt = np.array([0.0,0.0])             #coordinate where robot stops following obstacle
goalPos = np.array([(-1.1,0), (-2.88, 0.0)])    #was going to use intermediate goals to guide robot but was able to tune parameters to not need them
goal_ind = 1
robotState = "move to goal"
mLineFound = False

counter = 0
updateFreq = 20
while robot.step(timestep) != -1:    
    
    recObjs = camera.getRecognitionObjects()
    recObjsNum = camera.getRecognitionNumberOfObjects()
    z_pos = np.zeros((recObjsNum, 2))

    lidarInfo = lidar.getRangeImage()         #lidar readings
    # print("robot state = ", robotState, x_hat_t)
    #print("robot task = ", robotTask)
    #print("goal ind = ", goal_ind)
    
    front = np.array(lidarInfo[256])
    frontLeft = np.array(lidarInfo[128])
    frontRight = np.array(lidarInfo[384])
    left = np.array(lidarInfo[0:65])       #68 ~ 23 degrees
    right = np.array(lidarInfo[447:-1])       #157 ~ 180 degrees

    left = lidarInfo[0]                       #180 degrees - left side of robot 
    right = lidarInfo[511]                      #0 degrees - right side of robot
    x_hat_t_estimates = []
    
    
    #if recObjsNum > 0:
    if counter % updateFreq == 0:
        for i in range(0, recObjsNum):
            landmark = robot.getFromId(recObjs[i].get_id())
            G_p_L = landmark.getPosition()
            rel_lm_trans = landmark.getPose(robotNode)

            std_m = 0.05
            Sigma_m = [[std_m*std_m, 0], [0,std_m*std_m]]
            z_pos[i] = [rel_lm_trans[3]+np.random.normal(0,std_m), rel_lm_trans[7]+np.random.normal(0,std_m)]
            
            x_hat_t, Sigma_x_t = EKFRelPosUpdate(x_hat_t, Sigma_x_t, z_pos[i], Sigma_m, G_p_L, dt)    
     
    counter += 1
    if mLineFound == False:
        robotState = 2

        mLine_Robotx = x_hat_t[0]
        mLine_Goalx = goalPos[goal_ind][0]
        mLine_Roboty = x_hat_t[1]
        mLine_Goaly = goalPos[goal_ind][1]

        mLine_slope = (mLine_Goaly - mLine_Roboty) / (mLine_Goalx - mLine_Robotx)
        mLine_intercept = mLine_Goaly - (mLine_slope * mLine_Goalx)
        #print("slope = ", mLine_slope, "  intercept = ", mLine_intercept)
        
        mLineFound = True
        
    else:
        if robotState == 1:     #follow wall and check if curr robot position is near m-line
            #get point on mLine that corresponds to current x pos
            mLine_currPosx = x_hat_t[0]
            mLine_currPosy = mLine_slope * mLine_currPosx + mLine_intercept
            
            mLine_currPt = np.array([mLine_currPosx, mLine_currPosy])

            #dist from currPt to mLine
            dist2mLine_currPt = np.linalg.norm(mLine_currPt - x_hat_t[0:2])
            dist2hitPt = np.linalg.norm(x_hat_t[0:2] - hitPt)

            if dist2mLine_currPt < 0.15 and dist2hitPt > 0.5:
                #decide if robot should keep following wall by comparing dist between hitPt and leavePt
                leavePt[0], leavePt[1] = x_hat_t[0], x_hat_t[1]
                
                if (np.linalg.norm(leavePt - hitPt)) > 0.3:
                    dist2goal_leavePt = np.linalg.norm(goalPos[goal_ind] - leavePt)
                    dist_resid_hitPt_leavePt = dist2goal_hitPt - dist2goal_leavePt
                    
                    #if the difference between hitPt2goal and leavePt2goal is large then hitPt is closer to goal
                    #therefore, should stop following wall 
                    if dist_resid_hitPt_leavePt > 0.1:
                        robotState = 2
                    
                    continue
            
            #robot obstacle avoidance/wall-following
            v, omega = wallFollow(frontLeft, front, frontRight)
            
            u = np.array([v, omega])
            left_v, right_v = omegaToWheelSpeeds(omega+np.random.normal(0,std_n_omega), v+np.random.normal(0,std_n_v))
            x_hat_t, Sigma_x_t = EKFPropagate(x_hat_t, Sigma_x_t, u, Sigma_n, dt)
            leftMotor.setVelocity(left_v/wheelRadius)
            rightMotor.setVelocity(right_v/wheelRadius)
        

        elif robotState == 2:       #move towards goal
            
            thresh = 0.15
            if (front < thresh or frontLeft < thresh or frontRight < thresh):  #obstacle detected in front of robot
                robotState = 1
                
                hitPt[0], hitPt[1] = x_hat_t[0], x_hat_t[1]
                
                dist2goal_hitPt = np.linalg.norm(goalPos[goal_ind] - hitPt)
                
                #turn left
                omega = 1
                v = 0
                
                u = np.array([v, omega])
                left_v, right_v = omegaToWheelSpeeds(omega+np.random.normal(0,std_n_omega), v+np.random.normal(0,std_n_v))
                x_hat_t, Sigma_x_t = EKFPropagate(x_hat_t, Sigma_x_t, u, Sigma_n, dt)
                    
                leftMotor.setVelocity(left_v/wheelRadius)
                rightMotor.setVelocity(right_v/wheelRadius)
                continue 
            
            if robotTask == 1:          #adjust bearing if no obstacle in front of robot
                bearing2goal = math.atan2(goalPos[goal_ind][1] - x_hat_t[1], goalPos[goal_ind][0] - x_hat_t[0])
                bearing_resid = bearing2goal - x_hat_t[-1]
                
                if bearing_resid > 0.15:             #constant found through trial and error
                    v = 0
                    omega = 0.5
                    
                    u = np.array([v, omega])
                    left_v, right_v = omegaToWheelSpeeds(omega+np.random.normal(0,std_n_omega), v+np.random.normal(0,std_n_v))
                    x_hat_t, Sigma_x_t = EKFPropagate(x_hat_t, Sigma_x_t, u, Sigma_n, dt)
                    leftMotor.setVelocity(left_v/wheelRadius)
                    rightMotor.setVelocity(right_v/wheelRadius)
                
                elif bearing_resid < -0.15:
                    v = 0
                    omega = -0.5
                    u = np.array([v, omega])
                    
                    left_v, right_v = omegaToWheelSpeeds(omega+np.random.normal(0,std_n_omega), v+np.random.normal(0,std_n_v))
                    x_hat_t, Sigma_x_t = EKFPropagate(x_hat_t, Sigma_x_t, u, Sigma_n, dt)
                    leftMotor.setVelocity(left_v/wheelRadius)
                    rightMotor.setVelocity(right_v/wheelRadius)
                
                else:
                    robotTask = 2           #robot at desired bearing, change task to go fwd

            elif robotTask == 2:        #move robot fwd if robot has correct bearing 
                pos_resid = np.linalg.norm(goalPos[goal_ind] - x_hat_t[0:2])
                
                if pos_resid > 0.2:    
                    omega = 0
                    v = 0.06

                    bearing2goal = math.atan2(goalPos[goal_ind][1] - x_hat_t[1], goalPos[goal_ind][0] - x_hat_t[0])

                    bearing_resid = bearing2goal - x_hat_t[-1]
                    
                    u = np.array([v, omega])
                    left_v, right_v = omegaToWheelSpeeds(omega+np.random.normal(0,std_n_omega), v+np.random.normal(0,std_n_v))
                    x_hat_t, Sigma_x_t = EKFPropagate(x_hat_t, Sigma_x_t, u, Sigma_n, dt)
                    leftMotor.setVelocity(left_v/wheelRadius)
                    rightMotor.setVelocity(right_v/wheelRadius)
                    
                    if np.fabs(bearing_resid) > 2.0*(math.pi/180):
                        robotTask = 1                   #change task to adjust bearing

                else:   #reached goal 
                    print("goal reached!")
                    v = 0
                    omega = 0
                    u = np.array([v, omega])
                    left_v, right_v = omegaToWheelSpeeds(omega+np.random.normal(0,std_n_omega), v+np.random.normal(0,std_n_v))
                    x_hat_t, Sigma_x_t = EKFPropagate(x_hat_t, Sigma_x_t, u, Sigma_n, dt)
                    leftMotor.setVelocity(left_v/wheelRadius)
                    rightMotor.setVelocity(right_v/wheelRadius)
     
    pass
