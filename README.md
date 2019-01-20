# AI_Robotics
A Line follower robot using Micropython and Neural Networks with MicroMLP library 

To build you need 
ESP32 (Tested on ESP-WROOM-32) - 1 nos
FC-51 IR Sensor - 2 nos
Continuous rotation micro 9g servo - 2 nos
Robot chasis - 1 nos


Circuit Diagram

![alt text](https://github.com/AerostatDrones/AI_Robotics/blob/master/Line%20Follower%20Circuit.PNG)

First we train our Neural Network with example data. (Input - IR sensor data, Output - Servo motor) 

Once Trainig is over it generates weights for neurons, which is loaded for testing the robot.
