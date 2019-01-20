from microMLP import MicroMLP
from machine import Pin, PWM, reset
from servo import Servo
import time

mlp = MicroMLP.LoadFromFile('line_follower.json')

ir1 = Pin(15, Pin.IN, Pin.PULL_UP)
ir2 = Pin(4, Pin.IN, Pin.PULL_UP)

mR = Pin(12)
mR = Servo(mR)
mL = Pin(14)
mL = Servo(mL)

mR.angle = 100
mL.angle = 100

while 1:
  try:  
    ip1 = mlp.NNValue(0,1,ir1.value())
    ip2 = mlp.NNValue(0,1,ir2.value())
    #print("Line detected by Sensor 1 and Sensor 2 : ", mlp.Predict([ip1 , ip2])[0].AsPercent, mlp.Predict([ip1, ip2])[1].AsPercent)
    mR.write_angle(int(mlp.Predict([ip1 , ip2])[0].AsPercent))
    mL.write_angle(int(mlp.Predict([ip1 , ip2])[1].AsPercent))
    time.sleep_ms(100)
  except KeyboardInterrupt:
    mR.write_angle(50)
    mL.write_angle(50)
    time.sleep_ms(100)
    reset()
    


