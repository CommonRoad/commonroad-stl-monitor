*************************************************************
*                                                           *
*    This file contains brief notes about the scenarios:    *
*                                                           *
*************************************************************

-Note that the planning problem in the below scenarios is a dummy one, because we don't care about it in the current COP scope.
-For the notes related to the turnsignal scenarios (21-27), the duration of the lane change is configured as 5 steps. Also, saying in the description of the scenario that the turn signal is 'ON' n steps before the lane change means that the turn signal is 'ON' n steps before the BEGINNING of the lane change. 

Scenarios information:
**********************

1_A9_STOPMAINCARRIAGE.xml
*A9 network
*1 vehicle depspeed 2, maxspeed 10, in the right lane
=======================================================
2_A9_STOPMAINCARRIAGE.xml
*A9 network
*1 vehicle depspeed 0.1, maxspeed 0.1, in the right lane
=======================================================
3_A9_STOPMAINCARRIAGE.xml
*A9 network
*4 Vehicles
  -depspeed 4 maxspeed 10 stop on ordinary lane for 1 second
  -depspeed 2 maxspeed 15
  -depspeed 1 maxspeed 5
  -depspeed 3 maxspeed 8
=======================================================
4_A9_STOPMAINCARRIAGE.xml
*A9 network
*1 vehicle depspeed 2, maxspeed 10, stop 2 seconds the right lane (main carriage)
=======================================================
5_A9_STOPMAINCARRIAGE.xml
*A9 network
*1 vehicle depspeed 3, maxspeed 15, stop twice for 1 second in the right lane (main carriage)
=======================================================
6_A9_STOPMAINCARRIAGE.xml
*A9 network
*4 Vehicles
  -depspeed 4 maxspeed 10 stop 3 seconds in lane 0 (main carriage) 
  -depspeed 2 maxspeed 15
  -depspeed 1 maxspeed 5 stop 2 seconds in lane 2 (main carriage)
  -depspeed 5 maxspeed 8
=======================================================
7_A9_STOPRAMP.xml
*A9 network
*1 vehicle depspeed 2, maxspeed 10, in the right lane
=======================================================
8_A9_STOPRAMP.xml
*A9 network
*1 vehicle depspeed 0.1, maxspeed 0.1, in the right lane
=======================================================
9_A9_STOPRAMP.xml
*A9 network
*4 Vehicles
  -depspeed 4 maxspeed 10 stop on ordinary lane for 1 second
  -depspeed 2 maxspeed 15
  -depspeed 1 maxspeed 5
  -depspeed 3 maxspeed 8
=======================================================
10_A9_STOPRAMP.xml
*A9 network
*1 vehicle depspeed 2, maxspeed 10, stop 2 seconds the right lane (ramp)
=======================================================
11_A9_STOPRAMP.xml
*A9 network
*1 vehicle depspeed 3, maxspeed 15, stop twice for 1 second in the right lane (ramp)
=======================================================
12_A9_STOPRAMP.xml
*A9 network
*4 Vehicles
  -depspeed 4 maxspeed 10 stop 3 seconds in lane 0 (ramp) 
  -depspeed 2 maxspeed 15
  -depspeed 1 maxspeed 5 stop 2 seconds in lane 2 (main carriage)
  -depspeed 5 maxspeed 8
=======================================================
13_A9_CONGESTION.xml
*A9 network
*7 vehicles
  -depspeed 30 max speed 30 start on lane 0 (most right)
  -depspeed 6 max speed 6 start on lane 3 (most left) with departPos 20
  -depspeed 6 max speed 6 start on lane 3 (most left) with departPos 70
  -depspeed 6 max speed 6 start on lane 3 (most left) with departPos 120
  -depspeed 6 max speed 6 start on lane 3 (most left) with departPos 170
  -depspeed 6 max speed 6 start on lane 3 (most left) with departPos 220
  -depspeed 6 max speed 6 start on lane 3 (most left) with departPos 270
=======================================================
14_A9_CONGESTION.xml
*A9 network
*7 vehicles
  -depspeed 20 max speed 20 start on lane 0 (most right)
  -depspeed 0.5 max speed 6 start on lane 3 (most left) with departPos 20
  -depspeed 0.5 max speed 6 start on lane 3 (most left) with departPos 28 
  -depspeed 0.5 max speed 6 start on lane 3 (most left) with departPos 36
  -depspeed 0.5 max speed 6 start on lane 3 (most left) with departPos 44
  -depspeed 0.5 max speed 6 start on lane 3 (most left) with departPos 52
  -depspeed 0.5 max speed 6 start on lane 3 (most left) with departPos 60
=======================================================
15_A9_CONGESTION.xml
*A9 network
*7 vehicles
  -depspeed 5 max speed 6 start on lane 0 (most right) 
  -depspeed 6 max speed 8 start on lane 3 (most left) lane with departPos 0
  -depspeed 6 max speed 8 start on lane 3 (most left) lane with departPos 15 
  -depspeed 6 max speed 8 start on lane 3 (most left) lane with departPos 30
  -depspeed 6 max speed 8 start on lane 3 (most left) lane with departPos 45 
  -depspeed 6 max speed 8 start on lane 3 (most left) lane with departPos 60
  -depspeed 6 max speed 8 start on lane 3 (most left) lane with departPos 75
=======================================================
16_A9_CONGESTION.xml
*A9 network
*10 vehicles
  -depspeed 15 max speed 30 start on lane 0 (most right) 
  -depspeed 6 max speed 6 start on lane 1 with departPos 0
  -depspeed 6 max speed 6 start on lane 1 with departPos 50 
  -depspeed 6 max speed 6 start on lane 1 with departPos 100
  -depspeed 6 max speed 6 start on lane 2 with departPos 10 
  -depspeed 6 max speed 6 start on lane 2 with departPos 60
  -depspeed 6 max speed 6 start on lane 2 with departPos 110
  -depspeed 6 max speed 6 start on lane 3 with departPos 20 
  -depspeed 6 max speed 6 start on lane 3 with departPos 70
  -depspeed 6 max speed 6 start on lane 3 with departPos 120
=======================================================
17_A9_CONGESTION.xml
*A9 network
*10 vehicles
  -depspeed 6 max speed 6 start on lane 0 (most right) with departPos 0
  -depspeed 6 max speed 6 start on lane 0 (most right) with departPos 50 
  -depspeed 6 max speed 6 start on lane 0 (most right) with departPos 100
  -depspeed 15 max speed 18 start on lane 1 with departPos base 
  -depspeed 16 max speed 20 start on lane 2 with departPos 10 
  -depspeed 16 max speed 20 start on lane 2 with departPos 60
  -depspeed 16 max speed 20 start on lane 2 with departPos 110
  -depspeed 6 max speed 6 start on lane 3 with departPos 20 
  -depspeed 6 max speed 6 start on lane 3 with departPos 70
  -depspeed 6 max speed 6 start on lane 3 with departPos 120
=======================================================
18_A9_CONGESTION.xml
*A9 network
*10 vehicles
  -depspeed 0.5 max speed 6 start on lane 0 (most right) with departPos 0
  -depspeed 0.5 max speed 6 start on lane 0 (most right) with departPos 8 
  -depspeed 0.5 max speed 6 start on lane 0 (most right) with departPos 16 
  -depspeed 8 max speed 10 start on lane 1 with departPos base 
  -depspeed 9 max speed 12 start on lane 2 with departPos 10 
  -depspeed 9 max speed 12 start on lane 2 with departPos 30
  -depspeed 9 max speed 12 start on lane 2 with departPos 50
  -depspeed 0.5 max speed 6 start on lane 3 with departPos 20 
  -depspeed 0.5 max speed 6 start on lane 3 with departPos 29
  -depspeed 0.5 max speed 6 start on lane 3 with departPos 38
=======================================================
19_A9_CONGESTION.xml
*A9 network
*10 vehicles
  -depspeed 0.5 max speed 6 start on lane 0 (most right) with departPos 0
  -depspeed 0.5 max speed 6 start on lane 0 (most right) with departPos 8 
  -depspeed 0.5 max speed 6 start on lane 0 (most right) with departPos 16 
  -depspeed 3 max speed 5 start on lane 1 with departPos base 
  -depspeed 9 max speed 12 start on lane 2 with departPos 10 
  -depspeed 9 max speed 12 start on lane 2 with departPos 30
  -depspeed 9 max speed 12 start on lane 2 with departPos 50
  -depspeed 5 max speed 7 start on lane 3 with departPos 0 
  -depspeed 5 max speed 7 start on lane 3 with departPos 15
  -depspeed 5 max speed 7 start on lane 3 with departPos 30
=======================================================
20_A9_CONGESTION.xml
*A9 network
*10 vehicles
  -depspeed 0.5 max speed 6 start on lane 0 (most right) with departPos 0
  -depspeed 0.5 max speed 6 start on lane 0 (most right) with departPos 8 
  -depspeed 0.5 max speed 6 start on lane 0 (most right) with departPos 16 
  -depspeed 1 max speed 3 start on lane 1 with departPos base 
  -depspeed 0.1 max speed 4 start on lane 2 with departPos 0 
  -depspeed 0.1 max speed 4 start on lane 2 with departPos 9
  -depspeed 0.1 max speed 4 start on lane 2 with departPos 18
  -depspeed 0.1 max speed 2 start on lane 3 with departPos 0 
  -depspeed 0.1 max speed 2 start on lane 3 with departPos 8
  -depspeed 0.1 max speed 2 start on lane 3 with departPos 16
=======================================================
21_A9_TURNSIGNAL.xml
*A9 network
*Vehicle on exit ramp goto maincarriage without turnsignal then goto maincarriage without turnsignal
*1 Vehicle:
   -depspeed 0.5 maxspeed 7 start on lane3 (exitramp) (most left) with departpos base 
   -1st lane change to lane2 (maincarriage) begin at step 21 for 5 steps (no turnsignal)
   -2nd lane change to lane1 (maincarriage) begin at step 41 for 5 steps (no turnsignal)
=======================================================
22_A9_TURNSIGNAL.xml
*A9 network
*Vehicle on maincarriage goto exitramp with turn signal (3steps before) then goto maincarriage without turnsignal then goto exit ramp with wrong turnsignal (3steps before)
*1 Vehicle:
   -depspeed 0.5 maxspeed 7 start on lane1 (maincarriage) with departpos base 
   -1st lane change to lane0 (exitramp) begin at step 11 for 5 steps (turnsignal 3 steps before)
   -2nd lane change to lane1 (maincarriage) begin at step 21 for 5 steps (no turnsignal)
   -3nd lane change to lane0 (exitramp) begin at step 31 for 5 steps (turnsignal wrong direction 3 steps before)
=======================================================
23_A9_TURNSIGNAL.xml
*A9 network
*Vehicle on maincarriage goto exitramp with turn signal (1step before) then goto maincarriage without turnsignal then goto exit ramp with turnsignal (when change starts)
*1 Vehicle:
   -depspeed 0.5 maxspeed 7 start on lane1 (maincarriage) with departpos base 
   -1st lane change to lane0 (exitramp) begin at step 11 for 5 steps (turnsignal 1 step before)
   -2nd lane change to lane1 (maincarriage) begin at step 21 for 5 steps (no turnsignal)
   -3nd lane change to lane0 (exitramp) begin at step 31 for 5 steps (turnsignal when change starts)
=======================================================
24_A9_TURNSIGNAL.xml
*A9 network
*Vehicle on maincarriage goto exitramp with turn signal (2step before) then goto maincarriage without turnsignal then goto exit ramp with turnsignal (5step before)
*1 Vehicle:
   -depspeed 0.5 maxspeed 7 start on lane1 (maincarriage) with departpos base 
   -1st lane change to lane0 (exitramp) begin at step 11 for 5 steps (turnsignal 2 steps before)
   -2nd lane change to lane1 (maincarriage) begin at step 21 for 5 steps (no turnsignal)
   -3nd lane change to lane0 (exitramp) begin at step 41 for 5 steps (turnsignal 5 steps before)
=======================================================
25_A9_TURNSIGNAL.xml
*A9 network
*Two Vehicles on maincarriage goto 2 exitramps at the same time one with turn signal (3step before) and one with wrong signal
*2 Vehicle:
   -Vehicle1
      =depspeed 0.5 maxspeed 7 start on lane1 (maincarriage) with departpos base
      =1st lane change to lane0 (exitramp) begin at step 21 for 5 steps (wrong turn signal 3 steps before)
   -Vehicle2
      =depspeed 0.5 maxspeed 7 start on lane2 (maincarriage) with departpos base
      =1st lane change to lane3 (exitramp) begin at step 21 for 5 steps (turnsignal 3 steps before)
=======================================================
26_A9_TURNSIGNAL.xml
*A9 network
*Two Vehicles on maincarriage goto 2 exitramps at different times one with turn signal (4step before) and one without turn signal
*2 Vehicle:
   -Vehicle1
      =depspeed 0.5 maxspeed 7 start on lane1 (maincarriage) with departpos base
      =1st lane change to lane0 (exitramp) begin at step 11 for 5 steps (turn signal 4 steps before)
   -Vehicle2
      =depspeed 0.5 maxspeed 7 start on lane2 (maincarriage) with departpos base
      =1st lane change to lane3 (exitramp) begin at step 21 for 5 steps (without turn signal)
=======================================================
27_A9_TURNSIGNAL.xml
*A9 network
*Two Vehicles on maincarriage goto 2 exitramps at different times one with turn signal (3step before) and one with turn signal (5steps before)
*2 Vehicle:
   -Vehicle1
      =depspeed 0.5 maxspeed 7 start on lane1 (maincarriage) with departpos base
      =1st lane change to lane0 (exitramp) begin at step 11 for 5 steps (turn signal 3 steps before)
   -Vehicle2
      =depspeed 0.5 maxspeed 7 start on lane2 (maincarriage) with departpos base
      =1st lane change to lane3 (exitramp) begin at step 21 for 5 steps (turn signal 5 steps before)
=======================================================
28_A9_STOPEXITRAMP.xml
*A9 network
*4 Vehicles
  -depspeed 4 maxspeed 10 stop 3 seconds in lane 0 (exitramp)
  -depspeed 2 maxspeed 15
  -depspeed 1 maxspeed 5 stop 2 seconds in lane 2 (exitramp)
  -depspeed 5 maxspeed 8
=======================================================
29_U_TURN.xml
*arbitrary network
*1 Vehicle makes u-turn
=======================================================
30_A9_REVERSING.xml
*A9 network
*4 Vehicles
  -depspeed 4 maxspeed 10 reverses 3 seconds in lane 0 (main carriage)
  -depspeed 2 maxspeed 15
  -depspeed 1 maxspeed 5 reverses 2 seconds in lane 2 (main carriage)
  -depspeed 5 maxspeed 8
=======================================================
31_A9_REVERSING.xml
*A9 network
*1 vehicle depspeed 3, maxspeed 15, reverses twice for 1 second in the right lane (main carriage)
=======================================================
32_A9_VEHICLEENTERING.xml
*A9 network
*Vehicle on ramp goto maincarriage then goto next lane on maincarriage
*1 Vehicle:
   -depspeed 0.5 maxspeed 7 start on lane3 (ramp) (most left) with departpos base
   -1st lane change to lane2 (maincarriage) begin at step 21 for 5 steps
   -2nd lane change to lane1 (maincarriage) begin at step 41 for 5 steps
=======================================================
33_A9_VEHICLEENTERING.xml
*A9 network
*Vehicle on maincarriage goto ramp then goto maincarriage then goto ramp
*1 Vehicle:
   -depspeed 0.5 maxspeed 7 start on lane1 (maincarriage) with departpos base
   -1st lane change to lane0 (ramp) begin at step 11 for 5 steps
   -2nd lane change to lane1 (maincarriage) begin at step 21 for 5 steps
   -3nd lane change to lane0 (ramp) begin at step 31 for 5 steps
=======================================================
34_A9_SPEEDLIMIT.xml
*A9 network
*1 vehicle depspeed 2, maxspeed 10, in the right lane, speed limit 10
=======================================================
35_A9_SPEEDLIMIT.xml
*A9 network
*1 vehicle depspeed 2, maxspeed 10, in the right lane, speed limit 0
=======================================================
36_A9_SAFEDISTANCE.xml
*A9 network
*7 vehicles
  -start on lane 0 (most right)
  -start on lane 3 (most left) with departPos 20
  -start on lane 3 (most left) with departPos 28
  -start on lane 3 (most left) with departPos 36
  -start on lane 3 (most left) with departPos 44
  -start on lane 3 (most left) with departPos 52
  -start on lane 3 (most left) with departPos 60
=======================================================
37_BRAKES_ABRUPTLY.xml
*arbitrary network
*1 vehicle brakes abruptly without front vehicle, acceleration is set in xml
=======================================================
38_BRAKES_ABRUPTLY.xml
*arbitrary network
*1 vehicle brakes abruptly without front vehicle, no acceleration is set
=======================================================
39_BRAKES_ABRUPTLY.xml
*arbitrary network
*2 vehicles
*1 vehicle brakes abruptly without front vehicle braking
=======================================================
40_BRAKES_ABRUPTLY.xml
*arbitrary network
*2 vehicles
*one vehicle brakes abruptly, then other has to brake too - not abrupt since caused by other vehicle braking
