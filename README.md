# selfdrivingcar

2D camera-over-top car game developed with PyGame and AI agent created with Tensorflow.

I started this project for a presentation on Reinforcement Learning I was supposed to give, inspired by some Youtube videos that did the same project (self-driving car using Deep-Q-Learning.). All of the code (game and ML model) was written by me from scratch just based on the concepts I saw on the very interesting videos combined with some new nuances I added to it that I didn't see on the videos I watched like being able to draw new tracks to test the model in different scenarios. This is still an ongoing pet project as there are some elements I want to incorporate to the game i.e.: being able to retrieve saved models. Most of the basic functionalities are operational as described below:

1- Start game by running sfc.py

2- Tracks

  2.a - Click on the "Load Track" button to browse through some of the preset tracks. 
  
  2.b.1 - You can also draw your own tracks using the "Draw Track" button. 
  
  2.b.2 - In drawing mode, use the mouse to create the tracks. Press the S key to increase the width of the track and X to decrease. Make sure you close the loop in the track so the learning agent can revolve around multiple times. You can save your track by clicking "Save Track" in drawing mode and then "Done" once you are done with drawing.
  
3. Reward gates: If you draw your own track, the learning agent requires you put "reward gates" in the track so the learning agent gets rewarded for driving correctly. On drawing mode, click "Add Reward Gate" to add as many reward gates as you wish. Make sure you them in the order you want the agent to get "points" for.

4. The game is in "Gaming Mode" by default, so with the track visible you can drive the car by pressing the up, down, left and right keys on your keyboard. Avoid touching the sides of the track. As you complete the track, the maximum speed of the car increases to increase difficulty.

5. If you want to have the AI agent to start learning, click on "Learning Mode" then "Learn!" and you will see the car will start to drive by itself.

