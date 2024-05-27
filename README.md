# Lunar_Landing
In this task, I will train an agent by utilze the Double Deep Q Networks to land a lunar lander safely on a landing pad on the surface of the moon.

Outline
1. Lunar Lander Environment
In this notebook, we'll be utilizing the Gym Library by OpenAI, which offers a diverse range of environments suitable for reinforcement learning experiments. It's important to note that all development of Gym has now transitioned to Gymnasium, a continuation of the project under the stewardship of the Farama Foundation.For more detailed information, please visit the specified website: https://www.gymlibrary.dev/environments/box2d/index.html

We will focus on the Lunar Lander environment within this framework. The objective here is to achieve a safe landing for the lunar module on a specified landing pad, marked by two flag poles, located at the zero coordinates (0,0). Although landing precisely on the pad is ideal, the lander can also touch down safely elsewhere. The scenario begins with the lander at the center-top of the environment, subjected to a random initial force acting on its center of mass. The lander has access to unlimited fuel. The task is considered successfully completed when a score of 200 points is achieved.


1.1 Action Space:
The agent has four discrete actions available:

Do nothing.
Fire right engine.
Fire main engine.
Fire left engine.
Each action has a corresponding numerical value:

Do nothing = 0
Fire right engine = 1
Fire main engine = 2
Fire left engine = 3

1.2 Action Space
The agent has four discrete actions available:

Do nothing.
Fire right engine.
Fire main engine.
Fire left engine.
Each action has a corresponding numerical value:

Do nothing = 0
Fire right engine = 1
Fire main engine = 2
Fire left engine = 3

1.3 Observation Space

![observations](https://github.com/xiaorui0409/Lunar_Landing/assets/151507129/2cf93749-a674-4418-948a-01944a11a207)









All development of Gym has been moved to Gymnasium, a new package in the Farama Foundation that's maintained by the same team of developers who have maintained Gym for the past 18 months.

https://github.com/xiaorui0409/Lunar_Landing/assets/151507129/48fb2ea9-0bd4-47b7-916a-c77d1200b25e

