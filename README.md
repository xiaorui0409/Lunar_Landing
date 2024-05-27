# Lunar_Landing
In this task, I will train an agent by utilze the Double Deep Q Networks to land a lunar lander safely on a landing pad on the surface of the moon.

Outline
1. Lunar Lander Environment
In this notebook, we'll be utilizing the Gym Library by OpenAI, which offers a diverse range of environments suitable for reinforcement learning experiments. It's important to note that all development of Gym has now transitioned to Gymnasium, a continuation of the project under the stewardship of the Farama Foundation.For more detailed information, please visit the specified website: https://www.gymlibrary.dev/environments/box2d/index.html

We will focus on the Lunar Lander environment within this framework. The objective here is to achieve a safe landing for the lunar module on a specified landing pad, marked by two flag poles, located at the zero coordinates (0,0). Although landing precisely on the pad is ideal, the lander can also touch down safely elsewhere. The scenario begins with the lander at the center-top of the environment, subjected to a random initial force acting on its center of mass. The lander has access to unlimited fuel. The task is considered successfully completed when a score of 200 points is achieved.

The agent has four discrete actions available:
Action Space:

Do nothing.
Fire right engine.
Fire main engine.
Fire left engine.
Each action has a corresponding numerical value:

Do nothing = 0
Fire right engine = 1
Fire main engine = 2
Fire left engine = 3

3 - The Lunar Lander Environment
In this notebook we will be using OpenAI's Gym Library. The Gym library provides a wide variety of environments for reinforcement learning. To put it simply, an environment represents a problem or task to be solved. In this notebook, we will try to solve the Lunar Lander environment using reinforcement learning.

The goal of the Lunar Lander environment is to land the lunar lander safely on the landing pad on the surface of the moon. The landing pad is designated by two flag poles and it is always at coordinates (0,0) but the lander is also allowed to land outside of the landing pad. The lander starts at the top center of the environment with a random initial force applied to its center of mass and has infinite fuel. The environment is considered solved if you get 200 points.



Fig 1. Lunar Lander Environment.

3.1 Action Space
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

Observation Space








All development of Gym has been moved to Gymnasium, a new package in the Farama Foundation that's maintained by the same team of developers who have maintained Gym for the past 18 months.

https://github.com/xiaorui0409/Lunar_Landing/assets/151507129/48fb2ea9-0bd4-47b7-916a-c77d1200b25e

