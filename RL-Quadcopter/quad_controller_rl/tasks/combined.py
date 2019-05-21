"""Combined task."""

import numpy as np
from quad_controller_rl.tasks.base_task import BaseTask
from quad_controller_rl.tasks.takeoff import Takeoff
from quad_controller_rl.tasks.hover import Hover
from quad_controller_rl.tasks.landing import Landing

class Combined(BaseTask):
    """Simple task where the goal is to make the agent learn to settle down gently."""

    def __init__(self):
        # Task-specific parameters
        self.tasks = {'takeoff' : Takeoff(),
                    'hover' : Hover(),
                    'landing' : Landing()}
        self.current = 'takeoff'

    def set_agent(self, agent):
        self.agent = agent
        {task.set_agent(agent) for task in self.tasks.values()}

    def reset(self):
        # Nothing to reset; just return initial condition
        self.current = 'takeoff'
        return self.tasks[self.current].reset()

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        wrench, done = self.tasks[self.current].update(timestamp, pose, angular_velocity, linear_acceleration)

        if self.has_takeoff(timestamp, pose, angular_velocity, linear_acceleration):
            self.current = 'hover'
        elif self.has_hover(timestamp, pose, angular_velocity, linear_acceleration):
            self.current = 'landing'
        elif self.has_landing(timestamp, pose, angular_velocity, linear_acceleration):
            return wrench, True
        
        return wrench, done

    def has_takeoff(self, timestamp, pose, angular_velocity, linear_acceleration):
        return pose.position.z >= self.tasks['takeoff'].target_z

    def has_hover(self, timestamp, pose, angular_velocity, linear_acceleration):
        return timestamp > self.tasks['hover'].max_duration

    def has_landing(self, timestamp, pose, angular_velocity, linear_acceleration):
        return pose.position.z < 0.5
