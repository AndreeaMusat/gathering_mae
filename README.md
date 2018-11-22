# gathering_mae

Install with `python3 -m pip install .`

## Config example & explanations

name: "SingleAgentGatheringEnv" `# Name of the environment class`

verbose: 0

base_map: "<path>"  `# Path to a txt map of NxM elements | ".": empty | "w": wall | "o": R | "x"
 -R`

map_size: [32,32] `# Size of previously defined map (from base_map path)`

map_view_extend: 0 `# Add padding to full map ( if more than 0 pixels)`

env_max_steps_no: 1000  `# Max number of steps that can be played in the env -> Reset after.`

no_envs: 1

no_agents: 1

agents_collide: no

hide_other_agents: no

partial_observable: [1, 4]  

                                # [0 (no) / 1 (yes), radius] If to use partial observability and
                                # size of view image considering a radius in pixels around the agent

agents_init_pos: [[-1, 5, 5, 4]] 
                                
                                 # [ [no_agents, x_coord, y_coord, radius] 
                                 # Position no_agents around point (x_coord, y_coord)  within
                                 # a radius of "radius"

agents_init_colors: [[-1, 0]] # [ [no_agents, colord_id] ]

use_laser: no

agents_laser_size: 2

reward_distance: no  # Reward based on distance to first reward found on map

reward_value: [1., -1.]  # [ reward_values, ... ] Reward value for each type of reward defined

reward_respawn_time: [4, 4]  # [ reward_respawn_time, ...] No steps to wait for each type of

                             # reward to respawn

reward_init_pos: [[0, 10, 25, 25, 5], [1, 50, 16, 16, 20]]

                  # [ [Reward type, no of reward elements, x_coord, y_coord, radius] ...]
                  # Define zones to spawn / re-spawn rewards

use_cuda: no

visualize_rgb: yes  `# Render in RGB mode (obs returned by .render()`

store_agents_trace: no

visualize: yes  `# Should be yes if we want to render`

record_data:
  turned_on: no
  only_coord: no
