# Village People, 2018
"""
  name: "GatheringEnv"
  verbose: *verbose
  type_acr: "map32_4rooms"
  base_map: "map32_4rooms"
  map_size: &map_size [32,32]
  map_view_extend: 0
  env_max_steps_no: &env_max_steps_no 100
  no_envs: 1
  no_agents: 10
  agents_collide: yes
  hide_other_agents: no
  partial_observable: [0, 4]
  agents_init_pos: [[-1, 5, 5, 6]]
  agents_init_colors: [[-1, 0]]
  use_laser: no
  agents_laser_size: 2
  reward_distance: yes # IF reward distance true (give reward based on distance to first reward
                                                on map)

  reward_value: 1.
  reward_respawn_time: 0
  use_cuda: no
  visualize_rgb: yes
  visualize: no
"""

import torch
import cv2
import numpy as np
from collections import deque
from gym import spaces
import copy
import os

from . import util
from .util import get_gray_color
from .base_env import MultiAgentBaseEnv
from .elements_definitions import *


def get_linear_pos(x, y, width):
    return x * width + y


class MapManager:

    # Elements definitions
    no_static_el = len(STATIC_ELEMENTS)
    no_special_el = len(SPECIAL_ELEMENTS)
    all_elements = STATIC_ELEMENTS + SPECIAL_ELEMENTS

    # Reverse to have ascii to Element id mapping
    ascii_code_index = dict()
    # Elements colors (normalized 0-1)
    elements_colors = torch.FloatTensor(no_static_el + no_special_el)
    el_color_convert_g_rgb = dict()

    for k, v in enumerate(all_elements):
        ascii_ = v[E_ASCII_K]
        assert ascii_ not in ascii_code_index.keys(), "Ascii key already defined"
        ascii_code_index[ascii_] = k
        gray_color = get_gray_color(v[E_RGB_K])
        assert gray_color not in el_color_convert_g_rgb.keys(), \
            "Converted grayscale color for element  duplicate {}".format(v[E_RGB_K])

        el_color_convert_g_rgb[gray_color] = v[E_RGB_K]
        elements_colors[k] = float(gray_color)
    elements_colors.div_(255.)

    direction_color = get_gray_color(DIRECTION_EL[E_RGB_K])
    assert direction_color not in el_color_convert_g_rgb.keys(), \
        "Converted grayscale color for element  duplicate {}".format(v[E_RGB_K])
    el_color_convert_g_rgb[direction_color] = DIRECTION_EL[E_RGB_K]
    direction_color = direction_color / 255.

    static_el_color = elements_colors[:no_static_el]
    special_el_color = elements_colors[no_static_el:]

    # Agents colors
    agents_all_colors_g = torch.FloatTensor(len(AGENTS_COLORS))
    for i, ag_cl in enumerate(AGENTS_COLORS):
        gray_color = get_gray_color(ag_cl)
        assert gray_color not in el_color_convert_g_rgb.keys(), \
            "Converted grayscale color for element  duplicate {}".format(v[E_RGB_K])

        el_color_convert_g_rgb[gray_color] = ag_cl
        agents_all_colors_g[i] = float(gray_color)
    agents_all_colors_g.div_(255.)

    # # Color convertor GRAY to RGB
    gray_to_bgr = cv2.cvtColor(np.arange(256).astype(np.uint8).reshape(1, 256), cv2.COLOR_GRAY2BGR)
    gray_to_bgr = torch.ByteTensor(gray_to_bgr).squeeze(0)

    for gray_color, rgb_color in el_color_convert_g_rgb.items():
        gray_to_bgr[gray_color] = torch.ByteTensor(rgb_color)

    gray_to_bgr = gray_to_bgr[:, [2, 1, 0]]

    def __init__(self, cfg):
        if os.path.isfile(cfg.base_map):
            base_map_path = cfg.base_map
        else:
            from .maps.utils import get_map_path
            base_map_path = get_map_path(cfg.base_map)

        self.map_size = map_size = cfg.map_size
        self.map_view_extend = cfg.map_view_extend
        self.use_cuda = cfg.use_cuda
        self.reward_distance = cfg.reward_distance
        self.visualize_rgb = cfg.visualize_rgb
        self.visualize = cfg.visualize
        self.store_agents_trace = cfg.store_agents_trace

        self.no_agents = cfg.no_agents
        self.agents_init_pos = cfg.agents_init_pos

        self.agents_colors, self.agents_group_id, self.agents_color_map = \
            self.generate_agents_colors(cfg.agents_init_colors)

        self.hide_other_agents = cfg.hide_other_agents
        self.agents_collide = cfg.agents_collide

        self.partial_observable = (cfg.partial_observable[0] == 1)
        self.partial_observable_radius = cfg.partial_observable[1]

        self.agents_range = range(self.no_agents)

        # Base Map loaded from file (torch coordinate map)
        self.base_static_map, self.base_special_map = self.load_base_map(base_map_path)

        # Build base variables used for map views (observations & for visualization)
        self.special_el_color_map = self.special_el_color.unsqueeze(1).unsqueeze(2)\
            .expand(self.no_special_el, *self.map_size)

        self.static_map_view, full_view_map, view_offset_r = self.build_static_map_views()
        self.view_offset_r = view_offset_r

        # x - r + r : x + r + r (full view map will be offset by partial_observable_radius)
        # so will have to cut from x: x + 2*r
        self.partial_observable_offset = self.partial_observable_radius * 2 + 1

        self.view_offset_p = [view_offset_r, view_offset_r+map_size[0],
                              view_offset_r, view_offset_r+map_size[1]]

        # Expand to number agents (Full view might contain extra border)
        full_view_map = full_view_map.unsqueeze(0).expand(self.no_agents,
                                                          full_view_map.size(0),
                                                          full_view_map.size(1)).contiguous()

        if self.use_cuda:
            if self.partial_observable:
                self.observation_buffer = torch.zeros([self.no_agents,
                                                       self.partial_observable_offset,
                                                       self.partial_observable_offset
                                                       ]).cuda(async=True)
            else:
                self.observation_buffer = full_view_map.cuda(async=True)

        self.full_view_map = full_view_map

        self.map_view_size = torch.Size([self.no_agents] + self.map_size)
        self.full_view_map_size = full_view_map.size()

        self.prev_build_map = None
        self.move_map = None
        self.prev_build_base_map = None

    def get_observation_size(self):
        if self.partial_observable:
            return torch.Size([self.no_agents,
                               self.partial_observable_offset,
                               self.partial_observable_offset])
        else:
            return self.full_view_map.size()

    def generate_agents_colors(self, agents_init_colors):
        """
        agents_colors: [(no_agents, color_id)]
        no_agents == -1 -> fill for the rest of agents
        color_id == -1  -> random from available list of colors
        """
        no_agents = self.no_agents
        agents_all_colors_g = self.agents_all_colors_g

        colored_agents = 0
        agents_colors = []
        agents_group_id = []
        agents_color_map = torch.FloatTensor(no_agents, *self.map_size)

        for group_id, (group_no, color_id) in enumerate(agents_init_colors):
            if group_no == -1:
                # Color all other agents
                group_no = no_agents - colored_agents
            else:
                group_no = min(no_agents - colored_agents, group_no)

            if color_id == -1:
                color_id = np.random.randint(0, agents_all_colors_g.size(0))
            elif color_id >= agents_all_colors_g.size(0):
                raise ValueError('No color for color id: {}'.format(color_id))

            for i in range(group_no):
                agents_group_id.append(group_id)
                agents_colors.append(agents_all_colors_g[color_id])
                agents_color_map[colored_agents] = agents_all_colors_g[color_id]
                colored_agents += 1

        return agents_colors, agents_group_id, agents_color_map

    def generate_operation_savers(self):
        """Generate stuff that save calculations during run, considering the static map only"""
        map_ = self.base_static_map

        map_size = self.map_size

        # move_mapper[coord_r, coord_c, action<0..3>] -> (new_r, new_c), direction
        move_mapper = []
        for r in range(map_size[0]):
            coord_c_ = []
            for c in range(map_size[1]):
                act_ = []

                for a in range(4):
                    new_r, new_c = util.add_tuple((r, c), ACTIONS_MOVE[a])
                    not_valid_move = True

                    if 0 <= new_r < map_size[0] and 0 <= new_c < map_size[1]:
                        if map_[new_r, new_c] == DEFAULT_ELEMENT_ID:
                            # Direction is set to movement direction
                            # (key: 0 .. "up" .. direction: 0)
                            act_.append(((new_r, new_c), a))
                            not_valid_move = False

                    if not_valid_move:
                        act_.append(((r, c), a))

                coord_c_.append(act_)
            move_mapper.append(coord_c_)

        # Generate Color maps for agents and
        self.move_map = move_mapper
        return move_mapper

    def load_base_map(self, base_map_path):
        """Load map from text file"""
        ascii_code_index = self.ascii_code_index
        map_size = self.map_size
        no_static_el = self.no_static_el
        no_special_el = self.no_special_el

        map_file = open(base_map_path, 'r')
        map_ = map_file.readlines()
        map_ = list(map(str.strip, map_))
        torch_static_map = torch.zeros(map_size)
        torch_special_map = torch.zeros(no_special_el, *map_size)

        if len(map_) != map_size[1]:
            raise ValueError('Base map bad height. ({})'.format(base_map_path))

        # Generate torch map from ASCII
        for ir, row in enumerate(map_):
            if len(row) != map_size[0]:
                raise ValueError('Base map bad width. ({})'.format(base_map_path))

            for ic, x in enumerate(row):
                if x not in ascii_code_index.keys():
                    raise ValueError('Base map bad ASCII at ({}, {}) - {}. ({})'
                                     .format(ir, ic, x, base_map_path))
                el_idx = ascii_code_index[x]
                if el_idx < no_static_el:
                    torch_static_map[ir, ic] = ascii_code_index[x]
                else:
                    torch_special_map[ascii_code_index[x] - no_static_el, ir, ic] = 1

        return torch_static_map, torch_special_map

    def reset_map(self):
        # Should consider base map and initialize other elements.
        static_map = self.base_static_map.clone()
        special_map = self.base_special_map.clone()

        return static_map, special_map

    def init_agents(self, map_, init_pos, agents_b_map, agents_collide):
        """
        Agents_map is changed inplace
        init_pos: [(no_agents, x, y, radius zone to place randomly)]
        if radius is not enough -> radius * 2

        E.g. for 10 agents & map (9x9)- [(10, 5,5, 8)] -> agents will be placed randomly everywhere
        """
        # Should return closest available position
        map_x, map_y = self.map_size
        move_map = self.move_map

        no_agents = agents_b_map.size(0)
        agents_b_map.zero_()
        agents_d = []
        agents_d_c = []
        agents_coord = []
        agents_dir = torch.from_numpy(np.random.randint(0, AGENT_MAX_STATES, size=no_agents)).byte()

        agents_placed = 0

        for no_ag, x, y, r in init_pos:
            # Place all other agents
            if no_ag == -1:
                no_ag = no_agents - agents_placed
            else:
                no_ag = min(no_agents - agents_placed, no_ag)

            while no_ag > 0:
                x0 = max(0, x - r)
                x1 = min(map_x, x + r)
                y0 = max(0, y - r)
                y1 = min(map_y, y + r)

                if agents_collide:
                    empty_coord = ((map_[x0:x1, y0:y1] + agents_b_map[:, x0:x1, y0:y1].sum(0)) ==
                                   DEFAULT_ELEMENT_ID).nonzero()
                else:
                    empty_coord = ((map_[x0:x1, y0:y1]) == DEFAULT_ELEMENT_ID).nonzero()

                if empty_coord.nelement() == 0:
                    if r > map_x and r > map_y:
                        raise ValueError("Cannot place all agents: {}".format(r))

                    r *= 2
                    continue

                empty_coord[:, 0].add_(x0)
                empty_coord[:, 1].add_(y0)

                no_empty = empty_coord.size(0)
                select = np.arange(no_empty)
                np.random.shuffle(select)
                select_idx = 0

                while no_ag > 0 and select_idx < no_empty:
                    idx = select[select_idx]
                    ir, ic = empty_coord[idx, 0], empty_coord[idx, 1]
                    agents_coord.append((ir, ic))
                    agents_b_map[agents_placed, ir, ic] = 1.
                    agents_d.append(agents_dir[agents_placed])
                    d_coord, _ = move_map[ir][ic][agents_dir[agents_placed]]
                    agents_d_c.append(d_coord)

                    select_idx += 1
                    agents_placed += 1
                    no_ag -= 1

        if agents_placed != no_agents:
            raise ValueError("Did not place all agents")

        return agents_b_map, agents_d, agents_d_c, agents_coord

    def init_elements(self, map_, occupancy_map, init_pos):
        """
        init_pos: [(idx_map, no_items, x, y, radius zone to place randomly)]
        """
        # Should return closest available position
        map_x, map_y = self.map_size

        # Generate for each initialized element an entry point with initialization
        el_init = dict()

        for ix_map, no_items, x, y, r in init_pos:

            x0 = max(0, x - r)
            x1 = min(map_x, x + r)
            y0 = max(0, y - r)
            y1 = min(map_y, y + r)

            empty_coord = (occupancy_map[x0:x1, y0:y1] == DEFAULT_ELEMENT_ID).nonzero()

            select_idxs = torch.randperm(empty_coord.size(0))
            if no_items == -1:
                # Fill zone
                no_items = empty_coord.size(0)

            select_idxs = select_idxs[:no_items]
            if select_idxs.size(0) > 0:
                for iselect in select_idxs:
                    x_p, y_p = empty_coord[iselect]
                    map_[ix_map, x_p, y_p] = 1
                    el_init[get_linear_pos(x_p, y_p, map_y).item()] = (ix_map, 1, x, y, r)

        return el_init

    def build_static_map_views(self):
        static_map = self.base_static_map
        use_cuda = self.use_cuda
        map_size = self.map_size
        elements_colors = self.elements_colors

        # Calculate boarder map extension for either partial observability or settings of extension
        extend_view_r = self.map_view_extend
        if self.partial_observable:
            extend_view_r = max(extend_view_r, self.partial_observable_radius)

        map_full_size = (map_size[0] + extend_view_r * 2, map_size[1] + extend_view_r * 2)
        map_static_view = torch.zeros(map_full_size[0], map_full_size[1])
        rm = map_size[0] + extend_view_r
        cm = map_size[1] + extend_view_r

        # Create map border with wall
        if extend_view_r > 0:
            map_static_view[extend_view_r-1: rm+1,
                            extend_view_r-1: cm+1].fill_(elements_colors[WALL_ID])

        map_static_view[extend_view_r: rm, extend_view_r: cm] = \
            elements_colors[static_map.view(-1).long()].view(*map_size)

        full_view = map_static_view
        map_static_view = map_static_view[extend_view_r: rm, extend_view_r: cm]
        return map_static_view.clone(), full_view.clone(), extend_view_r

    def build_maps(self, special_map, agents_dir, agents_coord):
        # We consider static map unchanged and already loaded
        base_map = self.static_map_view.clone()
        hide_other_agents = self.hide_other_agents
        direction_color = self.direction_color

        # Place agents on map
        if hide_other_agents:
            # Place special elements (Special Symbols do not collide)
            # Hide shadow for others
            base_map.add_((self.special_el_color_map[:-1] * special_map[:-1]).sum(0))

            # Faster if few agents ~ <20
            base_map = base_map.unsqueeze(0).expand(self.map_view_size).contiguous()

            if self.visualize:
                self.prev_build_map = base_map[0].clone()
                self.prev_build_base_map = base_map[0].clone()
            else:
                self.prev_build_map = base_map[0]

            for i in self.agents_range:
                ag_map = base_map[i]
                ag_map[agents_coord[i]] = self.agents_colors[i]

                # draw direction
                if ag_map[agents_dir[i]] == 0:
                    ag_map[agents_dir[i]] = direction_color

                if self.visualize:
                    self.prev_build_map[agents_coord[i]] = self.agents_colors[i]
        else:
            # Place special elements (Special Symbols do not collide)
            base_map.add_((self.special_el_color_map * special_map).sum(0))

            if self.visualize:
                self.prev_build_base_map = base_map[0].clone()

            # Faster if few agents ~ <20
            for i in self.agents_range:
                base_map[agents_coord[i]] = self.agents_colors[i]

                # draw direction
                if base_map[agents_dir[i]] == 0:
                    base_map[agents_dir[i]] = direction_color

            # Save for visualization
            if self.visualize:
                self.prev_build_map = base_map

            base_map = base_map.unsqueeze(0).expand(self.map_view_size).contiguous()

            # Scales better to many agents
            # if agents_collide:
            #     # Remove other elements (agents go on top)
            #     g_color = (self.agents_color_map * agents_map).sum(2)
            #     base_map.mul_((g_color == 0).float())
            #     base_map.add_(g_color)
            # else:
            #     # Get only max agent
            #     g_color = (self.agents_color_map * agents_map).gather(2, agents_map.max(2)[
            #         1].unsqueeze(2))
            #     base_map.mul_((g_color == 0).float())
            #     base_map.add_(g_color)

        if self.view_offset_r > 0:
            # Add map to allocated buffer which has border
            view_offset = self.view_offset_p

            self.full_view_map[:, view_offset[0]:view_offset[1],
                               view_offset[2]:view_offset[3]].copy_(base_map)
            base_map = self.full_view_map

            if self.partial_observable:
                r = self.partial_observable_offset
                partial_view = []
                for i in self.agents_range:
                    x, y = agents_coord[i]
                    partial_view.append(base_map[i, x:x+r, y:y+r].unsqueeze(0))
                base_map = torch.cat(partial_view, dim=0)

        # If observation output required on CUDA use buffer
        if self.use_cuda:
            self.observation_buffer.copy_(base_map, async=True)
            base_map = self.observation_buffer

        return base_map

    def transform_gray_to_rgb(self, gray_color):
        x = (gray_color * 255).byte()
        return self.gray_to_bgr[int(x)]

    def transform_map_to_rgb(self, map):
        map = (map.cpu() * 255).byte()

        gray_to_bgr = self.gray_to_bgr
        new_map = gray_to_bgr[map.view(-1).long()].view(map.size() + torch.Size((3,)))
        return new_map

    def transform_observation_to_view(self, observation, rgb=True):
        observation = (observation.cpu() * 255).byte()
        prev_build_map = (self.prev_build_map.cpu() * 255).byte()

        if self.visualize_rgb:
            gray_to_bgr = self.gray_to_bgr
            observation = gray_to_bgr[observation.view(-1).long()].view(observation.size() +
                                                                        torch.Size((3,)))
            prev_build_map = gray_to_bgr[prev_build_map.view(-1).long()]\
                .view(prev_build_map.size() + torch.Size((3,)))
            add_shape = (3, )
        else:
            add_shape = ()

        if self.partial_observable or self.hide_other_agents:
            full_obs = observation.numpy()

            empty_line = np.zeros((1, PARTIAL_VISUALIZE_SIZE[1]) + add_shape, dtype=np.uint8)
            empty_line[:] = 255
            p_obs = []
            for i in self.agents_range:
                p_obs.append(cv2.resize(full_obs[i], PARTIAL_VISUALIZE_SIZE,
                                        interpolation=cv2.INTER_NEAREST))
                p_obs.append(empty_line)

            p_obs = np.vstack(p_obs)
            full_map = prev_build_map.numpy()
            full_map = cv2.resize(full_map, FULL_VISUALIZE_SIZE,
                                  interpolation=cv2.INTER_NEAREST)

            w = max(p_obs.shape[0], full_map.shape[0])
            h = p_obs.shape[1] + 20 + full_map.shape[1]

            full_image = np.zeros((w, h) + add_shape, dtype=np.uint8)
            full_image[:] = 255

            full_image[:full_map.shape[0], :full_map.shape[1]] = full_map
            full_image[: p_obs.shape[0],
                       full_map.shape[1]+20: full_map.shape[1]+20+p_obs.shape[1]] = p_obs
        else:
            full_image = observation[0].numpy()
            full_image = cv2.resize(full_image, FULL_VISUALIZE_SIZE,
                                    interpolation=cv2.INTER_NEAREST)

        return full_image


class GatheringEnv(MultiAgentBaseEnv):
    def __init__(self, cfg):
        super(GatheringEnv, self).__init__(cfg)
        self.no_agents = no_agents = cfg.no_agents
        self.map_size = map_size = cfg.map_size

        self.agents_collide = cfg.agents_collide
        self.agent_init_pos = cfg.agents_init_pos
        self.use_laser = use_laser = cfg.use_laser
        self.agents_laser_size = laser_size = cfg.agents_laser_size

        self.rewards_value = cfg.reward_value
        self.rewards_respawn_time = cfg.reward_respawn_time
        self.reward_distance = cfg.reward_distance
        self.reward_init_pos = cfg.reward_init_pos

        # Agent stuff
        # agents_map[x,y, agent_idx] != 0 -> [0,1,2,3] (direction N, E, S, V)
        self.agents_binary_map = torch.zeros(no_agents, *map_size)
        self.agents_direction = []
        self.agents_direction_coord = []
        self.agents_coord = []

        self.range_agents = range(no_agents)

        # Map stuff
        self.map_manager = map_manager = MapManager(cfg)
        self.static_map = None
        self.special_map = None

        # Util variables to save computation (Based on base Map/ init stuff)
        self.move_map = map_manager.generate_operation_savers()

        # Special elements variables
        self.el_rewards_respawn = None
        self.el_init = None

        self.laser_coord = laser_coord = []
        for x, y in ACTIONS_MOVE:
            if x == 0:
                if y > 0:
                    laser_coord.append([0, 1, y, y * laser_size+1])
                else:
                    laser_coord.append([0, 1, y * laser_size, y+1])
            else:
                if x > 0:
                    laser_coord.append([x, x * laser_size+1, 0, 1])
                else:
                    laser_coord.append([x * laser_size, x+1, 0, 1])

        self.no_actions = len(ACTIONS)
        if not use_laser:
            self.no_actions -= 1

        self.action_space_ = spaces.MultiDiscrete([self.no_actions] * no_agents)
        self.observation_space_ = map_manager.get_observation_size()

        if self.reward_distance:
            self.reward_distance_ref = None
            self.max_reward_distance = np.sqrt(self.map_size[0] ** 2 + self.map_size[1] ** 2)

    def _step(self, action):
        no_agents = self.no_agents
        special_map = self.special_map
        agents_b_map = self.agents_binary_map
        agents_d = self.agents_direction
        agents_d_coord = self.agents_direction_coord
        agents_coord = self.agents_coord
        agents_collide = self.agents_collide
        reward = torch.zeros(no_agents)
        rewards_respawn_time = self.rewards_respawn_time
        rewards_val = self.rewards_value
        move_map = self.move_map
        el_rewards_respawn = self.el_rewards_respawn
        step_cnt = self.step_cnt
        map_manager = self.map_manager

        reward_distance = self.reward_distance
        if reward_distance:
            reward_ref = self.reward_distance_ref
            max_reward_distance = self.max_reward_distance

        # Reset direction & laser special maps map
        rewards_ids = ALL_REWARD_IDS
        laser_map = special_map[LASER_ID]
        laser_map.zero_()

        for ix, el_reward_respawn in zip(rewards_ids, el_rewards_respawn):
            if rewards_respawn_time[ix] != 0:
                if len(el_reward_respawn) > 0:
                    while el_reward_respawn[0][0] < step_cnt:
                        _, coord, init_gen = el_reward_respawn.popleft()
                        if init_gen == -1:
                            special_map[ix][coord] = 1.
                        else:
                            self.el_init.update(map_manager.init_elements(special_map,
                                                                          self.static_map,
                                                                          init_gen))

                        if len(el_reward_respawn) <= 0:
                            break

        for ag_idx in self.range_agents:
            act = action[ag_idx]
            x, y = agents_coord[ag_idx]

            if act < ACTION_M_MOVE:
                # Action to move to new space

                new_coord, new_d = move_map[x][y][act]

                if agents_collide:
                    if new_coord in agents_coord:
                        # Cannot move
                        continue

                # Check interactions with SPECIAL_ELEMENTS
                for imap in rewards_ids:
                    if special_map[imap][new_coord] != 0:
                        reward[ag_idx] += rewards_val[imap]
                        if rewards_respawn_time[imap] != 0:
                            special_map[imap][new_coord] = 0.

                            # Check generator
                            next_stp = step_cnt + rewards_respawn_time[imap]
                            if self.el_init:
                                lcoord = get_linear_pos(new_coord[0], new_coord[1],
                                                        special_map.size(2))
                                init_gen = self.el_init.pop(lcoord, -1)
                                if init_gen != -1:
                                    init_gen = [init_gen]
                                el_rewards_respawn[imap].append((next_stp, new_coord, init_gen))
                            else:
                                el_rewards_respawn[imap].append((next_stp, new_coord, -1))
                # Move agent
                agents_b_map[ag_idx, x, y] = 0.
                agents_b_map[ag_idx][new_coord] = 1.
                agents_d[ag_idx] = new_d
                agents_coord[ag_idx] = new_coord
                x, y = new_coord
                agents_d_coord[ag_idx], _ = move_map[x][y][new_d]

            # elif act == ACTION_NULL:
            elif act == ACTION_TURN_C:
                agents_d[ag_idx] = TURN_C[agents_d[ag_idx]]
                agents_d_coord[ag_idx], _ = move_map[x][y][agents_d[ag_idx]]
            elif act == ACTION_TURN_CC:
                agents_d[ag_idx] = TURN_CC[agents_d[ag_idx]]
                agents_d_coord[ag_idx], _ = move_map[x][y][agents_d[ag_idx]]
            elif act == ACTION_ACTION:
                a, b, c, d = self.laser_coord[agents_d[ag_idx]]
                laser_map[max(0, x+a): max(0, x+b), max(0, y+c): max(0, y+d)] = 1.

            if reward_distance:
                r = np.sqrt((reward_ref[0] - x) ** 2 + (reward_ref[1] - y) ** 2)
                reward[ag_idx] += float((max_reward_distance - r) / max_reward_distance *
                                        rewards_val)

        # Should calculate effect of laser after all agents have moved
        # TODO
        done = torch.zeros(no_agents).byte()
        observation = self.map_manager.build_maps(special_map, agents_d_coord, agents_coord)
        return observation, reward, done

    def record_step_data(self, observation, reward, done):
        mm = self.map_manager
        map_view = mm.transform_observation_to_view(self.prev_observation)
        agent_coord = copy.deepcopy(self.agents_coord)
        agent_direction = copy.deepcopy(self.agents_direction)
        if self.record_data_prop.only_coord:
            data = dict({"agent_coord": agent_coord})
            if self.step_cnt == 1:
                data["no_agents_map"] = mm.transform_map_to_rgb(mm.prev_build_base_map)
        else:
            data = dict({"map_view": map_view,
                         "agent_coord": agent_coord,
                         "agent_direction": agent_direction,
                         "no_agents_map": mm.transform_map_to_rgb(mm.prev_build_base_map),
                         "done": done})

        self.record_data["step_data"].append(data)

    def init_ep_record_data(self):
        mm = self.map_manager
        data = dict({
            "type": self.record_data_prop.only_coord,
            "episode": self.ep_cnt, "step_data": [],
            "agents_colors": list(map(mm.transform_gray_to_rgb, mm.agents_colors)),
            "base_static_map": mm.transform_map_to_rgb(mm.base_static_map),
            "base_special_map": mm.transform_map_to_rgb(mm.base_special_map),
            "no_agents": self.no_agents,
            "map_size": self.map_size
        })
        self.record_data = data

    def _reset(self):
        self.static_map, self.special_map = self.map_manager.reset_map()

        if self.reward_distance:
            self.reward_distance_ref = self.special_map[REWARD_ID].nonzero()[0]

        # Position agents only on empty spaces
        check_occupancy = self.static_map
        _, self.agents_direction, self.agents_direction_coord, self.agents_coord = \
            self.map_manager.init_agents(check_occupancy, self.agent_init_pos,
                                         self.agents_binary_map, self.agents_collide)

        if self.reward_init_pos:
            self.el_init = self.map_manager.init_elements(self.special_map, self.static_map,
                                                          self.reward_init_pos)
        else:
            self.el_init = dict()

        self.el_rewards_respawn = [deque() for _ in range(len(self.rewards_value))]

        return self.map_manager.build_maps(self.special_map, self.agents_direction_coord,
                                           self.agents_coord)

    def _render(self):
        view = self.map_manager.transform_observation_to_view(self.prev_observation)
        cv2.imshow("Game", view)
        return view

    @property
    def action_space(self):
        return self.action_space_

    @property
    def observation_space(self):
        return self.observation_space_
