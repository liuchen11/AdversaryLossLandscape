import random

global_color_map = {}

def get_color(color_idx):
    if color_idx in global_color_map:
        return global_color_map[color_idx]

    base_color = ['b', 'y', 'c', 'm', 'g', 'r']
    if color_idx < 6:
        global_color_map[color_idx] = base_color[color_idx]
        return base_color[color_idx]
    else:
        dex = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']
        ret_color = '#'
        for _ in range(6):
            token_idx = random.randint(0,15)
            ret_color += dex[token_idx]
        global_color_map[color_idx] = ret_color
        return ret_color