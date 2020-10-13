import numpy as np

h_message = '''
>>> You can use "min" or "max" to control the minimum and maximum values.
>>> constant
y = c
>>> linear
y = start_v + x * slope
>>> cycle_linear
y = start_v +  slope * (x - LAST_CKPT) / (NEXT_CKPT - LAST_CKPT)
>>> exp / exponential
y = start_v * power ** (x / interval)
>>> cycle_exp
y = start_v * power ** ((x - LAST_CKPT) / (NEXT_CKPT - LAST_CKPT))
>>> jump
y = start_v * power ** (max(idx - min_jump_pt, 0) // jump_freq)
>>> cycle_jump
y = start_v * power ** (#DROP_POINT passed in drop_point_list since the latest CKPT in ckpt_list)
>>> cos_cycle_jump
y = (cos ( (max(0, x - min_jump_pt) %% cycle_freq) / cycle_freq * Pi) + 1) * (up_v - low_v) / 2 + low_v
>>> cycle_cos
y = 1/2 * (eps_max - eps_min) * [1 - cos((x - LAST_CKPT) / (NEXT_CKPT - LAST_CKPT) * pi)] + eps_min
'''


def continuous_seq(*args, **kwargs):
    '''
    >>> return a float to float mapping
    '''
    name = kwargs['name']
    max_v = kwargs['max'] if 'max' in kwargs else np.inf
    min_v = kwargs['min'] if 'min' in kwargs else -np.inf

    if name.lower() in ['h', 'help']:
        print(h_message)
        exit(0)
    elif name.lower() in ['constant',]:
        start_v = float(kwargs['start_v'])
        return lambda x: np.clip(start_v, a_min = min_v, a_max = max_v)
    elif name.lower() in ['linear',]:
        start_v = float(kwargs['start_v'])
        slope = float(kwargs['slope'])
        return lambda x: np.clip(start_v + x * slope, a_min = min_v, a_max = max_v)
    elif name.lower() in ['cycle_linear',]:
        start_v = float(kwargs['start_v'])
        slope = float(kwargs['slope'])
        ckpt_list = list(map(float, kwargs['ckpt_list'].split(':')))
        ckpt_list = list(sorted(ckpt_list))
        if ckpt_list[0] > 0.:
            ckpt_list = [0.,] + ckpt_list
        def local_linear_warmup(start_v, slope, ckpt_list, min_v, max_v, x):
            if x > ckpt_list[-1]:
                return np.clip(start_v + slope, a_min = min_v, a_max = max_v)
            if x <= ckpt_list[0]:
                return np.clip(start_v, a_min = min_v, a_max = max_v)
            for l_ckpt, r_ckpt in zip(ckpt_list[:-1], ckpt_list[1:]):
                if l_ckpt <= x and x < r_ckpt:
                    ratio = (x - l_ckpt) / (r_ckpt - l_ckpt)
            return np.clip(start_v + ratio * slope, a_min = min_v, a_max = max_v)
        return lambda x: local_linear_warmup(start_v, slope, ckpt_list, min_v, max_v, x)
    elif name.lower() in ['exp, exponential',]:
        start_v = float(kwargs['start_v'])
        power = float(kwargs['power'])
        interval = int(kwargs['interval']) if 'interval' in kwargs else 1
        return lambda x: np.clip(start_v * power ** (x / float(interval)), a_min = min_v, a_max = max_v)
    elif name.lower() in ['cycle_exp',]:
        start_v = float(kwargs['start_v'])
        power = float(kwargs['power'])
        ckpt_list = list(map(float, kwargs['ckpt_list'].split(':')))
        ckpt_list = list(sorted(ckpt_list))
        if ckpt_list[0] > 0.:
            ckpt_list = [0, ] + ckpt_list
        def local_cycle_exp(start_v, power, min_v, max_v, x):
            if x > ckpt_list[-1]:
                ratio = 1.
            elif x <= ckpt_list[0]:
                ratio = 0.
            else:
                for l_ckpt, r_ckpt in zip(ckpt_list[:-1], ckpt_list[1:]):
                    if l_ckpt <= x and x < r_ckpt:
                        ratio = (x - l_ckpt) / (r_ckpt - l_ckpt)
            return np.clip(start_v * power ** ratio, a_min = min_v, a_max = max_v)
        return lambda x: local_cycle_exp(start_v, power, min_v, max_v, x)
    elif name.lower() in ['jump',]:
        start_v = float(kwargs['start_v'])
        power = float(kwargs['power'])
        min_jump_pt = int(kwargs['min_jump_pt'])
        jump_freq = int(kwargs['jump_freq'])
        return lambda x: np.clip(start_v * power ** (max(x - min_jump_pt + jump_freq, 0) // jump_freq), a_min = min_v, a_max = max_v)
    elif name.lower() in ['cycle_jump',]:
        start_v = float(kwargs['start_v'])
        power = float(kwargs['power'])
        drop_point_list = list(map(float, kwargs['drop_point_list'].split(':')))
        ckpt_list = list(map(float, kwargs['ckpt_list'].split(':')))
        drop_point_list = list(sorted(drop_point_list))
        ckpt_list = list(sorted(ckpt_list))
        def local_cycle_jump(start_v, power, drop_point_list, ckpt_list, max_v, min_v, x):
            assert x >= ckpt_list[0] and x < ckpt_list[-1], 'x = %f should be between %.4f and %.4f as defined' % (x, ckpt_list[0], ckpt_list[-1])
            for l_ckpt, r_ckpt in zip(ckpt_list[:-1], ckpt_list[1:]):
                if l_ckpt <= x and x < r_ckpt:
                    ratio = (x - l_ckpt) / (r_ckpt - l_ckpt)
            value = start_v
            for drop_point in drop_point_list:
                if ratio >= drop_point:
                    value *= power
            return np.clip(value, a_min = min_v, a_max = max_v)
        return lambda x: local_cycle_jump(start_v, power, drop_point_list, ckpt_list, max_v, min_v, x)
    elif name.lower() in ['cos_cycle_jump',]:
        low_v = float(kwargs['low_v'])
        up_v = float(kwargs['up_v'])
        cycle_freq = float(kwargs['cycle_freq'])
        min_jump_pt = float(kwargs['min_jump_pt'])
        return lambda x: np.clip((np.cos((max(x - min_jump_pt, 0) % cycle_freq) / cycle_freq * np.pi) + 1) * (up_v - low_v) / 2. + low_v, a_min = min_v, a_max = max_v)
    elif name.lower() in ['cycle_cos',]:
        eps_min = float(kwargs['eps_min'])
        eps_max = float(kwargs['eps_max'])
        ckpt_list = list(map(float, kwargs['ckpt_list'].split(':')))
        ckpt_list = list(sorted(ckpt_list))
        if ckpt_list[0] > 0.:
            ckpt_list = [0.,] + ckpt_list
        def local_cos_eps_warmup(eps_min, eps_max, ckpt_list, min_v, max_v, x):
            if x > ckpt_list[-1]:
                return np.clip(eps_max, a_min = min_v, a_max = max_v)
            if x <= ckpt_list[0]:
                return np.clip(eps_min, a_min = min_v, a_max = max_v)
            for l_ckpt, r_ckpt in zip(ckpt_list[:-1], ckpt_list[1:]):
                if l_ckpt <= x and x < r_ckpt:
                    ratio = (x - l_ckpt) / (r_ckpt - l_ckpt)
            return np.clip(eps_min + 0.5 * (eps_max - eps_min) * (1 - np.cos(ratio * np.pi)), a_min = min_v, a_max = max_v)
        return lambda x: local_cos_eps_warmup(eps_min, eps_max, ckpt_list, min_v, max_v, x)
    else:
        raise ValueError('Unrecognized name: %s'%name)

def discrete_seq(*args, **kwargs):
    '''
    >>> return a list of values
    '''
    name = kwargs['name']
    func = continuous_seq(*args, **kwargs)

    pt_num = int(kwargs['pt_num'])
    return [func(idx) for idx in range(pt_num)]

