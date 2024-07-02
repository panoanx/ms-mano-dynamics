import numpy as np
import json
import os
from scipy.spatial.transform import Rotation as R


def load_mim_data(
    path,
    mim_range=None,
    joint_indexes=range(1, 16),
    form='plain',
    interval=1,
    reverse=False
):
    format_list = ["HO3D", "plain", "json", "arctic"]
    assert form in format_list, f"type must be in {format_list}"
    data = {}

    if isinstance(path, str) and os.path.isdir(path):
        path_list = [os.path.join(path, file) for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
    elif isinstance(path, str) and os.path.isfile(path):
        path_list = [path]
    elif isinstance(path, list):
        path_list = path
    else:
        raise ValueError("Path must be a file, a directory or a list of paths.")

    for file_path in path_list:
        if form == 'HO3D':
            data_temp = load_HO3D(file_path, mim_range, joint_indexes)
        elif form == 'plain':
            data_temp = load_plain(file_path, mim_range)
        elif form == 'json':
            data_temp = load_rot_json(file_path, mim_range)
        elif form == 'arctic':
            data_temp = load_arctic(file_path, mim_range)

        if data == {}:
            data = data_temp
        else:
            for key in data_temp:
                if key in data:
                    if type(data_temp[key]) == np.ndarray:
                        data[key] = np.concatenate((data[key], data_temp[key]))
                    elif type(data_temp[key]) == list:
                        data[key] = data[key] + data_temp[key]
                    elif data_temp[key] == data[key]:
                        pass
                    elif key == 'len':
                        data[key] += data_temp[key]
                    else:
                        raise ValueError(f"Key {key} has different values in different files.")
                else:
                    raise ValueError(f"Key {key} does not exist in the first file.")

    selected_range = np.arange(data['len'])[::interval]
    for key in data:
        data[key] = data[key][selected_range] if type(data[key]) == np.ndarray else data[key]
    data['len'] = len(selected_range)
    
    # TODO vel reverse handling
    if reverse:
        for key in data:
            if key in ['rot', 'pos', 'jb']:
                data[key] = np.concatenate((data[key], data[key][::-1]))
            elif key == 'len':
                data[key] *= 2

    return data


def load_arctic(path, mim_range):
    mimicker = np.load(path, allow_pickle=True).item()
    mim_rot = mimicker['right_hand_pose'].reshape(-1, 15, 3) * (1, -1, -1)
    mim_shape = mim_rot.shape
    if mim_range is not None:
        mim_rot = mim_rot[mim_range]
    return {
        'rot': mim_rot,
        'len': len(mim_rot)
    }


def load_HO3D(path, mim_range, joint_indexes):
    mimicker = np.load(path, allow_pickle=True)
    mim_rot = np.array([mimicker[i]['handPose'] for i in range(len(mimicker))]) * 180 / np.pi
    if mim_range is not None:
        mim_rot = mim_rot[mim_range]
    mim_rot = (mim_rot.reshape(-1, 16, 3) * (1, -1, -1))[:, joint_indexes, :]
    return {
        'rot': mim_rot,
        'len': len(mim_rot)
    }


def load_rot_json(path, mim_range):
    j = json.load(open(path, 'r'))
    rot = np.array(j)
    if mim_range is not None:
        rot = rot[mim_range]
    return {
        'rot': rot,
        'len': len(rot)
    }


def load_plain(path, mim_range):
    rot_pos = np.load(path, allow_pickle=True).tolist()
    rot = rot_pos['rot']
    pos = rot_pos['pos']
    jb = rot_pos['jb']
    # vel = rot_pos['vel']
    dt = rot_pos['dt']
    if mim_range is not None:
        rot = rot[mim_range]
        pos = pos[mim_range]
        jb = jb[mim_range]
        vel = vel[mim_range]
    rot_pos = {
        'rot': rot,
        'pos': pos,
        'jb': jb,
        'len': len(rot),
        # 'vel': vel,
        'dt': dt
    }
    return rot_pos


if __name__ == "__main__":
    load_mim_data("../glove/processed/")
    pass
