import subprocess
import operator
import re

from os import listdir
from os.path import isfile, join, isdir, splitext

def sort_dict_by_value(dictionary, reverse=False):
    return sorted(dictionary.items(), key=operator.itemgetter(1), reverse=reverse)

def get_files_paths_for_folder(path):
    files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    return files

def get_folder_paths_for_folder(path):
    files = [join(path, f) for f in listdir(path) if isdir(join(path, f))]
    return files

def wget(url, path):
    strCMD = 'wget {0} -P {1}'.format(url, path)
    return execute(strCMD)

def unzip(input_path, output_path=None):
    if output_path is None:
        strCMD = 'unzip {0}'.format(input_path)
    else:
        strCMD = 'unzip {0} -d {1}'.format(input_path, output_path)
    return execute(strCMD)

def execute(strCMD):
    try:
        return subprocess.check_output(strCMD, shell=True, universal_newlines=True)
    except:
        return None

def execute_and_return(strCMD):
    proc = subprocess.Popen(shlex.split(strCMD), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    out, err = out.decode("UTF-8").strip(), err.decode("UTF-8").strip()
    if err != '':
        return err
    else:
        return out

def get_files_by_filetype(path):
    filetype2paths = {}
    folders = get_folder_paths_for_folder(path)
    for folder in folders:
        files = get_files_paths_for_folder(folder)
        for f in files:
            _, ext = splitext(f)
            if ext not in filetype2paths: filetype2paths[ext] = []
            filetype2paths[ext].append(f)
    return filetype2paths


def get_active_window_coordinates():
    data = execute('xdotool getactivewindow getwindowgeometry')
    data = data.split('\n')
    data = data[2][12:]
    x, y = data.split('x')
    return int(x),int(y)


def get_active_window_name():
    return execute('xdotool getactivewindow getwindowname')

def get_active_window_path():
    pid = execute('xdotool getactivewindow getwindowpid').strip()
    cmd = 'ps -ef | grep {0}'.format(pid)
    results = execute(cmd).split('\n')
    for r in results:
        r = re.sub('\s+', ' ', r)
        values = r.split(' ')
        if len(values) < 2: continue
        if values[1] == pid:
            path = values[7]
    return path

