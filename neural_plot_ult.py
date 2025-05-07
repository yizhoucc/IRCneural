
from datetime import datetime, date
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from IPython.display import clear_output
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter1d
import pickle
import torch
from numpy import pi
from pathlib import Path
import seaborn as sns
import matplotlib.patches as mpatches
import time
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.decomposition import PCA
from sklearn.metrics import explained_variance_score
from scipy.stats import pearsonr
import neo
from scipy.signal import medfilt
from scipy.stats import norm

# plot setting ------------------------
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['figure.facecolor'] = 'none' 
plt.rcParams['axes.facecolor'] = 'none' 
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({
    'font.size': 15, 'lines.linewidth': 2,
    'xtick.labelsize': 13, 'ytick.labelsize': 13,
    'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.dpi': 1200,
})
 
# colors
state_color='blue'
belief_color='purple'
eye_color='red'
cmap_eye = mcolors.LinearSegmentedColormap.from_list('eye_cmap', ['white', eye_color])
cmap_state = mcolors.LinearSegmentedColormap.from_list('state_cmap', ['white', state_color])
cmap_belief = mcolors.LinearSegmentedColormap.from_list('belief_cmap', ['white', belief_color])


# notify ------------------------
import requests
import configparser
config = configparser.ConfigParser()
config.read_file(open('privateconfig'))
token=config['Notification']['token']

def notify(msg='plots ready', group='lab',title='plot'):
    notification="https://api.day.app/{}/{}/{}?group={}".format(token,title, msg, group)
    requests.get(notification)


def compute_scatter_density(x,y):
    '''return density of scatters.'''
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    return z


def solidcbar(cbar):
    cbar.solids.set_edgecolor("face")
    cbar.solids.set_alpha(1) 

def plot_best_fit(x,y, ax, color='black'):
    slope, intercept = np.polyfit(x,y, 1)
    best_fit_line = slope * x + intercept
    ax.plot(x, best_fit_line, color=color)
    return slope


def quicklegend(names, colors,ax):
    '''generate legend given list of labels and list of colors'''
    legend_handles = [mpatches.Patch(color=color, label=name) for name, color in zip(names, colors)]
    ax.legend(handles=legend_handles, loc='upper right')


def quickallspine(ax):
    '''remove all spines'''
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


def plot_gradient_line(ax, x, y, cmap, linewidth=3):
    '''plot x y data as a gradient color line. usage is same as plt.plot(x,y)'''
    norm = plt.Normalize(0, len(x))
    for i in range(len(x) - 1):
        ax.plot(x[i:i+2], y[i:i+2], color=cmap(norm(i)), alpha= np.clip(i /len(x)+0.6, 0,1), linewidth=linewidth)

# from cebra embedding plots -------------------------------
def plot_embedding_contrast(ax, embedding, label, gray=False, beh_idx=(0, 1), idx_order=(0, 1, 2)):
    '''plot the embeeding and color by the difference between the beh_idx task varaibles'''
    if not gray:
        r_cmap = 'cool'
        r_c = label[:, beh_idx[0]]-label[:, beh_idx[1]]
    else:
        r_cmap = None
        r_c = 'gray'
    idx1, idx2, idx3 = idx_order
    r = ax.scatter(embedding[:, idx1],
                   embedding[:, idx2],
                   embedding[:, idx3],
                   c=r_c,
                   #    vmin=0,
                   #    vmax=1,
                   cmap=r_cmap, s=0.5)
    plt.colorbar(r)
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.set_ticks([-1, 0, 1])
    ax.yaxis.set_ticks([-1, 0, 1])
    ax.zaxis.set_ticks([-1, 0, 1])
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')
    ax.set_zlabel('dim 3')
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.axis('equal')
    return ax


def plot_embedding(ax, embedding, label, gray=False, beh_idx=0, idx_order=(0, 1, 2)):
    '''cebra 3d embedding'''
    if not gray:
        r_cmap = 'cool'
        r_c = label[:, beh_idx]
    else:
        r_cmap = None
        r_c = 'gray'
    idx1, idx2, idx3 = idx_order
    r = ax.scatter(embedding[:, idx1],
                   embedding[:, idx2],
                   embedding[:, idx3],
                   c=r_c,
                   vmin=0,
                   vmax=1,
                   cmap=r_cmap, s=0.5)
    plt.colorbar(r)
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.set_ticks([-1, 0, 1])
    ax.yaxis.set_ticks([-1, 0, 1])
    ax.zaxis.set_ticks([-1, 0, 1])
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')
    ax.set_zlabel('dim 3')
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.axis('equal')
    return ax


def plot_embedding2d(ax, embedding, label, gray=False, beh_idx=0, idx_order=(0, 1)):
    '''cebra 2d embedding'''
    if not gray:
        r_cmap = 'cool'
        r_c = label[:, beh_idx]
    else:
        r_cmap = None
        r_c = 'gray'
    idx1, idx2 = idx_order
    r = ax.scatter(embedding[:, idx1],
                   embedding[:, idx2],
                   c=r_c,
                   #    vmin=0,
                   #    vmax=1,
                   cmap=r_cmap, s=0.5)
    plt.colorbar(r)
    ax.grid(False)
    ax.xaxis.set_ticks([-1, 0, 1])
    ax.yaxis.set_ticks([-1, 0, 1])
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')
    ax.axis('equal')
    return ax


def project_and_unfold(x, y):
    # Step 1: Calculate distance of each point from the origin
    distance = np.sqrt(x**2 + y**2)

    # Step 2: Find nearest point on the circle
    radius = 1
    x_projected = x / distance
    y_projected = y / distance

    # Step 3: Unfold the circle onto a line
    angle = np.arctan2(y_projected, x_projected)

    x_unfolded = angle

    y_unfolded = distance - radius

    return x_unfolded, y_unfolded


def plot_embedding2d_unflold_line(ax, embedding, label, gray=False, beh_idx=0, idx_order=(0, 1)):
    '''convert 3d to 2d by mapping dots to ring'''
    if not gray:
        r_cmap = 'cool'
        r_c = label[:, beh_idx]
    else:
        r_cmap = None
        r_c = 'gray'
    idx1, idx2 = idx_order
    x, y = embedding[:, idx1], embedding[:, idx2]
    x_unfolded, y_unfolded = project_and_unfold(x, y)

    r = ax.scatter(x_unfolded,
                   y_unfolded,
                   c=r_c,
                   #    vmin=0,
                   #    vmax=1,
                   cmap=r_cmap, s=0.5)
    plt.colorbar(r)
    ax.grid(False)
    ax.xaxis.set_ticks([-1, 0, 1])
    ax.yaxis.set_ticks([-1, 0, 1])
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')
    ax.axis('equal')
    return ax


def plot_embedding2d_unflold(ax, embedding, label, gray=False, beh_idx=0, idx_order=(0, 1)):
    '''convert 3d to 2d by mapping dots to ring, ignoring the mapping'''
    if not gray:
        r_cmap = 'cool'
        r_c = label[:, beh_idx]
    else:
        r_cmap = None
        r_c = 'gray'
    idx1, idx2 = idx_order
    x, y = embedding[:, idx1], embedding[:, idx2]
    x_unfolded, y_unfolded = project_and_unfold(x, y)

    r = ax.scatter(x_unfolded,
                   r_c,
                   c=r_c,
                   #    vmin=0,
                   #    vmax=1,
                   cmap=r_cmap, s=0.5)

    ax.grid(False)
    ax.xaxis.set_ticks([-1, 0, 1])
    ax.yaxis.set_ticks([-1, 0, 1])
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')
    ax.axis('equal')
    corr_coef = np.corrcoef(x_unfolded.squeeze(), r_c.squeeze())[0, 1].item()
    ax.set_title(f'corr = {corr_coef:.2f}')
    return ax


def plot_embedding2d_contrast(ax, embedding, label, gray=False, beh_idx=(0, 1), idx_order=(0, 1), contrast=lambda x, y: x - y, vmin=None, vmax=None):
    '''plot the embeeding and color by the difference between the beh_idx task varaibles'''
    if not gray:
        r_cmap = 'bwr'
        # r_c = label[:, beh_idx[0]] - label[:, beh_idx[1]]
        r_c = contrast(label[:, beh_idx[0]], label[:, beh_idx[1]])
    else:
        r_cmap = None
        r_c = 'gray'
    idx1, idx2 = idx_order
    if not vmin and not vmax:
        norm = mcolors.CenteredNorm(0)
        r = ax.scatter(embedding[:, idx1],
                       embedding[:, idx2],
                       c=r_c,
                       cmap=r_cmap, s=0.5,
                       norm=norm)
    else:
        vmin = -1*max(-vmin, vmax)
        vmax = max(-vmin, vmax)
        r = ax.scatter(embedding[:, idx1],
                       embedding[:, idx2],
                       c=r_c,
                       vmin=vmin,
                       vmax=vmax,
                       cmap=r_cmap, s=0.5)

    plt.colorbar(r)
    ax.grid(False)
    ax.xaxis.set_ticks([-1, 0, 1])
    ax.yaxis.set_ticks([-1, 0, 1])
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')
    ax.axis('equal')
    return ax


# from ruiyi neural eye analysis plot -------------------------------
monkey_height = 10
DT = 0.006  # DT for raw data


def distance(dx, dy):
    '''simple 2d norm distantce given dx and dy'''
    return (dx**2+dy**2)**0.5


def set_violin_plot(vp, facecolor, edgecolor, linewidth=1, alpha=1, ls='-', hatch=r''):
    plt.setp(vp['bodies'], facecolor=facecolor, edgecolor=edgecolor,
             linewidth=linewidth, alpha=alpha, ls=ls, hatch=hatch)
    plt.setp(vp['cmins'], facecolor=facecolor, edgecolor=edgecolor,
             linewidth=linewidth, alpha=alpha)
    plt.setp(vp['cmaxes'], facecolor=facecolor, edgecolor=edgecolor,
             linewidth=linewidth, alpha=alpha)
    plt.setp(vp['cbars'], facecolor=facecolor, edgecolor=edgecolor,
             linewidth=linewidth, alpha=alpha)

    linecolor = 'k' if facecolor == 'None' else 'snow'
    if 'cmedians' in vp:
        plt.setp(vp['cmedians'], facecolor=linecolor, edgecolor=linecolor,
                 linewidth=linewidth, alpha=alpha)
    if 'cmeans' in vp:
        plt.setp(vp['cmeans'], facecolor=linecolor, edgecolor=linecolor,
                 linewidth=linewidth, alpha=alpha)


def downsample(data, bin_size=17):
    num_bin = data.shape[0] // bin_size
    data_ = data[:bin_size * num_bin]
    data_ = data_.reshape(num_bin, bin_size, data.shape[-1])
    data_ = np.nanmean(data_, axis=1)
    return data_

def downsample_variance(data, bin_size=17):
    '''return the downsampled data variance'''
    num_bin = data.shape[0] // bin_size
    data_ = data[:bin_size * num_bin] # time, unit
    data_ = data_.reshape(num_bin, bin_size, data.shape[-1]) # numbins, binsize, unit
    data_ = np.nanvar(data_, axis=1) # apply function within each bin
    return data_

def convert_egolocation_to_angle(dx, dy, monkey_height=10):
    """
    Convert relative gaze position (dx, dy) in egocentric coordinates to 
    horizontal and vertical angles in degrees.

    dx: target x relative to body (lateral)
    dy: target y relative to body (forward)
    monkey_height: vertical distance from eye to ground plane

    Returns:
        hor_theta: horizontal angle (degrees)
        ver_theta: vertical angle (degrees)
    """
    hor_theta = -np.rad2deg(np.arctan2(-dx, np.sqrt(dy**2 + monkey_height**2))).reshape(-1, 1)
    ver_theta = -np.rad2deg(np.arctan2(monkey_height, dy)).reshape(-1, 1)
    return hor_theta, ver_theta

def convert_location_to_angle_(gaze_r, gaze_x, gaze_y, body_theta, body_x, body_y, hor_theta_eye, ver_theta_eye, monkey_height=monkey_height, DT=DT, remove_pre=True, remove_post=True):
    '''
        convert the world overhead view location of the 'gaze' location to eye coord. 

        gaze location, the target
        gaze_r, relative distance
        gaze_x, gaze location x
        gaze_y,

        body_theta, heading direction
        body_x, monkey location x
        body_y, 

        hor_theta_eye, actual eye location in eye coord. used here to remove pre saccade (when monkey hasnt seen the target yet)
        ver_theta_eye
    '''

    # hor_theta = -np.rad2deg(np.arctan2(-(gaze_x - body_x), gaze_y - body_y) - (body_theta-np.deg2rad(90))).reshape(-1, 1)
    hor_theta = -np.rad2deg(np.arctan2(-(gaze_x - body_x), np.sqrt((gaze_y - body_y)**2 + monkey_height**2))
                            - (body_theta-np.deg2rad(90))).reshape(-1, 1)

    k = -1 / np.tan(body_theta)
    b = body_y - k * body_x
    gaze_r_sign = (k * gaze_x + b < gaze_y).astype(int)
    gaze_r_sign[gaze_r_sign == 0] = -1
    # ver_theta = -np.rad2deg(np.arctan2(monkey_height,
    #                         gaze_r_sign * gaze_r)).reshape(-1, 1)
    ver_theta = -np.rad2deg(np.arctan2(monkey_height, (gaze_y - body_y))).reshape(-1, 1) # more stable

    # remove overshooting
    if remove_post:
        overshoot_idx = np.where(((gaze_x - body_x) * gaze_x < 0) | (gaze_y < body_y)
                                 # | (abs(hor_theta.flatten()) > 60)
                                 )[0]
        if overshoot_idx.size > 0:
            hor_theta[overshoot_idx[0]:] = np.nan

        overshoot_idx = np.where((gaze_r_sign < 0)
                                 # | (abs(ver_theta.flatten()) > 60)
                                 )[0]
        if overshoot_idx.size > 0:
            ver_theta[overshoot_idx[0]:] = np.nan

    # detect saccade
    if remove_pre:
        if hor_theta_eye.size > 2:
            saccade = np.sqrt((np.gradient(hor_theta_eye) / DT)**2 +
                              (np.gradient(ver_theta_eye) / DT)**2)
            saccade_start_idx = np.where(saccade > 100)[0]
            saccade_start_idx = saccade_start_idx[0] + \
                16 if saccade_start_idx.size > 0 else None

            hor_theta[:saccade_start_idx] = np.nan
            ver_theta[:saccade_start_idx] = np.nan

    return hor_theta, ver_theta

def convert_location_to_angle(gaze_r, gaze_x, gaze_y, body_theta, body_x, body_y, hor_theta_eye, ver_theta_eye,
                              is_stop=False,DT=DT):
    #hor_theta = np.rad2deg(np.arctan2(-(gaze_x - body_x), gaze_y - body_y) - (body_theta-np.deg2rad(90))).reshape(-1, 1) 
    #hor_theta = np.rad2deg(np.arctan2(-(gaze_x - body_x), np.sqrt((gaze_y - body_y)**2 + monkey_height**2))
    #                        - (body_theta-np.deg2rad(90))).reshape(-1, 1)
    
    hor_theta = np.rad2deg(np.arctan2(-(gaze_x - body_x), gaze_y - body_y)
                           - (body_theta-np.deg2rad(90))).reshape(-1, 1) 
    if is_stop and hor_theta.size > 1:
        hor_theta[-1] = hor_theta[-2]
    overshoot_idx = np.where(((gaze_x - body_x) * gaze_x < 0) | (gaze_y < body_y)
                             #| (abs(hor_theta.flatten()) > 60)
                            )[0]
    #if overshoot_idx.size > 0:
    #    hor_theta[overshoot_idx[0]:] = np.nan

    k = -1 / np.tan(body_theta); b = body_y - k * body_x
    gaze_r_sign = (k * gaze_x + b < gaze_y).astype(int)
    gaze_r_sign[gaze_r_sign == 0] = -1
    ver_theta = -np.rad2deg(np.arctan2(monkey_height, gaze_r_sign * gaze_r)).reshape(-1, 1)
    overshoot_idx = np.where((gaze_r_sign < 0)
                             #| (abs(ver_theta.flatten()) > 60)
                            )[0]
    #if overshoot_idx.size > 0:
    #    ver_theta[overshoot_idx[0]:] = np.nan
        
        
    hor_theta_withpre = hor_theta.copy()
    ver_theta_withpre = ver_theta.copy() 
    # detect saccade
    if hor_theta_eye.size > 2:
        saccade = np.sqrt((np.gradient(hor_theta_eye) / DT)**2 + 
                          (np.gradient(ver_theta_eye) / DT)**2)
        saccade_start_idx = np.where(saccade > 100)[0]
        saccade_start_idx = saccade_start_idx[0] + 16 if saccade_start_idx.size > 0 else None

        # hor_theta[:saccade_start_idx] = np.nan
        # ver_theta[:saccade_start_idx] = np.nan
        
    return -hor_theta, ver_theta

def compute_error(data1, data2, mask):
    # data1 = data1[~mask]; data2 = data2[~mask]
    # corr = np.corrcoef(data1, data2)
    error = abs(data1 - data2)

    rng = np.random.default_rng(seed=0)
    data1_ = data1.copy()
    data2_ = data2.copy()
    rng.shuffle(data1_)
    rng.shuffle(data2_)
    error_shuffle = abs(data1_ - data2_)
    return error


# ---------------------
def mytime():
    '''get date as str'''
    current_date_time = datetime.now()
    current_date = current_date_time.date()
    formatted_date = current_date.strftime("%m%d")

    return formatted_date


def normalize_01(data, low=5, high=95):
    '''normalize the data vector or matrix to 0-1 range
    use percentile to avoid outliers.'''
    themin = np.percentile(data[~np.isnan(data)], low)
    themax = np.percentile(data[~np.isnan(data)], high)
    res = (data - themin) / (themax - themin)
    res[np.isnan(data)] = np.nan
    res = np.clip(res, 0, 1)
    return res


def normalize_z(data):
    '''normalize the data vector or matrix to have mean of 0 std of 1'''
    nanmask = ~np.isnan(data)
    validdata = data[nanmask]
    mean = sum(data[nanmask]) / len(data[nanmask])
    variance = sum((x - mean) ** 2 for x in data[nanmask]) / len(data[nanmask])
    std_deviation = variance ** 0.5
    normalized_data = [
        (x - mean) / std_deviation if x else np.nan for x in data]
    return normalized_data


def state_step2(px, py, heading, v, w, a, pro_gainv=1, pro_gainw=1, dt=0.006, userad=False):
    ''' run the task and get the state values.'''
    if not userad:
        w = w/180*pi

    # overall, x'=Ax+Bu+noise. here, noise=0

    # use current v and w to update x y and heading
    # (x'=Ax) part

    if v <= 0:
        pass
    elif w == 0:
        px = px + v*dt * np.cos(heading)
        py = py + v*dt * np.sin(heading)
    else:
        px = px-np.sin(heading)*(v/w-(v*np.cos(w*dt)/w)) + \
            np.cos(heading)*((v*np.sin(w*dt)/w))
        py = py+np.cos(heading)*(v/w-(v*np.cos(w*dt)/w)) + \
            np.sin(heading)*((v*np.sin(w*dt)/w))
    heading = heading + w*dt
    heading = np.clip(heading, -pi, pi)

    # apply the new control to state
    # (Bu) part
    v = pro_gainv * a[0]
    w = pro_gainw * a[1]
    return px, py, heading, v, w



# from inverse functions --------------
def process_inv(res, removegr=True, ci=5, ind=-1, usingbest=False):
    # get final theta and cov
    if type(res) == str:
        res = Path(res)
    print(res)
    with open(res, 'rb') as f:
        log = pickle.load(f)
    if ind >= len(log):
        ind = -1
    elif ind <= -len(log):
        ind = 1
    if usingbest:
        ind = np.argmin([np.mean([l[1] for l in eachlog[2]])
                        for eachlog in log[:ind]])
    print('using ind: ', ind, 'final logll : ',
          np.mean([l[1] for l in log[ind][2]]))
    finalcov = torch.tensor(log[ind][0]._C).float()
    finaltheta = torch.tensor(log[ind][0]._mean).view(-1, 1)
    theta = torch.cat([finaltheta[:6], finaltheta[-4:]])
    cov = finalcov[torch.arange(finalcov.size(0)) != 6]
    cov = cov[:, torch.arange(cov.size(1)) != 6]
    cirange = get_ci(log, low=ci, high=100-ci, ind=ind).astype('float32')
    if removegr:
        return theta, cov, np.delete(cirange, (6), axis=1)
    return finaltheta, finalcov, cirange


def monkeyloss_(agent=None,
                actions=None,
                tasks=None,
                phi=None,
                theta=None,
                env=None,
                num_iteration=1,
                states=None,
                samples=1,
                gpu=False,
                action_var=0.1,
                debug=False):
    if gpu:
        logPr = torch.zeros(1).cuda()[0]  # torch.FloatTensor([])
    else:
        logPr = torch.zeros(1)[0]  # torch.FloatTensor([])

    def _wrapped_call(ep, task):
        logPr_ep = torch.zeros(1).cuda()[0] if gpu else torch.zeros(1)[0]
        for sample_index in range(samples):
            mkactionep = actions[ep]
            if mkactionep == [] or mkactionep.shape[0] == 0:
                continue
            env.reset(theta=theta, phi=phi, goal_position=task,
                      vctrl=mkactionep[0][0], wctrl=mkactionep[0][1])
            numtime = len(mkactionep[1:])

            # compare mk data and agent actions
            # use a t and s t (treat st as st+1)
            for t, mk_action in enumerate(mkactionep[1:]):
                # agent's action
                action = agent(env.decision_info)
                # agent's obs, last step obs doesnt matter.
                if t < len(states[ep])-1:
                    if type(states[ep]) == list:
                        nextstate = states[ep][1:][t]
                    elif type(states[ep]) == torch.Tensor:
                        nextstate = states[ep][1:][t].view(-1, 1)
                    else:  # np array
                        nextstate = torch.tensor(states[ep])[1:][t].view(-1, 1)
                    obs = env.observations(nextstate)
                    # agent's belief
                    env.b, env.P = env.belief_step(
                        env.b, env.P, obs, torch.tensor(mk_action).view(1, -1))
                    previous_action = mk_action  # current action is prev action for next time
                    env.trial_timer += 1
                    env.decision_info = env.wrap_decision_info(
                        previous_action=torch.tensor(previous_action),
                        time=env.trial_timer)
                # loss
                action_loss = -1 * \
                    logll(torch.tensor(mk_action),
                          action, std=np.sqrt(action_var))
                obs_loss = -1*logll(error=env.obs_err(),
                                    std=theta[4:6].view(1, -1))
                logPr_ep = logPr_ep + action_loss.sum() + obs_loss.sum()
                del action_loss
                del obs_loss
            # if agent has not stop, compare agent action vs 0,0
            agentstop = torch.norm(action) < env.terminal_vel
            while not agentstop and env.trial_timer < 40:
                action = agent(env.decision_info)
                agentstop = torch.norm(action) < env.terminal_vel
                obs = (torch.tensor([0.5, pi/2])*action+env.obs_err()).t()
                env.b, env.P = env.belief_step(
                    env.b, env.P, obs, torch.tensor(action).view(1, -1))
                # previous_action=torch.tensor([0.,0.]) # current action is prev action for next time
                previous_action = action
                env.trial_timer += 1
                env.decision_info = env.wrap_decision_info(
                    previous_action=torch.tensor(previous_action),
                    time=env.trial_timer)
                # loss
                action_loss = -1 * \
                    logll(torch.tensor(torch.zeros(2)),
                          action, std=np.sqrt(action_var))
                obs_loss = -1*logll(error=env.obs_err(),
                                    std=theta[4:6].view(1, -1))
                logPr_ep = logPr_ep + action_loss.sum() + obs_loss.sum()
                del action_loss
                del obs_loss

        return logPr_ep/samples/env.trial_timer.item()

    tik = time.time()
    loglls = []
    for ep, task in enumerate(tasks):
        logPr_ep = _wrapped_call(ep, task)
        logPr += logPr_ep
        loglls.append(logPr_ep)
        del logPr_ep
    regularization = torch.sum(1/(theta+1e-4))
    # print('calculate loss time {:.0f}'.format(time.time()-tik))
    if debug:
        return loglls
    return logPr/len(tasks)+0.01*regularization


def logll(true=None, estimate=None, std=0.3, error=None, prob=False):
    # print(error)
    var = std**2
    if error is not None:  # use for point eval, obs
        def g(x): return 1/torch.sqrt(2*pi*torch.ones(1)) * \
            torch.exp(-0.5*x**2/var)
        z = 1/g(torch.zeros(1)+1e-8)
        loss = torch.log(g(error)*z+1e-8)
    else:  # use for distribution eval, aciton
        c = torch.abs(true-estimate)
        def gi(x): return -(torch.erf(x/torch.sqrt(torch.tensor([2]))/std)-1)/2
        loss = torch.log(gi(c)*2+1e-16)
    if prob:
        return torch.exp(loss)
    return loss

def run_trial(agent=None, env=None, given_action=None, given_state=None, action_noise=0.1, pert=None, stimdur=None):
    '''    
        # return epactions,epbliefs,epbcov,epstates
        # 10 a 10 s.
        # when both
        # use a1 and s2
        # at t1, use a1. results in s2
    '''

    def _collect():
        epactions.append(action)
        epbliefs.append(env.b)
        epbcov.append(env.P)
        epstates.append(env.s)
    # saves
    epactions, epbliefs, epbcov, epstates = [], [], [], []
    if given_action is not None:
        epactions.append(torch.tensor(given_action[0]))
    else:
        epactions.append(env.s[3:].view(-1))
    # print(env.s,epactions)
    with torch.no_grad():
        # if at least have something
        if given_action is not None and given_state is not None:  # have both
            t = 0
            while t < len(given_state):
                action = agent(env.decision_info)[0]
                _collect()
                # print(given_state)
                env.step(torch.tensor(given_action[t]).reshape(
                    1, -1), next_state=torch.tensor(given_state[t]).reshape(-1, 1))
                t += 1
                # print(env.s)
        elif given_state is not None:  # have states but no actions
            t = 0
            while t < len(given_state):
                action = agent(env.decision_info)[0]
                _collect()
                env.step(torch.tensor(action).reshape(1, -1),
                         next_state=given_state[t].view(-1, 1))
                t += 1

        elif given_action is not None:  # have actions but no states
            t = 0
            while t < len(given_action):
                action = agent(env.decision_info)[0]
                _collect()
                noise = torch.normal(torch.zeros(2), action_noise)
                _action = (action+noise).clamp(-1, 1)
                if pert is not None and int(env.trial_timer) < len(pert):
                    _action = (given_action[t]).reshape(
                        1, -1)+pert[int(env.trial_timer)]
                env.step(_action)
                t += 1

        else:  # nothing
            done = False
            t = 0
            while not done:
                action = agent(env.decision_info)[0]
                _collect()
                noise = torch.normal(torch.zeros(2), action_noise)
                _action = (action+noise).clamp(-1, 1)
                if pert is not None and int(env.trial_timer) < len(pert):
                    _action += pert[int(env.trial_timer)]
                if stimdur is not None:
                    _, _, done, _ = env.step(torch.tensor(_action).reshape(
                        1, -1), predictiononly=(t >= stimdur))
                else:
                    _, _, done, _ = env.step(
                        torch.tensor(_action).reshape(1, -1))
                t += 1
    return epactions, epbliefs, epbcov, epstates


def run_trials(agent, env, phi, theta, task, ntrials=10, stimdur=None, given_obs=None, action_noise=0.1, pert=None, return_belief=False, given_action=None, given_state=None):
    '''
    # sample ntrials for same task and return states and actions

    initialize the env, by (theta, phi, task)
    then call run single trial function
    till we have enough data to return
    '''
    states = []
    actions = []
    beliefs = []
    covs = []

    while len(states) < ntrials:
        if given_action is not None:
            env.debug=True
            env.reset(phi=phi, theta=theta, goal_position=task, pro_traj=None,
                        vctrl=given_action[0, 0], wctrl=given_action[0, 1], obs_traj=given_obs)
        else:
            print('given action', given_action)
            env.reset(phi=phi, theta=theta, goal_position=task,
                        pro_traj=None, vctrl=0., wctrl=0., obs_traj=given_obs)
            print('init s', env.s)

        epactions, epbliefs, epbcov, epstates = run_trial(
            agent, env, given_action=given_action, given_state=given_state, pert=pert, action_noise=action_noise, stimdur=stimdur,)
    
        states.append(torch.stack(epstates)[:, :, 0])
        actions.append(torch.stack(epactions))
        beliefs.append(torch.stack(epbliefs))
        covs.append((torch.stack(epbcov)))

    if return_belief:
        return states, actions, beliefs, covs
    else:
        return states, actions

# end from inverse functions --------------
def get_ci(log, low=5, high=95, threshold=2, ind=-1):
    res = [l[2] for l in log[:ind//threshold]]
    mean = log[ind][0]._mean
    allsamples = []
    for r in res:
        for point in r:
            allsamples.append([point[1], point[0]])
    allsamples.sort(key=lambda x: x[0])
    aroundsolution = allsamples[:ind//threshold]
    aroundsolution.sort(key=lambda x: x[0])
    alltheta = np.vstack([x[1] for x in aroundsolution])

    lower_ci = [np.percentile(alltheta[:, i], low)
                for i in range(alltheta.shape[1])]
    upper_ci = [np.percentile(alltheta[:, i], high)
                for i in range(alltheta.shape[1])]
    asymmetric_error = np.array(list(zip(lower_ci, upper_ci))).T
    res = np.array([np.abs(mean.T-asymmetric_error[0, :]),
                   np.abs(asymmetric_error[1, :]-mean.T)])
    # res=asymmetric_error
    return res


def quickspine(ax):
    '''remove the top right spine and center ax'''
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


def overheadbase(ax, fontsize=9,notations=True):
    ''' plot a base overhead view
    return fig and ax to add new stuff
    
    # template for overheadbase bwr
    fig, ax = overheadbase(figsize=(3,3))
    vm=np.max(np.abs(scatterv))
    cax=ax.scatter(scatterx*worldscale, scattery*worldscale,s=5, c=scatterv,cmap='bwr', vmin=-vm, vmax=vm)
    cbar = fig.colorbar(cax,shrink=0.6, label='colorbar title')
    '''
    ax.set_aspect('equal')
    ax.set_facecolor('none')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    ax.set_xlim([-235, 235])
    ax.set_ylim([-2, 430])
    x_temp = np.linspace(-235, 235)
    ax.plot(x_temp, np.sqrt(420**2 - x_temp**2), c='k', ls=':')
    if notations:
        ax.text(-10, 425, s=r'$70\degree$', fontsize=fontsize)
        ax.text(130, 150, s=r'$400cm$', fontsize=fontsize)
        ax.text(-130, 0, s=r'$100cm$', fontsize=fontsize)
        ax.plot(np.linspace(-230, -130), np.linspace(0, 0), c='k')
        ax.plot(np.linspace(0, 230 + 7),
                np.tan(np.deg2rad(55)) * np.linspace(0, 230 + 7) - 10, c='k', ls=':')
    ax.text(-230, 100, s=r'$100cm$', fontsize=fontsize)
    ax.plot(np.linspace(-230, -230), np.linspace(0, 100), c='k')


def getcbarnorm(min, mid, max):
    '''center the mid color to zero'''
    divnorm = mcolors.TwoSlopeNorm(vmin=min, vcenter=mid, vmax=max)
    return divnorm




def bypass(input):
    return input


class MonkeyDataExtractor():

    # ruiyi example
    # data_path=Path.cwd().parents[1]/'mkdata/bcm/bruno'
    # ext=MonkeyDataExtractor(folder_path=data_path)
    # trajectory=ext()
    # trajectory.to_pickle(data_path/'test.pkl')

    def __init__(self, folder_path):
        self.monkey_class = 'BCM'
        self.folder_path = folder_path
        self.smr_full_file_path = sorted(self.folder_path.glob('*.smr'))
        self.log_full_file_path = [file.parent / (file.stem + '.log') 
                                   for file in self.smr_full_file_path]
        if self.monkey_class == 'NYU':
            self.marker_memo = {'file_start': 1, 'trial_start': 2, 'trial_end': 3,
                            'juice' :4, 'perturb_start': 8}
        else:
            self.marker_memo = {'file_start': 1, 'trial_start': 2, 'trial_end': 3,
                            'juice' :4, 'perturb_start': 8, 'perturb_start2': 5}
            
        self.y_offset = 32.5
    
    def __call__(self, downsample_fun=bypass, saving_fun=bypass, returndata=True):
        if self.monkey_class == 'NYU':
            self.nyu_extract_smr()
            self.nyu_extract_log()
            self.nyu_segment()
        else:
            self.bcm_extract_smr()
            self.bcm_extract_log()
            self.bcm_segment()
            downsample_fun(self.monkey_trajectory)
            saving_fun(self.monkey_trajectory)
        if returndata:
            return self.monkey_trajectory

    def nyu_extract_smr(self):
        """
        Read Spike2 .smr files and store:
            • self.channel_signal_all  – list of DataFrames, one per file
            • self.marker_all          – list of dicts {'key', 'time'}
            • self.SAMPLING_RATE       – Hz (taken from the first file)
        Written for Neo ≥ 0.14.
        """
        channel_signal_all = []
        marker_all = []

        for file_idx, file_name in enumerate(self.smr_full_file_path):
            # ---- 1.  read the file -------------------------------------------------
            # ‘try_signal_grouping=False’ forces Neo to create ONE AnalogSignal per
            # physical channel (the behaviour your 0.9‑era code expected):contentReference[oaicite:0]{index=0}
            seg = neo.io.Spike2IO(
                filename=file_name,
                try_signal_grouping=False
            ).read_segment(lazy=False)

            # ---- 2.  sampling rate (once) ------------------------------------------
            if file_idx == 0:
                self.SAMPLING_RATE = float(seg.analogsignals[0].sampling_rate)

            # ---- 3.  assemble the channel matrix -----------------------------------
            analog_length = min(sig.shape[0] for sig in seg.analogsignals)
            n_channels    = len(seg.analogsignals)

            channel_signal = np.empty((analog_length, n_channels + 1))
            channel_names  = []

            for ch_idx, sig in enumerate(seg.analogsignals):
                # sig.magnitude gives raw numbers, [:,0] flattens the (N,1) array
                channel_signal[:, ch_idx] = sig.magnitude[:analog_length, 0]
                channel_names.append(str(sig.name))   # Spike2 title

            # add a time column (seconds)
            channel_signal[:, -1] = (
                seg.analogsignals[0].times.rescale("s").magnitude[:analog_length]
            )
            channel_names.append("Time")

            channel_signal_all.append(
                pd.DataFrame(channel_signal, columns=channel_names)
            )

            # ---- 4.  marker channel (if present) -----------------------------------
            try:
                marker_ev = next(ev for ev in seg.events if ev.name == "marker")
                marker_key  = marker_ev.labels.astype(int)
                marker_time = marker_ev.times.rescale("s").magnitude
                marker_all.append({"key": marker_key, "time": marker_time})
            except StopIteration:
                # file has no marker track – keep list positions aligned
                marker_all.append({"key": np.array([]), "time": np.array([])})

        # ---- 5.  expose results ----------------------------------------------------
        self.channel_signal_all = channel_signal_all
        self.marker_all         = marker_all

    # def nyu_extract_smr(self):
    #     channel_signal_all = []
    #     marker_all = []
        
    #     for idx, file_name in enumerate(self.smr_full_file_path):
    #         seg_reader = neo.io.Spike2IO(filename=file_name).read_segment()
            
    #         if idx == 0: # only get sampling rate once
    #             self.SAMPLING_RATE = seg_reader.analogsignals[0].sampling_rate.item()
             
    #         # Sometimes the length across channels varies a bit
    #         analog_length = min([i.size for i in seg_reader.analogsignals])
    #         channel_signal = np.ones((analog_length, seg_reader.size['analogsignals'] + 1))
            
    #         channel_names = []
    #         try:
    #             for ch_idx, ch_data in enumerate(seg_reader.analogsignals):
    #                 channel_signal[:, ch_idx] = ch_data.as_array()[:analog_length].T
    #                 print(ch_data.annotations)
    #                 channel_names.append(ch_data.annotations['channel_names'][0])
    #         except:
    #             print('no channel names')
    #             continue

    #         # Add a time channel
    #         channel_signal[:, -1] = seg_reader.analogsignals[0].times[:analog_length]
    #         channel_names.append('Time') 
            
    #         channel_signal_all.append(pd.DataFrame(channel_signal, columns=channel_names))
            
    #         marker_channel_idx = [idx for idx, value 
    #                                 in enumerate(seg_reader.events)
    #                                 if value.name == 'marker'][0]
    #         marker_key, marker_time = (
    #             seg_reader.events[marker_channel_idx].get_labels().astype('int'),
    #             seg_reader.events[marker_channel_idx].as_array())
    #         marker = {'key': marker_key, 'time': marker_time}
    #         marker_all.append(marker)
            
    #         self.channel_signal_all = channel_signal_all
    #         self.marker_all = marker_all
            
    def nyu_extract_log(self):
        log_data_all = []
        
        for file_name in self.log_full_file_path:
            with open(file_name, 'r', encoding='UTF-8') as content:
                log_content = content.readlines()
            
            floor_density = []
            perturb_vpeak = []; perturb_wpeak = []; perturb_start_time_ori = []
            full_on = []; target_x = []; target_y = []
            for line in log_content:
                if 'Joy Stick Max Velocity' in line:
                    gain_v = float(line.split(': ')[1])
                    
                if 'Joy Stick Max Angular Velocity' in line:
                    gain_w = float(line.split(': ')[1])
                    
                if 'Perturb Max Velocity' in line:
                    perturb_vpeakmax = float(line.split(': ')[1])
                    
                if 'Perturb Max Angular Velocity' in line:
                    perturb_wpeakmax = float(line.split(': ')[1])
                    
                if 'Perturbation Sigma' in line:
                    perturb_sigma = float(line.split(': ')[1])
                    
                if 'Perturbation Duration' in line:
                    perturb_dur = float(line.split(': ')[1])
                    
                if 'Floor Density' in line:
                    content_temp = float(line.split(': ')[1])
                    floor_density.append(content_temp)
                    
                if 'Perturbation Linear Speed' in line:
                    content_temp = float(line.split(': ')[1])
                    perturb_vpeak.append(content_temp)
                    
                if 'Perturbation Angular Speed' in line:
                    content_temp = float(line.split(': ')[1])
                    perturb_wpeak.append(- content_temp)
                    
                if 'Perturbation Delay Time' in line:
                    content_temp = float(line.split(': ')[1])
                    perturb_start_time_ori.append(content_temp / 1000)  # ms to s
                    
                if 'Firefly Full On' in line:
                    content_temp = bool(int(line.split(': ')[1]))
                    full_on.append(content_temp)
                
                if 'Position x/y(cm)' in line:
                    content_temp_x, content_temp_y = line.split(': ')[1].split(' ')
                    target_x.append(float(content_temp_x))
                    # Monkey data's y positions are reversed.
                    target_y.append(- float(content_temp_y) + self.y_offset)
                
            log_data_all.append({'gain_v': gain_v, 'gain_w': gain_w,
                                 'perturb_vpeakmax': perturb_vpeakmax, 'perturb_wpeakmax': perturb_wpeakmax,
                                 'perturb_sigma': perturb_sigma, 'perturb_dur': perturb_dur,
                                 'floor_density': floor_density,
                                 'perturb_vpeak': perturb_vpeak, 'perturb_wpeak': perturb_wpeak,
                                 'perturb_start_time_ori': perturb_start_time_ori,
                                  'full_on': full_on, 'target_x': target_x, 'target_y': target_y})
            
            self.log_data_all = log_data_all
            
    def nyu_segment(self, lazy_threshold=4000, skip_threshold=400, skip_r_threshold=30, 
                    crazy_threshold=200,
                    medfilt_kernel=5, v_threshold=1, reward_boundary=65, 
                    perturb_corr_threshold=50):
        # lazy_threshold (data points): Trial is too long.
        # skip_threshold (data points): Trial is too short.
        # skip_r_threshold (cm): Monkey did not move a lot.
        # crazy_threshold (cm): Monkey stopped too far.
        # v_threshold (cm/s): Velocity threshold for start and end.
        # reward_boundary (cm): Rewarded when stop inside this circular boundary.
        # perturb_corr_threshold (data points): Corrected perturbation start index should not be too biased.
        
        gain_v = []; gain_w = []; perturb_vpeakmax = []; perturb_wpeakmax = []
        perturb_sigma = []; perturb_dur = []; perturb_vpeak = []; perturb_wpeak = []
        perturb_v = []; perturb_w = []; perturb_v_gauss = []; perturb_w_gauss = []
        perturb_start_time = []; perturb_start_time_ori = []
        floor_density = []; pos_x = []; pos_y = []
        head_dir = []; head_dir_end = []; pos_r = []; pos_theta = []
        pos_r_end = []; pos_theta_end = []
        pos_v = []; pos_w = []; target_x = []; target_y = []
        target_r = []; target_theta = []; full_on = []; rewarded = []
        relative_radius = []; relative_angle = []; time = []
        trial_dur = []; action_v = []; action_w = []
        relative_radius_end = []; relative_angle_end = []; category = []

        for session_idx, session_data in enumerate(self.channel_signal_all):
            log_data = self.log_data_all[session_idx]
            marker_data = self.marker_all[session_idx]
            start_marker_times = marker_data['time'][
                            marker_data['key'] == self.marker_memo['trial_start']]
            end_marker_times = marker_data['time'][
                            marker_data['key'] == self.marker_memo['trial_end']]
            perturb_marker_times = marker_data['time'][
                            marker_data['key'] == self.marker_memo['perturb_start']]

            # segment trials
            for trial_idx in range(end_marker_times.size):
                trial_data = session_data[np.logical_and(
                    session_data.Time > start_marker_times[trial_idx],
                    session_data.Time < end_marker_times[trial_idx])].copy()
                
                # # Use median filter kernel size as 5 to remove spike noise first.
                # trial_data['ForwardV'] = medfilt(trial_data['ForwardV'], medfilt_kernel)
                # trial_data['AngularV'] = medfilt(trial_data['AngularV'], medfilt_kernel)
                
                # cut non-moving head and tail
                moving_period = np.where(trial_data['ForwardV'].abs() > v_threshold)[0]
                if moving_period.size > 0:
                    start_idx = moving_period[0]
                    end_idx = moving_period[-1] + 2
                else:
                    start_idx = 0
                    end_idx = None
                  
                # store trial data
                trial_data = trial_data.iloc[start_idx : end_idx]
                trial_data['AngularV'] = - trial_data['AngularV']
                trial_data['MonkeyYa'] = np.cumsum(trial_data['AngularV']) / self.SAMPLING_RATE + 90
                trial_data['MonkeyX'] = np.cumsum(trial_data['ForwardV']
                                            * np.cos(np.deg2rad(trial_data['MonkeyYa']))) / self.SAMPLING_RATE
                trial_data['MonkeyY'] = np.cumsum(trial_data['ForwardV']
                                            * np.sin(np.deg2rad(trial_data['MonkeyYa']))) / self.SAMPLING_RATE
                
                gain_v.append(log_data['gain_v'])
                gain_w.append(log_data['gain_w'])
                perturb_vpeakmax.append(log_data['perturb_vpeakmax'])
                perturb_wpeakmax.append(log_data['perturb_wpeakmax'])
                perturb_vpeak.append(log_data['perturb_vpeak'][trial_idx])
                perturb_wpeak.append(log_data['perturb_wpeak'][trial_idx])
                perturb_start_time_ori.append(log_data['perturb_start_time_ori'][trial_idx])
                perturb_sigma.append(log_data['perturb_sigma'])
                perturb_dur.append(log_data['perturb_dur'])
                pos_x.append(trial_data['MonkeyX'].values)
                pos_y.append(trial_data['MonkeyY'].values)
                head_dir.append(trial_data['MonkeyYa'].values)
                head_dir_end.append(trial_data['MonkeyYa'].values[-1])
                floor_density.append(log_data['floor_density'][trial_idx])
                full_on.append(log_data['full_on'][trial_idx])

                rho, phi = cart2pol(pos_x[-1], pos_y[-1])
                pos_r.append(rho)
                pos_theta.append(np.rad2deg(phi))
                pos_r_end.append(rho[-1])
                pos_theta_end.append(np.rad2deg(phi[-1]))
                
                
                # determine if it is a perturbation trial
                perturb_start_time_temp = perturb_marker_times[(np.logical_and(
                                            perturb_marker_times > start_marker_times[trial_idx],
                                            perturb_marker_times < end_marker_times[trial_idx]))]
                if bool(perturb_start_time_temp.size):
                    assert perturb_start_time_temp.size == 1
                    pos_v.append(trial_data['ForwardV'].values)
                    pos_w.append(trial_data['AngularV'].values)
                    
                    # construct perturbation curves
                    perturb_xaxis = np.linspace(0, perturb_dur[-1], round(self.SAMPLING_RATE))
                    perturb_temp = norm.pdf(perturb_xaxis, loc=perturb_dur[-1] / 2, scale=perturb_sigma[-1])
                    perturb_temp /= perturb_temp.max()
                    perturb_v_temp = perturb_temp * perturb_vpeak[-1]
                    perturb_w_temp = perturb_temp * perturb_wpeak[-1]
                    perturb_v_gauss.append(perturb_v_temp)
                    perturb_w_gauss.append(perturb_w_temp)
                    
                    # use obvious angular perturbation curve as a template
                    if abs(perturb_wpeak[-1]) / perturb_wpeakmax[-1] > 0.1:
                        perturb_template = perturb_w_temp
                        original_vel = pos_w[-1]
                    else:
                        perturb_template = perturb_v_temp
                        original_vel = pos_v[-1]
                        
                    # use the template to do cross-correlation to find perturbation start time
                    perturb_start_idx_mark = int((perturb_start_time_temp
                                                - trial_data['Time'].values[0]) * self.SAMPLING_RATE)
                    perturb_start_idx_mark = np.clip(perturb_start_idx_mark, 0, None)
                    perturb_peak_idx = np.correlate(original_vel, perturb_template, mode='same').argsort()[::-1]
                    perturb_start_idx_corr = perturb_peak_idx - perturb_dur[-1] / 2 * self.SAMPLING_RATE
                    mask = (perturb_start_idx_corr > 0) \
                           & (perturb_start_idx_corr > perturb_start_idx_mark) \
                           & (perturb_start_idx_corr - perturb_start_idx_mark < perturb_corr_threshold)
                    
                    if mask.sum() == 0 or original_vel.size < perturb_template.size:
                        perturb_start_idx = np.clip(perturb_start_idx_mark, None, pos_v[-1].size - 1)
                    else:
                        perturb_start_idx = int(perturb_start_idx_corr[mask][0])
                    perturb_start_time.append(perturb_start_idx / self.SAMPLING_RATE)
                    
                    # get pure actions
                    perturb_v_full = np.zeros_like(pos_v[-1])
                    perturb_v_full[perturb_start_idx:perturb_start_idx + perturb_v_temp.size] = \
                                                            perturb_v_temp[:perturb_v_full.size - perturb_start_idx]
                    perturb_w_full = np.zeros_like(pos_w[-1])
                    perturb_w_full[perturb_start_idx:perturb_start_idx + perturb_w_temp.size] = \
                                                            perturb_w_temp[:perturb_w_full.size - perturb_start_idx]
                    
                    perturb_v.append(perturb_v_full); perturb_w.append(perturb_w_full)
                    action_v.append((pos_v[-1] - perturb_v_full).clip(-gain_v[-1], gain_v[-1]) / gain_v[-1])
                    action_w.append((pos_w[-1] - perturb_w_full).clip(-gain_w[-1], gain_w[-1]) / gain_w[-1])
                else:
                    pos_v.append(trial_data['ForwardV'].values.clip(-gain_v[-1], gain_v[-1]))
                    pos_w.append(trial_data['AngularV'].values.clip(-gain_w[-1], gain_w[-1]))
                    perturb_v_gauss.append(np.zeros(round(self.SAMPLING_RATE)))
                    perturb_w_gauss.append(np.zeros(round(self.SAMPLING_RATE)))
                    perturb_start_time.append(np.nan)
                    perturb_v.append(np.zeros_like(pos_v[-1])); perturb_w.append(np.zeros_like(pos_w[-1]))
                    action_v.append(pos_v[-1] / gain_v[-1])
                    action_w.append(pos_w[-1] / gain_w[-1])
                
                trial_data['Time'] -= trial_data['Time'].iloc[0]
                time.append(trial_data['Time'].values)
                trial_dur.append(trial_data['Time'].values[-1])
                target_x.append(log_data['target_x'][trial_idx])
                target_y.append(log_data['target_y'][trial_idx])
                tar_rho, tar_phi = cart2pol(target_x[-1], target_y[-1])
                target_r.append(tar_rho)
                target_theta.append(np.rad2deg(tar_phi))
                
                relative_r, relative_ang = get_relative_r_ang(
                                pos_x[-1], pos_y[-1], head_dir[-1], target_x[-1], target_y[-1])
                relative_radius.append(relative_r)
                relative_angle.append(np.rad2deg(relative_ang))
                relative_radius_end.append(relative_r[-1])
                relative_angle_end.append(np.rad2deg(relative_ang[-1]))
                rewarded.append(relative_r[-1] < reward_boundary)

                # Categorize trials
                if rewarded[-1]:
                    category.append('normal')
                else:
                    if trial_data['ForwardV'].size < skip_threshold or\
                       pos_r_end[-1] < skip_r_threshold:
                        category.append('skip')
                    elif trial_data['ForwardV'].size > lazy_threshold:
                        category.append('lazy')
                    elif relative_r[-1] > crazy_threshold:
                        category.append('crazy')
                    else:
                        category.append('normal')


        # Construct a dataframe   
        self.monkey_trajectory = pd.DataFrame().assign(gain_v=gain_v, gain_w=gain_w, 
                                 perturb_vpeakmax=perturb_vpeakmax, perturb_wpeakmax=perturb_wpeakmax,
                                 perturb_sigma=perturb_sigma, perturb_dur=perturb_dur,
                                 perturb_vpeak=perturb_vpeak, perturb_wpeak=perturb_wpeak,
                                 perturb_start_time=perturb_start_time,
                                 perturb_start_time_ori=perturb_start_time_ori,
                                 perturb_v_gauss=perturb_v_gauss, perturb_w_gauss=perturb_w_gauss,
                                 perturb_v=perturb_v, perturb_w=perturb_w,
                                 floor_density=floor_density, pos_x=pos_x,
                                 pos_y=pos_y, head_dir=head_dir, head_dir_end=head_dir_end,
                                 pos_r=pos_r, pos_theta=pos_theta, pos_r_end=pos_r_end,
                                 pos_theta_end=pos_theta_end, pos_v=pos_v, pos_w=pos_w, 
                                 target_x=target_x, target_y=target_y, target_r=target_r,
                                 target_theta=target_theta, full_on=full_on, rewarded=rewarded,
                                 relative_radius=relative_radius, relative_angle=relative_angle,
                                 time=time, trial_dur=trial_dur, 
                                 action_v=action_v, action_w=action_w, 
                                 relative_radius_end=relative_radius_end,
                                 relative_angle_end=relative_angle_end, category=category)

    def bcm_extract_smr(self):
        print('starting ext')
        channel_signal_all = []
        marker_all = []
        
        for idx, file_name in enumerate(self.smr_full_file_path):
            seg_reader = neo.io.Spike2IO(filename=str(file_name)).read_segment()
            
            if idx == 0: # only get sampling rate once
                self.SAMPLING_RATE = seg_reader.analogsignals[0].sampling_rate.item()
                
            # Sometimes the length across channels varies a bit
            analog_length = min([i.size for i in seg_reader.analogsignals])
            channel_signal = np.ones((analog_length, seg_reader.size['analogsignals']))
            
            channel_names = []

            # Do not read the last channel as it has a unique shape.
            for ch_idx, ch_data in enumerate(seg_reader.analogsignals[:-1]):
                channel_signal[:, ch_idx] = ch_data.as_array()[:analog_length].T
                channel_names.append(ch_data.annotations['channel_names'][0])

            # Add a time channel
            channel_signal[:, -1] = seg_reader.analogsignals[0].times[:analog_length]
            channel_names.append('Time') 
            
            channel_signal_all.append(pd.DataFrame(channel_signal,columns=channel_names))
            
            marker_channel_idx = [idx for idx, value 
                                    in enumerate(seg_reader.events)
                                    if value.name == 'marker'][0]
            marker_key, marker_time = (
                seg_reader.events[marker_channel_idx].get_labels().astype('int'),
                seg_reader.events[marker_channel_idx].as_array())
            marker = {'key': marker_key, 'time': marker_time}
            marker_all.append(marker)
            
            self.channel_signal_all = channel_signal_all
            self.marker_all = marker_all
            
    def bcm_extract_log(self):
        log_data_all = []
        
        for file_name in self.log_full_file_path:
            with open(file_name, 'r', encoding='UTF-8') as content:
                log_content = content.readlines()
                
            floor_density = []
            perturb_vpeak = []; perturb_wpeak = []
            perturb_start_time_ori = []
            full_on = []
            for line in log_content:
                if 'Joy Stick Max Velocity' in line:
                    gain_v = float(line.split(': ')[1])
                    
                if 'Joy Stick Max Angular Velocity' in line:
                    gain_w = float(line.split(': ')[1])
                    
                if 'Perturb Max Velocity' in line:
                    perturb_vpeakmax = float(line.split(': ')[1])
                    
                if 'Perturb Max Angular Velocity' in line:
                    perturb_wpeakmax = float(line.split(': ')[1])
                    
                if 'Perturbation Sigma' in line:
                    perturb_sigma = float(line.split(': ')[1])
                    
                if 'Perturbation Duration' in line:
                    perturb_dur = float(line.split(': ')[1])
                    
                if 'Floor Density' in line:
                    content_temp = float(line.split(': ')[1])
                    floor_density.append(content_temp)
                    
                if 'Perturbation Linear Speed' in line:
                    content_temp = float(line.split(': ')[1])
                    perturb_vpeak.append(content_temp)
                    
                if 'Perturbation Angular Speed' in line:
                    content_temp = float(line.split(': ')[1])
                    perturb_wpeak.append(- content_temp)
                    
                if 'Perturbation Delay Time' in line:
                    content_temp = float(line.split(': ')[1])
                    perturb_start_time_ori.append(content_temp / 1000)
                    
                if 'Firefly Full ON' in line:
                    content_temp = bool(int(line.split(': ')[1]))
                    full_on.append(content_temp)
            
            if len(full_on) == 0: # Quigley's perturbation sessions
                full_on = [False] * len(floor_density)
            
            log_data_all.append({'gain_v': gain_v, 'gain_w': gain_w, 
                                 'perturb_vpeakmax': perturb_vpeakmax, 'perturb_wpeakmax': perturb_wpeakmax,
                                 'perturb_sigma': perturb_sigma, 'perturb_dur': perturb_dur,
                                 'floor_density': floor_density, 
                                 'perturb_vpeak': perturb_vpeak, 'perturb_wpeak': perturb_wpeak,
                                 'perturb_start_time_ori': perturb_start_time_ori,
                                 'full_on': full_on})
            self.log_data_all = log_data_all
            
    def bcm_segment(self, lazy_threshold=4000, skip_threshold=400, skip_r_threshold=30, 
                    crazy_threshold=200,
                    medfilt_kernel=5, v_threshold=1, reward_boundary=65,
                    target_r_range=[100, 400], target_theta_range=[55, 125], 
                    target_tolerance=1, perturb_corr_threshold=100):
        print('starting segmenting')
        # lazy_threshold (time): Trial is too long.

        # skip_threshold (time): Trial is too short.
        # and
        # skip_r_threshold (cm): Monkey did not move a lot.

        # crazy_threshold (cm): Monkey stopped too far, very wrong.

        # medfilt_kernel (data points): Remove spikes from raw data.
        # v_threshold (cm/s): Threshold for end point.
        # reward_boundary (cm): Rewarded when stop inside this circular boundary.
        # target_r_range (cm): Radius of target distribution.
        # target_theta_range (deg): Angle of target distribution.
        # target_tolerance (cm or deg): Max tolerance for targets out of distribution.
        # perturb_corr_threshold (data points): Corrected perturbation start index should not be too biased.

        gain_v = []; gain_w = []; perturb_vpeakmax = []; perturb_wpeakmax = []
        perturb_sigma = []; perturb_dur = []; perturb_vpeak = []; perturb_wpeak = []
        perturb_v = []; perturb_w = []; perturb_v_gauss = []; perturb_w_gauss = []
        perturb_start_time = []; perturb_start_time_ori = []
        floor_density = []; pos_x = []; pos_y = []
        head_dir = []; head_dir_end = []; pos_r = []; pos_theta = []; 
        pos_r_end = []; pos_theta_end = []
        pos_v = []; pos_w = []; target_x = []; target_y = []
        target_r = []; target_theta = []; full_on = []; rewarded = []
        relative_radius = []; relative_angle = []; time = []; 
        trial_dur = []; action_v = []; action_w = []; 
        relative_radius_end = []; relative_angle_end = []; category = []

        for session_idx, session_data in enumerate(self.channel_signal_all):
            log_data = self.log_data_all[session_idx]
            marker_data = self.marker_all[session_idx]
            start_marker_times = marker_data['time'][
                            marker_data['key'] == self.marker_memo['trial_start']]
            end_marker_times = marker_data['time'][
                            marker_data['key'] == self.marker_memo['trial_end']]
            perturb_marker_times = marker_data['time'][
                            marker_data['key'] == self.marker_memo['perturb_start']]
            if perturb_marker_times.size == 0:
                perturb_marker_times = marker_data['time'][
                            marker_data['key'] == self.marker_memo['perturb_start2']]

            # segment trials
            for trial_idx in range(end_marker_times.size):
                trial_data = session_data[np.logical_and(
                    session_data.Time > start_marker_times[trial_idx],
                    session_data.Time < end_marker_times[trial_idx])].copy()

                # Use median filter kernel size as 5 to remove spike noise first.
                trial_data['ForwardV'] = medfilt(trial_data['ForwardV'], medfilt_kernel)
                trial_data['AngularV'] = medfilt(trial_data['AngularV'], medfilt_kernel)
                
                # cut non-moving head and tail
                moving_period = np.where(trial_data['ForwardV'].abs() > v_threshold)[0]
                if moving_period.size > 0:
                    start_idx = moving_period[0]
                    end_idx = moving_period[-1] + 2
                else:
                    start_idx = 0
                    end_idx = None
                    
                # store trial data
                trial_data = trial_data.iloc[start_idx : end_idx]
                trial_data['AngularV'] = - trial_data['AngularV']
                trial_data['MonkeyYa'] = np.cumsum(trial_data['AngularV']) / self.SAMPLING_RATE + 90
                trial_data['MonkeyX'] = np.cumsum(trial_data['ForwardV']
                                            * np.cos(np.deg2rad(trial_data['MonkeyYa']))) / self.SAMPLING_RATE
                trial_data['MonkeyY'] = np.cumsum(trial_data['ForwardV']
                                            * np.sin(np.deg2rad(trial_data['MonkeyYa']))) / self.SAMPLING_RATE
                
                gain_v.append(log_data['gain_v'])
                gain_w.append(log_data['gain_w'])
                perturb_vpeakmax.append(log_data['perturb_vpeakmax'])
                perturb_wpeakmax.append(log_data['perturb_wpeakmax'])
                perturb_vpeak.append(log_data['perturb_vpeak'][trial_idx])
                perturb_wpeak.append(log_data['perturb_wpeak'][trial_idx])
                perturb_start_time_ori.append(log_data['perturb_start_time_ori'][trial_idx])
                perturb_sigma.append(log_data['perturb_sigma'])
                perturb_dur.append(log_data['perturb_dur'])
                pos_x.append(trial_data['MonkeyX'].values)
                pos_y.append(trial_data['MonkeyY'].values)
                head_dir.append(trial_data['MonkeyYa'].values)
                head_dir_end.append(trial_data['MonkeyYa'].values[-1])
                floor_density.append(log_data['floor_density'][trial_idx])
                full_on.append(log_data['full_on'][trial_idx])
                
                rho, phi = cart2pol(pos_x[-1], pos_y[-1])
                pos_r.append(rho)
                pos_theta.append(np.rad2deg(phi))
                pos_r_end.append(rho[-1])
                pos_theta_end.append(np.rad2deg(phi[-1]))
                
                
                # determine if it is a perturbation trial
                perturb_start_time_temp = perturb_marker_times[(np.logical_and(
                                            perturb_marker_times > start_marker_times[trial_idx],
                                            perturb_marker_times < end_marker_times[trial_idx]))]
                if bool(perturb_start_time_temp.size):
                    assert perturb_start_time_temp.size == 1
                    pos_v.append(trial_data['ForwardV'].values)
                    pos_w.append(trial_data['AngularV'].values)
                
                    # construct perturbation curves
                    perturb_xaxis = np.linspace(0, perturb_dur[-1], round(self.SAMPLING_RATE))
                    perturb_temp = norm.pdf(perturb_xaxis, loc=perturb_dur[-1] / 2, scale=perturb_sigma[-1])
                    perturb_temp /= perturb_temp.max()
                    perturb_v_temp = perturb_temp * perturb_vpeak[-1]
                    perturb_w_temp = perturb_temp * perturb_wpeak[-1]
                    perturb_v_gauss.append(perturb_v_temp)
                    perturb_w_gauss.append(perturb_w_temp)
                    
                    # use the more obvious perturbation curve as a template
                    corrcoef_v = np.correlate(pos_v[-1] - pos_v[-1].mean(), 
                                              perturb_v_temp - perturb_v_temp.mean(),
                                              mode='same') / (pos_v[-1].std() * perturb_v_temp.std())
                    corrcoef_w = np.correlate(pos_w[-1] - pos_w[-1].mean(), 
                                              perturb_w_temp - perturb_w_temp.mean(),
                                              mode='same') / (pos_w[-1].std() * perturb_w_temp.std())
                    if corrcoef_v.max() > corrcoef_w.max():
                        perturb_template = perturb_v_temp
                        original_vel = pos_v[-1]
                    else:
                        perturb_template = perturb_w_temp
                        original_vel = pos_w[-1]
                        
                    # use the template to do cross-correlation to find perturbation start time
                    perturb_start_idx_mark = int((perturb_start_time_temp
                                                - trial_data['Time'].values[0]) * self.SAMPLING_RATE)
                    perturb_start_idx_mark = np.clip(perturb_start_idx_mark, 0, None)
                    perturb_peak_idx = np.correlate(original_vel, perturb_template, mode='same').argsort()[::-1]
                    perturb_start_idx_corr = perturb_peak_idx - perturb_dur[-1] / 2 * self.SAMPLING_RATE
                    mask = (perturb_start_idx_corr > 0) \
                           & (perturb_start_idx_corr > perturb_start_idx_mark) \
                           & (perturb_start_idx_corr - perturb_start_idx_mark < perturb_corr_threshold)
                    
                    if mask.sum() == 0 or original_vel.size < perturb_template.size:
                        perturb_start_idx = np.clip(perturb_start_idx_mark, None, pos_v[-1].size - 1)
                    else:
                        perturb_start_idx = int(perturb_start_idx_corr[mask][0])
                    perturb_start_time.append(perturb_start_idx / self.SAMPLING_RATE)
                    
                    # get pure actions
                    perturb_v_full = np.zeros_like(pos_v[-1])
                    perturb_v_full[perturb_start_idx:perturb_start_idx + perturb_v_temp.size] = \
                                                            perturb_v_temp[:perturb_v_full.size - perturb_start_idx]
                    perturb_w_full = np.zeros_like(pos_w[-1])
                    perturb_w_full[perturb_start_idx:perturb_start_idx + perturb_w_temp.size] = \
                                                            perturb_w_temp[:perturb_w_full.size - perturb_start_idx]
                    
                    perturb_v.append(perturb_v_full); perturb_w.append(perturb_w_full)
                    action_v.append((pos_v[-1] - perturb_v_full).clip(-gain_v[-1], gain_v[-1]) / gain_v[-1])
                    action_w.append((pos_w[-1] - perturb_w_full).clip(-gain_w[-1], gain_w[-1]) / gain_w[-1])
                else:
                    pos_v.append(trial_data['ForwardV'].values.clip(-gain_v[-1], gain_v[-1]))
                    pos_w.append(trial_data['AngularV'].values.clip(-gain_w[-1], gain_w[-1]))
                    perturb_v_gauss.append(np.zeros(round(self.SAMPLING_RATE)))
                    perturb_w_gauss.append(np.zeros(round(self.SAMPLING_RATE)))
                    perturb_start_time.append(np.nan)
                    perturb_v.append(np.zeros_like(pos_v[-1])); perturb_w.append(np.zeros_like(pos_w[-1]))
                    action_v.append(pos_v[-1] / gain_v[-1])
                    action_w.append(pos_w[-1] / gain_w[-1])
                

                trial_data['Time'] -= trial_data['Time'].iloc[0]
                time.append(trial_data['Time'].values)
                trial_dur.append(trial_data['Time'].values[-1])
                
                
                # target position is analog in BCM data, I bin target channels
                # and find the mode of bins
                targetx_bins = np.arange(my_floor(trial_data['FireflyX'].min(), 1),
                                         my_ceil(trial_data['FireflyX'].max(), 1), 0.1)
                targetx_idxes = np.digitize(trial_data['FireflyX'], targetx_bins)
                targetx_hist, _ = np.histogram(trial_data['FireflyX'], targetx_bins)
                try:
                    tar_x = trial_data['FireflyX'][
                                    targetx_idxes == targetx_hist.argmax()+1].mean()
                except: # when start_idx == end_idx, they are bad trials that not matter
                    tar_x = trial_data['FireflyX'].mean()

                targety_bins = np.arange(my_floor(trial_data['FireflyY'].min(), 1),
                                         my_ceil(trial_data['FireflyY'].max(), 1), 0.1)
                targety_idxes = np.digitize(trial_data['FireflyY'], targety_bins)
                targety_hist, _ = np.histogram(trial_data['FireflyY'], targety_bins)
                try:
                    tar_y = trial_data['FireflyY'][
                                    targety_idxes == targety_hist.argmax()+1].mean()
                except:
                    tar_y = trial_data['FireflyY'].mean()

                target_x.append(tar_x)
                target_y.append(- tar_y + self.y_offset)
                tar_rho, tar_phi = cart2pol(target_x[-1], target_y[-1])
                target_r.append(tar_rho)
                target_theta.append(np.rad2deg(tar_phi))

                relative_r, relative_ang = get_relative_r_ang(
                                pos_x[-1], pos_y[-1], head_dir[-1], target_x[-1], target_y[-1])
                relative_radius.append(relative_r)
                relative_angle.append(np.rad2deg(relative_ang))
                relative_radius_end.append(relative_r[-1])
                relative_angle_end.append(np.rad2deg(relative_ang[-1]))
                rewarded.append(relative_r[-1] < reward_boundary)

                #juice_time = marker_data['time'][marker_data['key'] == marker_memo['juice']]
                #j_marker = np.where(np.logical_and(juice_time > start_marker_times[trial_idx],
                #               juice_time < end_marker_times[trial_idx]))[0]

                # Categorize trials
                # Note that few targets in BCM data are out of the distribution
                # for unknown reason, I just label and ignore them.
                if target_r[-1] < target_r_range[0] - target_tolerance or\
                   target_r[-1] > target_r_range[1] + target_tolerance or\
                   target_theta[-1] < target_theta_range[0] - target_tolerance or\
                   target_theta[-1] > target_theta_range[1] + target_tolerance:
                    category.append('wrong_target')
                else:
                    if rewarded[-1]:
                        category.append('normal')
                    else:
                        if trial_data['ForwardV'].size < skip_threshold and\
                           pos_r_end[-1] < skip_r_threshold:
                            category.append('skip')
                        elif trial_data['ForwardV'].size > lazy_threshold:
                            category.append('lazy')
                        elif relative_r[-1] > crazy_threshold:
                            category.append('crazy')
                        else:
                            category.append('normal')

        # Construct a dataframe   
        self.monkey_trajectory = pd.DataFrame().assign(gain_v=gain_v, gain_w=gain_w, 
                                 perturb_vpeakmax=perturb_vpeakmax, perturb_wpeakmax=perturb_wpeakmax,
                                 perturb_sigma=perturb_sigma, perturb_dur=perturb_dur,
                                 perturb_vpeak=perturb_vpeak, perturb_wpeak=perturb_wpeak,
                                 perturb_start_time=perturb_start_time,
                                 perturb_start_time_ori=perturb_start_time_ori,
                                 perturb_v_gauss=perturb_v_gauss, perturb_w_gauss=perturb_w_gauss,
                                 perturb_v=perturb_v, perturb_w=perturb_w,
                                 floor_density=floor_density, pos_x=pos_x,
                                 pos_y=pos_y, head_dir=head_dir, head_dir_end=head_dir_end,
                                 pos_r=pos_r, pos_theta=pos_theta, pos_r_end=pos_r_end,
                                 pos_theta_end=pos_theta_end, pos_v=pos_v, pos_w=pos_w, 
                                 target_x=target_x, target_y=target_y, target_r=target_r,
                                 target_theta=target_theta, full_on=full_on, rewarded=rewarded,
                                 relative_radius=relative_radius, relative_angle=relative_angle,
                                 time=time, trial_dur=trial_dur, 
                                 action_v=action_v, action_w=action_w, 
                                 relative_radius_end=relative_radius_end,
                                 relative_angle_end=relative_angle_end, category=category)

def cart2pol(*args):
    if type(args[0]) == list:
        x = args[0][0]
        y = args[0][1]
    else:
        x = args[0]
        y = args[1]
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def xy2pol(*args, rotation=True):
    # return distance and angle. default rotated left 90 degree for the task
    x = args[0][0]
    y = args[0][1]
    d = np.sqrt(x**2 + y**2)
    a = np.arctan2(y, x)+pi/2 if rotation else np.arctan2(y, x)
    return d, a

def get_relative_r_ang(px, py, heading_angle, target_x, target_y):
    heading_angle = np.deg2rad(heading_angle)
    distance_vector = np.vstack([px - target_x, py - target_y])
    relative_r = np.linalg.norm(distance_vector, axis=0)

    relative_ang = heading_angle - np.arctan2(distance_vector[1],
                                              distance_vector[0])
    # make the relative angle range [-pi, pi]
    relative_ang = np.remainder(relative_ang, 2 * np.pi)
    relative_ang[relative_ang >= np.pi] -= 2 * np.pi
    return relative_r, relative_ang


def world_to_egocentric(gaze_x, gaze_y, body_x, body_y, body_theta):
    """
    Convert world coordinates to egocentric coordinates through rotation.
    
    This version carefully matches the original function's calculations.
    """
    # Calculate relative position in world frame
    dx = gaze_x - body_x
    dy = gaze_y - body_y
    
    # Calculate rotation angle exactly as in the original
    rotation_angle = body_theta - np.deg2rad(90)
    
    # Rotate coordinates to match the original function
    # The negative sign on ego_x matches the negative sign in the arctan2 call
    ego_x = -(dx * np.cos(-rotation_angle) - dy * np.sin(-rotation_angle))
    ego_y = dx * np.sin(-rotation_angle) + dy * np.cos(-rotation_angle)
    
    return ego_x, ego_y

def convert_egolocation_to_angle(ego_x, ego_y, monkey_height=10):
    """
    Convert egocentric coordinates to angles, exactly matching the original.
    """
    # Use the same formula but without the rotation adjustment
    # Note that ego_x is already negated in the coordinate transform
    hor_theta = -np.rad2deg(np.arctan2(ego_x, np.sqrt(ego_y**2 + monkey_height**2))).reshape(-1, 1)
    
    # Vertical angle calculation
    ver_theta = -np.rad2deg(np.arctan2(monkey_height, ego_y)).reshape(-1, 1)
    
    return hor_theta, ver_theta


def world2irc(fx, fy, mx, my, worldscale=200):

    # Calculate relative position
    task_x = (fx - mx).astype('float32')
    task_y = (fy - my).astype('float32')
    
    # Apply scaling and return in the format [y_coord, x_coord]
    return [task_y/worldscale, task_x/worldscale]

def irc2world(irc_y, irc_x, mx, my, worldscale=200):

    # Reverse the scaling
    task_y = irc_y * worldscale
    task_x = irc_x * worldscale
    
    # Reverse the coordinate shift
    fx = task_x + mx
    fy = task_y + my
    
    return fx, fy


def action2irc(mv, mw, worldscale=200):

    # Stack linear and angular velocities into a 2D array
    action = np.stack([mv, mw]).T
    
    # Scale linear velocity by worldscale
    action[:, 0] = action[:, 0] / worldscale
    
    # Convert angular velocity from degrees to radians
    action[:, 1] = action[:, 1] / 180 * np.pi
    
    # Convert to float32
    return action.astype('float32')

def moment_matched_cov(cov, radius):
    """
    Compute the moment-matched covariance for a circular target
    with uncertain center location.

    Args:
        cov (2,2)   : original covariance matrix (e.g. target center uncertainty)
        radius (float): radius of the target circle (meters)

    Returns:
        (2,2) numpy array: augmented covariance matrix
    """
    return cov + (radius ** 2 / 4.0) * np.eye(2)