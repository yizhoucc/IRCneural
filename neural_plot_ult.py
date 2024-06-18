
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

state_color='blue'
belief_color='purple'
eye_color='red'

# notify ------------------------
import requests
import configparser
config = configparser.ConfigParser()
config.read_file(open('privateconfig'))
token=config['Notification']['token']

def notify(msg='plots ready', group='lab',title='plot'):
    notification="https://api.day.app/{}/{}/{}?group={}".format(token,title, msg, group)
    requests.get(notification)



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

# cebra embedding plots -------------------------------
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


# ruiyi neural eye analysis plot -------------------------------
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


def downsample(data, bin_size=20):
    num_bin = data.shape[0] // bin_size
    data_ = data[:bin_size * num_bin]
    data_ = data_.reshape(num_bin, bin_size, data.shape[-1])
    data_ = np.nanmean(data_, axis=1)
    return data_


def convert_location_to_angle(gaze_r, gaze_x, gaze_y, body_theta, body_x, body_y, hor_theta_eye, ver_theta_eye, monkey_height=monkey_height, DT=DT, remove_pre=True, remove_post=True):
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
    ver_theta = -np.rad2deg(np.arctan2(monkey_height,
                            gaze_r_sign * gaze_r)).reshape(-1, 1)

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