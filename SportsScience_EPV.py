'''


Credits to Laurie Shaw's tutorial on Friends of Tracking: https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking
'''

import Metrica_IO as mio
import Metrica_Viz as mviz
import Metrica_Velocities as mvel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Metrica_PitchControl as mpc
import Metrica_EPV as mepv
import sklearn
from sklearn import linear_model

DATADIR = 'C:/Users/sgopaladesikan/PycharmProjects/MMoF/Metrica/data'
game_id = 2  # let's look at sample match 2

# read in the event data
events = mio.read_event_data(DATADIR, game_id)

# read in tracking data
tracking_home = mio.tracking_data(DATADIR, game_id, 'Home')
tracking_away = mio.tracking_data(DATADIR, game_id, 'Away')

# Convert positions from metrica units to meters (note change in Metrica's coordinate system since the last lesson)
tracking_home = mio.to_metric_coordinates(tracking_home)
tracking_away = mio.to_metric_coordinates(tracking_away)
events = mio.to_metric_coordinates(events)

# reverse direction of play in the second half so that home team is always attacking from right->left
tracking_home, tracking_away, events = mio.to_single_playing_direction(tracking_home, tracking_away, events)

GK_numbers = [mio.find_goalkeeper(tracking_home),mio.find_goalkeeper(tracking_away)]
home_attack_direction = mio.find_playing_direction(tracking_home,'Home') # 1 if shooting left-right, else -1

#Set some global variables
player_ids = np.unique(list(c[:-2] for c in tracking_home.columns if c[:4] in ['Home', 'Away']))
maxspeed = 12
dt = tracking_home['Time [s]'].diff()
second_half_idx = tracking_home.Period.idxmax(2)

# Using Laurie's smoothing code
tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True)
tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True)

#Obtain the Unique Players
home_players = np.unique(list(c.split('_')[1] for c in tracking_home.columns if c[:4] == 'Home'))
away_players = np.unique(list(c.split('_')[1] for c in tracking_away.columns if c[:4] == 'Away'))

# Calculate these measures while in possession and out of possession
# Calculate the physical metrics of high or low EPV possessions (calculate each possession)
params = mpc.default_model_params()

EPV = mepv.load_EPV_grid(DATADIR+'/EPV_grid.csv')
mviz.plot_EPV(EPV,field_dimen=(106.0,68),attack_direction=home_attack_direction)

pass_events = events[events['Type'] == 'PASS']
pass_events['Poss_Seq'] = pass_events['Team'].ne(
    pass_events['Team'].shift()).cumsum()

home_poss = pass_events[pass_events['Team']=='Home']

home_poss_list = []
for i in np.unique(home_poss['Poss_Seq']):
    print(i)
    start_time = min(home_poss[home_poss['Poss_Seq']==i]['Start Time [s]'])
    end_time = max(home_poss[home_poss['Poss_Seq']==i]['End Time [s]'])
    half_temp = np.unique(home_poss[home_poss['Poss_Seq']==i]['Period'])
    #Get the total distance of both teams as well as the total EPV
    pass_poss = home_poss[home_poss['Poss_Seq']==i]

    poss_distance = []
    tracking_poss = tracking_home[(tracking_home['Time [s]']>=start_time) & (tracking_home['Time [s]']<=end_time) & (tracking_home['Period'].isin(half_temp))]
    for player in home_players:
        column = 'Home_' + player + '_speed'
        player_distance = tracking_poss.loc[tracking_poss[column] >= 3,column].sum() / 25. / 1000
        poss_distance.append(player_distance)

    opp_distance = []
    tracking_opp = tracking_away[
        (tracking_away['Time [s]'] >= start_time) & (tracking_away['Time [s]'] <= end_time) & (
            tracking_away['Period'].isin(half_temp))]
    for player in away_players:
        column = 'Away_' + player + '_speed'
        player_distance = tracking_opp.loc[tracking_opp[column] >= 3,column].sum() / 25. / 1000
        opp_distance.append(player_distance)
    eepv_added = []
    for i in pass_poss.index:
        EEPV_added, EPV_diff = mepv.calculate_epv_added(i, events, tracking_home, tracking_away, GK_numbers,
                                                        EPV, params)
        eepv_added.append(EEPV_added)
    total_dist = np.sum(poss_distance)
    total_opp_dist = np.sum(opp_distance)
    total_eepv = np.sum(eepv_added)
    home_poss_list.append([total_dist,total_opp_dist,total_eepv])

home_eepv_df = pd.DataFrame(np.array(home_poss_list).reshape(68,3), columns = ['HomeDist','AwayDist','EEPV'])

home_eepv_df.plot.scatter(x='HomeDist',
                      y='EEPV')

lm = sklearn.linear_model.LinearRegression().fit(np.array(home_eepv_df['HomeDist']).reshape(-1,1),np.array(home_eepv_df['EEPV']).reshape(-1,1))
lm_score = lm.score(np.array(home_eepv_df['HomeDist']).reshape(-1,1),np.array(home_eepv_df['EEPV']).reshape(-1,1))
#0.6397945808713286, 0.6730582132032926, 0.7568845621055171, 0.7869140810900896, 0.752852377695006, 0.6046637586000496
y = np.array(home_eepv_df['EEPV']).reshape(-1,1)
yhat = lm.predict(np.array(home_eepv_df['HomeDist']).reshape(-1,1))

plt.scatter(home_eepv_df['HomeDist'],home_eepv_df['EEPV'])
plt.plot(home_eepv_df['HomeDist'],yhat,color="red")
plt.title("Total Distance [>= 3m/s]")
plt.annotate(lm_score,xy=(1,.2))

#manually calculate lm_score
#SS_Residual = sum((y-yhat)**2)
#SS_Total = sum((y-np.mean(y))**2)
#r_squared = 1 - (float(SS_Residual))/SS_Total

##Optional: Are the EPV chances statistically better when the team is pacing above average together?

