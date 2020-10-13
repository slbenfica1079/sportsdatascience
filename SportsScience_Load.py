'''


Credits to Laurie Shaw's tutorial on Friends of Tracking: https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking
'''

import Metrica_IO as mio
import Metrica_Viz as mviz
import Metrica_Velocities as mvel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
import statsmodels.formula.api as smf
import scipy as sp
import Metrica_PitchControl as mpc
import Metrica_EPV as mepv
import seaborn
from sklearn import linear_model
import os


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

# Calculate the Player Velocities (Explain why it is important and the drawbacks of optical based physical metrics)
player_ids = np.unique(list(c[:-2] for c in tracking_home.columns if c[:4] in ['Home', 'Away']))
maxspeed = 12
dt = tracking_home['Time [s]'].diff()
second_half_idx = tracking_home.Period.idxmax(2)

for player in player_ids:
    vx = tracking_home[player + "_x"].diff() / dt
    vy = tracking_home[player + "_y"].diff() / dt

    if maxspeed > 0:
        # remove unsmoothed data points that exceed the maximum speed (these are most likely position errors)
        raw_speed = np.sqrt(vx ** 2 + vy ** 2)
        vx[raw_speed > maxspeed] = np.nan
        vy[raw_speed > maxspeed] = np.nan

    raw_speed = np.sqrt(vx ** 2 + vy ** 2)
    tracking_home[player + "_speed"] = raw_speed

fig, ax = plt.subplots(figsize=(12, 8))
#ax.plot(range(1, second_half_idx), tracking_home.loc[1:67941]['Home_5_speed'])
ax.plot(range(1, 9001), tracking_home.loc[1:9000]['Home_5_speed'])
ax.title.set_text('Unsmoothed Velocities (Home_5)')
#tracking_home.loc[1:67941][['Home_5_speed']].boxplot().set_title('Unsmoothed Velocities (Home_5)')

unsmoothed_vel = tracking_home.loc[1:9000][['Home_5_speed']]

# Using Laurie's smoothing code
tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True)
tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(range(1, 9001), tracking_home.loc[1:9000]['Home_5_speed'])
ax.title.set_text('Smoothed Velocities (Home_5)')
#tracking_home.loc[1:67941][['Home_5_speed']].boxplot().set_title('Smoothed Velocities (Home_5)')

smoothed_vel = tracking_home.loc[1:9000][['Home_5_speed']]

plt.plot(unsmoothed_vel,label="Unsmoothed")
plt.plot(smoothed_vel, label = "Smoothed")
plt.title('Home_5 Velocities')
plt.legend()

# Calculate some simple measures (distance covered, high speed distance,
# distance covered in high acceleration/deceleration, # of accelerations/decelerations, what is in Laurie's code)

# Total Distance
home_players = np.unique(list(c.split('_')[1] for c in tracking_home.columns if c[:4] == 'Home'))
home_summary = pd.DataFrame(index=home_players)

minutes_home = []
for player in home_players:
    # search for first and last frames that we have a position observation for each player (when a player is not on the pitch positions are NaN)
    column = 'Home_' + player + '_x' # use player x-position coordinate
    player_minutes = ( tracking_home[column].last_valid_index() - tracking_home[column].first_valid_index() + 1 ) / 25 / 60. # convert to minutes
    minutes_home.append( player_minutes )
home_summary['Minutes Played'] = minutes_home
home_summary = home_summary.sort_values(['Minutes Played'], ascending=False)

distance_home = []
for player in home_summary.index:
    column = 'Home_' + player + '_speed'
    player_distance = tracking_home[
                          column].sum() / 25. / 1000  # this is the sum of the distance travelled from one observation to the next (1/25 = 40ms) in km.
    distance_home.append(player_distance)
home_summary['Distance [km]'] = distance_home

# home_summary['Distance [km]'][home_summary.index.values == '8'] / home_summary['Minutes Played'][home_summary.index.values == '8']

away_players = np.unique(list(c.split('_')[1] for c in tracking_away.columns if c[:4] == 'Away'))
away_summary = pd.DataFrame(index=away_players)

minutes_away = []
for player in away_players:
    # search for first and last frames that we have a position observation for each player (when a player is not on the pitch positions are NaN)
    column = 'Away_' + player + '_x' # use player x-position coordinate
    player_minutes = (tracking_away[column].last_valid_index() - tracking_away[column].first_valid_index() + 1 ) / 25 / 60. # convert to minutes
    minutes_away.append( player_minutes )
away_summary['Minutes Played'] = minutes_away
away_summary = away_summary.sort_values(['Minutes Played'], ascending=False)

distance_away = []
for player in away_summary.index:
    column = 'Away_' + player + '_speed'
    player_distance = tracking_away[
                          column].sum() / 25. / 1000  # this is the sum of the distance travelled from one observation to the next (1/25 = 40ms) in km.
    distance_away.append(player_distance)
away_summary['Distance [km]'] = distance_away

home_summary['Team'] = 'Home'
away_summary['Team'] = 'Away'

game_summary = home_summary.append(away_summary)
game_summary['isSub'] = np.where(game_summary['Minutes Played']==94.104,0,1)
game_summary_sorted = game_summary.sort_values(by=['Distance [km]'], ascending=False)
game_summary_sorted['Player'] = game_summary_sorted.index
game_summary_sorted['Player'] = np.where(game_summary_sorted['isSub']==0,game_summary_sorted['Player'],game_summary_sorted['Player']+'*')
fg = seaborn.factorplot(x='Player', y='Distance [km]', hue='Team', kind='bar', data=game_summary_sorted,legend=True).ax.set_title("Distance Covered by Player by Team [km]")

#mviz.plot_frame( tracking_home.loc[51], tracking_away.loc[51], include_player_velocities=False, annotate=True)

# Distance at certain speed bands + high acceleration
walking = []
jogging = []
running = []
sprinting = []
for player in home_summary.index:
    column = 'Home_' + player + '_speed'
    # walking (less than 2 m/s)
    player_distance = tracking_home.loc[tracking_home[column] < 2, column].sum() / 25. / 1000
    walking.append(player_distance)
    # jogging (between 2 and 4 m/s)
    player_distance = tracking_home.loc[
                          (tracking_home[column] >= 2) & (tracking_home[column] < 4), column].sum() / 25. / 1000
    jogging.append(player_distance)
    # running (between 4 and 7 m/s)
    player_distance = tracking_home.loc[
                          (tracking_home[column] >= 4) & (tracking_home[column] < 7), column].sum() / 25. / 1000
    running.append(player_distance)
    # sprinting (greater than 7 m/s)
    player_distance = tracking_home.loc[tracking_home[column] >= 7, column].sum() / 25. / 1000
    sprinting.append(player_distance)

home_summary['Walking [km]'] = walking
home_summary['Jogging [km]'] = jogging
home_summary['Running [km]'] = running
home_summary['Sprinting [km]'] = sprinting

ax = home_summary[['Walking [km]','Jogging [km]','Running [km]','Sprinting [km]']].plot.bar(colormap='coolwarm')
ax.set_xlabel('Player')
ax.set_ylabel('Distance covered [m]')
ax.set_title('Distance Covered At Various Velocity Bands')

# Calculate # of Accelerations and Decelerations
pd.options.mode.chained_assignment = None  # default='warn'
maxacc = 6
home_acc_dict = {}
for player in home_players:
    print(player)
    tracking_home['Home_' + player + '_Acc'] = tracking_home['Home_' + player + '_speed'].diff() / dt
    tracking_home['Home_' + player + '_Acc'].loc[np.absolute(tracking_home['Home_' + player + '_Acc']) > maxacc] = np.nan
    tracking_home['Home_' + player + '_Acc_type'] = np.where(np.absolute(tracking_home['Home_' + player + '_Acc']) >= 2,
                                                             "High", "Low")
    tracking_home['Home_' + player + '_Acc_g'] = tracking_home['Home_' + player + '_Acc_type'].ne(
        tracking_home['Home_' + player + '_Acc_type'].shift()).cumsum()

    for g in np.unique(tracking_home['Home_' + player + '_Acc_g']):
        acc_temp = tracking_home[tracking_home['Home_' + player + '_Acc_g'] == g]
        if acc_temp['Home_' + player + '_Acc_type'].iloc[0] == 'High':
            acc_duration = round(max(acc_temp['Time [s]']) - min(acc_temp['Time [s]']), 2)
            acc_or_dec = np.where(np.mean(acc_temp['Home_'+player+'_Acc']) > 0, "Acc", "Dec")
            home_acc_dict[len(home_acc_dict) + 1] = {'Player': player, 'Group': g, 'Duration': acc_duration,
                                                     'Type': acc_or_dec}

home_acc_df = pd.DataFrame.from_dict(home_acc_dict,orient='index')
home_acc_df['Duration'].describe()
plt.boxplot(home_acc_df['Duration'])
home_acc_df1 = home_acc_df[home_acc_df['Duration']>=.75]

accdec = []
for player in home_players:
    accs = home_acc_df1[(home_acc_df1['Player']==player) & (home_acc_df1['Type']=='Acc')].count()[0]
    decs = home_acc_df1[(home_acc_df1['Player']==player) & (home_acc_df1['Type']=='Dec')].count()[0]
    ac_ratio = accs / decs
    accdec.append(ac_ratio)

home_summary['AccDec'] = accdec
home_summary.plot.scatter(x='Distance [km]',y='AccDec')
for i in home_summary.index:
    plt.text(home_summary[home_summary.index==i]['Distance [km]'], home_summary[home_summary.index==i]['AccDec'], str(i))
plt.title("Acceleration - Deceleration Ratio")


# Introduce concept of metabolic power and SPI
def split_at(s, c, n):
    words = s.split(c)
    return c.join(words[:n]), c.join(words[n:])

def metabolic_cost(acc): #https://jeb.biologists.org/content/221/15/jeb182303
    if acc > 0:
        cost = 0.102 * ((acc ** 2 + 96.2) ** 0.5) * (4.03 * acc + 3.6 * np.exp(-0.408 * acc))
    elif acc < 0:
        cost = 0.102 * ((acc ** 2 + 96.2) ** 0.5) * (-0.85 * acc + 3.6 * np.exp(1.33 * acc))
    else:
        cost = 0
    return cost

team = tracking_home

#def metabolic_power(team):
    playerids = np.unique(list(c[:-2] for c in team.columns if c[:4] in ['Home', 'Away']))
    playerids = np.unique(list(map(lambda x: split_at(x, '_', 2)[0], playerids)))

    #for player in playerids:
        player = 'Home_6'
        mc_temp = list(map(lambda x: metabolic_cost(team[player + '_Acc'][x]), range(1, len(team[player + '_Acc'])+1)))
        #team[player+'_MP'] = mc_temp * team[player+'_speed']
        mp_temp = mc_temp * team[player+'_speed']
        test_mp = mp_temp.rolling(7500,min_periods=1).apply(lambda x : np.nansum(x)) #Use Changepoint Detection Here
        plt.plot(test_mp)
        plt.title('Metabolic Power Output [5 min Rolling Window]')

        #Bin Seg
        signal = np.array(test_mp[7500:len(test_mp)]).reshape((len(test_mp[7500:len(test_mp)]),1))
        algo = rpt.Binseg(model="l2").fit(signal)  ##potentially finding spot where substitution should happen
        result = algo.predict(n_bkps=1)  # big_seg
        rpt.show.display(signal, result, figsize=(10, 6))
        plt.title('Metabolic Power Output [5 min Rolling Window]')

        #PELT
        algo = rpt.Pelt(model="l2",min_size=7500).fit(signal)
        result = algo.predict(pen=np.log(len(signal))*1*np.std(signal)**2) ##Potentially pacing strategy or identifying moments in the game that are slower
        rpt.show.display(signal, result, figsize=(10, 6))
        plt.title('Metabolic Power Output [5 min Rolling Window]')

#SPI and Measure the minute after (Show it first for Home_6 before running the loop)
home_spi_list = []

for player in home_players:
    print(player)
    test_spi = tracking_home['Home_'+player+'_speed'].rolling(1500,min_periods=1).apply(lambda x : np.nansum(x)) / 25.
    xcoords = sp.signal.find_peaks(test_spi, distance=1500)
    spi_values = list(map(lambda x: test_spi[x], xcoords[0]))
    spi_values_index = np.argsort(spi_values)[-3:]
    spi_index = xcoords[0][spi_values_index]
    for i in range(len(spi_index)):
        spi_temp = spi_index[i]
        spi_value_temp = spi_values[spi_values_index[i]]
        spi_min_after = sum(tracking_home['Home_'+player+'_speed'][spi_temp+2:spi_temp+1502]) / 25. # Find the top 3 for each player and then can do a lmm (Diff From Avg ~ 1, group == Player)
        spi_append = [player,'Dist',spi_value_temp,spi_min_after]
        home_spi_list.append(spi_append)

    test_hsd_spi = pd.Series(np.where(tracking_home['Home_'+player+'_speed'] >= 5,tracking_home['Home_'+player+'_speed'],0)).rolling(1500,min_periods=1).apply(lambda x : np.nansum(x)) / 25.
    xcoords = sp.signal.find_peaks(test_hsd_spi, distance=1500)
    hsd_values = list(map(lambda x: test_hsd_spi[x], xcoords[0]))
    hsd_values_index = np.argsort(hsd_values)[-3:]
    hsd_index = xcoords[0][hsd_values_index]
    for i in range(len(hsd_index)):
        hsd_temp = hsd_index[i]
        hsd_value_temp = hsd_values[hsd_values_index[i]]
        hsd_min_after = sum(tracking_home['Home_' + player + '_speed'][hsd_temp+ 2:hsd_temp+ 1502]) / 25.
        hsd_append = [player,'HSD',hsd_value_temp,hsd_min_after]
        home_spi_list.append(hsd_append)

home_summary['DPM'] = 1000*(home_summary['Distance [km]'] / home_summary['Minutes Played'])

spi_df = pd.DataFrame(np.array(home_spi_list).reshape(83,4), columns = ['Player','Type','SPI','MinAfter'])
merged = pd.merge(spi_df, home_summary[['DPM']], left_on='Player', right_index=True)
hsd_df = merged[merged['Player']!='11']
hsd_df_lmm = hsd_df[~hsd_df['MinAfter'].str.contains("nan")]
hsd_df_lmm['MinAfter'] = pd.to_numeric(hsd_df_lmm['MinAfter'])
hsd_df_lmm['Diff'] = hsd_df_lmm['MinAfter'] - hsd_df_lmm['DPM']
hsd_df_lmm['Team'] = 'Home'
away_spi_list = []

for player in away_players:
    print(player)
    test_spi = tracking_away['Away_'+player+'_speed'].rolling(1500,min_periods=1).apply(lambda x : np.nansum(x)) / 25.
    xcoords = sp.signal.find_peaks(test_spi, distance=1500)
    spi_values = list(map(lambda x: test_spi[x], xcoords[0]))
    spi_values_index = np.argsort(spi_values)[-3:]
    spi_index = xcoords[0][spi_values_index]
    for i in range(len(spi_index)):
        spi_temp = spi_index[i]
        spi_value_temp = spi_values[spi_values_index[i]]
        spi_min_after = sum(tracking_away['Away_'+player+'_speed'][spi_temp+2:spi_temp+1502]) / 25. # Find the top 3 for each player and then can do a lmm (Diff From Avg ~ 1, group == Player)
        spi_append = [player,'Dist',spi_value_temp,spi_min_after]
        away_spi_list.append(spi_append)

    test_hsd_spi = pd.Series(np.where(tracking_away['Away_'+player+'_speed'] >= 5,tracking_away['Away_'+player+'_speed'],0)).rolling(1500,min_periods=1).apply(lambda x : np.nansum(x)) / 25.
    xcoords = sp.signal.find_peaks(test_hsd_spi, distance=1500)
    hsd_values = list(map(lambda x: test_hsd_spi[x], xcoords[0]))
    hsd_values_index = np.argsort(hsd_values)[-3:]
    hsd_index = xcoords[0][hsd_values_index]
    for i in range(len(hsd_index)):
        hsd_temp = hsd_index[i]
        hsd_value_temp = hsd_values[hsd_values_index[i]]
        hsd_min_after = sum(tracking_away['Away_' + player + '_speed'][hsd_temp+ 2:hsd_temp+ 1502]) / 25.
        hsd_append = [player,'HSD',hsd_value_temp,hsd_min_after]
        away_spi_list.append(hsd_append)

away_summary['DPM'] = 1000*(away_summary['Distance [km]'] / away_summary['Minutes Played'])

spi_df = pd.DataFrame(np.array(away_spi_list).reshape(72,4), columns = ['Player','Type','SPI','MinAfter'])
merged = pd.merge(spi_df, away_summary[['DPM']], left_on='Player', right_index=True)
hsd_df = merged[merged['Player']!='25']
hsd_df_lmm_away = hsd_df[~hsd_df['MinAfter'].str.contains("nan")]
hsd_df_lmm_away['MinAfter'] = pd.to_numeric(hsd_df_lmm_away['MinAfter'])
hsd_df_lmm_away['Diff'] = hsd_df_lmm_away['MinAfter'] - hsd_df_lmm_away['DPM']
hsd_df_lmm_away['Team'] = 'Away'

hsd_full = hsd_df_lmm.append(hsd_df_lmm_away)

md_hsd = smf.mixedlm("Diff ~ 1", hsd_full[hsd_full['Type']=='HSD'], groups=hsd_full[hsd_full['Type']=='HSD']['Player'])
md_dist = smf.mixedlm("Diff ~ 1", hsd_full[hsd_full['Type']=='Dist'], groups=hsd_full[hsd_full['Type']=='Dist']['Player'])

mdf_hsd = md_hsd.fit(method='cg')
print(mdf_hsd.summary())
#np.mean(hsd_full[hsd_full['Type']=='HSD']['Diff'])

mdf_dist = md_dist.fit(method='cg')
print(mdf_dist.summary())

# Combine the above concepts with pitch control to measure what space is available or not available
# Take a certain threshold and measure what % of what he normally covers in possession or out possession




