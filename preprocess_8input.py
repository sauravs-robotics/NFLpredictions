from bs4 import BeautifulSoup
import requests
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd

#THIS USES THE SAME LOGIC AS preprocess_4input.py, except now I am using
#4 different rankings instead of 2
def teamrankings(url):
    teams = []
    ranks = []
    html_content = requests.get(url).text
    soup = BeautifulSoup(html_content, "lxml")
    #print(soup.prettify())
    #print(soup.title)
    rank_table = soup.find("table")
    for row in rank_table.tbody.find_all("tr"):
        team = row.find('td', attrs = {"class": "text-left"}).text
        #format = city name w abbresviations for LA Chargers/Rams
        rank = row.find('td', attrs = {"class": "text-right"}).text
        teams.append(team)
        ranks.append(float(rank))
    return teams, ranks

def gitweeklyscores(year):
    url = 'https://raw.githubusercontent.com/ryurko/nflscrapR-data/master/games_data/regular_season/reg_games_' + str(year) + '.csv'
    df = pd.read_csv(url)
    drop_columns = ['type', 'state_of_game', 'game_url','game_id','week', 'season']
    df.drop(columns=drop_columns, axis=1, inplace=True)
    nugames = df.shape[0]
    ateam = df['away_team']
    hteam = df['home_team']
    away_teams_score = df['away_score']
    home_teams_score = df['home_score']
    results = np.zeros((1,nugames))
    for i in range(0,nugames):
        results[0][i] = home_teams_score[i] > away_teams_score[i]

    return ateam, hteam, results

def weeklyscores(year):
    urlp1 = "https://www.cbssports.com/nfl/scoreboard/all/" + str(year) + "/regular/"
    teams = []
    scores = []
    for i in range(1,18):
        urlp2 = str(i) + "/"
        url = urlp1 + urlp2
        time.sleep(5)
        html_content1 = requests.get(url).text
        time.sleep(5)
        soup2 = BeautifulSoup(html_content1, "lxml")
        for mdata in soup2.find_all("table"):
            try:
                gg = mdata.find("tbody")
                for boxscore in gg.find_all("tr"):

                    try:
                        t = boxscore.find("td", attrs={"class": "team"})
                        team = t.find("a", attrs={"class": "team"}).text
                        score = boxscore.find("td", attrs={"class": "total-score"}).text
                        score = int(score)
                    except Exception:
                        uselessvar = 0
                    else:
                        teams.append(team)
                        scores.append(score)
            except:
                uselessvar = 0
    totalteams = len(teams)
    away_teams = []
    home_teams = []
    n = int(totalteams/2)
    results = np.zeros((1,n))
    for j in range(0,totalteams):
        if j%2 == 0:
            away_teams.append(teams[j])
            idx = int(j/2)
            if j<totalteams-2:
                results[0][idx] = scores[j+1] > scores[j]
        else:
            home_teams.append(teams[j])
    return away_teams, home_teams, results

y = int(input("Please enter NFL year to examine:\n"))
tr_year = y + 1
url1 = "https://www.teamrankings.com/nfl/stat/yards-per-rush-attempt?date=" + str(tr_year) + "-02-09"
url2 = "https://www.teamrankings.com/nfl/stat/yards-per-pass-attempt?date=" + str(tr_year) + "-02-09"
url3 = "https://www.teamrankings.com/nfl/stat/opponent-yards-per-rush-attempt?date=" + str(tr_year) + "-02-09"
url4 = "https://www.teamrankings.com/nfl/stat/opponent-yards-per-pass-attempt?date=" + str(tr_year) + "-02-09"
t1ns, r1 = teamrankings(url1)
t2ns, r2 = teamrankings(url2)
t3ns, r3 = teamrankings(url3)
t4ns, r4 = teamrankings(url4)
away_teams_all_ns, home_teams_all_ns, results_all = gitweeklyscores(y)
 #ns = not sorted

def team_mapping(teams):
    for i in range(0,len(teams)):
        if ("zona" in teams[i]) or ("Card" in teams[i]):
            teams[i] = "AZ"
        elif ("Atl" in teams[i]) or ("Falc" in teams[i]):
            teams[i] = "ATL"
        elif ("Bal" in teams[i]) or ("Rav" in teams[i]):
            teams[i] = "BAL"
        elif ("Buf" in teams[i]) or ("Bil" in teams[i]):
            teams[i] = "BUF"
        elif ("Caro" in teams[i]) or ("Pant" in teams[i]):
            teams[i] = "CAR"
        elif ("Cin" or "Beng") in teams[i] or ("Beng" in teams[i]):
            teams[i] = "CIN"
        elif ("Chic" in teams[i]) or ("Bear" in teams[i]):
            teams[i] = "CHI"
        elif ("Cle" in teams[i]) or ("Brown" in teams[i]):
            teams[i] = "CLE"
        elif ("Dal" in teams[i]) or ("Cow" in teams[i]):
            teams[i] = "DAL"
        elif ("Den" in teams[i]) or ("Bro" in teams[i]):
            teams[i] = "DEN"
        elif ("Det" in teams[i]) or ("Lio" in teams[i]):
            teams[i] = "DET"
        elif ("Gre" or "Pack") in teams[i] or ("Pack" in teams[i]):
            teams[i] = "GB"
        elif ("Hou" or "Tex") in teams[i] or ("Tex" in teams[i]):
            teams[i] = "HOU"
        elif ("Ind" or "Colt") in teams[i] or ("Colt" in teams[i]):
            teams[i] = "IND"
        elif ("Kan" or "Chief") in teams[i] or ("Chief" in teams[i]):
            teams[i] = "KAN"
        elif "Charg"  in teams[i]:
            teams[i] = "LAC"
        elif "Rams" in teams[i]:
            teams[i] = "LAR"
        elif ("Jack" in teams[i]) or ("Jag" in teams[i]):
            teams[i] = "JAX"
        elif ("Mia" in teams[i]) or ("Dol" in teams[i]):
            teams[i] = "MIA"
        elif ("Min" in teams[i]) or ("Vik" in teams[i]):
            teams[i] = "MIN"
        elif ("Engl" in teams[i]) or ("Pat" in teams[i]):
            teams[i] = "NE"
        elif ("Orl" in teams[i]) or ("Sai" in teams[i]):
            teams[i] = "NO"
        elif "Giants" in teams[i]:
            teams[i] = "NYG"
        elif "Jets" in teams[i]:
            teams[i] = "NYJ"
        elif ("Oak" in teams[i]) or ("Rai" in teams[i]):
            teams[i] = "OAK"
        elif ("Phil" in teams[i]) or ("Eag" in teams[i]):
            teams[i] = "PHI"
        elif ("Fran" in teams[i]) or ("49" in teams[i]):
            teams[i] = "SF"
        elif "Sea" in teams[i]:
            teams[i] = "SEA"
        elif ("Pit" in teams[i]) or ("Stee" in teams[i]):
            teams[i] = "PIT"
        elif ("Tam" in teams[i]) or ("Buc" in teams[i]):
            teams[i] = "TB"
        elif ("Ten" in teams[i]) or ("Tit" in teams[i]):
            teams[i] = "TEN"
        elif ("Wash" in teams[i]) or ("Red" in teams[i]):
            teams[i] = "WAS"
    return teams

t1 = team_mapping(t1ns)
t2 = team_mapping(t2ns)
t3 = team_mapping(t3ns)
t4 = team_mapping(t4ns)

away_teams_all = team_mapping(away_teams_all_ns)
home_teams_all = team_mapping(home_teams_all_ns)

def align_ranks(team1,team2,team3, team4, rank1, rank2, rank3, rank4):
    nu_teams = len(team1)
    rank_mat = np.zeros((32,4))
    for i in range(0,nu_teams):
        for j in range(nu_teams):
            if team1[i] == team2[j]:
                rank_mat[i][0] = rank1[i]
                rank_mat[i][1] = rank2[j]
    for i in range(0,nu_teams):
        for j in range(nu_teams):
            if team1[i] == team3[j]:
                rank_mat[i][2] = rank3[j]
    for i in range(0,nu_teams):
        for j in range(nu_teams):
            if team1[i] == team4[j]:
                rank_mat[i][3] = rank4[j]
    return rank_mat

rank_mat = align_ranks(t1,t2,t3, t4, r1,r2, r3, r4) # team order same as t1

def create_inputs(t1,rank_mat,away_teams, home_teams):
    nugames = len(away_teams)
    feature_mat = np.zeros((nugames,8))
    holder = 0
    for i in range(0,nugames):
        for j in range (0,len(t1)):
            if away_teams[i] == t1[j]:
                feature_mat[i][4] = rank_mat[j][0]
                feature_mat[i][5] = rank_mat[j][1]
                feature_mat[i][6]= rank_mat[j][2]
                feature_mat[i][7] = rank_mat[j][3]
                holder +=1
            elif home_teams[i] == t1[j]:
                feature_mat[i][0] = rank_mat[j][0]
                feature_mat[i][1] = rank_mat[j][1]
                feature_mat[i][2] = rank_mat[j][2]
                feature_mat[i][3] = rank_mat[j][3]
                holder +=1
            if holder ==2:
                break
        holder = 0
    return feature_mat


X = create_inputs(t1,rank_mat, away_teams_all, home_teams_all)
y_actual = results_all.T

#debugging
#for i in range(0,11):
#    print("Away team" + str(away_teams_all[i]) + "vs Home team"  + str(home_teams_all[i]))
#    print(results_all.shape)
#    print(X_wb[i][:])

#bias = np.zeros((256,1))
#X = np.hstack((X_wb,bias))
n1 = "2orderedrankings" + str(y) + '.csv'
n2 = "2orderedresults" + str(y) + '.csv'
np.savetxt(n1, X, delimiter=",")
np.savetxt(n2, y_actual, delimiter=",")
