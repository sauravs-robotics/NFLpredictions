from bs4 import BeautifulSoup
import requests
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd

#using teamrankings.com to gather different NFL rankings
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

#two different ways to get NFL scores, one using web scraping (weeklyscores),
#one using a github link that already gathered the games_data (gitweeklyscores)
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
    #weeks 1-18
    for i in range(1,18):
        urlp2 = str(i) + "/"
        url = urlp1 + urlp2
        time.sleep(5)
        html_content1 = requests.get(url).text
        time.sleep(5)
        soup2 = BeautifulSoup(html_content1, "lxml")
        #finds within the tables where the team names and scores are
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

#pick what year to run data on, to gather multiple years run program multiple times
y = int(input("Please enter NFL year to examine:\n"))
print(type(y))
tr_year = y + 1
url1 = "https://www.teamrankings.com/nfl/stat/yards-per-play?date=" + str(tr_year) + "-02-09"
url2 = "https://www.teamrankings.com/nfl/stat/opponent-yards-per-play?date=" + str(tr_year) + "-02-09"
t1ns, r1 = teamrankings(url1) #gather rankings of choice
t2ns, r2 = teamrankings(url2)
away_teams_all_ns, home_teams_all_ns, results_all = gitweeklyscores(y)
print()
 #ns = not sorted
print(results_all)

#manually making all team names the same because some websites use full team names
#,others use abbreviations; more useful if gathering rankings from multiple sites
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

away_teams_all = team_mapping(away_teams_all_ns)
home_teams_all = team_mapping(home_teams_all_ns)

#aligning all the ranking vectors into one correctly ordered matrix of 32*2 (rows= teams, columns=rankings)
def align_ranks(team1,team2,rank1, rank2):
    nu_teams = len(team1)
    rank_mat = np.zeros((32,2))
    for i in range(0,nu_teams):
        for j in range(nu_teams):
            if team1[i] == team2[j]:
                rank_mat[i][0] = rank1[i]
                rank_mat[i][1] = rank2[j]
    return rank_mat

rank_mat = align_ranks(t1,t2,r1,r2) # team order same as t1

#iterating through all NFL games and creating input for NN consisting of all rankings for both teams
#matrix = 256*4 (rows = number of games, columns = all rankings for both teams)
def create_inputs(t1,rank_mat,away_teams, home_teams):
    nugames = len(away_teams)
    feature_mat = np.zeros((nugames,4))
    holder = 0
    for i in range(0,nugames):
        for j in range (0,len(t1)):
            if away_teams[i] == t1[j]:
                feature_mat[i][2] = rank_mat[j][0]
                feature_mat[i][3] = rank_mat[j][1]
                holder +=1
            elif home_teams[i] == t1[j]:
                feature_mat[i][0] = rank_mat[j][0]
                feature_mat[i][1] = rank_mat[j][1]
                holder +=1
            if holder ==2:
                break
        holder = 0
    return feature_mat

X = create_inputs(t1,rank_mat, away_teams_all, home_teams_all)
y_actual = results_all.T #making row vector



#saving input data into csv file for other python files to grab and enter into NN
n1 = "1orderedrankings" + str(y) + '.csv'
n2 = "1orderedresults" + str(y) + '.csv'
np.savetxt(n1, X, delimiter=",")
np.savetxt(n2, y_actual, delimiter=",")
