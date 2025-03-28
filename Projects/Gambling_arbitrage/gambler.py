def are_we_happy(losses_list, gains_list, odds_list):
    total_losses = 0
    for i in losses_list:
        total_losses += i
    count = 0
    for n in zip(losses_list, gains_list, odds_list):
        if n[1] >= total_losses:
            count +=1
        
        if n[1] < total_losses:
            if n[0] != 0:
                return (count,0)
        if n[0] == 0:
            return (count,1)

        if count == len(odds_list):
            if n[0] != 0:
                return (count,3)

def update_bets_when_happy(which_bet, losses_list, gains_list, odds_list):
    total_losses2 = 0
    for i in losses_list: total_losses2 += i
    odd_in_question = odds_list[which_bet]
    bet_in_question = round((total_losses2 / odd_in_question),2)
    if bet_in_question % 1>0:
        bet_in_question = round(bet_in_question)+1
    losses_list[which_bet] = bet_in_question
    gain_in_question = round(bet_in_question * (odd_in_question + 1),2)
    gains_list[which_bet] = gain_in_question
    return [losses_list, gains_list]

def update_when_unhappy(which_bet, losses_list, gains_list, odds_list):
    total_losses3 = 0
    for i in losses_list: total_losses3 += i
    wrong_bet = losses_list[which_bet]
    odd_in_question = odds_list[which_bet]
    new_bet = round(((total_losses3 - wrong_bet) / odd_in_question),2)
    if new_bet % 1>0:
        new_bet = round(new_bet)+1
    new_gain = round(new_bet * (odd_in_question + 1),2)
    gains_list[which_bet] = new_gain
    losses_list[which_bet] = new_bet
    return [losses_list, gains_list]

def statistics(odds_list):
    limit = 0
    for odd in odds_list:
        percent = 100/odd
        limit += percent
    final_limit = limit/100
    print("number of odds:",len(odds_list))
    print("total percent of odds", final_limit)

def profit_distribution(losses_list, gains_list, odds_list):
    profits_by_bet = []
    total_cost = 0
    for m in losses_list:
        total_cost += m
    for d in gains_list:
        profit_by_bet = d - total_cost
        profits_by_bet.append(profit_by_bet)
    for g in zip(odds_list, losses_list, profits_by_bet):
        print(f"Bet ${g[1]} on the bet with {g[0]}:1 odds to make ${g[2]} profit")
    total_money_available=0
    for a in profits_by_bet:
        total_money_available += a
    print("Your total profit available is: $", total_money_available)
    print("Your total money risked is: $",total_cost)


odds_list = [7,5,2,7,8]
odds_list.sort()
bets_list = []
prizes_list = []

for n in range(len(odds_list)):
    bets_list.append(0)
    prizes_list.append(0)

bets_list[0] = 1000
prizes_list[0] = ((odds_list[0]+1)*bets_list[0])

print('odds list:', odds_list)
print('bets_list', bets_list)
print('prizes list:', prizes_list)

progress = 0
x = True
while x == True:
    progress+=1

    answer=are_we_happy(bets_list,prizes_list,odds_list)
    print(answer)
    if answer[1] == 1:
        new_lists = update_bets_when_happy(answer[0], bets_list, prizes_list, odds_list)
        new_lists[0] = bets_list
        new_lists[1] = prizes_list
       

    if answer[1] == 0:
        new_lists2 = update_when_unhappy(answer[0], bets_list, prizes_list, odds_list)
        new_lists2[0] = bets_list
        new_lists2[1] = prizes_list
        print("updated")

    if answer == (0,0):
        print("This will be impossible")
        print("Here were your odds", odds_list)
        statistics(odds_list)
        x = False

    if answer == ((len(odds_list)),3):
        print("This is possible. Here are your bets:")
        print(bets_list)
        print("Here are the odds")
        print(odds_list)
        print("Here are the payouts")
        print(prizes_list)
        statistics(odds_list)
        profit_distribution(bets_list, prizes_list, odds_list)
        x = False

    print(f"bets on round {progress}:", bets_list)
    print(f"payouts on round {progress}:", prizes_list)