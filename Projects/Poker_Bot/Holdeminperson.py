import Dealer as D

hearts = ['AH','KH','QH','JH','TH','9H','8H','7H','6H','5H','4H','3H','2H']
spades = ['AS','KS','QS','JS','TS','9S','8S','7S','6S','5S','4S','3S','2S']
clubs = ['AC','KC','QC','JC','TC','9C','8C','7C','6C','5C','4C','3C','2C']
diamonds = ['AD','KD','QD','JD','TD','9D','8D','7D','6D','5D','4D','3D','2D']

deck = hearts+clubs+spades+diamonds

def ask_bot(bot):
    pass


def ask_decision(name,playertype):
    if playertype == 'human':
        choice = input('What would you like to do?')
        bet_amount = input('How much would you like to bet?')
    if playertype == 'bot':
        choice = ask_bot(name)
        bet_amount =ask_bot(name)
    return name,[choice,bet_amount]

def error_check(name,prev_dec,playertype):
    x = 1
    dec = prev_dec[0]
    bet = prev_dec[1]
    answer = ask_decision(playertype)
    decnow = answer[0]
    betnow = int(answer[1])
    if dec == 'Check':
         decs = ['Bet','Check','Call','Raise','Fold']
    if dec == 'Bet':
        decs = ['Call','Raise','Fold']
    if dec == 'Raise':
         decs = ['Call','Raise','Fold']  
    if dec == 'First':
        decs = ['Bet','Check','Call','Raise','Fold']
    if dec =='Call':
        decs = ['Call','Raise','Fold']
    print(bet)
    print(betnow)
    if decnow not in decs:
        x = 0
    if decnow == 'Check':
        if betnow !=0:
             x=0
    if decnow == 'Bet':
        if betnow <=0:
            x=0
    if decnow =='Call':
        if betnow != bet:
            x=0
    if decnow =='Raise':
        if betnow < 2*bet:
            x=0
    if decnow =='Fold':
        if betnow !=0:
            x = 0
    if type(betnow) != int:
        x=0

    if x ==0:
        return [False]
    if x==1: return [True,[decnow,betnow],name]

def five_chances(name,prev_dec,playertype):
    answer = error_check(name,prev_dec,playertype)
    if not answer[0]:
        print("You can't do that. (Mistake 1/5)")
        answer = error_check(name,prev_dec,playertype)
        if not answer[0]:
            print("Try Again. (Mistake 2/5)")
            answer = error_check(name,prev_dec,playertype)
            if not answer[0]:
                print("Are you trying to cheat? (Mistake 3/5)")
                answer = error_check(name,prev_dec,playertype)
                if not answer[0]:
                    print("If you mess up again, consider yourself folded. (Mistake 4/5)")
                    answer = error_check(name,prev_dec,playertype)
                    if not answer[0]:
                        print("I warned you. (Mistake 5/5)")
                        returner = ['Fold',0]
                        return returner,name
    return [answer[1],name]


#playerdata is a list that includes name, bot/human, chip amount. chair can be deterined by list order
#the ways to end a turn for two players are: 2 checks, any call, any fold

players = [['Coby','human',1000]]

def begin_hand(player_data,BBamount,SBamount):

    decisions_list = [['First',0]]
    playercount = len(player_data)
    hands = D.deal_cards(deck,playernum=playercount)
    pot = SBamount+BBamount
    player_data[0][2] = player_data[0][2]-SBamount
    player_data[1][2] = player_data[1][2]-BBamount
    community = []
    if playercount ==2:
        players = [player_data[0][0],player_data[1][0]]
        for n in range(len(player_data)):
            response = five_chances(player_data[n][0],decisions_list[-1],player_data[n][1])
            decisions_list.append(response[0])
            if response[0][0] == 'Fold':
                players.remove(player_data[n][0])
                return players
        if decisions_list[-1][0] in ['Call','Check']:
            
        if decisions_list[-1][0] == ['Raise']:

    


