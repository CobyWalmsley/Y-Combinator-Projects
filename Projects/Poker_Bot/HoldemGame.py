#players each get their hole cards
#bet
#flop is revealed
#bet
#turn is revealed
#bet
#river is revealed
#bet
#show hands
#decide a winner

import Dealer as D
hearts = ['AH','KH','QH','JH','TH','9H','8H','7H','6H','5H','4H','3H','2H']
spades = ['AS','KS','QS','JS','TS','9S','8S','7S','6S','5S','4S','3S','2S']
clubs = ['AC','KC','QC','JC','TC','9C','8C','7C','6C','5C','4C','3C','2C']
diamonds = ['AD','KD','QD','JD','TD','9D','8D','7D','6D','5D','4D','3D','2D']

deck = hearts+clubs+spades+diamonds

def update_turn(actionlist,community,flop,turn,river,player1hand,player2hand,p1chips,p2chips,potsize):
    if actionlist[-1][0]==1:
        player = 2
    if actionlist[-1][0]==2:
        player = 1
    turn = actionlist[-1][1]
    if actionlist[-1][2] == 'First':
        action = askdecision(actionlist)
        actionlist.append([player,turn]+action)
    if actionlist[-1][2] == 'Check':
        if actionlist[-2][2] == 'Check':
            turn+=1
            if turn==2:
                community = community+flop
            if turn==3:
                community = community+turn
            if turn ==4:
                community = community +river
            if turn ==5:
                return D.whowins(player1hand,player2hand)
        if actionlist[-2][2] != 'Check':
            action = askdecision(actionlist)
            actionlist.append([player,turn]+action)

    if actionlist[-1][2] == 'Bet':
        action = askdecision(actionlist)
        actionlist.append([player,turn]+action)

    if actionlist[-1][2] == 'Call':
        turn+=1
        if turn==2:
            community = community+flop
        if turn==3:
            community = community+turn
        if turn ==4:
            community = community +river
        if turn ==5:
            return D.whowins(player1hand,player2hand)

    if actionlist[-1][2] == 'Raise':
        action = askdecision(actionlist)
        actionlist.append([player,turn]+action)
        #flip player, dont flip turn

    if actionlist[-1][2] == 'Fold':
        if player == 1:
            p1chips+=potsize
            potsize = 0
        if player ==2:
            p2chips+=potsize
            potsize = 0
    

    
        


#actionlist is a list of actions denoted [playernum,turnnum,action,bet_amount]      
#There are 4 actions the game can take depending on what just happened. If the previous player called, update turn
    #reveal another card, ask SB to bet.
    #If the previous player checked, see if its the second check of this turn. If not, ask the second player if they want to check or bet.
    #if it is, update the turn, revel another card.
    #If the previous player raised, do not change the turn and ask the original better to call
    #If the previous player bet, do not change the turn and ask the other player to call.
    #If its the first turn, ask the other player to call.
        



def play_one_hand(player1,player2,deck,bigblindamount,smallblindamount,player1chips,player2chips,BB,SB):
    hands = D.deal_cards(deck)
    decisionslist = []
    pocketSB = hands[0]
    pocketBB = hands[1]
    flop = hands[5]
    turn = hands[6]
    river = hands[7]
    fullhandSB = hands[2]
    fullhandBB=hands[3]

    potsize = 0
    if BB == player1:player1chips = player1chips-bigblindamount
    if BB == player2:player2chips = player2chips-bigblindamount
    if SB == player1:player1chips = player1chips-smallblindamount
    if SB == player2: player2chips = player2chips-smallblindamount

    potsize = potsize +bigblindamount+smallblindamount

    
    SBdecision1 = askdecision(SB,BB,1,potsize,pocketSB,[],'SB',('BB',bigblindamount),player1chips,smallblindamount)
    if SBdecision1[0] == 'Raise':
        player1chips = player1chips-SBdecision1[1]
        potsize = potsize+SBdecision1[1]
    if SBdecision1[0] == 'Fold':
        player2chips = player2chips+potsize
    if SBdecision1[0] == 'Call':
        player1chips = player1chips-SBdecision1[1]
        potsize = potsize+SBdecision1[1]
    decisionslist.append[('BB',bigblindamount),SBdecision1]

    if decisionslist[-1][0] == 'Raise':
        updateturn()
    else





def askdecision(playername,opponentname,betnum,potsize,holecards,communitycards,chair,opponentchoice,mychipstack,mypottotal):

    #return('Call',1200)

#playername = 'Coby'
#opponentname = 'Parker'
#betnum = what turn is it? 1= hole cards only, 2=flop showing, 3=turn showing, 4=river showing
#potsize: how many chips are currently in the pot?
#what are your hole cards? ['AH','2S']
#what are the community cards showing?
# what chair are you sitting in? (SB or BB?)
#What was your opponents last choice? If you are going first, it will be a tuple ('First',bigblindamount). Otherwise it will be ('Check',0,0), ('Bet',1200,0), ('Call',1200,0), (Raise,3200,1),('Fold',0,0).
#How many chips are in your stack?
#how many chips have I put in so far?