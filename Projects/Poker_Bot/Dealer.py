import random as rd

hearts = ['AH','KH','QH','JH','TH','9H','8H','7H','6H','5H','4H','3H','2H']
spades = ['AS','KS','QS','JS','TS','9S','8S','7S','6S','5S','4S','3S','2S']
clubs = ['AC','KC','QC','JC','TC','9C','8C','7C','6C','5C','4C','3C','2C']
diamonds = ['AD','KD','QD','JD','TD','9D','8D','7D','6D','5D','4D','3D','2D']

deck = hearts+clubs+spades+diamonds

def pair_checker(cards):
    vals_list = []
    vals = []
    pairs_list = []
    trips_list = []
    quads_list = []
    for card in cards:
        vals_list.append(card[0])
    vals = set(vals_list)
    vals_nodupes = list(vals)
    for n in vals_nodupes:
        num = vals_list.count(n)
        if num ==2:
            pairs_list.append(n)
        if num ==3:
            trips_list.append(n)
        if num ==4:
            quads_list.append(n)
    return(pairs_list,trips_list,quads_list)

def flush_check(cards):
    flush = False
    suits = []
    flush_list=[]
    for card in cards:
        suits.append(card[1])
    suits_set = set(suits)
    suits_nodupes = list(suits_set)
    for s in suits_nodupes:
        num = suits.count(s)
        if num > 4:
            flush =True
            for card in cards:
                if card[1] ==s:
                    flush_list.append(card)
    return(flush,flush_list)    

                 
def check_straight(cards):
    straight = False
    straight_check = []
    for card in cards:
        straight_check.append(card[0])
    for n in range(len(straight_check)):
        t = straight_check[n]
        if t == 'A':
            straight_check[n] = 14
            straight_check.append(1)
        if t == 'K':
            straight_check[n] = 13
        if t == 'Q':
            straight_check[n] = 12
        if t == 'J':
            straight_check[n] = 11
        if t == 'T':
            straight_check[n] = 10
    for n in range (len(straight_check)):
        t = straight_check[n]
        straight_check[n] = int(t)
        
    straight_check.sort()
    count = 0
    straight_vals = []
    for n in range (len(straight_check)-1):
        if straight_check[n+1] == straight_check[n]+1:
            count+=1
            straight_vals.append(straight_check[n])
       
        if count <4:
            if straight_check[n+1] > straight_check[n]+1:
                count = 0
                straight_vals = []
        
    if count >= 4:
        if straight_check[-1] == straight_check[-2]+1:
            count+=1
            straight_vals.append(straight_check[-1])
    if count >4:
        straight = True
    if straight==False:
        straight_vals = []
 
    return(straight, straight_vals)


def check_hand(cards):
    pairs = pair_checker(cards)[0]
    trips = pair_checker(cards)[1]
    quads = pair_checker(cards)[2]
    straight = check_straight(cards)
    flush = flush_check(cards)
    #print(pairs,trips,quads,straight,flush)

    if flush[0]:
        val = check_straight(flush[1])
        if val[0] == True:
            hand = 'Straight Flush'
            handval = 1
            return hand,handval
        else:
            hand = 'Flush'
            handval = 4
            return hand,handval
    if straight[0]:
        hand = 'straight'
        handval = 5
        return hand,handval
    if len(quads) == 1:
        hand = '4 of a kind'
        handval = 2
        return hand,handval
    if len(trips) == 2:
        hand = 'Full House'
        handval = 3
        return hand,handval
    if len(trips) == 1:
        if len(pairs) >0:
            hand = 'Full House'
            handval = 3
            return hand,handval
    if len(pairs) >= 2:
        if len(trips) == 0:
            hand = '2 Pair'
            handval = 7
            return hand,handval
    if len(trips) ==1:
        if len(pairs)<1:
            if len(quads)<1:
                hand = '3 of a Kind'
                handval = 6
                return hand,handval
    if len(pairs) ==1:
        if len(trips)==0:
            if len(quads) ==0:
                hand = '1 Pair'
                handval = 8
                return hand,handval
    if len(pairs)==0:
        if len(trips)==0:
            hand = "High Card"
            handval = 9
            return hand,handval

def numerize(cards):
    cards1vals = []
    for card in cards:
        val=card[0]
        if val == 'A':
            cards1vals.append(14)
            cards1vals.append(1)
        elif val == 'K':
            cards1vals.append(13)
        elif val == 'Q':
            cards1vals.append(12)
        elif val == 'J':
            cards1vals.append(11)
        elif val == 'T':
            cards1vals.append(10)
        else: 
            cards1vals.append(int(val))
    cards1vals.sort(reverse=True)
    return(cards1vals)
    
def highcard(p1nums,p2nums,tie):
    p1nums.sort(reverse = True)
    p2nums.sort(reverse = True)
    for a,b in zip(p1nums,p2nums):
            count = 0
            if a > b:
                winner = 1
                return winner
            if b>a:
                winner = 2
                return winner
            count+=1
            if count == tie:
                winner = 0
                return winner

def tiebreak(cards1,cards2):
    hand = check_hand(cards1)[1]
    
    p1nums = numerize(cards1)
    p2nums = numerize(cards2)
            
    if hand ==9:
        winner = highcard(p1nums,p2nums,5)
        return winner
        
    if hand ==8:
        pairlist1 = []
        pairlist2 = []
        blanklist1 = []
        blanklist2 = []
        for a,b in zip(p1nums,p2nums):
            if p1nums.count(a) == 2:
                pairlist1.append(a)
            if p1nums.count(a) ==1:
                blanklist1.append(a)
            if p2nums.count(b) == 2:
                pairlist2.append(b)
            if p2nums.count(b) ==1:
                blanklist2.append(b)
        if pairlist1[0] > pairlist2[0]:
            winner = 1
            return winner
        if pairlist2[0] > pairlist1[0]:
            winner = 2
            return winner
        if pairlist1[0] == pairlist2[0]:
            winner = highcard(blanklist1,blanklist2,3)
            return winner
        
    if hand == 7:
        pairlist1 = []
        pairlist2 = []
        blanklist1 = []
        blanklist2 = []
        for a,b in zip(p1nums,p2nums):
            if p1nums.count(a) == 2:
                pairlist1.append(a)
            if p1nums.count(a) ==1:
                blanklist1.append(a)
            if p2nums.count(b) == 2:
                pairlist2.append(b)
            if p2nums.count(b) ==1:
                blanklist2.append(b)
        if highcard(pairlist1,pairlist2,2) ==1:
            winner = 1
            return winner
        if highcard(pairlist1,pairlist2,2) ==2:
            winner = 2
            return winner
        if highcard(pairlist1,pairlist2,1) ==0:
            if highcard(blanklist1,blanklist2,1) ==1:
                winner = 1
                return winner
            if highcard(blanklist1,blanklist2,1) ==2:
                winner = 2
                return winner
            if highcard(blanklist1,blanklist2,1) ==0:
                winner = 0
                return winner
    if hand ==6:   
        triplist1 = []
        triplist2 = []
        blanklist1 = []
        blanklist2 = []
        for a,b in zip(p1nums,p2nums):
            if p1nums.count(a) == 3:
                triplist1.append(a)
            if p1nums.count(a) ==1:
                blanklist1.append(a)
            if p2nums.count(b) == 3:
                triplist2.append(b)
            if p2nums.count(b) ==1:
                blanklist2.append(b)
        if highcard(triplist1,triplist2,1) ==1:
            winner = 1
            return winner
        if highcard(triplist1,triplist2,1) ==2:
            winner = 2
            return winner
        if highcard(triplist1,triplist2,1) ==0:
            if highcard(blanklist1,blanklist2,2) ==1:
                winner = 1
                return winner
            if highcard(blanklist1,blanklist2,2) ==2:
                winner = 2
                return winner
            if highcard(blanklist1,blanklist2,2) ==0:
                winner = 0
                return winner
            
    if hand ==5:
        hand1 = check_straight(cards1)[1]
        hand2 = check_straight(cards2)[1]
        
        hand1.sort(reverse = True)
        hand2.sort(reverse = True)
        
        for n in range(3):
            if len(hand1) !=5:
                hand1.pop(-1)
            if len(hand2) !=5:
                hand2.pop(-1)
        if highcard(hand1,hand2,5) == 1:
            winner = 1
            return winner
        if highcard(hand1,hand2,5) ==2:
            winner = 2
            return winner
        if highcard(hand1,hand2,5)==0:
            winner = 0
            return winner

    if hand ==4:
        hand1 = flush_check(cards1)[1]
        hand2 = flush_check(cards2)[1]
        hand1num = numerize(hand1)
        hand2num = numerize(hand2)
        if highcard(hand1num,hand2num,5) ==1:
            winner = 1
            return winner
        if highcard(hand1num,hand2num,5) ==2:
            winner = 2
            return winner
        if highcard(hand1num,hand2num,5) ==0:
            winner = 0
            return winner

    if hand ==3:
        pairlist1 = []
        pairlist2 = []
        triplist1 = []
        triplist2 = []
        for a,b in zip(p1nums,p2nums):
            if p1nums.count(a) == 2:
                pairlist1.append(a)
            if p1nums.count(a) ==3:
                triplist1.append(a)
            if p2nums.count(b) == 2:
                pairlist2.append(b)
            if p2nums.count(b) ==3:
                triplist2.append(b)
        if highcard(triplist1,triplist2,1) ==1:
            winner = 1
            return winner
        if highcard(triplist1,triplist2,1) ==2:
            winner = 2
            return winner
        if highcard(triplist1,triplist2,1) ==0:
            if highcard(pairlist1,pairlist2,1) ==1:
                winner = 1
                return winner
            if highcard(pairlist1,pairlist2,1) ==2:
                winner = 2
                return winner
            if highcard(pairlist1,pairlist2,1) ==0:
                winner = 0
                return winner
    if hand == 2:
        quadlist1 = []
        quadlist2 = []
        blanklist1 = []
        blanklist2 = []
        for a,b in zip(p1nums,p2nums):
            if p1nums.count(a) == 4:
                quadlist1.append(a)
            if p1nums.count(a) ==1:
                blanklist1.append(a)
            if p2nums.count(b) == 4:
                quadlist2.append(b)
            if p2nums.count(b) ==1:
                blanklist2.append(b)
        if highcard(quadlist1,quadlist2,1) ==1:
            winner = 1
            return winner
        if highcard(quadlist1,quadlist2,1) ==2:
            winner = 2
            return winner
        if highcard(quadlist1,quadlist2,1) ==0:
            if highcard(blanklist1,blanklist2,1) ==1:
                winner = 1
                return winner
            if highcard(blanklist1,blanklist2,1) ==2:
                winner = 2
                return winner
            if highcard(blanklist1,blanklist2,1) ==0:
                winner = 0
                return winner
    if hand ==1:
        hand1 = check_straight(cards1)[1]
        hand2 = check_straight(cards2)[1]
        
        hand1.sort(reverse = True)
        hand2.sort(reverse = True)
        for n in range(3):
            if len(hand1) !=5:
                hand1.pop(-1)
            if len(hand2) !=5:
                hand2.pop(-1)
        if highcard(hand1,hand2,5) == 1:
            winner = 1
            return winner
        if highcard(hand1,hand2,5) ==2:
            winner = 2
            return winner
        if highcard(hand1,hand2,5)==0:
            winner = 0
            return winner

def whowins(cards1,cards2):
    hand1 = check_hand(cards1)
    hand2 = check_hand(cards2)
    if hand1[1]==hand2[1]:
        winner = tiebreak(cards1,cards2)
        return winner
    if hand1[1] < hand2[1]:
        winner = 1
        return winner
    if hand1[1] > hand2[1]:
        winner = 2
        return winner
    
    
def deal_cards(deck,playernum=2):
    rd.shuffle(deck)
    
    hand1 = [deck[0],deck[1]]
    hand2 = [deck[2],deck[3]]
    hand3 = [deck[9],deck[10]]
    hand4 = [deck[11],deck[12]]
    hand5 = [deck[13],deck[14]]
    hand6 = [deck[15],deck[16]]
    hand7 = [deck[17],deck[18]]
    hand8 = [deck[19],deck[20]]
    flop = [deck[4],deck[5],deck[6]]
    turn = [deck[7]]
    river = [deck[8]]

    hands_list = [hand1,hand2]
    if playernum==3:
        hands_list.append(hand3)
    if playernum==4:
        hands_list.append(hand3,hand4)
    if playernum==5:
        hands_list.append(hand3,hand4,hand5)    
    if playernum==6:
        hands_list.append(hand3,hand4,hand5,hand6)
    if playernum==7:
        hands_list.append(hand3,hand4,hand5,hand6,hand7)
    if playernum==8:
        hands_list.append(hand3,hand4,hand5,hand6,hand7,hand8)

    return flop,turn,river,hands_list
def nonsense_test():
    hands = deal_cards(deck)
    pocket1 = hands[3][0]
    pocket2 = hands[3][1]
    community = hands[0]+hands[1]+hands[2]
    hand1 = pocket1+community
    hand2 = pocket2+community

    print('Coby:',pocket1)
    print('Parker', pocket2)
    print(community)
    print("Coby has hand number",check_hand(hand1))
    print("Parker has hand number", check_hand(hand2))

    result = whowins(hand1,hand2)

    print(result)

def who_is_luckier(player1,player2,deck):
    player1count = 0
    player2count = 0
    for n in range(1000):
        hands = deal_cards(deck)
        hand1 = hands[2]
        hand2 = hands[3]
        result = whowins(hand1,hand2)
        if result ==1:
            player1count +=1
        if result ==2:
            player2count+=1
    print(f'{player1} won {player1count} hands')
    print(f'{player2} won {player2count} hands')

def once_over(hands_list):
    for n in hands_list:
        matchup = 0
        for f in hands_list:
            if n!=f:
                matchup = whowins(n,f)
            if matchup ==2:
                hands_list.remove(n)
                break
    return(hands_list)

def last_standing(hands_list):
    for n in range(len(hands_list)):
        hands_list = once_over(hands_list)
    return(hands_list)

def multi_winner(hands_list):
    list_pres = hands_list
    winner_list = []
    final_list = last_standing(list_pres)
    if len(final_list) == 1:
        winner = final_list[0]
        return winner,1   ##It chooses the right winner but cant communicate that
    if len(final_list) >1:
        for n in range(len(list_pres)):
            for f in final_list:
                if list_pres[n] ==f:
                    winner_list.append(list_pres[n])
        return winner_list,2
    
def who_wins_multi(hands_list,hands_list2):
    split_pot = []
    result = multi_winner(hands_list) 
    print(result)  
    winners = result[0]
    if result[1]==1:
        for n in range(len(hands_list2)):
            if hands_list2[n] == winners:
                final_winner = [n]
                return final_winner
    if result[1]==2:
        for n in range(len(hands_list2)):
            for f in winners:
                if f == hands_list2[n]:
                    split_pot.append(n)
        return split_pot
    

community = ['8C','2C','QH','KH','AS']
pocket1 = ['8S','JH']
pocket2 = ['KH','2D']
pocket3 = ['KS','2S']
pocket4 = ['KS','2H']
hand1 = pocket1+community
hand2 = pocket2+community
hand3 = pocket3+community
hand4 = pocket4+community
hands_list = [hand1,hand2,hand3,hand4]
hands_list2 = [hand1,hand2,hand3,hand4]



            







