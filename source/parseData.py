import pdb
import os
import sys
import chess
import chess.pgn as pgn
import numpy as np
import random
from io import StringIO
PIECE_TYPES = [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING] = range(1, 7)
#TOTAL 378 CLASS
pawnDictOne = {
    8:0,
    16:1,
    7:2,
    9:3,
    -8:0,
    -16:1,
    -7:2,
    -9:3,
    2:4,
    3:5,
    4:6,
    5:7
}
pawnDictTwo = {
    8:8,
    16:9,
    7:10,
    9:11,
    -8:8,
    -16:9,
    -7:10,
    -9:11,
    2:12,
    3:13,
    4:14,
    5:15
}
pawnDictThree = {
    8:16,
    16:17,
    7:18,
    9:19,
    -8:16,
    -16:17,
    -7:18,
    -9:19,
    2:20,
    3:21,
    4:22,
    5:23
}
pawnDictFour = {
    8:24,
    16:25,
    7:26,
    9:27,
    -8:24,
    -16:25,
    -7:26,
    -9:27,
    2:28,
    3:29,
    4:30,
    5:31
}
pawnDictFive = {
    8:32,
    16:33,
    7:34,
    9:35,
    -8:32,
    -16:33,
    -7:34,
    -9:35,
    2:36,
    3:37,
    4:38,
    5:39
}
pawnDictSix = {
    8:40,
    16:41,
    7:42,
    9:43,
    -8:40,
    -16:41,
    -7:42,
    -9:43,
    2:44,
    3:45,
    4:46,
    5:47
}
pawnDictSeven = {
    8:48,
    16:49,
    7:50,
    9:51,
    -8:48,
    -16:49,
    -7:50,
    -9:51,
    2:52,
    3:53,
    4:54,
    5:55
}
pawnDictEight = {
    8:56,
    16:57,
    7:58,
    9:59,
    -8:56,
    -16:57,
    -7:58,
    -9:59,
    2:60,
    3:61,
    4:62,
    5:63
}
knightDictOne = {
    6:64,
    15:65,
    17:66,
    10:67,
    -6:68,
    -15:69,
    -17:70,
    -10:71
}
knightDictTwo = {
    6:72,
    15:73,
    17:74,
    10:75,
    -6:76,
    -15:77,
    -17:78,
    -10:79
}
knightDictNew = {
    6:80,
    15:81,
    17:82,
    10:83,
    -6:84,
    -15:85,
    -17:86,
    -10:87
}
bishopDictOne = {
    49:88,
    42:89,
    35:90,
    28:91,
    21:92,
    14:93,
    7:94,
    9:95,
    18:96,
    27:97,
    36:98,
    45:99,
    54:100,
    63:101,
    -7:102,
    -14:103,
    -21:104,
    -28:105,
    -35:106,
    -42:107,
    -49:108,
    -9:109,
    -18:110,
    -27:111,
    -36:112,
    -45:113,
    -54:114,
    -63:115
}
bishopDictTwo = {
    49:116,
    42:117,
    35:118,
    28:119,
    21:120,
    14:121,
    7:122,
    9:123,
    18:124,
    27:125,
    36:126,
    45:127,
    54:128,
    63:129,
    -7:130,
    -14:131,
    -21:132,
    -28:133,
    -35:134,
    -42:135,
    -49:136,
    -9:137,
    -18:138,
    -27:139,
    -36:140,
    -45:141,
    -54:142,
    -63:143
}
bishopDictNew = {
    49:322,
    42:323,
    35:324,
    28:325,
    21:326,
    14:327,
    7:328,
    9:329,
    18:330,
    27:331,
    36:332,
    45:333,
    54:334,
    63:335,
    -7:336,
    -14:337,
    -21:338,
    -28:339,
    -35:340,
    -42:341,
    -49:342,
    -9:343,
    -18:344,
    -27:345,
    -36:346,
    -45:347,
    -54:348,
    -63:349
}
rookDictOne = {
    -7:144,
    -6:145,
    -5:146,
    -4:147,
    -3:148,
    -2:149,
    -1:150,
    56:151,
    48:152,
    40:153,
    32:154,
    24:155,
    16:156,
    8:157,
    1:158,
    2:159,
    3:160,
    4:161,
    5:162,
    6:163,
    7:164,
    -8:165,
    -16:166,
    -24:167,
    -32:168,
    -40:169,
    -48:170,
    -56:171
}
rookDictTwo = {
    -7:172,
    -6:173,
    -5:174,
    -4:175,
    -3:176,
    -2:177,
    -1:178,
    56:179,
    48:180,
    40:181,
    32:182,
    24:183,
    16:184,
    8:185,
    1:186,
    2:187,
    3:188,
    4:189,
    5:190,
    6:191,
    7:192,
    -8:193,
    -16:194,
    -24:195,
    -32:196,
    -40:197,
    -48:198,
    -56:199
}
rookDictNew = {
    -7:350,
    -6:351,
    -5:352,
    -4:353,
    -3:354,
    -2:355,
    -1:356,
    56:357,
    48:358,
    40:359,
    32:360,
    24:361,
    16:362,
    8:363,
    1:364,
    2:365,
    3:366,
    4:367,
    5:368,
    6:369,
    7:370,
    -8:371,
    -16:372,
    -24:373,
    -32:374,
    -40:375,
    -48:376,
    -56:377
}
queenDictOne_left = {
    49:200,
    42:201,
    35:202,
    28:203,
    21:204,
    14:205,
    7:206,
    -9:207,
    -18:208,
    -27:209,
    -36:210,
    -45:211,
    -54:212,
    -63:213,
    -7:214,
    -6:215,
    -5:216,
    -4:217,
    -3:218,
    -2:219,
    -1:220
}
queenDictOne_right = {
    -7:221,
    -14:222,
    -21:223,
    -28:224,
    -35:225,
    -42:226,
    -49:227,
    9:228,
    18:229,
    27:230,
    36:231,
    45:232,
    54:233,
    63:234,
    56:235,
    48:236,
    40:237,
    32:238,
    24:239,
    16:240,
    8:241,
    1:242,
    2:243,
    3:244,
    4:245,
    5:246,
    6:247,
    7:248,
    -8:249,
    -16:250,
    -24:251,
    -32:252,
    -40:253,
    -48:254,
    -56:255
}
queenDictNew_left = {
    49:256,
    42:257,
    35:258,
    28:259,
    21:260,
    14:261,
    7:262,
    -9:263,
    -18:264,
    -27:265,
    -36:266,
    -45:267,
    -54:268,
    -63:269,
    -7:270,
    -6:271,
    -5:272,
    -4:273,
    -3:274,
    -2:275,
    -1:276
}
queenDictNew_right = {
    -7:277,
    -14:278,
    -21:279,
    -28:280,
    -35:281,
    -42:282,
    -49:283,
    9:284,
    18:285,
    27:286,
    36:287,
    45:288,
    54:289,
    63:290,
    56:291,
    48:292,
    40:293,
    32:294,
    24:295,
    16:296,
    8:297,
    1:298,
    2:299,
    3:300,
    4:301,
    5:302,
    6:303,
    7:304,
    -8:305,
    -16:306,
    -24:307,
    -32:308,
    -40:309,
    -48:310,
    -56:311
}
kingDict = {
    7:312,
    8:313,
    9:314,
    1:315,
    -7:316,
    -8:317,
    -9:318,
    -1:319,
    2:320,
    -2:321
}

abstractMove = {
    'pawnone': pawnDictOne,
    'pawntwo': pawnDictTwo,
    'pawnthree': pawnDictThree,
    'pawnfour': pawnDictFour,
    'pawnfive': pawnDictFive,
    'pawnsix': pawnDictSix,
    'pawnseven': pawnDictSeven,
    'pawneight': pawnDictEight,
    'knightone':knightDictOne,
    'knighttwo': knightDictTwo,
    'knightnew': knightDictNew,
    'bishopone': bishopDictOne,
    'bishoptwo':bishopDictTwo,
    'bishopnew':bishopDictNew,
    'rookone':rookDictOne,
    'rooktwo':rookDictTwo,
    'rooknew':rookDictNew,
    'queen': (queenDictOne_left, queenDictOne_right),
    'queennew': (queenDictNew_left, queenDictNew_right),
    KING: kingDict
}


def parseData_policy(PGNFilePath, npzFilePath, boardShape=(8, 8)):
    if os.path.exists(npzFilePath):
        print("error! the Path has existed a npz file!!")
        return
    pgnFile = open(PGNFilePath,encoding="utf-8-sig")
    games = []
    game = pgn.read_game(pgnFile)
    if game is None:
        print("empty pgnfiles")
        return
    else:
        ## be careful with the maximum size of memory 
        # data = []
        features = []
        labels = []
        num = 0
        while game is not None:
            
            position_piece_white_dict = {
                PAWN :{
                    'a2':'pawnone',
                    'b2':'pawntwo',
                    'c2':'pawnthree',
                    'd2':'pawnfour',
                    'e2':'pawnfive',
                    'f2':'pawnsix',
                    'g2':'pawnseven',
                    'h2':'pawneight'
                },
                ROOK : {
                    'a1': 'rookone',
                    'h1': 'rooktwo'
                },
                BISHOP : {
                    'c1': 'bishopone',
                    'f1': 'bishoptwo'
                },
                KNIGHT : {
                    'b1': 'knightone',
                    'g1': 'knighttwo'
                },
                QUEEN:{
                    'd1':'queen'
                }
            }
            position_piece_black_dict = {
                PAWN: {
                    'a7': 'pawnone',
                    'b7': 'pawntwo',
                    'c7': 'pawnthree',
                    'd7': 'pawnfour',
                    'e7': 'pawnfive',
                    'f7': 'pawnsix',
                    'g7': 'pawnseven',
                    'h7': 'pawneight'
                },
                ROOK: {
                    'a8': 'rookone',
                    'h8': 'rooktwo'
                },
                BISHOP: {
                    'c8': 'bishopone',
                    'f8': 'bishoptwo'
                },
                KNIGHT: {
                    'b8': 'knightone',
                    'g8': 'knighttwo'
                },
                QUEEN: {
                    'd8': 'queen'
                }
            }
            root = game
            board = root.board()
            if len(game.variations) == 0:
                game = pgn.read_game(pgnFile)
                continue
            nextNode = game.variation(0)
            turn = board.turn
            move = nextNode.move
            while not nextNode.is_end():
                
                
                locationArray = get_location_array_chw(board, boardShape)
                from_square = move.from_square
                color = board.piece_at(from_square).color
                if color == True:
                    abstractMove = get_abstract_move(
                    board, move, position_piece_white_dict,color)
                else:
                    try:
                        abstractMove = get_abstract_move(
                            board, move, position_piece_black_dict,color)
                    except Exception:
                        print(game.headers)
                        return
                print(abstractMove)
                features.append(np.array(locationArray))
                labels.append(abstractMove)
                board.push(move)
                nextNode = nextNode.variation(0)
                turn = board.turn
                move = nextNode.move
               
                # locationArray = get_location_concate_array_chw(
                #     board, boardShape)
                # turnArray = bool2concatearray(turn,boardShape)
                # label = get_abstract_move(board,move)
                # feature = np.append(locationArray,turnArray)
                # data.append(np.append(feature,np.array(label,dtype=np.int8)))

                
            else:
                # np.savetxt("test.dat", data,
                #            delimiter=",", newline="\n")
                if len(labels)>10000000:
                    save2npzfile(npzFilePath, np.array(features),
                                 np.array(labels, dtype=np.int16))
                    break
                    
                #     save2npzfile(npzFileDir+str(num)+".npz",np.array(features),np.array(labels, dtype=np.int8))
                #     features = []
                #     labels = []
                #     num+=1
                game = pgn.read_game(pgnFile)
                
                # print(sys.getsizeof(features),len(labels))
        else:
            # with open("test.dat","ab") as file:
            #     np.save(file,data)
            save2npzfile(npzFilePath,np.array(features),np.array(labels, dtype=np.int16))
            print("==============================================")
            print("succeed!    the dir is"+npzFilePath)
            print("==============================================")
        return


def parseData_value(PGNFileDir, npzFilePath, boardShape=(8, 8)):
    if os.path.exists(npzFilePath):
        print("error! the Path has existed a npz file!!")
        return
    pgnfileNames = os.listdir(PGNFileDir)
    pgnfilePaths = list(map(lambda x: PGNFileDir+"/"+x, pgnfileNames))
    labels = []
    features = []
    for path in pgnfilePaths:
        pgnFile = open(path, encoding="utf-8-sig")
        game = pgn.read_game(pgnFile)
        if game == None:
            print("empty pgn file")
            continue
        else:
            random.seed()
            while game is not None:
                result = game.headers.get("Result")
                if result == "*":
                    game = pgn.read_game(pgnFile)
                    continue
                root = game
                board = game.board()
                lens = len(list(game.main_line()))
                if lens == 0:
                    print(game)
                    print("no move game")
                    game = pgn.read_game(pgnFile)
                    continue
                index = random.randint(0, lens)
                nextNode = root
                for x in range(index):
                    nextNode = nextNode.variation(0)
                    move = nextNode.move
                    board.push(move)
                else:
                    feature_board = get_location_array_chw(board, boardShape)
                    feature_turn = bool2array(board.turn, boardShape)
                    feature_board.append(feature_turn)
                    features.append(np.array(feature_board))
                    label = -1.0
                    if result == "1-0" and board.turn == True:
                        label = 1.0
                    if result == "0-1" and board.turn == False:
                        label = 1.0
                    if result == "1/2-1/2":
                        label = 0.0
                    labels.append(np.array([label]))
                    game = pgn.read_game(pgnFile)
                    print(feature_turn)
    else:
        # save2npzfile(npzFilePath, np.array(features), np.array(labels))
        pass
    return
def test(PGNFileDir, npzFilePath, boardShape=(8, 8)):
    pgnfileNames = os.listdir(PGNFileDir)
    pgnfilePaths = list(map(lambda x:PGNFileDir+"/"+x,pgnfileNames))
    labels = []
    features = []
    for path in pgnfilePaths:
        pgnFile = open(path,encoding="utf-8-sig")
        game = pgn.read_game(pgnFile)
        if game == None:
            print("empty pgn file")
            continue
        else:
            random.seed()
            while game is not None:
                result = game.headers.get("Result")
                if result == "*":
                    game = pgn.read_game(pgnFile)
                    continue
                root = game
                board = game.board()
                lens = len(list(game.main_line()))
                if  lens == 0:
                    print(game)
                    print("no move game")
                    game = pgn.read_game(pgnFile)
                    continue
                index = random.randint(0,lens)
                nextNode = root
                for x in range(index):
                    nextNode = nextNode.variation(0)
                    move = nextNode.move
                    board.push(move)
                else:
                    print(board)
                    
                    game = pgn.read_game(pgnFile)
        print(len(labels))   
    return





def save2npzfile(npzFilePath,features,labels):
    np.savez_compressed(npzFilePath, feature=features, label=labels)
    return

def get_game(PGNFilePath):
    pgnFile = open(PGNFilePath)
    games = []
    game = pgn.read_game(pgnFile)
    if game is None:
        print("empty pgnfiles")
        return
    else:
        while game is not None:
            games.append(game)
            game = pgn.read_game(pgnFile)
        return games
def get_location_array_chw(board,boardShape=(8,8)):
    pieces = [
        board.pawns,   board.knights,
        board.bishops, board.rooks,
        board.queens,  board.kings
    ]
    pieceLocations_square = list(map(chess.SquareSet,pieces))
    playerLocations_square = list(map(chess.SquareSet, board.occupied_co))
    # Color specific piece boards
    pieceLocations_black_square = map(lambda bb: bb & playerLocations_square[
        chess.BLACK], pieceLocations_square)
    pieceLocations_white_square = map(lambda bb: bb & playerLocations_square[
        chess.WHITE], pieceLocations_square)
    pieceLocations_black_coordi = map(lambda ss: intList2CoordiSet(
        ss, boardShape), pieceLocations_black_square)
    pieceLocations_white_coordi = map(lambda ss: intList2CoordiSet(
        ss, boardShape), pieceLocations_white_square)
    data = []
    for b_w in zip(pieceLocations_black_coordi, pieceLocations_white_coordi):
        if len(b_w) == 2:
            bb = np.zeros(boardShape,dtype=np.float32)
            for b in b_w[0]:
                bb[b] = -1
            for w in b_w[1]:
                bb[w] = 1
            else:
                data.append(bb)
        else:
            print("error in coordinate")
            break
    else:
        if len(data) != 6:
            print("error, position data channel less than 6")
            return
        else:
            return data


def get_location_concate_array_chw(board, boardShape=(8, 8)):
    pieces = [
        board.pawns,   board.knights,
        board.bishops, board.rooks,
        board.queens,  board.kings
    ]
    pieceLocations_square = list(map(chess.SquareSet, pieces))
    playerLocations_square = list(map(chess.SquareSet, board.occupied_co))
    # Color specific piece boards
    pieceLocations_black_square = map(lambda bb: bb & playerLocations_square[
        chess.BLACK], pieceLocations_square)
    pieceLocations_white_square = map(lambda bb: bb & playerLocations_square[
        chess.WHITE], pieceLocations_square)
    pieceLocations_black_coordi = map(lambda ss: intList2CoordiSet(
        ss, boardShape), pieceLocations_black_square)
    pieceLocations_white_coordi = map(lambda ss: intList2CoordiSet(
        ss, boardShape), pieceLocations_white_square)
    data = []
    for b_w in zip(pieceLocations_black_coordi, pieceLocations_white_coordi):
        if len(b_w) == 2:
            bb = np.zeros(boardShape)
            for b in b_w[0]:
                bb[b] = -1
            for w in b_w[1]:
                bb[w] = 1
            else:
                data.append(np.array(bb,dtype=np.float32))
        else:
            print("error in coordinate")
            break
    else:
        if len(data) != 6:
            print("error, position data channel less than 6")
            return
        else:
            return np.concatenate(np.concatenate(data))


def get_abstract_move(board, move, position_piece_dict,color):
    from_square = move.from_square
    to_square = move.to_square
    move_uci = move.uci()
    from_uci = move_uci[0:2]
    to_uci = move_uci[2:4]
    pieceType = board.piece_type_at(from_square)
    if board.is_castling(move):
        if board.is_kingside_castling(move):
            if color == True:
                position_piece_dict[ROOK]['f1'] = 'rooktwo'
                del position_piece_dict[ROOK]['h1']
            elif color ==False:
                position_piece_dict[ROOK]['f8'] = 'rooktwo'
                del position_piece_dict[ROOK]['h8']
            return abstractMove[pieceType][to_square-from_square]
        else:
            if color == True:
                position_piece_dict[ROOK]['d1'] = 'rookone'
                del position_piece_dict[ROOK]['a1']
            elif color ==False:
                position_piece_dict[ROOK]['d8'] = 'rookone'
                del position_piece_dict[ROOK]['a8']
            return abstractMove[pieceType][to_square-from_square]

    elif move.promotion != None:
        piecePosition = position_piece_dict[pieceType][from_uci]
        if move.promotion == QUEEN:
            position_piece_dict[move.promotion][to_uci] = 'queennew'
        if move.promotion ==KNIGHT:
            position_piece_dict[move.promotion][to_uci] = 'knightnew'
        if move.promotion == ROOK:
            position_piece_dict[move.promotion][to_uci] = 'rooknew'
        if move.promotion ==BISHOP:
            position_piece_dict[move.promotion][to_uci] = 'bishopnew'
        del position_piece_dict[pieceType][from_uci]
        return abstractMove[piecePosition][move.promotion]
    elif pieceType == QUEEN:
        piecePosition = position_piece_dict[pieceType][from_uci]
        position_piece_dict[pieceType][to_uci] = piecePosition
        del position_piece_dict[pieceType][from_uci]
        if from_square % 8 > to_square % 8:
            return abstractMove[piecePosition][0][to_square-from_square]
        else:
            return abstractMove[piecePosition][1][to_square-from_square]
    elif pieceType ==KING:
            return abstractMove[pieceType][to_square-from_square]
    elif pieceType!=None:
        piecePosition = position_piece_dict[pieceType][from_uci]
        position_piece_dict[pieceType][to_uci] = piecePosition

        del position_piece_dict[pieceType][from_uci]
        return abstractMove[piecePosition][to_square-from_square]
    return 

def get_move_array(move, boardShape=(8, 8)):
    sq_f = move.from_square
    sq_t = move.to_square
    data = []
    sq_f_coordi = createBitboardWithCoordi([int2coordi(sq_f)])
    sq_t_coordi = createBitboardWithCoordi([int2coordi(sq_t)])
    data.append(sq_f_coordi)
    data.append(sq_t_coordi)
    return data
    

def bool2array(boolean,boardShape=(8,8),dt=np.float32):
    if boolean:
        return np.ones(boardShape,dtype=dt)
    else:
        return np.zeros(boardShape,dtype=dt)
    return data


def bool2concatearray(boolean, boardShape=(8, 8), dt=np.byte):
    if boolean:
        return np.concatenate(np.ones(boardShape, dtype=dt))
    else:
        return np.concatenate(np.zeros(boardShape, dtype=dt))


def createBitboardWithCoordi(coordiList,boardShape=(8,8),dt=np.byte,defaultValue=1):
    if len(coordiList) ==0:
        print("error,empty coordinate list")
    else:
        bb = np.zeros(boardShape, dtype=dt)
        for cd in coordiList:
            bb[cd] = defaultValue
        else:
            return bb

def intList2CoordiSet(intList,coordiShape=(8,8)):
    return map(lambda i: int2coordi(i, coordiShape),intList)

def int2coordi(intNum, coordiShape=(8, 8)):
    coordi = np.unravel_index(intNum,coordiShape)
    return (7-coordi[0],coordi[1])

def coordi2int(coordi,coordiShape=(8,8)):
    return np.ravel_multi_index(coordi,coordiShape)

if __name__ == '__main__':
    # games = get_game('../data/master_games_1_one.pgn')
    # game = games[0]
    # board = game.board()

    # parseData_policy("../data/ficsgamesdb_2017_chess2000_nomovetimes_1540646.pgn",
    #                  '../data/policynetwork/data_policy_l_232_float32.npz')
    parseData_value(
        "../data/data_value_l","../data/data_value_lld.npz")


# value_l store 0.0 for lose and 1.0 for winning
#value _ll store -1.0 for losing and 1.0 for winning

    # with open("data_l","ab") as file:
    #     a = np.array([1,2,3,4,5])
    #     np.save(file,a)
    #     file.close()
    # with open("data_l","ab") as file:
    #     b = np.array([6, 7, 8, 9, 10])
    #     np.save(file,b)
    #     file.close(

    pgn_string = "d4 g6 2. Nf3 Bg7 3. c3 c5 4. Nbd2 cxd4 5. cxd4 Nc6 6. e3 d6 7. Be2 Nf6 8. O-O O-O 9. Re1 e5 10. Rb1 e4 11. Ng5 d5 12. h4 h6 13. Nh3 Bxh3 14. gxh3 h5 15. f3 Bh6 16. Nf1 Re8 17. Bb5 Qb6 18. Bxc6 bxc6 19. b4 exf3 20. Qxf3 Ne4 21. Nd2 Nc3 22. Rb2 Na4 23. Rb1 Re6 24. Nf1 Rae8 25. Rb3 Kg7 26. Ng3 Rf6 27. Qg2 Qb5 28. e4 Qc4 29. Bxh6+ Kxh6 30. e5 Qxd4+ 31. Kh1 Rxe5 32. Rg1 Nb6 33. Rf3 Rxf3 34. Qxf3 f5 35. Ne2 Qxh4 36. Nf4 g5 37. Ng2 Qe4 38. Qg3 f4 39. Qh2 f3 40. Ne1 f2+ 41. Rg2 fxe1=R+ 42. Qg1 Rxg1+ 43. Kxg1 Qxg2+ 44. Kxg2 Re2+ 45. Kg3 Re4 46. Kf3 Rd4 47. Kg2 Rd3 {White forfeits on time} 0-1"

    # position_piece_white_dict = {
    #       PAWN: {
    #           'a2': 'pawnone',
    #                 'b2': 'pawntwo',
    #                 'c2': 'pawnthree',
    #                 'd2': 'pawnfour',
    #                 'e2': 'pawnfive',
    #                 'f2': 'pawnsix',
    #                 'g2': 'pawnseven',
    #                 'h2': 'pawneight'
    #       },
    #       ROOK: {
    #           'a1': 'rookone',
    #           'h1': 'rooktwo'
    #       },
    #       BISHOP: {
    #           'c1': 'bishopone',
    #                 'f1': 'bishoptwo'
    #       },
    #       KNIGHT: {
    #           'b1': 'knightone',
    #                 'g1': 'knighttwo'
    #       },
    #       QUEEN: {
    #           'd1': 'queen'
    #       }
    #   }
    # position_piece_black_dict = {
    #             PAWN: {
    #                 'a7': 'pawnone',
    #                 'b7': 'pawntwo',
    #                 'c7': 'pawnthree',
    #                 'd7': 'pawnfour',
    #                 'e7': 'pawnfive',
    #                 'f7': 'pawnsix',
    #                 'g7': 'pawnseven',
    #                 'h7': 'pawneight'
    #             },
    #             ROOK: {
    #                 'a8': 'rookone',
    #                 'h8': 'rooktwo'
    #             },
    #             BISHOP: {
    #                 'c8': 'bishopone',
    #                 'f8': 'bishoptwo'
    #             },
    #             KNIGHT: {
    #                 'b8': 'knightone',
    #                 'g8': 'knighttwo'
    #             },
    #             QUEEN: {
    #                 'd8': 'queen'
    #             }
    #         }


    pgn = StringIO(pgn_string)
    game = chess.pgn.read_game(pgn)
    root = game
    board = game.board()

    # for move in game.main_line():
    #     print(board)
    #     from_square = move.from_square
    #     print(move,from_square,board.piece_at(from_square))
    #     color = board.piece_at(from_square).color
    #     if color==True:
    #         abmove = get_abstract_move(board, move, position_piece_white_dict,color)
    #         print(abmove)
    #     else:
    #         abmove = get_abstract_move(board, move, position_piece_black_dict,color)
    #         print(abmove)
    #     board.push(move)
    # # # num = 0
    # board = game.board()
    # board.piece_at()
    # root = game
    # print(board)
    # print()
    # move = root.variation(0).variation(0).move
    # print(move)
    # for move in moves:
    #     print(move.promotion)
    #     num +=1
    #     print(str(num)+":"+str(move),get_abstract_move(board,move))
    #     board.push(move)


    # move1 = game.variation(0)
    # move2 = move1.move
    # print(move2)

    # node = game.variation(0)
    # move = node.move
    # board.push(move)
    # for x in game.main_line():
    #     print(x,end=',')
    # for i in range(10):
    #     node = node.variation(0)
    #     board.push(node.move)
    # print(node.variation(0).move,end='       ')
    # print()
    

    # gameNode = game.variation(0)
    # move = gameNode.move
    # print(board,board.turn)
    # print(move)
    # board.push(move)
    # print(board)
    # data = np.load('../data/data_m.npz')
    # features = data['feature']
    # labels = data['label']
    # x = features[9]
    # print(x)

        

    

