import pdb
import os
import sys
import chess
import chess.pgn as pgn
import numpy as np
from io import StringIO
PIECE_TYPES = [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING] = range(1, 7)
#TOTAL 137 CLASS
pawnDict = {
    8:0,
    16:1,
    7:2,
    9:3,
    -8:0,
    -16:1,
    -7:2,
    -9:3,
    2:134,
    3:135,
    4:136,
    5:137
}
knightDict = {
    6:4,
    15:5,
    17:6,
    10:7,
    -6:8,
    -15:9,
    -17:10,
    -10:11
}
bishopDict = {
    49:12,
    42:13,
    35:14,
    28:15,
    21:16,
    14:17,
    7:18,
    9:19,
    18:20,
    27:21,
    36:22,
    45:23,
    54:24,
    63:25,
    -7:26,
    -14:27,
    -21:28,
    -28:29,
    -35:30,
    -42:31,
    -49:32,
    -9:33,
    -18:34,
    -27:35,
    -36:36,
    -45:37,
    -54:38,
    -63:39
}
rookDict = {
    -7:40,
    -6:41,
    -5:42,
    -4:43,
    -3:44,
    -2:45,
    -1:46,
    56:47,
    48:48,
    40:49,
    32:50,
    24:51,
    16:52,
    8:53,
    1:54,
    2:55,
    3:56,
    4:57,
    5:58,
    6:59,
    7:60,
    -8:61,
    -16:63,
    -24:63,
    -32:64,
    -40:65,
    -48:66,
    -56:67
}
queenDict_left = {
    49:68,
    42:69,
    35:70,
    28:71,
    21:72,
    14:73,
    7:74,
    -9:89,
    -18:90,
    -27:91,
    -36:92,
    -45:93,
    -54:94,
    -63:95,
    -7:96,
    -6:97,
    -5:98,
    -4:99,
    -3:100,
    -2:101,
    -1:102
}
queenDict_right = {
    -7:82,
    -14:83,
    -21:84,
    -28:85,
    -35:86,
    -42:87,
    -49:88,
    9:75,
    18:76,
    27:77,
    36:78,
    45:79,
    54:80,
    63:81,
    56:103,
    48:104,
    40:105,
    32:106,
    24:107,
    16:108,
    8:109,
    1:110,
    2:111,
    3:112,
    4:113,
    5:114,
    6:115,
    7:116,
    -8:117,
    -16:118,
    -24:119,
    -32:120,
    -40:121,
    -48:122,
    -56:123
}

kingDict = {
    7:124,
    8:125,
    9:126,
    1:127,
    -7:128,
    -8:129,
    -9:130,
    -1:131,
    2:132,
    -2:133
}

abstractMove = {
    PAWN: pawnDict,
    KNIGHT:knightDict,
    BISHOP: bishopDict,
    ROOK: rookDict,
    QUEEN: (queenDict_left, queenDict_right),
    KING: kingDict
}

def parsePGN(PGNFilePath, npzFileDir,boardShape=(8, 8)):
    if len(os.listdir(npzFileDir)) > 0:
        print("error! the Path has existed a npz file!!")
        return
    pgnFile = open(PGNFilePath,encoding="utf-8-sig")
    games = []
    npzIndex = 0
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
            root = game
            board = root.board()
            if len(game.variations) == 0:
                game = pgn.read_game(pgnFile)
                continue
            nextNode = game.variation(0)
            turn = board.turn
            move = nextNode.move
            while not nextNode.is_end():
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

                locationArray = get_location_array_chw(board,boardShape)
                turnArray = bool2array(turn, boardShape)
                abstractMove = get_abstract_move(board,move)
                locationArray.append(turnArray)
                features.append(locationArray)
                labels.append(abstractMove)
            else:
                # np.savetxt("test.dat", data,
                #            delimiter=",", newline="\n")
                # if len(labels)>400000:
                #     save2npzfile(npzFileDir+str(num)+".npz",np.array(features),np.array(labels, dtype=np.int8))
                #     features = []
                #     labels = []
                #     num+=1
                game = pgn.read_game(pgnFile)
                
                # print(sys.getsizeof(features),len(labels))
        else:
            # with open("test.dat","ab") as file:
            #     np.save(file,data)
            print(len(labels))
            save2npzfile(npzFileDir+str(num)+".npz",np.array(features),np.array(labels, dtype=np.int16))
            print("==============================================")
            print("succeed!    the dir is"+npzFileDir)
            print("==============================================")
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


def get_abstract_move(board,move):
    from_square = move.from_square
    to_square = move.to_square
    pieceType = board.piece_type_at(from_square)
    if move.promotion != None:
        return abstractMove[pieceType][move.promotion]
    elif pieceType == QUEEN:
        if from_square % 8 > to_square % 8:
            return abstractMove[pieceType][0][to_square-from_square]
        else:
            return abstractMove[pieceType][1][to_square-from_square]
    elif pieceType != None:
            return abstractMove[pieceType][to_square-from_square]
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
    # games = get_game('../data/master_games.pgn')
    # game = games[0]
    # board = game.board()

    parsePGN("../data/master_games_2.pgn", '../data/test/')


    # with open("data_l","ab") as file:
    #     a = np.array([1,2,3,4,5])
    #     np.save(file,a)
    #     file.close()
    # with open("data_l","ab") as file:
    #     b = np.array([6, 7, 8, 9, 10])
    #     np.save(file,b)
    #     file.close()

    


    # pgn_string = "1.c4 c5 2.a4 e5 3.Ra3 f5 4.Rb3 d6 5.Rb5 Ne7 6.d4 Nd7 7.dxe5 Nc6 8.Rb6 Ndxe5 9.Rxc6 Be6 10.b4 Be7 11.bxc5 Nf7 12.cxd6 Nxd6 13.a5 bxc6 14.a6 O-O 15.c5 Nb5 16.Qxd8 Bxd8 17.Bg5 Rb8 18.Bxd8 Ba2 19.Nd2 Rfxd8 20.Ne4 fxe4 21.f4 e3 22.f5 Nc3 23.g4 Bd5 24.f6 Bxh1 25.Bh3 Be4 26.g5 Kf7 27.Be6+ Ke8 28.Kf1 gxf6 29.gxf6 Bh1 30.Bf7+ Kf8 31.Be6 Rd5 32.Bd7 Rg5 33.Bxc6 Bxc6 34.Nf3 Rg4 35.Ne5 Ne4 36.Nd7+ Kf7 37.Nxb8 Bb5 38.Nd7 Rg5 39.Ne5+ Ke8 40.Nd7 Kf7 41.Ne5+ Ke8 42.f7+ Ke7 43.f8=R+ Kxf8  *"
    # pgn = StringIO(pgn_string)
    # game = chess.pgn.read_game(pgn)
    # moves = game.main_line()
    # num = 0
    # board = game.board()
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

        

    

