from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gensim
import numpy
import tensorflow as tf

WORDMAPPING = {
    'of': 'Of',
    'and': 'And',
    'a': 'A',
    'to': 'To',
    'applecare': 'Applecare'
}

def readSentiWordNet():
    arrDatas = []
    mapDatas = {}
    doc = open('./SentiWordNet.txt', 'r+')
    labtopfi = open('./labtop.xml', 'r+')
    labtopDoc = labtopfi.read()
    labtopfi.close()
    for line in doc:
        line = line.strip()
        if not line.startswith('#'):
            cols = line.split('\t')
            if len(cols) != 6:
                print('%%%%%%%%%%%%%%%%%%parse SentiWordNet error')
                exit(1)
            POS = cols[0]
            posScore = float(cols[2])
            negScore = float(cols[3])
            words = []
            
            tmpWords = cols[4].split(' ')
            for tmpWord in tmpWords:
                tmpWord = tmpWord.strip()
                if tmpWord == '':
                    continue
                tmpWord = tmpWord.split('#')[0]
                if labtopDoc.find(tmpWord) == -1:
                    continue
                # print(tmpWord)
                words.append(tmpWord)

            tmpSentences = []
            gloss = cols[5]
            lenOfGloss = len(gloss)
            index = -1
            while index < lenOfGloss:
                try:
                    startIndex = gloss.index('"', index + 1)
                    endIndex = gloss.index('"', startIndex + 1)
                    tmpSentence = gloss[startIndex + 1:endIndex].lower()
                    tmpSentence = tmpSentence.replace('?', '')
                    tmpSentence = tmpSentence.replace(',', ' ')
                    tmpSentence = tmpSentence.replace('!', '')
                    tmpSentence = tmpSentence.replace('(', '')
                    tmpSentence = tmpSentence.replace(')', '')
                    tmpSentence = tmpSentence.replace('.', '')
                    tmpSentence = tmpSentence.replace(';', '')
                    tmpSentence = tmpSentence.replace('--', ' ')
                    tmpSentence = tmpSentence.replace('-', ' ')
                    tmpSentence = tmpSentence.strip()
                    tmpSentences.append(tmpSentence)
                    index = endIndex + 1
                except Exception as e:
                    break

            for tmpSentence in tmpSentences:
                for word in words:
                    word = word.strip()
                    if word == '':
                        continue
                    if tmpSentence.startswith(word + ' '):
                        tmpData = {
                            'POS': POS,
                            'posScore': posScore,
                            'negScore': -negScore,
                            'word': word,
                            'sentence': tmpSentence,
                            'prevSentence': None,
                            'nextSentence': tmpSentence
                        }

                        arrDatas.append(tmpData)

                        if not word in mapDatas:
                            mapDatas[word] = {}
                        if not POS in mapDatas[word]:
                            mapDatas[word][POS] = []
                        mapDatas[word][POS].append(tmpData)

                    if tmpSentence.endswith(' ' + word):
                        tmpData = {
                            'POS': POS,
                            'posScore': posScore,
                            'negScore': -negScore,
                            'word': word,
                            'sentence': tmpSentence,
                            'prevSentence': tmpSentence,
                            'nextSentence': None
                        }

                        arrDatas.append(tmpData)

                        if not word in mapDatas:
                            mapDatas[word] = {}
                        if not POS in mapDatas[word]:
                            mapDatas[word][POS] = []
                        mapDatas[word][POS].append(tmpData)
                    
                    if tmpSentence.find(' ' + word + ' ') != -1:
                        tmpIndex = tmpSentence.index(' ' + word + ' ')
                        tmpData = {
                            'POS': POS,
                            'posScore': posScore,
                            'negScore': negScore,
                            'word': word,
                            'sentence': tmpSentence,
                            'prevSentence': tmpSentence[:tmpIndex + len(word) + 1].strip(),
                            'nextSentence': tmpSentence[tmpIndex:].strip()
                        }

                        arrDatas.append(tmpData)

                        if not word in mapDatas:
                            mapDatas[word] = {}
                        if not POS in mapDatas[word]:
                            mapDatas[word][POS] = []
                        mapDatas[word][POS].append(tmpData)
    doc.close()

    return {
        'arrDatas': arrDatas,
        'mapDatas': mapDatas
    }

def wordEmbedding(datas):
    tDatas = []
    count = 0

    sentences = []
    for data in datas:
        sentence = []
        for tmpWord in data['sentence'].split(' '):
            tmpWord = tmpWord.strip()
            if tmpWord == '':
                continue
            sentence.append(tmpWord)
        sentences.append(sentence)
    newModel = gensim.models.Word2Vec(sentences, min_count=1, size=300)
    model = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)

    for data in datas:
        tWord = data['word']
        if not tWord in model.wv:
            continue

        prevVects = None
        if data['prevSentence'] != None:
            prevVects = []
            prevWords = data['prevSentence'].split(' ')
            for tmpWord in prevWords:
                tmpWord = tmpWord.strip()
                if tmpWord == '':
                    continue
                if tmpWord in model.wv:
                    prevVects.append(model.wv[tmpWord].tolist())
                elif tmpWord in WORDMAPPING:
                    prevVects.append(model.wv[WORDMAPPING[tmpWord]].tolist())
                elif tmpWord.endswith("'s"):
                    tmpWord = tmpWord.replace("'s", '')
                    if tmpWord in model.wv:
                        prevVects.append(model.wv[tmpWord].tolist())
                    else:
                        count += 1
                        # print(tmpWord, count, 1)
                        prevVects.append(newModel.wv[tmpWord + "'s"].tolist())
                elif tmpWord.startswith("`"):
                    tmpWord = tmpWord.replace("`", '')
                    if tmpWord in model.wv:
                        prevVects.append(model.wv[tmpWord].tolist())
                    else:
                        count += 1
                        # print(tmpWord, count, 2.1)
                        prevVects.append(newModel.wv["`" + tmpWord].tolist())
                elif tmpWord.endswith("'"):
                    tmpWord = tmpWord.replace("'", '')
                    if tmpWord in model.wv:
                        prevVects.append(model.wv[tmpWord].tolist())
                    else:
                        count += 1
                        # print(tmpWord, count, 2.2)
                        prevVects.append(newModel.wv[tmpWord + "'"].tolist())
                elif tmpWord.isdigit():
                    prevVects.append(model.wv['one'].tolist())
                else:
                    count += 1
                    # print(tmpWord, count, 3)
                    prevVects.append(newModel.wv[tmpWord].tolist())
        data['prevVects'] = prevVects  
        
        nextVects = None
        if data['nextSentence'] != None:
            nextVects = []
            nextWords = data['nextSentence'].split(' ')
            for tmpWord in nextWords:
                tmpWord = tmpWord.strip()
                if tmpWord == '':
                    continue
                if tmpWord in model.wv:
                    nextVects.append(model.wv[tmpWord].tolist())
                elif tmpWord in WORDMAPPING:
                    nextVects.append(model.wv[WORDMAPPING[tmpWord]].tolist())
                elif tmpWord.endswith("'s"):
                    tmpWord = tmpWord.replace("'s", '')
                    if tmpWord in model.wv:
                        nextVects.append(model.wv[tmpWord].tolist())
                    else:
                        count += 1
                        # print(tmpWord, count, 1)
                        nextVects.append(newModel.wv[tmpWord + "'s"].tolist())
                elif tmpWord.startswith("`"):
                    tmpWord = tmpWord.replace("`", '')
                    if tmpWord in model.wv:
                        nextVects.append(model.wv[tmpWord].tolist())
                    else:
                        count += 1
                        # print(tmpWord, count, 2.1)
                        nextVects.append(newModel.wv["`" + tmpWord].tolist())
                elif tmpWord.endswith("'"):
                    tmpWord = tmpWord.replace("'", '')
                    if tmpWord in model.wv:
                        nextVects.append(model.wv[tmpWord].tolist())
                    else:
                        count += 1
                        # print(tmpWord, count, 2.2)
                        nextVects.append(newModel.wv[tmpWord + "'"].tolist())
                elif tmpWord.isdigit():
                    nextVects.append(model.wv['one'].tolist())
                else:
                    count += 1
                    # print(tmpWord, count, 3)
                    nextVects.append(newModel.wv[tmpWord].tolist())
        if nextVects != None:
            nextVects.reverse()
        data['nextVects'] = nextVects
        tDatas.append(data)
    return tDatas

def paddingWordEmbedding(datas):
    prevMaxLen = 0
    nextMaxLen = 0
    for data in datas:
        if data['prevVects'] != None:
            prevMaxLen = max(prevMaxLen, len(data['prevVects']))
        if data['nextVects'] != None:
            nextMaxLen = max(nextMaxLen, len(data['nextVects']))
    
    for data in datas:
        prevLenToPadding = prevMaxLen
        if data['prevVects'] != None:
            prevLenToPadding = prevMaxLen - len(data['prevVects'])
        else:
            data['prevVects'] = []
        nextLenToPadding = nextMaxLen
        if data['nextVects'] != None:
            nextLenToPadding = nextMaxLen - len(data['nextVects'])
        else:
            data['nextVects'] = []
        data['prevVects'] = [[0.0] * 300] * prevLenToPadding + data['prevVects']
        data['nextVects'] = [[0.0] * 300] * nextLenToPadding + data['nextVects']
    
    print('prevMaxLen', prevMaxLen)
    print('nextMaxLen', nextMaxLen)
    print('************************')
    return datas

def getTrainDatas(datas, batchSize):
    prevVects = []
    nextVects = []
    scores = []
    words = []
    POSs = []
    for data in datas:
        prevVects.append(data['prevVects'])
        nextVects.append(data['nextVects'])
        scores.append([
            data['posScore'] * 1000,
            -data['negScore'] * 1000,
            # 1 - data['posScore'] - data['negScore']
        ])
        words.append(data['word'])
        POSs.append(data['POS'])
    size = len(prevVects)
    size = size - size % batchSize
    return {
        'size': size,
        'prevVects': prevVects[:size],
        'nextVects': nextVects[:size],
        'scores': scores[:size],
        'words': words[:size],
        'POSs': POSs[:size]
    }

def outputTSV(datas, filename='./wordembedding.tsv'):
    ret = ''
    for i, data in enumerate(datas):
        tmpData = ''
        for j, tmp in enumerate(data):
            if j != 0:
                tmpData = tmpData + '	'
            tmpData = tmpData + str(tmp)
        if i != 0:
            ret = ret + os.linesep
        ret = ret + tmpData
    fo = open(filename, 'w+')
    fo.write(ret)
    fo.close()

def outputWordEmbeddingMeta(datas, filename='./wordembedding_meta.tsv'):
    ret = ''
    for i, data in enumerate(datas):
        if i != 0:
            ret = ret + os.linesep
        ret = ret + data
    fo = open(filename, 'w+')
    fo.write(ret)
    fo.close()

datas = readSentiWordNet()
print('size:', len(datas['arrDatas']))
datas['arrDatas'] = wordEmbedding(datas['arrDatas'])
datas['arrDatas'] = paddingWordEmbedding(datas['arrDatas'])

WORD_EMBEDDING_SIZE = 300
WORD_EMBEDDING_SIZE_TARGET = 300

STEP_SIZE_L = 25
STEP_SIZE_R = 27

OUTPUT_SIZE = 2

TRAIN_BATCH_SIZE = 50

LEARN_RATE = 0.01

MAX_EPOCH = 20

def main():
    input_l = tf.placeholder(tf.float32, [TRAIN_BATCH_SIZE, STEP_SIZE_L, WORD_EMBEDDING_SIZE], name="input_l")
    input_r = tf.placeholder(tf.float32, [TRAIN_BATCH_SIZE, STEP_SIZE_R, WORD_EMBEDDING_SIZE], name='input_r')
    input_y = tf.placeholder(tf.float32, [TRAIN_BATCH_SIZE, OUTPUT_SIZE])

    with tf.variable_scope('left'):
        lstm_cell_l = tf.nn.rnn_cell.BasicLSTMCell(WORD_EMBEDDING_SIZE_TARGET, forget_bias=1.0, state_is_tuple=True)
        init_state_l = lstm_cell_l.zero_state(TRAIN_BATCH_SIZE, dtype=tf.float32)
        outputs_l, state_l = tf.nn.dynamic_rnn(lstm_cell_l, input_l, initial_state=init_state_l, dtype=tf.float32)
    
    with tf.variable_scope('right'):
        lstm_cell_r = tf.nn.rnn_cell.BasicLSTMCell(WORD_EMBEDDING_SIZE_TARGET, forget_bias=1.0, state_is_tuple=True)
        init_state_r = lstm_cell_r.zero_state(TRAIN_BATCH_SIZE, dtype=tf.float32)
        outputs_r, state_r = tf.nn.dynamic_rnn(lstm_cell_r, input_r, initial_state=init_state_r, dtype=tf.float32)
    
    state_concat = tf.concat([state_l[1], state_r][1], 1)

    state_transform_weight = tf.Variable(tf.truncated_normal(
        shape=[2 * WORD_EMBEDDING_SIZE_TARGET, WORD_EMBEDDING_SIZE_TARGET],
        mean=0,
        stddev=1), 
        'state_transform_weight',
        dtype=tf.float32
    )
    # target_state = tf.matmul(state_concat, state_transform_weight)

    # output_weight = tf.Variable(tf.truncated_normal(
    #     shape=[WORD_EMBEDDING_SIZE_TARGET, OUTPUT_SIZE],
    #     mean=0,
    #     stddev=1), 
    #     'output_weight',
    #     dtype=tf.float32
    # )
    
    # output = tf.matmul(target_state, output_weight)
    target_state = tf.matmul(state_concat, state_transform_weight)

    low_dimen_state_weight = tf.Variable(tf.truncated_normal(
        shape=[WORD_EMBEDDING_SIZE_TARGET, 3],
        mean=0,
        stddev=1), 
        'output_weight',
        dtype=tf.float32
    )
    low_dimen_state = tf.matmul(target_state, low_dimen_state_weight)

    output_weight = tf.Variable(tf.truncated_normal(
        shape=[3, OUTPUT_SIZE],
        mean=0,
        stddev=1), 
        'output_weight',
        dtype=tf.float32
    )
    
    output = tf.matmul(low_dimen_state, output_weight)
    # output = tf.nn.softmax(output)

    # cost = tf.reduce_mean(tf.square(tf.reshape(input_y - output, [-1])))
    cost = tf.reduce_mean(tf.square(input_y - output))
    train_op = tf.train.AdamOptimizer(LEARN_RATE).minimize(cost)

    init = tf.global_variables_initializer()

    session = tf.InteractiveSession()
    session.run(init)

    fileCount = 0

    trainDatas = getTrainDatas(datas['arrDatas'],TRAIN_BATCH_SIZE)
    for i in range(MAX_EPOCH):
        print('#################### new epoch:', i)
        wordEmbedding = {}
        lowDimenWordEmbedding = {}
        outputRet = []

        index = 0
        size = trainDatas['size']
        while index < size:
            if index + TRAIN_BATCH_SIZE > size:
                break

            prevTrainVects = trainDatas['prevVects'][index:index + TRAIN_BATCH_SIZE]
            nextTrainVects = trainDatas['nextVects'][index:index + TRAIN_BATCH_SIZE]
            trainScores = trainDatas['scores'][index:index + TRAIN_BATCH_SIZE]
            trainWords = trainDatas['words'][index:index + TRAIN_BATCH_SIZE]
            trainPOSs = trainDatas['POSs'][index:index + TRAIN_BATCH_SIZE]

            for k in range(150):
                print(session.run([train_op, cost], feed_dict={
                    input_l: prevTrainVects,
                    input_r: nextTrainVects,
                    input_y: trainScores
                }))
            index += TRAIN_BATCH_SIZE

            tmpNewWordEmbedding, tmpLowDimenWordEmbedding, tmpOutputRet = session.run(
                [
                    target_state,
                    low_dimen_state,
                    output
                ], 
                feed_dict={
                    input_l: prevTrainVects,
                    input_r: nextTrainVects,
                    input_y: trainScores
                }
            )
            tmpNewWordEmbedding = tmpNewWordEmbedding.tolist()
            tmpLowDimenWordEmbedding = tmpLowDimenWordEmbedding.tolist()
            tmpOutputRet = tmpOutputRet.tolist()

            outputRet = outputRet + tmpOutputRet
            for iii, word in enumerate(trainWords):
                POS = trainPOSs[iii]
                tmpWord = word + '_' + POS
                wordEmbedding[tmpWord] = tmpNewWordEmbedding[iii]
                lowDimenWordEmbedding[tmpWord] = tmpLowDimenWordEmbedding[iii]
            
            fileCount = fileCount + 1
            outputTSV(outputRet, 'output' + str(fileCount) + '.tsv')
            outputTSV(wordEmbedding.values(), './wordembedding' + str(fileCount) + '.tsv')
            outputTSV(lowDimenWordEmbedding.values(), './low-dimen-wordembedding' + str(fileCount) + '.tsv')
            outputWordEmbeddingMeta(wordEmbedding.keys(), './wordembedding_meta' + str(fileCount) + '.tsv')
        
        # tmpNewWordEmbedding = []
        # tmpLowDimenWordEmbedding = []
        # outputRet = []
        # index = 0
        # size = trainDatas['size']
        # while index < size:
        #     if index + TRAIN_BATCH_SIZE > size:
        #         break

        #     prevTrainVects = trainDatas['prevVects'][index:index + TRAIN_BATCH_SIZE]
        #     nextTrainVects = trainDatas['nextVects'][index:index + TRAIN_BATCH_SIZE]
        #     trainScores = trainDatas['scores'][index:index + TRAIN_BATCH_SIZE]

        #     tmpNewWordEmbedding = tmpNewWordEmbedding + session.run(target_state, feed_dict={
        #         input_l: prevTrainVects,
        #         input_r: nextTrainVects,
        #         input_y: trainScores
        #     }).tolist()
        #     tmpLowDimenWordEmbedding = tmpLowDimenWordEmbedding + session.run(low_dimen_state, feed_dict={
        #         input_l: prevTrainVects,
        #         input_r: nextTrainVects,
        #         input_y: trainScores
        #     }).tolist()
        #     outputRet = outputRet + session.run(output, feed_dict={
        #         input_l: prevTrainVects,
        #         input_r: nextTrainVects,
        #         input_y: trainScores
        #     }).tolist()
        #     index += TRAIN_BATCH_SIZE
        
        # for ii, word in enumerate(trainDatas['words']):
        #     POS = trainDatas['POSs'][ii]
        #     tmpWord = word + '_' + POS
        #     wordEmbedding[tmpWord] = tmpNewWordEmbedding[ii]
        #     lowDimenWordEmbedding[tmpWord] = tmpLowDimenWordEmbedding[ii]
        
        # outputTSV(outputRet, 'output.tsv')
        # outputTSV(wordEmbedding.values(), './wordembedding.tsv')
        # outputTSV(wordEmbedding.values(), './low-dimen-wordembedding.tsv')
        # outputWordEmbeddingMeta(wordEmbedding.keys())
        

    # tmpNewWordEmbedding = []
    # outputRet = []
    # index = 0
    # size = trainDatas['size']
    # while index < size:
    #     if index + TRAIN_BATCH_SIZE > size:
    #         break

    #     prevTrainVects = trainDatas['prevVects'][index:index + TRAIN_BATCH_SIZE]
    #     nextTrainVects = trainDatas['nextVects'][index:index + TRAIN_BATCH_SIZE]
    #     trainScores = trainDatas['scores'][index:index + TRAIN_BATCH_SIZE]

    #     tmpNewWordEmbedding = tmpNewWordEmbedding + session.run(low_dimen_state, feed_dict={
    #         input_l: prevTrainVects,
    #         input_r: nextTrainVects,
    #         input_y: trainScores
    #     }).tolist()
    #     outputRet = outputRet + session.run(output, feed_dict={
    #         input_l: prevTrainVects,
    #         input_r: nextTrainVects,
    #         input_y: trainScores
    #     }).tolist()
    #     index += TRAIN_BATCH_SIZE
    
    # wordEmbedding = {}
    # for i, word in enumerate(trainDatas['words']):
    #     POS = trainDatas['POSs'][i]
    #     tmpWord = word + '_' + POS
    #     wordEmbedding[tmpWord] = tmpNewWordEmbedding[i]
    
    # outputTSV(outputRet, 'output.tsv')
    # outputTSV(wordEmbedding.values())
    # outputWordEmbeddingMeta(wordEmbedding.keys())
    

main()

