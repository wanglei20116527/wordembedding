from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import tensorflow as tf
import re, gensim, logging, os, pdb
import xml.etree.cElementTree as ET

tf.logging.set_verbosity(tf.logging.INFO)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

POLARITY_EMBEDDING = {
	'positive': [1, 0, 0, 0],
	'negative': [0, 1, 0, 0],
	'neutral':  [0, 0, 1, 0],
	'conflict': [0, 0, 0, 1]
}

def makeOneHot(datas):
	oneHots = {}
	size = len(datas)
	for i, data in enumerate(datas):
		oneHot = [0] * size
		oneHot[i] = 1
		oneHots[data] = oneHot
	return oneHots

POSVectorMap = makeOneHot([
	'CC', 'CD', 'DT', 'EX', 'FW', 'IN',
	'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN',
	'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP',
	'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM',
	'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN',
	'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB',
])

WORD_EMBEDDING_SIZE = 300
POLARITY_EMBEDDING_SIZE = 4
POS_EMBEDDING_SIZE = len(POSVectorMap.keys())
CONV1_SIZE = 3

def readDatas(filePath):
	sentences = []
	tree = ET.ElementTree(file=filePath)
	for sts in tree.getroot():
		sentence = {}
		for child in sts:
			if child.tag == 'text':
				sentence['text'] = child.text.lower()
				continue

			if child.tag == 'aspectTerms':
				aspects = []
				for aspect in child:
					aspects.append({
						'text': aspect.attrib['term'].lower(),
						'polarity': aspect.attrib['polarity'].lower()
					})
				sentence['aspects'] = aspects
		if 'aspects' in sentence:
			sentences.append(sentence)

	datas = []
	for sts in sentences:
		for aspect in sts['aspects']:
			datas.append({
				'text': sts['text'],
				'aspect': aspect['text'],
				'polarity': aspect['polarity']
			})
	return datas

def word2vec(datas):
	sentences = []
	for data in datas:
		sentence = data['text']
		sentence = ' ' + sentence + ' '
		sentence = sentence.replace('- ', ' ')
		sentence = sentence.replace(' -', ' ')
		tmpWords = re.split(r'[^\w\_\-]', sentence)
		words = []
		for tmpWord in tmpWords:
			if tmpWord != '':
				words.append(tmpWord)
		sentences.append(words)
	# model = gensim.models.Word2Vec(sentences, min_count=1, size=WORD_EMBEDDING_SIZE)
	model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
	rets = []
	for i, sentence in enumerate(sentences):
		tmpSentence = []
		for word in sentence:
			if word != '':
				try:
					tmpSentence.append(model.wv[word].tolist())
				except Exception as e:
					tmpSentence.append([0] * WORD_EMBEDDING_SIZE)
					print('error word&&&&&&&&: ', word)
		
		aspect = datas[i]['aspect']
		aspect = ' ' + aspect + ' '
		aspect = aspect.replace('- ', ' ')
		aspect = aspect.replace(' -', ' ')
		# tmpAspectWords可能含有empty string
		tmpAspectWords = re.split(r'[^\w\_\-]', aspect) 
		aspectWords = []
		aspectVects = []
		for word in tmpAspectWords:
			if word != '':
				try:
					aspectWords.append(word)
					aspectVects.append(model.wv[word].tolist())
				except Exception as e:
					aspectVects.append([0] * WORD_EMBEDDING_SIZE)
					print('error: #', word)
		# 下面计算aspect是句子中的开始下标和结束下标（单词级别）
		sentenceString = '#'.join(sentence)
		aspectString = '#'.join(aspectWords)
		try:
			tmpIndex = sentenceString.index(aspectString)
		except Exception as e:
			print('sentence:#', sentenceString, '#')
			print('aspect:#', aspectString, '#')
			print(aspectWords)
			print('*******************')

		startIndex = 0
		prevSubStr = sentenceString[:tmpIndex]
		for word in prevSubStr.split('#'):
			if word != '':
				startIndex += 1
		
		endIndex = startIndex - 1
		for word in aspectWords:
			if word != '':
				endIndex += 1
		
		rets.append({
			'words': sentence,
			'vects': tmpSentence,
			'aspectVects': aspectVects,
			'aspectWords': aspectWords,
			'polarity': datas[i]['polarity'],
			'startIndex': startIndex,
			'endIndex': endIndex
		})
	return rets

def writeDatasToFileForJavaPOS(datas, filePath):
	doc = ''
	for data in datas:
		line = ''
		for word in data['words']:
			line += word + ' '
		line = line.strip()

		doc += os.linesep + line
	doc = doc.replace(os.linesep, '', 1)

	fo = open(filePath, "w+")
	fo.write(doc)
	fo.close()

def readPOSToDatas(file, datas):
	f = open(file, 'r+')
	doc = f.read()
	sentences = doc.split(os.linesep)
	for i, sentence in enumerate(sentences):
		words = sentence.split(' ')
		pos = []
		for word in words:
			components = word.split('_')
			pos.append(components[1])
		datas[i]['pos'] = pos

		posVects = []
		try:
			for tag in pos:
				posVects.append(POSVectorMap[tag])
			datas[i]['posVects'] = posVects
		except Exception as e:
			print('i:', i)
			print(datas[i]['words'])
			print(pos)

		aspectStartIndex = datas[i]['startIndex']
		aspectEndIndex = datas[i]['endIndex']
		datas[i]['aspectPos'] = pos[aspectStartIndex:aspectEndIndex + 1]
		datas[i]['aspectPosVects'] = posVects[aspectStartIndex:aspectEndIndex + 1]
	return datas

def getDistsToAspectInSentence(datas):
	for data in datas:
		startIndex = data['startIndex']
		endIndex = data['endIndex']

		maxDist = max(startIndex, len(data['words']) - endIndex - 1)

		dists = []
		for i, word in enumerate(data['words']):
			if i < startIndex:
				dists.append([(maxDist - startIndex + i + 1) * 0.1])
			elif i > endIndex:
				dists.append([(maxDist - i + endIndex + 1) * 0.1])
			else:
				dists.append([(maxDist + 1) * 0.1])
		data['dists'] = dists
	return datas

def markWordPolarityInSentence(datas):
	TAGMAP = {
		'NN': 'n',
		'NNS': 'n',
		'NNP': 'n',
		'NNPS': 'n',
		'PRP': 'n',
		'PRP$': 'n',
		'WP': 'n',
		'WP$': 'n',
		'RB': 'r',
		'RBR': 'r',
		'RBS': 'r',
		'WRB': 'r',
		'VB': 'v',
		'VBD': 'v',
		'VBG': 'v',
		'VBN': 'v',
		'VBP': 'v',
		'VBZ': 'v',
		'JJ': 'a',
		'JJR': 'a',
		'JJS': 'a',
	}
	
	doc = open('./SentiWordNet.txt', 'r+')
	sentimentScore = {}
	for line in doc:
		line = line.strip()
		if not line.startswith('#'):
			cols = line.split('\t')
			if len(cols) != 6:
				print('parse SentiWordNet error')
				exit(1)
			POS = cols[0]
			posScore = float(cols[2])
			negScore = float(cols[3])
			objScore = 1 - posScore - negScore
			words = cols[4].split(' ')
			for word in words:
				if word == '':
					continue
				word = word.split('#')[0]
				if not word in sentimentScore:
					sentimentScore[word] = {}
				if not POS in sentimentScore[word]:
					sentimentScore[word][POS] = []
				sentimentScore[word][POS].append({
					'posScore': posScore,
					'negScore': negScore
				})
	doc.close()

	tmpResult = [0, 0]
	print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
	print(sentimentScore['no'])
	# for tmp in sentimentScore['no']['a']:
	# 	tmpResult[0] += tmp['posScore']
	# 	tmpResult[1] += tmp['negScore']
	
	# print(tmpResult[0] / len(sentimentScore['no']['a']), tmpResult[1] / len(sentimentScore['no']['a']))
	# print('********************************')
	

	for data in datas:
		dataScores = []
		for i, word in enumerate(data['words']):
			pos = data['pos'][i]
			if not word in sentimentScore:
				# 最后一位表示该polarity score是否有效
				dataScores.append([0, 0])
				continue

			if (pos in TAGMAP) and (TAGMAP[pos] in sentimentScore[word]):
				pos = TAGMAP[pos]
				posScoreSum = 0
				negScoreSum = 0
				for score in sentimentScore[word][pos]:
					posScoreSum += score['posScore']
					negScoreSum += score['negScore']
				posScore = posScoreSum / len(sentimentScore[word][pos])
				negScore = negScoreSum / len(sentimentScore[word][pos])
				objScore = 1 - posScore - negScore
				dataScores.append([posScore, -negScore])
			else:
				count = 0
				posScoreSum = 0
				negScoreSum = 0
				for pos in sentimentScore[word]:
					scores = sentimentScore[word][pos]
					count += len(scores)
					for score in scores:
						posScoreSum += score['posScore']
						negScoreSum += score['negScore']
				posScore = posScoreSum / len(sentimentScore[word][pos])
				negScore = negScoreSum / len(sentimentScore[word][pos])
				objScore = 1 - posScore - negScore
				dataScores.append([posScore, -negScore])
		data['sentimentScores'] = dataScores
		startIndex = data['startIndex']
		endIndex = data['endIndex']
		data['aspectSentimentScores'] = dataScores[startIndex:endIndex + 1]
	return datas

def paddingDatas(datas):
	maxLenOfSentence = 0
	maxLenOfAspect = 0
	for data in datas:
		maxLenOfSentence = max(maxLenOfSentence, len(data['words']))
		maxLenOfAspect = max(maxLenOfAspect, len(data['aspectWords']))
	
	for data in datas:
		vects = data['vects']
		vectDimen = len(vects[0])
		posVects = data['posVects']
		posDimen = len(POSVectorMap.keys())
		dists = data['dists']
		aspectVects = data['aspectVects']
		aspectPosVects = data['aspectPosVects']
		sentimentScores = data['sentimentScores']
		aspectSentimentScores = data['aspectSentimentScores']
		
		startWordIndex = len(vects)
		while startWordIndex < maxLenOfSentence:
			# padding sentence word embedding
			vects.append([0] * vectDimen)
			# padding sentence pos
			posVects.append([0] * posDimen)
			# padding sentence dists
			dists.append([0])
			# padding sentence polarity
			sentimentScores.append([0] * 2)			

			startWordIndex += 1

		
		startAspectIndex = len(aspectVects)
		while startAspectIndex < maxLenOfAspect:
			# padding aspect embedding
			aspectVects.append([0] * vectDimen)
			# padding aspect pos
			aspectPosVects.append([0] * posDimen)
			# padding aspect polarity
			aspectSentimentScores.append([0] * 2)

			startAspectIndex += 1

	return datas

def getMemory(data):
	if len(data['vects']) != len(data['sentimentScores']):
		print('vects dimen is not equal to sentimentScores dimen')
		exit(1)

	memory = []
	for i, tmp in enumerate(data['vects']):
		memory.append(data['vects'][i] + data['sentimentScores'][i])
	return memory

def makeMemory(datas):
	for data in datas:
		data['memory'] = getMemory(data)
	return datas

datas = readDatas('./labtop.xml')
datas = word2vec(datas)

# 这里是用来将预处理好的句子按照指定的格式（每句话用“.”分割）保存到指定文件，用于stanford POS处理
# writeDatasToFileForJavaPOS(datas, './labtop-data-to-pos.txt')

# 这里是用来读取standford PSO处理后的文件，在使用之前请先调用writeDatasToFileForJavaPOS生产文件供standford PSO处理
datas = readPOSToDatas('./labtop-data-posed.txt', datas)
datas = getDistsToAspectInSentence(datas)
datas = markWordPolarityInSentence(datas)
datas = paddingDatas(datas)
# datas = makeMemory(datas)

def getInputDatas(datas):
	# memorys = []
	words = []
	aspects = []
	dists = []
	POSs = []
	polarities = []
	sentimentScores=[]
	aspectSentimentScores=[]
	for data in datas:
		# memorys.append(data['memory'])
		words.append(data['vects'])
		aspects.append(data['aspectVects'])
		dists.append(data['dists'])
		POSs.append(data['posVects'])
		sentimentScores.append(data['sentimentScores'])
		aspectSentimentScores.append(data['aspectSentimentScores'])
		polarities.append(POLARITY_EMBEDDING[data['polarity']])
	inputDatas = {
		'words': words,
		# 'memorys': memorys,
		'aspects': aspects,
		'dists': dists,
		'POSs': POSs,
		'sentimentScores': sentimentScores,
		'aspectSentimentScores': aspectSentimentScores,
		'polarities': polarities
	}
	return inputDatas

def buildConvNet(words_placeholder, poses_placeholder, sentiment_scores_placeholder, aspects_placeholder, words_filter, poses_filter, attention, wordsDimens, convSize):
	words_polarities = tf.concat([words_placeholder, aspects_placeholder, sentiment_scores_placeholder], 2)
	words_polarities = words_polarities * attention
	words_polarities_pad = tf.pad(words_polarities, [
		[0, 0],
		[convSize - 1, 0],
		[0, 0]
	])
	words_polarities_pad_NHWC = tf.transpose([
		words_polarities_pad
	], [1, 2, 3, 0])
	words_conv = tf.nn.conv2d(words_polarities_pad_NHWC, words_filter, [1, 1, 1, 1], padding='VALID')
	words_conv = tf.nn.relu(words_conv)
	words_conv = tf.reshape(words_conv, [-1, wordsDimens[0], 1])
	words_attention = tf.nn.softmax(words_conv, 1)

	poses_polarities = tf.concat([poses_placeholder, aspects_placeholder, sentiment_scores_placeholder], 2)
	poses_polarities = poses_polarities * attention
	poses_polarities_pad = tf.pad(poses_polarities, [
		[0, 0],
		[convSize - 1, 0],
		[0, 0]
	])
	poses_polarities_pad_NHWC = tf.transpose([
		poses_polarities_pad
	], [1, 2, 3, 0])
	poses_conv = tf.nn.conv2d(poses_polarities_pad_NHWC, poses_filter, [1, 1, 1, 1], padding='VALID')
	poses_conv = tf.nn.relu(poses_conv)
	poses_conv = tf.transpose(poses_conv, [
		0, 3, 1, 2
	])
	poses_conv = tf.reshape(poses_conv, [-1, wordsDimens[0], 1])
	poses_attention = tf.nn.softmax(poses_conv, 1)

	return words_attention, poses_attention
def fullyNet(x, y_, dimenOfInputData, numOfClasses):
	numNodesInLayer1 = 10
	w1 = tf.Variable(tf.truncated_normal(shape=[dimenOfInputData, numNodesInLayer1], mean=0, stddev=1), 'fullyNetWeight1', dtype=tf.float32)
	b1 = tf.Variable(tf.zeros([numNodesInLayer1]), 'fullyNetBias1', dtype=tf.float32)
	y1 = tf.nn.tanh(tf.matmul(x, w1) + b1)

	w2 = tf.Variable(tf.truncated_normal(shape=[numNodesInLayer1, numOfClasses], mean=0, stddev=1), 'fullyNetWeight2', dtype=tf.float32)
	b2 = tf.Variable(tf.zeros([numOfClasses]), 'fullyNetBias2', dtype=tf.float32)

	y = tf.matmul(y1, w2) + b2

	cross_entropy = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(
			labels=y_,
			logits=y
		)
	)

	train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)
	return y, train_step

def buildModel(
	wordsDimens,
	posesDimens,
	distsDimens,
	aspectsDimens,
	sentimentScoresDimens,
	aspectSentimentScoresDimens,
	polaritiesDimens,
	convSize,
	words,
	poses,
	dists,
	aspects,
	polarities,
	sentimentScores,
	aspectSentimentScores):

	words_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, wordsDimens[0], wordsDimens[1]))
	poses_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, posesDimens[0], posesDimens[1]))
	dists_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, distsDimens[0], distsDimens[1]))
	sentiment_scores_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, sentimentScoresDimens[0], sentimentScoresDimens[1]))
	aspect_sentiment_scores_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, aspectSentimentScoresDimens[0], aspectSentimentScoresDimens[1]))
	aspects_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, aspectsDimens[0], aspectsDimens[1]))
	polarities_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, polaritiesDimens[0]))

	words_filter = tf.Variable(tf.truncated_normal(shape=[convSize, wordsDimens[1] + aspectsDimens[1] + sentimentScoresDimens[1], 1, 1], mean=0, stddev=1), 'words_filter', dtype=tf.float32)
	poses_filter = tf.Variable(tf.truncated_normal(shape=[convSize, posesDimens[1] + aspectsDimens[1] + sentimentScoresDimens[1], 1, 1], mean=0, stddev=1), 'poses_filter', dtype=tf.float32)

	attention = tf.constant([[1]] * wordsDimens[0], dtype=tf.float32)

	words_attention_percent = tf.Variable(random.random(), 'words_attention_percent', dtype=tf.float32)
	poses_attention_percent = tf.Variable(random.random(), 'poses_attention_percent', dtype=tf.float32)
	dists_attention_percent = 1 - words_attention_percent - poses_attention_percent

	aspects_placeholder1 = tf.reduce_sum(aspects_placeholder, 1, True)
	aspects_placeholder1 = tf.tile(aspects_placeholder1, [1, wordsDimens[0], 1])

	for i in range(20):
		# 这里可以重复几次，从而使得memory attention based CNN能够递归起来
		words_attention, poses_attention = buildConvNet(
			words_placeholder,
			poses_placeholder,
			sentiment_scores_placeholder,
			aspects_placeholder1,
			words_filter,
			poses_filter,
			attention,
			wordsDimens,
			convSize
		)
		attention = words_attention * words_attention_percent + poses_attention * poses_attention_percent
	
	words_polarities = tf.concat([words_placeholder, aspects_placeholder1, sentiment_scores_placeholder], 2)
	
	words_represents = tf.reduce_sum(words_polarities * attention, 1)
	y, train_step = fullyNet(words_represents, polarities_placeholder, wordsDimens[1] + aspectsDimens[1] + sentimentScoresDimens[1], polaritiesDimens[0])


	# dists_softmax = tf.nn.softmax(dists_placeholder, 1)
	# sentiment_scores_placeholder1 = sentiment_scores_placeholder * dists_softmax

	# aspect_sentiment_scores_placeholder1 = tf.reshape(aspect_sentiment_scores_placeholder, [-1, aspectSentimentScoresDimens[0] * aspectSentimentScoresDimens[1]])

	# words_represents = tf.reshape(sentiment_scores_placeholder1, [-1, sentimentScoresDimens[0] * sentimentScoresDimens[1]])

	# words_represents = tf.concat([words_represents, aspect_sentiment_scores_placeholder1], 1)
	
	# # words_represents = tf.reduce_sum(words_represents, 1)
	# y, train_step = fullyNet(words_represents, polarities_placeholder, sentimentScoresDimens[0] * sentimentScoresDimens[1] + aspectSentimentScoresDimens[0] * aspectSentimentScoresDimens[1], polaritiesDimens[0])
	
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())
	BATCH_SIZE = 100
	for index in range(100):
		# print('new epcho')
		i = 0
		size = len(words)
		# size = 500
		while i < size:
			startIndex = i
			endIndex = i + BATCH_SIZE
			if endIndex > size:
				endIndex = size
			
			wordsToTrain = words[startIndex:endIndex]
			posesToTrain = poses[startIndex:endIndex]
			distsToTran = dists[startIndex:endIndex]
			aspectsToTrain = aspects[startIndex:endIndex]
			polaritiesToTrain = polarities[startIndex:endIndex]
			sentimentScoresToTrain = sentimentScores[startIndex:endIndex]
			aspectSentimentScoresToTrain = aspectSentimentScores[startIndex:endIndex]

			sess.run(
				train_step,
				feed_dict={
					words_placeholder: wordsToTrain,
					poses_placeholder: posesToTrain,
					dists_placeholder: distsToTran,
					aspects_placeholder: aspectsToTrain,
					polarities_placeholder: polaritiesToTrain,
					sentiment_scores_placeholder: sentimentScoresToTrain,
					aspect_sentiment_scores_placeholder: aspectSentimentScoresToTrain
				}
			)
			i += BATCH_SIZE

			correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(polarities_placeholder, 1))
			
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
			print(
				sess.run([
						accuracy,
						correct_prediction,
						tf.argmax(y, 1),
						# dists_softmax,
						# y,
						# words_represents,
					],
					feed_dict = {
						words_placeholder: wordsToTrain,
						poses_placeholder: posesToTrain,
						dists_placeholder: distsToTran,
						aspects_placeholder: aspectsToTrain,
						polarities_placeholder: polaritiesToTrain,
						sentiment_scores_placeholder: sentimentScoresToTrain,
						aspect_sentiment_scores_placeholder: aspectSentimentScoresToTrain
					}		  
				)
			)

			# print('*******************************')

def main():
	inputDatas = getInputDatas(datas)

	words = inputDatas['words']
	poses = inputDatas['POSs']
	dists = inputDatas['dists']
	aspects = inputDatas['aspects']
	polarities = inputDatas['polarities']
	sentimentScores = inputDatas['sentimentScores']
	aspectSentimentScores = inputDatas['aspectSentimentScores']

	wordsDimens = [
		len(words[0]),
		len(words[0][0])
	]

	posesDimens = [
		len(poses[0]),
		len(poses[0][0])
	]

	distsDimens = [
		len(dists[0]),
		len(dists[0][0])
	]

	aspectsDimens = [
		len(aspects[0]),
		len(aspects[0][0])
	]

	sentimentScoresDimens = [
		len(sentimentScores[0]),
		len(sentimentScores[0][0])
	]

	aspectSentimentScoresDimens = [
		len(aspectSentimentScores[0]),
		len(aspectSentimentScores[0][0])
	]

	polaritiesDimens = [
		len(polarities[0])
	]

	print(dists[0])

	# buildModel(
	# 	wordsDimens,
	# 	posesDimens,
	# 	distsDimens,
	# 	aspectsDimens,
	# 	sentimentScoresDimens,
	# 	aspectSentimentScoresDimens,
	# 	polaritiesDimens,
	# 	3,
	# 	words,
	# 	poses,
	# 	dists,
	# 	aspects,
	# 	polarities,
	# 	sentimentScores,
	# 	aspectSentimentScores
	# )

main()



