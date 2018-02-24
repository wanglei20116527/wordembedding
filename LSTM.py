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

WORD_EMBEDDING_SIZE = 300

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

def readSentimentWordEmbedding():
    wvsArr = []
    fi1 = open('./wordembedding.tsv', 'r+')
    doc1 = fi1.read()
    sentences1 = doc1.split(os.linesep)
    for i, sentence in enumerate(sentences1):
        wv = []
        words = sentence.split('	')
        for word in words:
            wv.append(float(word))
        wvsArr.append(wv)
    fi1.close()

    fi2 = open('./wordembedding_meta.tsv', 'r+')
    doc2 = fi2.read()
    wordsArr = doc2.split(os.linesep)
    fi2.close()

    wvsMapping = {}
    for i, wv in enumerate(wvsArr):
        wvsMapping[wordsArr[i]] = wv
    
    return wvsMapping


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


# datas = readDatas('./labtop.xml')
# datas = word2vec(datas)
# print(datas[0])
