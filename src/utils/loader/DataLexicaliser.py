######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2016 #
######################################################################
######################################################################
import os
import sys
import operator
sys.path.append('./utils/')
import itertools
import nltk
import json
import datetime
from num2words import num2words

file = open
class DataLexicaliser(object):

    def __init__(self):
        fin = file('utils/resource/special_values.txt')
        self.special_values= list(json.load(fin).keys()) + ['?']
        fin.close()
        fin = file('utils/resource/special_slots.txt')
        self.special_slots = json.load(fin)
        fin.close()
        
    def delexicalise(self,sent,jssv):
        raise NotImplementedError('method delexicalise() hasn\'t been implemented')
    def lexicalise(self,sent,jssv):
        raise NotImplementedError('method lexicalise() hasn\'t been implemented')

class ExactMatchDataLexicaliser(DataLexicaliser):

    def __init__(self):
        DataLexicaliser.__init__(self)
        self.typetoken = ''
        fin = open('utils/nlp/mapping.pair')
        self.replacements = []
        for line in fin.readlines():
            tok_from, tok_to = line.replace('\n','').split('\t')
            self.replacements.append((' '+tok_from+' ',' '+tok_to+' '))

    def delexicalise(self,sent,jssv):
        # no slot values return directly
        if len(jssv)==1 and jssv[0][1]==None:
            return sent
        new_jssv = []
        for slot,value in jssv:
            if value in self.special_values : continue # special values, skip      
            if value is None: continue
            new_jssv.append((slot,value))
        for slot,value in sorted(new_jssv,key=lambda x:len(x[-1]),reverse=True): 
            for fromx, tox in self.replacements:
                value = ' '+value+' '
                value = value.replace(fromx,tox)[1:-1]

            # taking care of all possible permutations of multiple values
            vs = value.replace(' or ',' and ').split(' and ')
            permutations =  [' and '.join(x) for x in itertools.permutations(vs)]+\
                            [' or '.join(x) for x in itertools.permutations(vs)]
            
            # try to match for each possible permutation
            isMatched = False
            for p in permutations:
                if p in sent : # exact match , ends 
                    sent = (' '+sent+' ').replace(\
                            ' '+p+' ',' SLOT_'+slot.upper()+' ',1)[1:-1]
                    isMatched = True
                    break
                else:
                    if slot == 'phone':
                        p = p.replace(' ','')
                        p = p.replace('-','')
                        if p in sent : # exact match , ends 
                            sent = (' '+sent+' ').replace(\
                                    ' '+p+' ',' SLOT_'+slot.upper()+' ',1)[1:-1]
                            isMatched = True
                            break
                    elif slot == 'choice':
                        if p.isdigit():
                            p = num2words(p)
                            if p in sent : # exact match , ends 
                                sent = (' '+sent+' ').replace(\
                                        ' '+p+' ',' SLOT_'+slot.upper()+' ',1)[1:-1]
                                isMatched = True
                                break
            if not isMatched: 
                pass
                #raise ValueError('value "'+value+'" cannot be delexicalised!')
        return sent

    def lexicalise(self,sent,jssv):
        # no slot values return directly
        if len(jssv)==1 and jssv[0][1]==None:
            return sent
        
        # replace values 
        for slot,value in jssv:
            if value in self.special_values : continue # special values, skip     
            if value is None: continue
            if 'SLOT_'+slot.upper() not in sent : 
                pass
                #raise ValueError('slot "SLOT_'+slot.upper()+'" does not exist !')
            else: 
                sent=(' '+sent+' ').replace(' SLOT_'+slot.upper()+' ',' '+value+' ',1)[1:-1]
        sent = (' '+sent+' ').replace(' SLOT_TYPE ',' '+self.typetoken+' ')[1:-1]
        return sent    


#if __name__ == '__main__':

