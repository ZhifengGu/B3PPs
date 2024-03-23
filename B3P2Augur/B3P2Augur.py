import tkinter as tk
import os
import pandas
import re
import glob
import joblib
import numpy
import math
import time
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import csv

class Prediction(object):
    def __init__(self):
        self.model_list = []
        self.score = None
        self.x = 0
        self.frame = None
        self.column = 0
        self.row = 0
    
    def ModelLoading(self):
        model_dir = os.path.join(os.getcwd(), "models")
        files = glob.glob(os.path.join(model_dir, "*.pkl"))
        self.model_list = []
        for file in files:
            with open(file, 'rb') as f:
                load_model = joblib.load(f)
                self.model_list.append(load_model)
    
    def ModelPrediction(self):
        prediction_score = None
        for model in self.model_list:
            try:
                temp = model.predict_proba(self.frame.values)
            except AttributeError:
                temp = model.predict(self.frame.values)
            if prediction_score is None:
                prediction_score = temp
            else:
                prediction_score += temp
            
        prediction_score /= len(self.model_list)
        if self.score is None:
                self.score = prediction_score
        else:
            self.score += prediction_score
    
    def DataLoading(self, file):
        f = pandas.read_csv(file, sep=',', header=None)
        self.frame = f.iloc[:, 1:]
        self.row = self.frame.index.size
        self.column = self.frame.columns.size
        self.frame.index=['Sample_%s'%i for i in range(self.row)]
        self.frame.columns = ['F_%s'%i for i in range(self.column)]
        self.label = numpy.array(f.iloc[:, 0]).astype(int)

class FASTA(object):
    def __init__(self, file):
        self.file = file
        self.fasta_list = []
        self.number = 0
        self.encoding_array = numpy.array([])
        self.row = 0
        self.column = 0

        self.fasta_list = self.read_fasta(self.file)
        self.number = len(self.fasta_list)

    def read_fasta(self, file):
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
        text = text.split('>')[1:]
        fasta_sequences = []
        for fasta in text:
            array = fasta.split('\n')
            header = array[0].split()[0]
            header_array = header.split('|')
            sequence = re.sub('[^ACDEFGHIKLMNPQRSTVWY-]', '-', ''.join(array[1:]).upper())
            name = header_array[0]
            label = header_array[1]
            train = header_array[2]
            fasta_sequences.append([name, sequence, label, train])
        return fasta_sequences
    
    def save(self, FileName):
        file = self.encoding_array
        file_save = file[:, 1:]
        numpy.savetxt(FileName, file_save, fmt='%s', delimiter=',')

class AAC(FASTA):
    def __init__(self, file):
        super(AAC, self).__init__(file)

    def AAC(self):
        AA = 'ACDEFGHIKLMNPQRSTVWY'
        encodings = []
        self.encoding_array = numpy.array([])
        header = ['Name', 'label']
        for i in AA:
            header.append(i)
        encodings.append(header)

        for k in self.fasta_list:
            name = k[0]
            sequence = re.sub('-', '' , k[1])
            label = k[2]
            code = [name, label]
            occur = Counter(sequence)
            sum = len(sequence)
            for i in AA:
                code.append(occur[i] / sum)   
            encodings.append(code)

        self.encoding_array = numpy.array(encodings, dtype=str)
        self.row = self.encoding_array[0]
        self.column = self.encoding_array[1]
        del encodings
    
class CKSAAP(FASTA):
    def __init__(self, file):
        super(CKSAAP, self).__init__(file)

    def CKSAAP(self):
        gap = 3
        AA = 'ACDEFGHIKLMNPQRSTVWY'
        encodings = []
        AA_Pairs = []
        self.encoding_array = numpy.array([])
        for i in AA:
            for j in AA:
                AA_Pairs.append(i + j)
        header = ['Name', 'label']
        for g in range(gap + 1):
            for i in AA_Pairs:
                header.append(i + '.gap' + str(g))
        encodings.append(header)

        for k in self.fasta_list:
            name = k[0]
            sequence = re.sub('-', '' , k[1])
            label = k[2]
            code = [name, label]
            for g in range(gap + 1):
                occur = {}
                for pair in AA_Pairs:
                    occur[pair] = 0
                for i in range(len(sequence) - g - 1):
                    j = i + g + 1
                    occur[sequence[i] + sequence[j]] += 1
                for pair in AA_Pairs:
                    code.append(occur[pair] / (len(sequence) - g - 1))
            encodings.append(code)

        self.encoding_array = numpy.array(encodings, dtype=str)
        self.row = self.encoding_array[0]
        self.column = self.encoding_array[1]
        del encodings

class DDE(FASTA):
    def __init__(self, file):
        super(DDE, self).__init__(file)
    
    def DDE(self):
        AA = 'ACDEFGHIKLMNPQRSTVWY'
        encodings = []
        AA_Pairs = []
        self.encoding_array = numpy.array([])
        for i in AA:
            for j in AA:
                AA_Pairs.append(i + j)
        header = ['Name', 'label'] + AA_Pairs
        encodings.append(header)
        Codons = {'A': 4, 'C': 2, 'D': 2, 'E': 2, 'F': 2, 'G': 4, 'H': 2, 'I': 3, 'K': 2, 'L': 6, 
                  'M': 1, 'N': 2, 'P': 4, 'Q': 2, 'R': 6, 'S': 6, 'T': 4, 'V': 4, 'W': 1, 'Y': 2}
        
        Tm = []
        for i in AA_Pairs:
            Tm.append((Codons[i[0]] / 61) * (Codons[i[1]] / 61))
        
        for k in self.fasta_list:
            name = k[0]
            sequence = re.sub('-', '' , k[1])
            label = k[2]
            code = [name, label]
            temp = []
            occur = {}
            for pair in AA_Pairs:
                occur[pair] = 0
            for i in range(len(sequence) - 1):
                occur[sequence[i] + sequence[i + 1]] += 1
            for pair in AA_Pairs:
                temp.append(occur[pair] / (len(sequence) - 1))

            Tv = []
            for i in range(len(Tm)):
                Tv.append(Tm[i] * (1- Tm[i]) / (len(sequence) - 1))

            for i in range(len(temp)):
                temp[i] = (temp[i] - Tm[i]) / math.sqrt(Tv[i])
            code = code +temp
            encodings.append(code)

        self.encoding_array = numpy.array(encodings, dtype=str)
        self.row = self.encoding_array[0]
        self.column = self.encoding_array[1]
        del encodings

class APAAC(FASTA):
    def __init__(self, file):
        super(APAAC, self).__init__(file)
    
    def APAAC(self):
        lambdaValue = 2
        w = 0.05
        dataFile = os.path.join(os.getcwd(), 'data', 'PAAC.txt')
        with open(dataFile) as f:
            PAAC = f.readlines()

        AA = ''.join(PAAC[0].rstrip().split()[1:])
        dict = {}
        prop1 = []
        name = []
        encodings = []
        self.encoding_array = numpy.array([])
        header = ['Name', 'label']
        for i in range(len(AA)):
            dict[AA[i]] = i
        for i in range(1, len(PAAC) - 1):
            array = PAAC[i].rstrip().split() if PAAC[i].rstrip() != '' else None
            prop1.append([float(j) for j in array[1:]])
            name.append(array[0])
        for i in AA:
            header.append('Pc1.' + i)
        for j in range(1, lambdaValue + 1):
            for i in name:
                header.append('Pc2.' + i + '.' + str(j))
        encodings.append(header)
        
        prop2 = []
        for i in prop1:
            mean = sum(i) / 20
            den = math.sqrt(sum([(j - mean) ** 2 for j in i]) / 20)
            prop2.append([(j - mean) / den for j in i])
        
        for k in self.fasta_list:
            name = k[0]
            sequence = re.sub('-', '' , k[1])
            label = k[2]
            code = [name, label]
            theta = []
            for n in range(1, lambdaValue + 1):
                for i in range(len(prop2)):
                    theta.append(sum([prop2[i][dict[sequence[j]]] * prop2[i][dict[sequence[j + n]]] for j in range(len(sequence) - n)]) / (len(sequence) - n))
            
            occur = {}
            for i in AA:
                occur[i] = sequence.count(i)
            for i in AA:
                code = code + [occur[i] / (1 + w * sum(theta))]
            for i in theta:
                code = code + [w * i / (1 + w * sum(theta))]
            encodings.append(code)
        
        self.encoding_array = numpy.array(encodings, dtype=str)
        self.row = self.encoding_array.shape[0]
        self.column = self.encoding_array.shape[1]
        del encodings

class ASDC(FASTA):
    def __init__(self, file):
        super(ASDC, self).__init__(file)

    def ASDC(self):
        AA = 'ACDEFGHIKLMNPQRSTVWY'
        encodings = []
        AA_Pairs = []
        self.encoding_array = numpy.array([])
        for i in AA:
            for j in AA:
                AA_Pairs.append(i + j)
        header = ['Name', 'label'] + AA_Pairs
        encodings.append(header)

        for k in self.fasta_list:
            name = k[0]
            sequence = re.sub('-', '' , k[1])
            label = k[2]
            code = [name, label]
            occur = {}
            sum = 0
            for i in AA_Pairs:
                occur[i] = 0
            for i in range(len(sequence) - 1):
                for j in range(i + 1, len(sequence)):
                    occur[sequence[i] + sequence[j]] += 1
                    sum += 1
            for i in AA_Pairs:
                code.append(occur[i] / sum)
            encodings.append(code)

        self.encoding_array = numpy.array(encodings, dtype=str)
        self.row = self.encoding_array.shape[0]
        self.column = self.encoding_array.shape[1]
        del encodings

class CTDC(FASTA):
    def __init__(self, file):
        super(CTDC, self).__init__(file)
    
    def Count_C(self, s, t):
        sum = 0
        for i in s:
            sum = sum + t.count(i)
        return sum
    
    def CTDC(self):
        group1 = {
            'hydrophobicity_PRAM900101': 'RKEDQN',
            'hydrophobicity_ARGP820101': 'QSTNGDE',
            'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
            'hydrophobicity_PONP930101': 'KPDESNQT',
            'hydrophobicity_CASG920101': 'KDEQPSRNTG',
            'hydrophobicity_ENGD860101': 'RDKENQHYP',
            'hydrophobicity_FASG890101': 'KERSQD',
            'normwaalsvolume': 'GASTPDC',
            'polarity': 'LIFWCMVY',
            'polarizability': 'GASDT',
            'charge': 'KR',
            'secondarystruct': 'EALMQKRH',
            'solventaccess': 'ALFCGIVW'
        }
        group2 = {
            'hydrophobicity_PRAM900101': 'GASTPHY',
            'hydrophobicity_ARGP820101': 'RAHCKMV',
            'hydrophobicity_ZIMJ680101': 'HMCKV',
            'hydrophobicity_PONP930101': 'GRHA',
            'hydrophobicity_CASG920101': 'AHYMLV',
            'hydrophobicity_ENGD860101': 'SGTAW',
            'hydrophobicity_FASG890101': 'NTPG',
            'normwaalsvolume': 'NVEQIL',
            'polarity': 'PATGS',
            'polarizability': 'CPNVEQIL',
            'charge': 'ANCQGHILMFPSTWYV',
            'secondarystruct': 'VIYCWFT',
            'solventaccess': 'RKQEND'
        }
        group3 = {
            'hydrophobicity_PRAM900101': 'CLVIMFW',
            'hydrophobicity_ARGP820101': 'LYPFIW',
            'hydrophobicity_ZIMJ680101': 'LPFYI',
            'hydrophobicity_PONP930101': 'YMFWLCVI',
            'hydrophobicity_CASG920101': 'FIWC',
            'hydrophobicity_ENGD860101': 'CVLIMF',
            'hydrophobicity_FASG890101': 'AYHWVMFLIC',
            'normwaalsvolume': 'MHKFRYW',
            'polarity': 'HQRKNED',
            'polarizability': 'KMHFRYW',
            'charge': 'DE',
            'secondarystruct': 'GNPSD',
            'solventaccess': 'MSPTHY'
        }
        groups = [group1, group2, group3]
        property = (
            'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
            'hydrophobicity_PONP930101',
            'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
            'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')
        self.encoding_array = numpy.array([])
        encodings = []
        header = ['Name', 'label']
        for i in property:
            for j in range(1, len(groups) + 1):
                header.append(i + '.G' + str(j))
        encodings.append(header)

        for k in self.fasta_list:
            name = k[0]
            sequence = re.sub('-', '' , k[1])
            label = k[2]
            code = [name, label]
            for i in property:
                c1 = self.Count_C(group1[i], sequence) / len(sequence)
                c2 = self.Count_C(group2[i], sequence) / len(sequence)
                c3 = 1 - c1 - c2
                code = code + [c1, c2, c3]
            encodings.append(code)
        
        self.encoding_array = numpy.array(encodings, dtype=str)
        self.row = self.encoding_array.shape[0]
        self.column = self.encoding_array.shape[1]
        del encodings

class CTDT(FASTA):
    def __init__(self, file):
        super(CTDT, self).__init__(file)
    
    def CTDT(self):
        group1 = {
            'hydrophobicity_PRAM900101': 'RKEDQN',
            'hydrophobicity_ARGP820101': 'QSTNGDE',
            'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
            'hydrophobicity_PONP930101': 'KPDESNQT',
            'hydrophobicity_CASG920101': 'KDEQPSRNTG',
            'hydrophobicity_ENGD860101': 'RDKENQHYP',
            'hydrophobicity_FASG890101': 'KERSQD',
            'normwaalsvolume': 'GASTPDC',
            'polarity': 'LIFWCMVY',
            'polarizability': 'GASDT',
            'charge': 'KR',
            'secondarystruct': 'EALMQKRH',
            'solventaccess': 'ALFCGIVW'
        }
        group2 = {
            'hydrophobicity_PRAM900101': 'GASTPHY',
            'hydrophobicity_ARGP820101': 'RAHCKMV',
            'hydrophobicity_ZIMJ680101': 'HMCKV',
            'hydrophobicity_PONP930101': 'GRHA',
            'hydrophobicity_CASG920101': 'AHYMLV',
            'hydrophobicity_ENGD860101': 'SGTAW',
            'hydrophobicity_FASG890101': 'NTPG',
            'normwaalsvolume': 'NVEQIL',
            'polarity': 'PATGS',
            'polarizability': 'CPNVEQIL',
            'charge': 'ANCQGHILMFPSTWYV',
            'secondarystruct': 'VIYCWFT',
            'solventaccess': 'RKQEND'
        }
        group3 = {
            'hydrophobicity_PRAM900101': 'CLVIMFW',
            'hydrophobicity_ARGP820101': 'LYPFIW',
            'hydrophobicity_ZIMJ680101': 'LPFYI',
            'hydrophobicity_PONP930101': 'YMFWLCVI',
            'hydrophobicity_CASG920101': 'FIWC',
            'hydrophobicity_ENGD860101': 'CVLIMF',
            'hydrophobicity_FASG890101': 'AYHWVMFLIC',
            'normwaalsvolume': 'MHKFRYW',
            'polarity': 'HQRKNED',
            'polarizability': 'KMHFRYW',
            'charge': 'DE',
            'secondarystruct': 'GNPSD',
            'solventaccess': 'MSPTHY'
        }
        groups = [group1, group2, group3]
        property = (
            'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
            'hydrophobicity_PONP930101',
            'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
            'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')
        self.encoding_array = numpy.array([])
        encodings = []
        header = ['Name', 'label']
        for i in property:
            for j in ('Tr1221', 'Tr1331', 'Tr2332'):
                header.append(i + '.' + j)
        encodings.append(header)

        for k in self.fasta_list:
            name = k[0]
            sequence = re.sub('-', '' , k[1])
            label = k[2]
            code = [name, label]
            for i in range(len(sequence) - 1):
                AA_Pair = [sequence[i:i + 2]]
            for i in property:
                c1221 = c1331 = c2332 = 0
                for AA in AA_Pair:
                    if((AA[0] in group1[i]) and (AA[1] in group2[i])):
                        c1221 += 1
                        continue
                    if((AA[0] in group2[i]) and (AA[1] in group1[i])):
                        c1221 += 1
                        continue
                    if((AA[0] in group1[i]) and (AA[1] in group3[i])):
                        c1331 += 1
                        continue
                    if((AA[0] in group3[i]) and (AA[1] in group1[i])):
                        c1331 += 1
                        continue
                    if((AA[0] in group2[i]) and (AA[1] in group3[i])):
                        c2332 += 1
                        continue
                    if((AA[0] in group3[i]) and (AA[1] in group2[i])):
                        c2332 += 1
                        continue
                code = code + [c1221 / len(AA_Pair), c1331 / len(AA_Pair), c2332 / len(AA_Pair)]
            encodings.append(code)

        self.encoding_array = numpy.array(encodings, dtype=str)
        self.row = self.encoding_array.shape[0]
        self.column = self.encoding_array.shape[1]
        del encodings

class CTDD(FASTA):
    def __init__(self, file):
        super(CTDD, self).__init__(file)
    
    def Count_D(self, group, s):
        sum = 0
        for i in s:
            if i in group:
                sum += 1
        node = [1, math.floor(0.25 * sum), math.floor(0.5 * sum), math.floor(0.75 * sum), sum]
        node = [i if i >= 1 else 1 for i in node]
        
        code = []
        for n in node:
            sum = 0
            for i in range(len(s)):
                if s[i] in group:
                    sum += 1
                    if sum == n:
                        code.append((i + 1) / len(s) * 100)
                        break
            if sum == 0:
                code.append(0)
        return code 

    def CTDD(self):
        group1 = {
            'hydrophobicity_PRAM900101': 'RKEDQN',
            'hydrophobicity_ARGP820101': 'QSTNGDE',
            'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
            'hydrophobicity_PONP930101': 'KPDESNQT',
            'hydrophobicity_CASG920101': 'KDEQPSRNTG',
            'hydrophobicity_ENGD860101': 'RDKENQHYP',
            'hydrophobicity_FASG890101': 'KERSQD',
            'normwaalsvolume': 'GASTPDC',
            'polarity': 'LIFWCMVY',
            'polarizability': 'GASDT',
            'charge': 'KR',
            'secondarystruct': 'EALMQKRH',
            'solventaccess': 'ALFCGIVW'
        }
        group2 = {
            'hydrophobicity_PRAM900101': 'GASTPHY',
            'hydrophobicity_ARGP820101': 'RAHCKMV',
            'hydrophobicity_ZIMJ680101': 'HMCKV',
            'hydrophobicity_PONP930101': 'GRHA',
            'hydrophobicity_CASG920101': 'AHYMLV',
            'hydrophobicity_ENGD860101': 'SGTAW',
            'hydrophobicity_FASG890101': 'NTPG',
            'normwaalsvolume': 'NVEQIL',
            'polarity': 'PATGS',
            'polarizability': 'CPNVEQIL',
            'charge': 'ANCQGHILMFPSTWYV',
            'secondarystruct': 'VIYCWFT',
            'solventaccess': 'RKQEND'
        }
        group3 = {
            'hydrophobicity_PRAM900101': 'CLVIMFW',
            'hydrophobicity_ARGP820101': 'LYPFIW',
            'hydrophobicity_ZIMJ680101': 'LPFYI',
            'hydrophobicity_PONP930101': 'YMFWLCVI',
            'hydrophobicity_CASG920101': 'FIWC',
            'hydrophobicity_ENGD860101': 'CVLIMF',
            'hydrophobicity_FASG890101': 'AYHWVMFLIC',
            'normwaalsvolume': 'MHKFRYW',
            'polarity': 'HQRKNED',
            'polarizability': 'KMHFRYW',
            'charge': 'DE',
            'secondarystruct': 'GNPSD',
            'solventaccess': 'MSPTHY'
        }
        groups = [group1, group2, group3]
        property = (
            'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
            'hydrophobicity_PONP930101',
            'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
            'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')
        self.encoding_array = numpy.array([])
        encodings = []
        header = ['Name', 'label']
        for i in property:
            for j in ('1', '2', '3'):
                for k in ['0', '25', '50', '75', '100']:
                    header.append(i + '.' + j + '.residue' + k)
        encodings.append(header)

        for k in self.fasta_list:
            name = k[0]
            sequence = re.sub('-', '' , k[1])
            label = k[2]
            code = [name, label]
            for i in property:
                code = code + self.Count_D(group1[i], sequence) + self.Count_D(group2[i], sequence) + self.Count_D(group3[i], sequence)
            encodings.append(code)

        self.encoding_array = numpy.array(encodings, dtype=str)
        self.row = self.encoding_array.shape[0]
        self.column = self.encoding_array.shape[1]
        del encodings

class QSO(FASTA):
    def __init__(self, file):
        super(QSO, self).__init__(file)
    
    def QSO(self):
        nlag = 2
        w = 0.05
        dataFile_S = os.path.join(os.getcwd(), 'data', 'Schneider-Wrede.txt')
        dataFile_G = os.path.join(os.getcwd(), 'data', 'Grantham.txt')
        AA1 = 'ACDEFGHIKLMNPQRSTVWY'
        AA2 = 'ARNDCQEGHILKMFPSTWYV'
        self.encoding_array = numpy.array([])
        dict_S = {}
        dict_G = {}
        for i in range(len(AA1)):
            dict_S[AA1[i]] = i
            dict_G[AA2[i]] = i            
        
        with open(dataFile_S) as f:
            text = f.readlines()[1:]
        distance_S = []
        for i in text:
            array = i.rstrip().split()[1:] if i.rstrip() != '' else None
            distance_S.append(array)
        distance_S = numpy.array([float(distance_S[i][j]) for i in range(len(distance_S)) for j in range(len(distance_S[i]))]).reshape((20, 20))
        
        with open(dataFile_G) as f:
            text = f.readlines()[1:]
        distance_G = []
        for i in text:
            array = i.rstrip().split()[1:] if i.rstrip() != '' else None
            distance_G.append(array)
        distance_G = numpy.array([float(distance_G[i][j]) for i in range(len(distance_G)) for j in range(len(distance_G[i]))]).reshape((20, 20))
        
        encodings = []
        header = ['Name', 'label']
        for a in AA1:
            header.append('Schneider-Wrede.Xr.' + a)
        for a in AA2:
            header.append('Grantham.Xr.' + a)
        for n in range(1, nlag + 1):
            header.append('Schneider-Wrede.Xd.' + str(n))
        for n in range(1, nlag + 1):
            header.append('Grantham.Xd.' + str(n))
        encodings.append(header)

        for k in self.fasta_list:
            name = k[0]
            sequence = re.sub('-', '' , k[1])
            label = k[2]
            code = [name, label]
            array_S = []
            array_G = []
            for n in range(1, nlag + 1):
                array_S.append(sum([distance_S[dict_S[sequence[i]]][dict_S[sequence[i + n]]] ** 2 for i in range(len(sequence) - n)]))
                array_G.append(sum([distance_G[dict_G[sequence[i]]][dict_G[sequence[i + n]]] ** 2 for i in range(len(sequence) - n)]))
            
            occur = {}
            for a in AA2:
                occur[a] = sequence.count(a)
            for a in AA2:
                code.append(occur[a] / (1 + w * sum(array_S)))
            for a in AA2:
                code.append(occur[a] / (1 + w * sum(array_G)))
            for i in array_S:
                code.append((w * i) / (1 + w * sum(array_S)))
            for i in array_G:
                code.append((w * i) / (1 + w * sum(array_G)))
            encodings.append(code)

        self.encoding_array = numpy.array(encodings, dtype=str)
        self.row = self.encoding_array.shape[0]
        self.column = self.encoding_array.shape[1]
        del encodings

def FeatureMerge():
    file_list = [f for f in os.listdir('.') if f.endswith('.csv')]
    f = pandas.read_csv(file_list[0])

    for file_name in file_list[1:]:
        temp = pandas.read_csv(file_name, usecols=lambda col: col not in f.columns[:2])
        f = pandas.concat([f, temp], axis=1)

    f.to_csv('Merge.csv', header=False, index=False)

class Analysis(object):
    def __init__(self):
        self.frame = None
        self.label = None
        self.column = 0
        self.row = 0
        self.result = None
        self.data = None
        self.sample = None

    def data_import(self, file):
        f = pandas.read_csv(file, sep=',', header=None)
        self.frame = f.iloc[:, 1:]
        self.row = self.frame.index.size
        self.column = self.frame.columns.size
        self.frame.index=['Sample_%s'%i for i in range(self.row)]
        self.frame.columns = ['F_%s'%i for i in range(self.column)]
        self.label = numpy.array(f.iloc[:, 0]).astype(int)
        self.sample = numpy.array([True] * self.row)

def select(x):
    path_b = os.path.join('models', 'header.csv')
    with open('MergeIG.csv') as a_file, open(path_b) as b_file:
        a_reader = csv.reader(a_file)
        b_reader = csv.reader(b_file)
    
        a_first_row = next(a_reader)
        b_first_row = next(b_reader)
        intersecting_elements = [element for element in b_first_row if element in a_first_row]
        column_indices = [a_first_row.index(element) for element in intersecting_elements]
        path_c = os.path.join('models', 'feature.csv')
        with open(path_c, 'w', newline='') as c_file:
            writer = csv.writer(c_file)
            for row in a_reader:
                writer.writerow([row[index] for index in column_indices])

def testing_read():
    sequences = input_box.get("1.0", "end")
    new_sequence = sequences.split('\n')
    test_fasta = open(os.path.join(os.getcwd(), "peptide_sequences.txt"), 'w+')
    for i in range(len(new_sequence)):
        if len(new_sequence[i]) == 0:
            continue
        print('>sequence_%d|0|testing'%(i + 1), file=test_fasta)
        print(new_sequence[i], file=test_fasta)
    test_fasta.close()

    output_box.insert("end", "Now we have got those peptide sequences.\n")
    output_box.insert("end", "Please wait for feature extraction ...\n")

def get_merge():
    file_path = os.path.join(os.getcwd(), "peptide_sequences.txt")

    f_AAC = AAC(file_path)
    f_AAC.AAC()
    f_AAC.save('AAC.csv')

    f_CKSAAP = CKSAAP(file_path)
    f_CKSAAP.CKSAAP()
    f_CKSAAP.save('CKSAAP.csv')

    f_DDE = DDE(file_path)
    f_DDE.DDE()
    f_DDE.save('DDE.csv')

    f_APAAC = APAAC(file_path)
    f_APAAC.APAAC()
    f_APAAC.save('APAAC.csv')

    f_ASDC = ASDC(file_path)
    f_ASDC.ASDC()
    f_ASDC.save('ASDC.csv')

    f_CTDC = CTDC(file_path)
    f_CTDC.CTDC()
    f_CTDC.save('CTDC.csv')

    f_CTDT = CTDT(file_path)
    f_CTDT.CTDT()
    f_CTDT.save('CTDT.csv')

    f_CTDD = CTDD(file_path)
    f_CTDD.CTDD()
    f_CTDD.save('CTDD.csv')

    f_QSO = QSO(file_path)
    f_QSO.QSO()
    f_QSO.save('QSO.csv')
    
    FeatureMerge()

    file_paths = [
        './AAC.csv',
        './CKSAAP.csv',
        './DDE.csv',
        './APAAC.csv',
        './ASDC.csv',
        './CTDC.csv',
        './CTDT.csv',
        './CTDD.csv',
        './QSO.csv'
    ]
    for file_path in file_paths:
        path = os.path.join(os.getcwd(), file_path)
        os.remove(path)

def get_selection():
    data = Analysis()
    data.data_import(os.path.join('.', 'Merge.csv'))
    data.frame.insert(0, 'Labels', data.label)
    data.frame.to_csv(os.path.join('.', 'MergeIG.csv'), sep=',', header=True, index=False)
    
    i = 1
    select(i)
    
    os.remove(os.path.join('.', 'Merge.csv'))
    os.remove(os.path.join('.', 'MergeIG.csv'))

def get_result(fasta):
    test_dir = os.path.join(".", "models", "feature.csv")
    fasta.DataLoading(test_dir)
    fasta.ModelLoading()
    fasta.ModelPrediction()
    os.remove(test_dir)
    
    
def calculate():
    start_time = time.time()

    output_box.delete("1.0", "end")
    
    testing_read()
    get_merge()
    get_selection()
    output_box.insert("end", "Feature extraction successful!\n")

    fasta = Prediction()
    get_result(fasta)

    output_box.delete("1.0", "end")
    output_box.insert("end", "Here are the result!\n")
    for i in range(len(fasta.score)):
        if(fasta.score[i][0] > fasta.score[i][1]):
            output_box.insert("end", "The sequence_%d is a non-B3PP!\n"%(i + 1))
        else:
            output_box.insert("end", "The sequence_%d is a B3PP!\n"%(i + 1))
        output_box.insert("end", "    The possibility of being a non-B3PP: %.3f\n"%(fasta.score[i][0]))
        output_box.insert("end", "    The possibility of being a B3PP: %.3f\n"%(fasta.score[i][1]))
    
    os.remove(os.path.join(".", "peptide_sequences.txt"))

    end_time = time.time()
    execution_time = end_time - start_time
    print(execution_time, "s")

root = tk.Tk()
root.title("B3PPs Predictor")

input_label = tk.Label(root, text="Please enter your FASTA sequence：")
input_label.grid(row=0, column=0, sticky=tk.W)
input_box = tk.Text(root, height=10, width=50)
input_box.grid(row=1, column=0)

calculate_button = tk.Button(root, text="Predict", command=calculate)
calculate_button.grid(row=4, column=0)

output_label = tk.Label(root, text="Classification Results：")
output_label.grid(row=2, column=0, sticky=tk.W)
output_box = tk.Text(root, height=10, width=50)
output_box.grid(row=3, column=0)

root.mainloop()
