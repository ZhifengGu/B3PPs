#!/usr/bin/env python
# _*_coding:utf-8_*_

import re
import os
import math
import numpy
import pandas
from collections import Counter
import csv


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
            sequence = re.sub('-', '', k[1])
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
            sequence = re.sub('-', '', k[1])
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
                Tv.append(Tm[i] * (1 - Tm[i]) / (len(sequence) - 1))

            for i in range(len(temp)):
                temp[i] = (temp[i] - Tm[i]) / math.sqrt(Tv[i])
            code = code + temp
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
        dataFile = '.\\data\\PAAC.txt'
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
            sequence = re.sub('-', '', k[1])
            label = k[2]
            code = [name, label]
            theta = []
            for n in range(1, lambdaValue + 1):
                for i in range(len(prop2)):
                    theta.append(sum([prop2[i][dict[sequence[j]]] * prop2[i][dict[sequence[j + n]]] for j in
                                      range(len(sequence) - n)]) / (len(sequence) - n))

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
            sequence = re.sub('-', '', k[1])
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
            sequence = re.sub('-', '', k[1])
            label = k[2]
            code = [name, label]
            for i in range(len(sequence) - 1):
                AA_Pair = [sequence[i:i + 2]]
            for i in property:
                c1221 = c1331 = c2332 = 0
                for AA in AA_Pair:
                    if ((AA[0] in group1[i]) and (AA[1] in group2[i])):
                        c1221 += 1
                        continue
                    if ((AA[0] in group2[i]) and (AA[1] in group1[i])):
                        c1221 += 1
                        continue
                    if ((AA[0] in group1[i]) and (AA[1] in group3[i])):
                        c1331 += 1
                        continue
                    if ((AA[0] in group3[i]) and (AA[1] in group1[i])):
                        c1331 += 1
                        continue
                    if ((AA[0] in group2[i]) and (AA[1] in group3[i])):
                        c2332 += 1
                        continue
                    if ((AA[0] in group3[i]) and (AA[1] in group2[i])):
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
            sequence = re.sub('-', '', k[1])
            label = k[2]
            code = [name, label]
            for i in property:
                code = code + self.Count_D(group1[i], sequence) + self.Count_D(group2[i], sequence) + self.Count_D(
                    group3[i], sequence)
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
        dataFile_S = '.\\data\\Schneider-Wrede.txt'
        dataFile_G = '.\\data\\Grantham.txt'
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
        distance_S = numpy.array(
            [float(distance_S[i][j]) for i in range(len(distance_S)) for j in range(len(distance_S[i]))]).reshape(
            (20, 20))

        with open(dataFile_G) as f:
            text = f.readlines()[1:]
        distance_G = []
        for i in text:
            array = i.rstrip().split()[1:] if i.rstrip() != '' else None
            distance_G.append(array)
        distance_G = numpy.array(
            [float(distance_G[i][j]) for i in range(len(distance_G)) for j in range(len(distance_G[i]))]).reshape(
            (20, 20))

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
            sequence = re.sub('-', '', k[1])
            label = k[2]
            code = [name, label]
            array_S = []
            array_G = []
            for n in range(1, nlag + 1):
                array_S.append(sum([distance_S[dict_S[sequence[i]]][dict_S[sequence[i + n]]] ** 2 for i in
                                    range(len(sequence) - n)]))
                array_G.append(sum([distance_G[dict_G[sequence[i]]][dict_G[sequence[i + n]]] ** 2 for i in
                                    range(len(sequence) - n)]))

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


def FeatureExtraction(filename):
    f_AAC = AAC(filename)
    f_AAC.AAC()
    f_AAC.save('AAC.csv')

    f_CKSAAP = CKSAAP(filename)
    f_CKSAAP.CKSAAP()
    f_CKSAAP.save('CKSAAP.csv')

    f_DDE = DDE(filename)
    f_DDE.DDE()
    f_DDE.save('DDE.csv')

    f_APAAC = APAAC(filename)
    f_APAAC.APAAC()
    f_APAAC.save('APAAC.csv')

    f_ASDC = ASDC(filename)
    f_ASDC.ASDC()
    f_ASDC.save('ASDC.csv')

    f_CTDC = CTDC(filename)
    f_CTDC.CTDC()
    f_CTDC.save('CTDC.csv')

    f_CTDT = CTDT(filename)
    f_CTDT.CTDT()
    f_CTDT.save('CTDT.csv')

    f_CTDD = CTDD(filename)
    f_CTDD.CTDD()
    f_CTDD.save('CTDD.csv')

    f_QSO = QSO(filename)
    f_QSO.QSO()
    f_QSO.save('QSO.csv')

    FeatureMerge()

    # os.remove('./AAC.csv')
    # os.remove('./CKSAAP.csv')
    # os.remove('./DDE.csv')
    # os.remove('./APAAC.csv')
    # os.remove('./ASDC.csv')
    # os.remove('./CTDC.csv')
    # os.remove('./CTDT.csv')
    # os.remove('./CTDD.csv')
    # os.remove('./QSO.csv')


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
        self.frame.index = ['Sample_%s' % i for i in range(self.row)]
        self.frame.columns = ['F_%s' % i for i in range(self.column)]
        self.label = numpy.array(f.iloc[:, 0]).astype(int)
        self.sample = numpy.array([True] * self.row)

    def save(self, file):
        data = self.select_feature
        data.to_csv(file, sep=',', header=True, index=False)

class IG(Analysis):
    def __init__(self, file):
        super(IG, self).__init__(file)

    def calProb(self, array):
        prob = {}
        for i in set(array):
            prob[i] = array.count(i) / len(array)
        return prob

    def jointProb(self, array, labels):
        prob = {}
        for i in range(len(labels)):
            prob[str(array[i]) + '-' + str(labels[i])] = prob.get(str(array[i]) + '-' + str(labels[i]), 0) + 1
        for k in prob:
            prob[k] = prob[k] / len(labels)
        return prob

    def IG(self, number):
        binBox = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        labels = self.label.tolist()
        data = numpy.array(self.frame.values)
        features = self.frame.columns
        probY = self.calProb(labels)
        selected = {}
        for i in range(len(features)):
            array = data[:, i]
            newarray = list(pandas.cut(array, len(binBox), labels=binBox))
            probX = self.calProb(newarray)
            probXY = self.jointProb(newarray, labels)
            HX = -1 * sum([p * math.log(p, 2) for p in probX.values()])
            HXY = 0
            for y in probY.keys():
                for x in probX.keys():
                    if str(x) + '-' + str(y) in probXY:
                        HXY = HXY + (probXY[str(x) + '-' + str(y)] * math.log(probXY[str(x) + '-' + str(y)] / probY[y], 2))
            selected[features[i]] = HX + HXY

        res = []
        for k in sorted(selected.items(), key=lambda item:item[1], reverse=True):
            res.append([k[0], '{0:.3f}'.format(selected[k[0]])])
        self.result = pandas.DataFrame(res, columns=['Name', 'Values'])
        self.select_feature = self.frame.loc[:, self.result.Name[:number]]
        self.select_feature.insert(0, 'Labels', self.label)


class Pearsonr(Analysis):
    def __init__(self, file):
        super(Pearsonr, self).__init__(file)

    def mult(self, x, y):
        sum = 0.0
        for i in range(len(x)):
            temp = x[i] * y[i]
            sum += temp
        return sum

    def corrcoef(self, x, y):
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = self.mult(x, y)
        sum_x2 = sum([pow(i, 2) for i in x])
        sum_y2 = sum([pow(i, 2) for i in y])
        num = sum_xy - (float(sum_x) * float(sum_y) / n)
        den = math.sqrt((sum_x2 - float(sum_x ** 2) / n) * (sum_y2 - float(sum_y ** 2) / n))
        if den == 0:
            return 0
        return num / den

    def Pearsonr(self, number):
        labels = self.label.tolist()
        data = numpy.array(self.frame.values)
        features = self.frame.columns
        selected = {}
        for i in range(len(features)):
            array = list(data[:, i])
            selected[features[i]] = self.corrcoef(array, labels)

        res = []
        for k in sorted(selected.items(), key=lambda item: item[1], reverse=True):
            res.append([k[0], '{0:.3f}'.format(selected[k[0]])])
        self.result = pandas.DataFrame(res, columns=['Name', 'Values'])
        self.select_feature = self.frame.loc[:, self.result.Name[:number]]
        self.select_feature.insert(0, 'Labels', self.label)


def select(x):
    path_b = '.\\models\\' + str(x) + '\\header.csv'
    with open('MergeIG.csv') as a_file, open(path_b) as b_file:
        a_reader = csv.reader(a_file)
        b_reader = csv.reader(b_file)

        a_header = next(a_reader)
        b_header = next(b_reader)

        common_cols = [col for col in a_header if col in b_header]

        target_cols = [a_header.index(col) for col in common_cols]

        path_c = '.\\models\\' + str(x) + '\\test.csv'
        with open(path_c, 'w', newline='') as c_file:
            writer = csv.writer(c_file)
            writer.writerow(common_cols)

            for a_row in a_reader:
                keep_row = [a_row[i] for i in target_cols]
                writer.writerow(keep_row)

def FeatureSelection(filename, select_number):
    data = IG()
    data.data_import(filename)
    data.IG(select_number)
    data.save('IG.csv')

if __name__ == '__main__':
    FeatureExtraction('./Training.txt')
    FeatureSelection('.\\Merge.csv', 383)
