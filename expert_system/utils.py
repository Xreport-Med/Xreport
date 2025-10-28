# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import re

#TODO:处理“第4、5掌骨”
#TODO:部位和病理变化严格对应；
#思路为先检索病理变化，然后查找同一句（逗号分隔）中病理变化之前紧挨着的部位

#第二、三
def deal_chinese_num(Str):
    numdict = {'一':'1','二':'2','三':'3','四':'4','五':'5'}
    for chinese_num,num in numdict.items():
        if chinese_num in Str:
            Str = Str.replace(chinese_num,num)
    return Str

#第3-5,S4-5,C5-7
def deal_hyphen(Str):
    # print('deal_hyphen',Str)
    start_ends = re.findall('[1]?[0-9]\-[1]?[0-9]',Str)
    # print('start_ends',start_ends)
    for start_end in start_ends:
        # start,end = eval(start_end[0]),eval(start_end[-1])+1
        start,end = eval(start_end.split('-')[0]),eval(start_end.split('-')[-1])+1
        # print('start,end',start,end)
        new = '、'.join(map(str,list(range(start,end))))
        # print('new',new)
        Str = Str.replace(start_end,new)
    # print('Str',Str)
    return Str

#左手第2/3/4/5掌骨体部骨折
def deal_slash(Str):
    if len(re.findall('[1]?[0-9]/[1]?[0-9]',Str))!=0:
        Str = Str.replace('/','、')
    return Str

def deal_stop_sign_num(sentence):
    #input:"第3、4、5掌骨骨折"
    #output:"第3掌骨骨折，第4掌骨骨折，第5掌骨骨折"
    pattern_list = ['[1]?[0-9]、[1]?[0-9]','[1]?[0-9]、[1]?[0-9]、[1]?[0-9]','[1]?[0-9]、[1]?[0-9]、[1]?[0-9]、[1]?[0-9]']
    nums = []
    for pattern in pattern_list:
        # pattern = re.compile(pattern)
        temp = re.findall(pattern,sentence)
        # print(temp)
        if len(temp)!=0:
            nums = temp[0].split('、')
        else:
            break
    # print('nums',nums)
    single_list = []
    ori_str = '、'.join(nums)
    # print('ori_str',ori_str)
    for num in nums:
        single_list.append(sentence.replace(ori_str,num))
    # print('single_list',single_list)
    return '，'.join(single_list)
    

def deal_stop_sign_double(sentence,pattern_list = ['胫、腓','尺、桡','胫腓','尺桡']):
    #input:"胫、腓骨骨折"
    #output:"胫骨骨折，腓骨骨折"
    nums = []
    for pattern in pattern_list:
        # pattern = re.compile(pattern)
        temp = re.findall(pattern,sentence)
        # print(temp)
        if len(temp)!=0:
            nums = [pattern[0],pattern[1]]
            ori_str = pattern
    # print('nums',nums)
    single_list = []
    for num in nums:
        single_list.append(sentence.replace(ori_str,num))
    # print('single_list',single_list)
    return '，'.join(single_list)

def deal_scaphoid(sentence):

    scapoid_site = {
        '手':['手','腕'],
        '足':['踝']
    }
    for site,keywords in scapoid_site.items():
        for keyword in keywords:
            if len(re.findall(keyword,sentence))!=0:
                #标记左右双
                direction = re.findall('[左右双]',sentence)
                direction = '' if len(direction)==0 else direction[-1]
                # if len(direction)==0:
                # print('sentence',sentence)
                # print(direction+site+'舟骨')
                return direction+site+'舟骨'
    return ''

def find_str_list(Str,target_list):
    for target in target_list:
        if len(re.findall(target,Str))!=0:
            return True
    return False

class BaseReport():
    def __init__(self,report_path,label_path,result_path,diagnosis_col,description_col):
        self.info = pd.read_excel(report_path)
        self.lesion_xlsx = pd.read_excel(label_path,sheet_name='病理变化',index_col='名称')
        self.bone_xlsx = pd.read_excel(label_path,sheet_name='部位',index_col='名称')
        self.other_disease_xlsx = pd.read_excel(label_path,sheet_name='其他疾病',index_col='名称')
        self.articular_xlsx = pd.read_excel(label_path,sheet_name='近关节骨折',index_col='名称')
        self.result_path = result_path
        self.diagnosis_col = diagnosis_col
        self.description_col = description_col
    def String_Split(self,string, separators='\n，'):
        # 将传进来的列表放入统一的数组中
        result_split = [string]
        # 使用for循环每次处理一种分割符
        for sep in separators:
            # 使用map函数迭代result_split字符串数组
            string_temp = []    # 用于暂时存储分割中间列表变量
            list(
                  map(
                     lambda sub_string: string_temp.extend(sub_string.split(sep)),
                     result_split
                     )
                 )
            # 经过上面的指令，中间变量string_temp就被分割后的内容填充了，
            # 将分割后的内容赋值给result_split，并作为函数返回值即可
            result_split = string_temp
        return result_split
    #搜索过滤器 如"突出"只用搜索"椎间盘" 返回bool决定循环是否continue,True会使得循环continue
    def search_filter(self,lesion,lesion_xlsx):
        # print('lesion_xlsx.at[lesion,分类]',lesion_xlsx.at[lesion,'分类'])
        # print('lesion_xlsx[0]',lesion_xlsx[0])
        lesionClass = lesion_xlsx.at[lesion,'分类'].split(',')
        for i,lesion_site in enumerate(lesionClass):
            if lesion_site=='全部':
                return self.bone_xlsx
            # elif lesion_site=='肺':
                # return self.bone_xlsx[self.bone_xlsx['分类'].str.contains('肺')]
            lesion_site_xlsx = self.bone_xlsx[self.bone_xlsx['分类'].str.contains(lesion_site)]
            if i==0:
                site_xlsx = lesion_site_xlsx
            else:
                site_xlsx = pd.concat([site_xlsx,lesion_site_xlsx],axis=0)
        return site_xlsx
    def search_filter_articular(self,lesion,lesion_xlsx):
        # print('lesion_xlsx.at[lesion,分类]',lesion_xlsx.at[lesion,'分类'])
        # print('lesion_xlsx[0]',lesion_xlsx[0])
        lesionClass = lesion_xlsx.at[lesion,'分类'].split(',')
        for i,lesion_site in enumerate(lesionClass):
            if lesion_site=='全部':
                return self.articular_xlsx
            lesion_site_xlsx = self.articular_xlsx[self.articular_xlsx['分类'].str.contains(lesion_site)]
            if i==0:
                site_xlsx = lesion_site_xlsx
            else:
                site_xlsx = pd.concat([site_xlsx,lesion_site_xlsx],axis=0)
        return site_xlsx
    def other_disease_retrieval(self,sentence,sentence_list):
        other_diseases = []
        diagnosis_list = []
        for other_disease_name,row in self.other_disease_xlsx.iterrows():
            # print('row',row)
            # print('other_disease_name',other_disease_name)
            # print('匹配字',row['匹配字'])
            other_disease_list = row['匹配字'].split(',')
            for other_disease in other_disease_list:
                index = sentence.find(other_disease)
                if index!=-1:
                    other_diseases.append([other_disease_name,index])
        other_diseases = sorted(other_diseases,key=(lambda x:x[1]),reverse=False)
        #检索同一句（逗号分隔）中病理变化之前紧挨着的部位
        #找到本病理变化和上一病理变化之间的字符串
        for i,other_disease in enumerate(other_diseases):
            other_disease,index = other_disease
            if self.other_disease_xlsx.at[other_disease,'分类']=='无':
                diagnosis_list.append(other_disease)
            else:
                #标记左右双
                direction = re.findall('[左右双]','，'.join(sentence_list))
                direction = '' if len(direction)==0 else direction[0]
                if self.other_disease_xlsx.at[other_disease,'分类']=='左右':
                    diagnosis_list.append(direction+other_disease)
                else:
                    siteStr = sentence[:index]
                    #搜索过滤器
                    site_xlsx = self.search_filter(other_disease,self.other_disease_xlsx)
                    for site_name,row in site_xlsx.iterrows():
                        # print(siteStr)
                        # print('site_name',site_name)
                        # print('other_disease',other_disease)
                        bone_list = row['匹配字'].split(',')
                        for bone in bone_list:
                            index = siteStr.find(bone)
                            if index!=-1:
                                diagnosis_list.append(direction+site_name+other_disease)
        return diagnosis_list
    def sentence_retrieval(self,sentence,sentence_list):
        #检索病理变化
        lesions = []
        diagnosis_list = []
        # print('lesion_xlsx',lesion_xlsx)
        # print('bone_xlsx',bone_xlsx)
        for lesion_name,row in self.lesion_xlsx.iterrows():
            # print('row',row)
            # print('匹配字',row['匹配字'])
            lesion_list = row['匹配字'].split(',')
            for lesion in lesion_list:
                index = sentence.find(lesion)
                if index!=-1:
                    lesions.append([lesion_name,index])
        lesions = sorted(lesions,key=(lambda x:x[1]),reverse=False)
        #检索同一句（逗号分隔）中病理变化之前紧挨着的部位
        #找到本病理变化和上一病理变化之间的字符串
        for i,onelesion in enumerate(lesions):
            lesion,index = onelesion
            # if i==0:
            #     siteStr = sentence[:index]
            #     # if '掌骨' in sentence:
            #         # print('siteStr',siteStr)
            # else:
            #     siteStr = sentence[lesions[i-1][1]:index]
            #     #处理"，撕脱骨折"/"，游离体" 向前多取一句
            #     if '撕脱' in siteStr or '游离体' in siteStr:
            #         siteStr = sentence_list[sentence_list.index(sentence)-1]+sentence[lesions[i-1][1]:index]
                # if '掌骨' in sentence:
                    # print('siteStr',siteStr)
            siteStr = sentence[:index]
            if '撕脱' in siteStr or '游离体' in siteStr:
                siteStr = sentence_list[sentence_list.index(sentence)-1]+sentence[lesions[i-1][1]:index]
            #标记左右双
            direction = re.findall('[左右双]','，'.join(sentence_list))
            direction = '' if len(direction)==0 else direction[0]
            #处理"第二、三"
            if len(re.findall('第[一二三四五]',siteStr))!=0:
                siteStr = deal_chinese_num(siteStr)
            #处理"第3-5近节指骨"
            if '-' in siteStr:
                siteStr = deal_hyphen(siteStr)
            if '、' in siteStr:
                #处理“第4、5掌骨”
                if len(re.findall('[1-5]、[1-5]',siteStr))!=0:
                    siteStr = deal_stop_sign_num(siteStr)
                    # if '掌骨' in siteStr:
                        # print('processed siteStr',siteStr)
                #处理“胫、腓骨”
                if len(re.findall('胫、腓',siteStr))!=0 or len(re.findall('尺、桡',siteStr))!=0:
                    siteStr = deal_stop_sign_double(siteStr)
            #处理“胫腓骨”
            if len(re.findall('胫腓',siteStr))!=0 or len(re.findall('尺桡',siteStr))!=0:
                siteStr = deal_stop_sign_double(siteStr)
            #处理"右第5 近节指骨骨折"
            siteStr = siteStr.replace(' ','')
            #处理"左踝关节对位可，关节间隙未见明显增宽、狭窄，舟骨上缘见条状高密度影。"
            if '舟' in siteStr and '手' not in siteStr and '足' not in siteStr:
                siteStr = deal_scaphoid('，'.join(sentence_list))
            #搜索过滤器
            site_xlsx = self.search_filter(lesion,self.lesion_xlsx)
            for site_name,row in site_xlsx.iterrows():
                # print('j',j,'row',row)
                #搜索过滤器
                # print(siteStr)
                # print('site_name',site_name)
                # print('lesion',lesion)
                bone_list = row['匹配字'].split(',')
                for bone in bone_list:
                    index = siteStr.find(bone)
                    if index!=-1:
                        if site_name[0] not in ['左','右']:
                            diagnosis_list.append(direction+site_name+lesion)
                        else:
                            diagnosis_list.append(site_name+lesion)
        #检索其他疾病
        other_diagnosis_list = self.other_disease_retrieval(sentence,sentence_list)
        diagnosis_list.extend(other_diagnosis_list)
        diagnosis_list = list(set(diagnosis_list))
        if '足舟骨骨折' in diagnosis_list or '足舟骨撕脱骨折' in diagnosis_list:
            print('diagnosis_list',diagnosis_list)
            print('sentence',sentence)
            print('sentence_list',sentence_list)
        # if '掌骨' in sentence:
            # print(sentence)
            # print('diagnosis_list',diagnosis_list)
            # print('sentence.find',sentence.find('[1-5]、[1-5]'))
        return diagnosis_list

    def sentence_retrieval_articular(self,sentence,sentence_list):
        #检索病理变化
        lesions = []
        diagnosis_list = []
        bone_list = []
        # print('lesion_xlsx',lesion_xlsx)
        # print('bone_xlsx',bone_xlsx)
        target_lesions = ['骨折','撕脱骨折']
        for target_lesion in target_lesions:
            lesion_list = self.lesion_xlsx.at[target_lesion,'匹配字'].split(',')
            for lesion in lesion_list:
                index = sentence.find(lesion)
                if index!=-1:
                    lesions.append([target_lesion,index])
        lesions = sorted(lesions,key=(lambda x:x[1]),reverse=False)
        #检索同一句（逗号分隔）中病理变化之前紧挨着的部位
        #找到本病理变化和上一病理变化之间的字符串
        for i,onelesion in enumerate(lesions):
            lesion,index = onelesion
            # if i==0:
            #     siteStr = sentence[:index]
            #     # if '掌骨' in sentence:
            #         # print('siteStr',siteStr)
            # else:
            #     siteStr = sentence[lesions[i-1][1]:index]
            #     #处理"，撕脱骨折"/"，游离体" 向前多取一句
            #     if '撕脱' in siteStr or '游离体' in siteStr:
            #         siteStr = sentence_list[sentence_list.index(sentence)-1]+sentence[lesions[i-1][1]:index]
            # if '掌骨' in sentence:
            # print('siteStr',siteStr)
            siteStr = sentence[:index]
            if '撕脱' in siteStr or '游离体' in siteStr:
                siteStr = sentence_list[sentence_list.index(sentence)-1]+sentence[lesions[i-1][1]:index]
            #标记左右双
            direction = re.findall('[左右双]','，'.join(sentence_list))
            direction = '' if len(direction)==0 else direction[0]
            #处理"第二、三"
            if len(re.findall('第[一二三四五]',siteStr))!=0:
                siteStr = deal_chinese_num(siteStr)
            #处理"第3-5近节指骨"
            if '-' in siteStr:
                siteStr = deal_hyphen(siteStr)
            if '、' in siteStr:
                #处理“第4、5掌骨”
                if len(re.findall('[1-5]、[1-5]',siteStr))!=0:
                    siteStr = deal_stop_sign_num(siteStr)
                    # if '掌骨' in siteStr:
                    # print('processed siteStr',siteStr)
                #处理“胫、腓骨”
                if len(re.findall('胫、腓',siteStr))!=0 or len(re.findall('尺、桡',siteStr))!=0:
                    siteStr = deal_stop_sign_double(siteStr)
            #处理“胫腓骨”
            if len(re.findall('胫腓',siteStr))!=0 or len(re.findall('尺桡',siteStr))!=0:
                siteStr = deal_stop_sign_double(siteStr)
            #处理"右第5 近节指骨骨折"
            siteStr = siteStr.replace(' ','')
            #处理"左踝关节对位可，关节间隙未见明显增宽、狭窄，舟骨上缘见条状高密度影。"
            if '舟' in siteStr and '手' not in siteStr and '足' not in siteStr:
                siteStr = deal_scaphoid('，'.join(sentence_list))
            #搜索过滤器
            site_xlsx = self.search_filter_articular(lesion,self.lesion_xlsx)
            for site_name,row in site_xlsx.iterrows():
                # print('j',j,'row',row)
                #搜索过滤器
                # print(siteStr)
                # print('site_name',site_name)
                # print('lesion',lesion)
                bone_str_list = row['匹配字'].split(',')
                for bone in bone_str_list:
                    index = siteStr.find(bone)
                    if index!=-1:
                        if site_name[0] not in ['左','右']:
                            diagnosis_list.append(direction+site_name+lesion)
                            bone_list.append(site_name)
                        else:
                            diagnosis_list.append(site_name+lesion)
                            bone_list.append(site_name)
        # #检索其他疾病
        # other_diagnosis_list = self.other_disease_retrieval(sentence,sentence_list)
        # diagnosis_list.extend(other_diagnosis_list)
        # diagnosis_list = list(set(diagnosis_list))
        # if '足舟骨骨折' in diagnosis_list or '足舟骨撕脱骨折' in diagnosis_list:
        #     print('diagnosis_list',diagnosis_list)
        #     print('sentence',sentence)
        #     print('sentence_list',sentence_list)
        # if '掌骨' in sentence:
        # print(sentence)
        # if len(diagnosis_list)!=0:
        #     print('diagnosis_list',diagnosis_list)
        #     print('bone_list',bone_list)
        # print('sentence.find',sentence.find('[1-5]、[1-5]'))
        return diagnosis_list,bone_list

    def findDiagnosisByExcel(self,paragraph):
        sentence_list = self.String_Split(paragraph)
        diagnosis_list = []
        for sentence in sentence_list:
            sentence_result = self.sentence_retrieval(sentence,sentence_list)
            diagnosis_list.extend(sentence_result)
        if len(diagnosis_list)==0:
            diagnosis_list.append('正常')
        return '，'.join(diagnosis_list)
    def findArticularDiagnosisByExcel(self,paragraph):
        sentence_list = self.String_Split(paragraph)
        diagnosis_list = []
        joint_list = []
        for sentence in sentence_list:
            sentence_result,bone_list = self.sentence_retrieval_articular(sentence,sentence_list)
            diagnosis_list.extend(sentence_result)
            for bone in bone_list:
                joint_list.append(self.articular_xlsx.at[bone,'关节'])
        diagnosis_list = list(set(diagnosis_list))
        joint_list = list(set(joint_list))
        # if len(diagnosis_list)!=0:
        #     print('diagnosis_list',diagnosis_list)
        #     print('joint_list',joint_list)
        return '，'.join(diagnosis_list),'，'.join(joint_list)
    def cal_class_num(self,xlsx,col_name='class'):
        result = []
        result_without_direction = []
        for index,row in xlsx.iterrows():
            class_list = row[col_name].split('，')
            class_list_without_direction = [re.sub('[左右双]','',oneclass) for oneclass in class_list]
            result.extend(class_list)
            result_without_direction.extend(class_list_without_direction)
        print('class num',len(set(result)))
        print('class num without direction',len(set(result_without_direction)))
    def search_site(self,siteStr):
        siteDict = set()
        for site_name,row in self.bone_xlsx.iterrows():
            # print('j',j,'row',row)
            #搜索过滤器
            # print(siteStr)
            # print('site_name',site_name)
            # print('lesion',lesion)
            bone_list = row['匹配字'].split(',')
            for bone in bone_list:
                if len(re.findall(bone,str(siteStr)))!=0:
                    siteDict.add(site_name)
        return siteDict
    def match_site_report(self,siteStr,reportStr):
        siteDict = self.search_site(siteStr)
        reportDict = self.search_site(reportStr)
        if len(siteDict & reportDict)!=0:
            return True
        return False
    def classify(self):
        blank = 0
        self.info['class'] = ''
        self.info['bones'] = ''
        self.info['site_match_diagnosis'] = ''
        self.info['site_match_report'] = ''
        for index,row in self.info.iterrows():
            # print('str(row[self.diagnosis_col]),self.label_path',str(row[self.diagnosis_col]),self.label_path)
            self.info.at[index,'site_match_diagnosis'] = self.match_site_report(row['检查部位'],row['诊断结论'])
            self.info.at[index,'site_match_diagnosis'] = self.match_site_report(row['检查部位'],row['征象描述'])

            classes = self.findDiagnosisByExcel(str(row[self.diagnosis_col]))
            # print(bones)
            if len(classes)!=0:
                self.info.at[index,'class'] = classes
            else: #对'征象描述'再进行一轮检索
                classes = self.findDiagnosisByExcel(str(row[self.description_col]))
                if len(classes)!=0:
                    self.info.at[index,'class'] = classes
                elif isinstance(row[self.diagnosis_col],str):
                    self.info.at[index,'class'] = row[self.diagnosis_col]
                    blank += 1
                    self.info.at[index,'blank'] = 'True'
                elif isinstance(row[self.description_col],str):
                    self.info.at[index,'class'] = row[self.description_col]
                    blank += 1
                    self.info.at[index,'blank'] = 'True'

            if index%100==0:
                print('process',index,'/',len(self.info))
        print('blank',blank/len(self.info)*100,'%')

        empty_cols = list(self.info.filter(regex = r'Unnamed: ', axis=1))
        # print('empty_cols',empty_cols)
        self.info.drop(empty_cols,axis=1,inplace=True)
        self.info.to_excel(self.result_path,encoding='utf-8-sig')
        self.cal_class_num(self.info)
    def classify_articular(self):
        labeled_info = pd.read_excel(self.result_path)
        # labeled_info.drop_duplicates('seriesinstanceUID',inplace = True)
        labeled_info['articular'] = ''
        labeled_info['joint'] = ''
        for index,row in labeled_info.iterrows():
            # print('str(row[self.diagnosis_col]),self.label_path',str(row[self.diagnosis_col]),self.label_path)
            classes,joints = self.findArticularDiagnosisByExcel(str(row[self.diagnosis_col]))
            # print(bones)
            if len(classes)!=0:
                labeled_info.at[index,'articular'] = classes
                labeled_info.at[index,'joint'] = joints
            else: #对'征象描述'再进行一轮检索
                classes,joints = self.findArticularDiagnosisByExcel(str(row[self.description_col]))
                if len(classes)!=0:
                    labeled_info.at[index,'articular'] = classes
                    labeled_info.at[index,'joint'] = joints


            if index%100==0:
                print('process',index,'/',len(labeled_info))

        empty_cols = list(labeled_info.filter(regex = r'Unnamed: ', axis=1))
        # print('empty_cols',empty_cols)
        labeled_info.drop(empty_cols,axis=1,inplace=True)
        labeled_info.to_excel(self.result_path.replace('.xlsx','_近关节骨折.xlsx'),encoding='utf-8-sig')
        self.cal_class_num(labeled_info,col_name='articular')


class BYSY(BaseReport):
    def __init__(self,report_path,label_path,result_path,diagnosis_col,description_col):
        super(BaseReport, self).__init__()
        self.info = pd.read_excel(report_path)
        self.lesion_xlsx = pd.read_excel(label_path,sheet_name='病理变化',index_col='名称')
        self.bone_xlsx = pd.read_excel(label_path,sheet_name='部位',index_col='名称')
        self.other_disease_xlsx = pd.read_excel(label_path,sheet_name='其他疾病',index_col='名称')
        self.articular_xlsx = pd.read_excel(label_path,sheet_name='近关节骨折',index_col='名称')
        self.result_path = result_path
        self.diagnosis_col = diagnosis_col
        self.description_col = description_col
class LZSZWKYY(BaseReport):
    def __init__(self,report_path,label_path,result_path,diagnosis_col,description_col):
        super(BaseReport, self).__init__()
        self.info = pd.read_excel(report_path)
        # self.info = self.info[self.info['blank']==True]
        self.lesion_xlsx = pd.read_excel(label_path,sheet_name='病理变化',index_col='名称')
        self.bone_xlsx = pd.read_excel(label_path,sheet_name='部位',index_col='名称')
        self.other_disease_xlsx = pd.read_excel(label_path,sheet_name='其他疾病',index_col='名称')
        self.articular_xlsx = pd.read_excel(label_path,sheet_name='近关节骨折',index_col='名称')
        self.result_path = result_path
        self.diagnosis_col = diagnosis_col
        self.description_col = description_col
    def deal_non_sense(self,string):
        #处理无意义符号:<p></p><br>
        nonsenses = ['<p>','</p>','<br>','&nbsp;']
        for nonsense in nonsenses:
            string = string.replace(nonsense,'')
        return string
    def String_Split(self,string, separators=['\n','。','？','；']):
        string = self.deal_non_sense(string)
        # 将传进来的列表放入统一的数组中
        result_split = [string]
        # 使用for循环每次处理一种分割符
        for sep in separators:
            # 使用map函数迭代result_split字符串数组
            string_temp = []    # 用于暂时存储分割中间列表变量
            list(
                map(
                    lambda sub_string: string_temp.extend(sub_string.split(sep)),
                    result_split
                )
            )
            # 经过上面的指令，中间变量string_temp就被分割后的内容填充了，
            # 将分割后的内容赋值给result_split，并作为函数返回值即可
            result_split = string_temp

        return result_split
    def sentence_retrieval(self,sentence,sentence_list):
        #检索病理变化
        lesions = []
        diagnosis_list = []
        # print('lesion_xlsx',lesion_xlsx)
        # print('bone_xlsx',bone_xlsx)
        for lesion_name,row in self.lesion_xlsx.iterrows():
            # print('row',row)
            # print('匹配字',row['匹配字'])
            lesion_list = row['匹配字'].split(',')
            for lesion in lesion_list:
                index = sentence.find(lesion)
                if index!=-1:
                    lesions.append([lesion_name,index])
        lesions = sorted(lesions,key=(lambda x:x[1]),reverse=False)
        #检索同一句（逗号分隔）中病理变化之前紧挨着的部位
        #找到本病理变化和上一病理变化之间的字符串
        for i,onelesion in enumerate(lesions):
            lesion,index = onelesion
            if i==0:
                siteStr = sentence[:index]
                # if '掌骨' in sentence:
                    # print('siteStr',siteStr)
            else:
                siteStr = sentence[lesions[i-1][1]:index]
                #处理"，撕脱骨折"/"，游离体" 向前多取一句
                if '撕脱' in siteStr or '游离体' in siteStr:
                    siteStr = sentence_list[sentence_list.index(sentence)-1]+sentence[lesions[i-1][1]:index]
                # if '掌骨' in sentence:
                    # print('siteStr',siteStr)
            #标记左右双
            direction = re.findall('[左右双]','，'.join(sentence_list))
            direction = '' if len(direction)==0 else direction[0]
            #处理"第二、三"
            if len(re.findall('第[一二三四五]',siteStr))!=0:
                siteStr = deal_chinese_num(siteStr)

            #处理"第3-5近节指骨"
            if '-' in siteStr and ('指' in siteStr or '趾' in siteStr or '掌' in siteStr or '跖' in siteStr):
                # print('siteStr',siteStr)
                siteStr = deal_hyphen(siteStr)
                # print('deal_hyphen',siteStr)
            #处理"左手第2/3/4/5掌骨体部骨折"
            if '/' in siteStr:
                # print('siteStr',siteStr)
                siteStr = deal_slash(siteStr)
                # print('deal_slash',siteStr)
            if '、' in siteStr:
                #处理“第4、5掌骨”
                if len(re.findall('[1-5]、[1-5]',siteStr))!=0:
                    # print('siteStr',siteStr)
                    siteStr = deal_stop_sign_num(siteStr)
                    # print('deal_stop_sign_num',siteStr)
                    # if '掌骨' in siteStr:
                        # print('processed siteStr',siteStr)
                #处理“胫、腓骨”
                if len(re.findall('胫、腓',siteStr))!=0 or len(re.findall('尺、桡',siteStr))!=0:
                    siteStr = deal_stop_sign_double(siteStr,pattern_list = ['胫、腓','尺、桡'])
                if len(re.findall('距、舟',siteStr))!=0 or len(re.findall('中、环',siteStr))!=0:
                    siteStr = deal_stop_sign_double(siteStr,pattern_list = ['距、舟','中、环'])
            #处理“胫腓骨”
            if (len(re.findall('胫腓',siteStr))!=0 or len(re.findall('尺桡',siteStr))!=0) and len(re.findall('尺桡关节',siteStr))==0:
                siteStr = deal_stop_sign_double(siteStr,pattern_list = ['胫腓','尺桡'])
            #处理"左手示指中远节"
            if len(re.findall('中远节',siteStr))!=0 or len(re.findall('中环指',siteStr))!=0:
                siteStr = deal_stop_sign_double(siteStr,pattern_list = ['中远','中环'])
            #处理"右第5 近节指骨骨折"
            siteStr = siteStr.replace(' ','')
            #处理"左踝关节对位可，关节间隙未见明显增宽、狭窄，舟骨上缘见条状高密度影。"
            if '舟' in siteStr and '手' not in siteStr and '足' not in siteStr:
                siteStr = deal_scaphoid('，'.join(sentence_list))
            #搜索过滤器
            site_xlsx = self.search_filter(lesion,self.lesion_xlsx)
            for site_name,row in site_xlsx.iterrows():
                # print('j',j,'row',row)
                bone_list = row['匹配字'].split(',')
                for bone in bone_list:
                    index = siteStr.find(bone)
                    if index!=-1:
                        if site_name[0] not in ['左','右'] and not find_str_list(site_name,['颈椎','胸椎','腰椎','骨盆','胸部','颅内','软组织','颅脑','头皮']):
                            diagnosis_list.append(direction+site_name+lesion)
                        else:
                            diagnosis_list.append(site_name+lesion)
        #检索其他疾病
        other_diagnosis_list = self.other_disease_retrieval(sentence,sentence_list)
        # if len(other_diagnosis_list)!=0:
            # print('other diagnosis list',other_diagnosis_list)
            # print('sentence',sentence)
            # print('sentence_list',sentence_list)
        diagnosis_list.extend(other_diagnosis_list)
        diagnosis_list = list(set(diagnosis_list))
        # if '足舟骨骨折' in diagnosis_list or '足舟骨撕脱骨折' in diagnosis_list:
        #     print('diagnosis_list',diagnosis_list)
        #     print('sentence',sentence)
        #     print('sentence_list',sentence_list)
        # if '掌骨' in sentence:
            # print(sentence)
            # print('diagnosis_list',diagnosis_list)
            # print('sentence.find',sentence.find('[1-5]、[1-5]'))
        return diagnosis_list

class ZJYY(BaseReport):
    def __init__(self,report_path,label_path,result_path,diagnosis_col,description_col):
        super(BaseReport, self).__init__()
        try:
            self.info = pd.read_csv(report_path,encoding='gbk')#[257:258][:200]#[126000:126300]#
        except:
            self.info = pd.read_csv(report_path)#[257:258]
        # self.info = self.info[self.info['blank']==True]
        self.lesion_xlsx = pd.read_excel(label_path,sheet_name='病理变化',index_col='名称')
        self.bone_xlsx = pd.read_excel(label_path,sheet_name='部位',index_col='名称')
        self.other_disease_xlsx = pd.read_excel(label_path,sheet_name='其他疾病',index_col='名称')
        self.articular_xlsx = pd.read_excel(label_path,sheet_name='近关节骨折',index_col='名称')
        self.class_xlsx = pd.read_excel(label_path,sheet_name='大类',index_col='大类')
        self.result_path = result_path
        self.diagnosis_col = diagnosis_col
        self.description_col = description_col
    def deal_non_sense(self,string):
        #处理无意义符号:<p></p><br>
        nonsenses = ['<p>','</p>','<br>','&nbsp;']
        for nonsense in nonsenses:
            string = string.replace(nonsense,'')
        return string
    def String_Split(self,string, separators=['\n','。','？','；']):
        string = self.deal_non_sense(string)
        # 将传进来的列表放入统一的数组中
        result_split = [string]
        # 使用for循环每次处理一种分割符
        for sep in separators:
            # 使用map函数迭代result_split字符串数组
            string_temp = []    # 用于暂时存储分割中间列表变量
            list(
                map(
                    lambda sub_string: string_temp.extend(sub_string.split(sep)),
                    result_split
                )
            )
            # 经过上面的指令，中间变量string_temp就被分割后的内容填充了，
            # 将分割后的内容赋值给result_split，并作为函数返回值即可
            result_split = string_temp

        return result_split
    def other_disease_retrieval(self,sentence,sentence_list):
        other_diseases = []
        diagnosis_list = []
        class_list = []
        for other_disease_name,row in self.other_disease_xlsx.iterrows():
            # print('row',row)
            # print('other_disease_name',other_disease_name)
            # print('匹配字',row['匹配字'])
            other_disease_list = row['匹配字'].split(',')
            for other_disease in other_disease_list:
                index = sentence.find(other_disease)
                if index!=-1 and '未见' not in sentence and '无' not in sentence:
                    other_diseases.append([other_disease_name,index])
        other_diseases = sorted(other_diseases,key=(lambda x:x[1]),reverse=False)
        # print('other_diseases',other_diseases)
        #检索同一句（逗号分隔）中病理变化之前紧挨着的部位
        #找到本病理变化和上一病理变化之间的字符串
        for i,other_disease in enumerate(other_diseases):
            other_disease,index = other_disease
            if self.other_disease_xlsx.at[other_disease,'分类']=='无':
                diagnosis_list.append(other_disease)
                class_list.append(self.other_disease_xlsx.at[other_disease,'大类'])
            else:
                #标记左右双
                direction = re.findall('[左右双]','，'.join(sentence_list))
                direction = '' if len(direction)==0 else direction[0]
                if self.other_disease_xlsx.at[other_disease,'分类']=='左右':
                    diagnosis_list.append(direction+other_disease)
                    class_list.append(self.other_disease_xlsx.at[other_disease,'大类'])
                else:
                    siteStr = sentence[:index]
                    #搜索过滤器
                    site_xlsx = self.search_filter(other_disease,self.other_disease_xlsx)
                    # print('other_diseases siteStr',siteStr)
                    for site_name,row in site_xlsx.iterrows():
                        # print(siteStr)
                        # print('site_name',site_name)
                        # print('other_disease',other_disease)
                        bone_list = row['匹配字'].split(',')
                        for bone in bone_list:
                            index = siteStr.find(bone)
                            if index!=-1:
                                diagnosis_list.append(direction+site_name+other_disease)
                                class_list.append(self.other_disease_xlsx.at[other_disease,'大类'])
        return diagnosis_list,class_list
    def sentence_retrieval(self,sentence,sentence_list):
        #检索病理变化
        lesions = []
        diagnosis_list = []
        class_list = []
        
        # tests = ['颈椎退行性改变伴C5/6椎间盘病变',
                 # '右肺下叶小片小片感染']
        # print('lesion_xlsx',lesion_xlsx)
        # print('bone_xlsx',bone_xlsx)
        for lesion_name,row in self.lesion_xlsx.iterrows():
            # print('row',row)
            # print('匹配字',row['匹配字'])
            lesion_list = row['匹配字'].split(',')
            for lesion in lesion_list:
                index = sentence.find(lesion)
                if index!=-1:
                    lesions.append([lesion_name,index])
        lesions = sorted(lesions,key=(lambda x:x[1]),reverse=False)
        # print('lesions',lesions)
        # if sentence in tests or '右肺下叶小片小片感染' in sentence:
            # print('lesions',lesions)
        #检索同一句（逗号分隔）中病理变化之前紧挨着的部位
        #找到本病理变化和上一病理变化之间的字符串
        for i,onelesion in enumerate(lesions):
            lesion,index = onelesion
            if lesion in ['椎间盘病变']:
                siteStr = sentence[:index]
                direction = ''
            else:
                if i==0:
                    siteStr = sentence[:index]
                    # if '掌骨' in sentence:
                    # print('siteStr',siteStr)
                #右尺骨鹰嘴粉碎性骨折内固定术后。左桡骨远端骨折石膏外固定术后。
                elif lesion in ['内固定','外固定','术后']:
                    siteStr = sentence[:lesions[i-1][1]]
                else:
                    siteStr = sentence[lesions[i-1][1]:index]
                    #处理"，撕脱骨折"/"，游离体" 向前多取一句
                    if '撕脱' in siteStr or '游离体' in siteStr:
                        siteStr = sentence_list[sentence_list.index(sentence)-1]+sentence[lesions[i-1][1]:index]
                    # if '掌骨' in sentence:
                    # print('siteStr',siteStr)
                #标记左右双
                # direction = re.findall('[左右双]','，'.join(sentence_list))
                direction = re.findall('[左右双]',sentence)
                direction = '' if len(direction)==0 else direction[0]
                #处理"第二、三"
                if len(re.findall('第[一二三四五]',siteStr))!=0:
                    siteStr = deal_chinese_num(siteStr)

                #处理"第3-5近节指骨"
                if '-' in siteStr and ('指' in siteStr or '趾' in siteStr or '掌' in siteStr or '跖' in siteStr or '肋' in siteStr):
                    # print('siteStr',siteStr)
                    siteStr = deal_hyphen(siteStr)
                    # print('deal_hyphen',siteStr)
                #处理"左手第2/3/4/5掌骨体部骨折"
                if '/' in siteStr:
                    # print('siteStr',siteStr)
                    siteStr = deal_slash(siteStr)
                    # print('deal_slash',siteStr)
                if '、' in siteStr:
                    #处理“第4、5掌骨”
                    if len(re.findall('[1]?[0-9]、[1]?[0-9]',siteStr))!=0:
                        # print('siteStr',siteStr)
                        siteStr = deal_stop_sign_num(siteStr)
                        # print('deal_stop_sign_num',siteStr)
                        # if '掌骨' in siteStr:
                        # print('processed siteStr',siteStr)
                    # #处理“胫、腓骨”
                    # if len(re.findall('胫、腓',siteStr))!=0 or len(re.findall('尺、桡',siteStr))!=0:
                        # siteStr = deal_stop_sign_double(siteStr,pattern_list = ['胫、腓','尺、桡'])
                    if len(re.findall('距、舟',siteStr))!=0 or len(re.findall('中、环',siteStr))!=0:
                        siteStr = deal_stop_sign_double(siteStr,pattern_list = ['距、舟','中、环'])
                    
                # #处理“胫腓骨”
                # if (len(re.findall('胫腓',siteStr))!=0 or len(re.findall('尺桡',siteStr))!=0) and len(re.findall('尺桡关节',siteStr))==0:
                    # siteStr = deal_stop_sign_double(siteStr,pattern_list = ['胫腓','尺桡'])
                #处理"左手示指中远节"
                if len(re.findall('中远节',siteStr))!=0 or len(re.findall('中环指',siteStr))!=0:
                    siteStr = deal_stop_sign_double(siteStr,pattern_list = ['中远','中环'])
                #处理"右第5 近节指骨骨折"
                siteStr = siteStr.replace(' ','')
                #处理"左踝关节对位可，关节间隙未见明显增宽、狭窄，舟骨上缘见条状高密度影。"
                if '舟' in siteStr and '手' not in siteStr and '足' not in siteStr:
                    siteStr = deal_scaphoid('，'.join(sentence_list))
            # if sentence in tests or '右肺下叶小片小片感染' in sentence:
                # print('siteStr',siteStr)
            # print('siteStr',siteStr)
            #搜索过滤器
            site_xlsx = self.search_filter(lesion,self.lesion_xlsx)
            for site_name,row in site_xlsx.iterrows():
                # print('j',j,'row',row)
                bone_list = row['匹配字'].split(',')
                # print('site_name',site_name,'bone_list',bone_list)
                for bone in bone_list:
                    index = siteStr.find(bone)
                    if index!=-1 and lesion!='正常' and '未见' not in siteStr and '无' not in siteStr:
                        # print('site_name',site_name)
                        # print('self.lesion_xlsx.at[lesion,大类]',self.lesion_xlsx.at[lesion,'大类'])
                        class_list.append(self.lesion_xlsx.at[lesion,'大类'])
                        # if sentence in tests or '右肺下叶小片小片感染' in sentence:
                            # print('site_name',site_name,'lesion',lesion)
                        if site_name[0] not in ['左','右'] and not find_str_list(site_name,['颈椎','胸椎','腰椎','骨盆','胸部','颅内','软组织','颅脑','头皮']):
                            diagnosis_list.append(direction+site_name+lesion)
                        else:
                            diagnosis_list.append(site_name+lesion)
        # if sentence in tests or '右肺下叶小片小片感染' in sentence:
            # print('diagnosis_list',diagnosis_list)
        #检索其他疾病
        other_diagnosis_list,other_class_list = self.other_disease_retrieval(sentence,sentence_list)
        # if len(other_diagnosis_list)!=0:
        # print('other diagnosis list',other_diagnosis_list)
        # print('sentence',sentence)
        # print('sentence_list',sentence_list)
        diagnosis_list.extend(other_diagnosis_list)
        diagnosis_list = list(set(diagnosis_list))
        class_list.extend(other_class_list)
        class_list = list(set(class_list))
        
        # if '足舟骨骨折' in diagnosis_list or '足舟骨撕脱骨折' in diagnosis_list:
        #     print('diagnosis_list',diagnosis_list)
        #     print('sentence',sentence)
        #     print('sentence_list',sentence_list)
        # if '掌骨' in sentence:
        # print(sentence)
        # print('diagnosis_list',diagnosis_list)
        # print('class_list',class_list)
        # print('sentence.find',sentence.find('[1-5]、[1-5]'))
        return diagnosis_list,class_list
    def findDiagnosisByExcel(self,paragraph):
        sentence_list = self.String_Split(paragraph)
        diagnosis_list = []
        class_list = []
        for sentence in sentence_list:
            sentence_result,class_result = self.sentence_retrieval(sentence,sentence_list)
            diagnosis_list.extend(sentence_result)
            class_list.extend(class_result)
        if len(diagnosis_list)==0:
            diagnosis_list.append('正常')
            class_list.append('正常')
        #class_list转class_array
        # class_num_list = []
        # for oneclass in class_list:
            # index = self.class_xlsx.at[oneclass,'编号']
            # class_num_list.append(index)
        return '，'.join(diagnosis_list),class_list#,class_num_list
    def classify(self):
        blank = 0
        # self.info['diagnosis'] = ''
        self.info['bones'] = ''
        self.info['site_match_diagnosis'] = ''
        self.info['site_match_report'] = ''
        # tests = ['颈椎退行性改变伴C5/6椎间盘病变。',
                 # '右肺下叶小片小片感染，建议复查。']
        # for class_name,num in self.class_xlsx.iterrows():
            # self.info[class_name] = 0
        for index,row in self.info.iterrows():
            # print('影像所见',row['影像所见'])
            # print('意见建议',row['意见建议'])
            # print('str(row[self.diagnosis_col]),self.label_path',str(row[self.diagnosis_col]),self.label_path)
            # try:
            # if '椎间盘' in row['REPORTSCONCLUSION'] or '椎间盘' in row['REPORTSEVIDENCES']:
            if '椎间盘' in row[self.diagnosis_col] or '椎间盘' in row[self.description_col]:
                self.info.at[index,'site_match_diagnosis'] = self.match_site_report(row['STUDIESEXAMINEALIAS'],self.diagnosis_col)
                self.info.at[index,'site_match_diagnosis'] = self.match_site_report(row['STUDIESEXAMINEALIAS'],self.description_col)

            diagnosises,class_list = self.findDiagnosisByExcel(str(row[self.diagnosis_col]))
            # if row['REPORTSCONCLUSION'] in tests or '右肺下叶小片小片感染' in row['REPORTSCONCLUSION']:
                # print('diagnosises',diagnosises,'class_list',class_list)
            # print(bones)
            if len(diagnosises)!=0:
                self.info.at[index,'diagnosis'] = diagnosises
                for oneclass in class_list:
                    self.info.at[index,oneclass] = 1
            else: #对'征象描述'再进行一轮检索
                diagnosises,class_list = self.findDiagnosisByExcel(str(row[self.description_col]))
                # print('diagnosises',diagnosises,'class_list',class_list)
                if len(diagnosises)!=0:
                    self.info.at[index,'diagnosis'] = diagnosises
                    for oneclass in class_list:
                        self.info.at[index,oneclass] = 1
                elif isinstance(row[self.diagnosis_col],str):
                    self.info.at[index,'diagnosis'] = row[self.diagnosis_col]
                    blank += 1
                    self.info.at[index,'blank'] = 'True'
                elif isinstance(row[self.description_col],str):
                    self.info.at[index,'diagnosis'] = row[self.description_col]
                    blank += 1
                    self.info.at[index,'blank'] = 'True'
            # except:
                # continue

            if index%100==0:
                print('process',index,'/',len(self.info))
        print('blank',blank/len(self.info)*100,'%')

        empty_cols = list(self.info.filter(regex = r'Unnamed: ', axis=1))
        # print('empty_cols',empty_cols)
        self.info.drop(empty_cols,axis=1,inplace=True)
        # self.info.to_excel(self.result_path,encoding='utf-8-sig')
        self.info.to_csv(self.result_path,encoding='utf-8-sig')
        self.cal_class_num(self.info,col_name='diagnosis')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 先命令行启动服务
    # bert-serving-start -model_dir /home/zhaokeyang/code/bert/chinese_L-12_H-768_A-12/ -num_worker=1
    #
    # main()

    # BYSY = BYSY(report_path='data/北医三院/报告.xlsx',
    #              label_path = 'data/北医三院/标签.xlsx',
    #              result_path='data/北医三院/分类结果.xlsx',
    #              diagnosis_col='诊断结论',
    #              description_col='征象描述')
    # BYSY.classify()
    # BYSY.classify_articular()
    # LZSZWKYY = LZSZWKYY(report_path='data/兰州手足外科医院/检查级报告.xlsx',#'data/兰州手足外科医院/分类结果.xlsx',
    #                   label_path = 'data/兰州手足外科医院/标签.xlsx',
    #                   result_path='data/兰州手足外科医院/分类结果.xlsx',
    #                   diagnosis_col='意见建议',
    #                   description_col='影像所见')
    # LZSZWKYY.classify()
    # LZSZWKYY.classify_articular()

    # ZJYY = ZJYY(report_path='./data/ZJYY/classify_results_DR2_20240806.csv',#'./data/ZJYY/DR2.csv',#'data/诸暨市人民医院/分类结果.xlsx',
                # label_path = './data/ZJYY/tag.xlsx',
                # result_path='./data/ZJYY/classify_results_DR2_20240808.csv',
                # diagnosis_col='REPORTSCONCLUSION',
                # description_col='REPORTSEVIDENCES')
    # ZJYY.classify()
    
    
    # test = ZJYY(report_path="./data/ZJYY/test_info_en_delete_col_sort_check.csv",
                # label_path = './data/ZJYY/tag.xlsx',
                # result_path='./data/ZJYY/test_info_en_delete_col_sort_check_classify.csv',
                # diagnosis_col='REPORTSCONCLUSION',
                # description_col='REPORTSEVIDENCES')
    # test.classify()
    
    Dr_vs_AI_list = [
                    # "/data_hdd/zhaokeyang/code/sentence_classify/data/ZJYY_Dr_vs_AI/ZDN3.csv",
                    # "/data_hdd/zhaokeyang/code/sentence_classify/data/ZJYY_Dr_vs_AI/ZZT2.csv",
                    # "/data_hdd/zhaokeyang/code/sentence_classify/data/ZJYY_Dr_vs_AI/YHY5.csv",
                    # "/data_hdd/zhaokeyang/code/sentence_classify/data/ZJYY_Dr_vs_AI/GBH4.csv",
                    # "/data_hdd/zhaokeyang/code/sentence_classify/data/ZJYY_Dr_vs_AI/HTH2.csv",
                    # "/data_hdd/zhaokeyang/code/sentence_classify/data/ZJYY_Dr_vs_AI/WSJ7.csv",
                    # "/data_hdd/zhaokeyang/code/sentence_classify/data/ZJYY_Dr_vs_AI/ZZT2_result_label.csv",
                    # "/data_hdd/zhaokeyang/code/sentence_classify/data/ZJYY_Dr_vs_AI/YHY5_result_label.csv",
                    # "/data_hdd/zhaokeyang/code/sentence_classify/data/ZJYY_Dr_vs_AI/GBH4_result_label.csv",
                    
    ]
    # for Dr_vs_AI in Dr_vs_AI_list:
        # Dr_vs_AI_stat = ZJYY(report_path=Dr_vs_AI,#'data/诸暨市人民医院/分类结果.xlsx',
                    # label_path = './data/ZJYY/tag.xlsx',
                    # result_path=Dr_vs_AI.replace('.csv','_result.csv'),
                    # diagnosis_col='意见建议',
                    # description_col='影像所见')
        # Dr_vs_AI_stat.classify()

    ER_list = [
                    # "/data_hdd/zhaokeyang/code/sentence_classify/data/ZJYY_ER/ZZT2_ER.csv",
                    # "/data_hdd/zhaokeyang/code/sentence_classify/data/ZJYY_ER/GBH4_ER.csv",
                    # "/data_hdd/zhaokeyang/code/sentence_classify/data/ZJYY_ER/YHY5_ER.csv"
                    
    ]
    for Dr_vs_AI in ER_list:
        Dr_vs_AI_stat = ZJYY(report_path=Dr_vs_AI,#'data/诸暨市人民医院/分类结果.xlsx',
                    label_path = './data/ZJYY/tag.xlsx',
                    result_path=Dr_vs_AI.replace('.csv','_result.csv'),
                    diagnosis_col='意见建议',
                    description_col='影像所见')
        Dr_vs_AI_stat.classify()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
