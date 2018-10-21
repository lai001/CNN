'''
推荐书籍算法
'''
import pymysql
import numpy as np
import csv
import sys
import json
import chardet

#a:list
def set_user_tag(a):
    bookname=[]
    for i in range(1, len(sys.argv)):
        bookname.append(sys.argv[i])

    for i in a:
        if ''.join(bookname)==i[0]:
            return i[2]

try:
    # 打开数据库连接
    db = pymysql.connect(host="localhost",port=3306,user="root",passwd="",db="py_book",charset="utf8")

    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()

    # 使用 execute()  方法执行 SQL 查询
    cursor.execute("SELECT `book_name`, `average`, `tags` FROM `book` WHERE 1")

    # 使用 fetchone() 方法获取单条数据.
    data = cursor.fetchall()
    # 关闭数据库连接
    db.close()
except:
    pass
book_name=[]#所有书籍的书名
average=[]#所有书籍的评分
book_tags=[]#所有书籍的标签
book=[]#书籍列表
recommend_list=[]#推荐的书籍的列表

try:
    for row in data:
        book_name.append((row[0]))
        average.append(row[1])
        book_tags.append((row[2]))
    for i in range(len(book_name)):
        book.append([book_name[i],average[i],book_tags[i]])  #得到书籍列表
except:
    pass

user_tag=set_user_tag(book)

# # 创建停用词list
# def stopwordslist(filepath):
#     stopwords = [line.strip() for line in open(filepath, 'r').readlines()]
#     return stopwords
# # 去掉不相关的标签
# def seg_sentence(sentence):
#     stopwords = stopwordslist("去掉不相关的书籍标签.txt")  # 这里加载停用词的路径
#     outstr = ''
#     for word in sentence:
#         if word not in stopwords:
#             if word != '\t':
#                 outstr += word
#                 outstr += " "
#     return outstr

#欧式距离
def euclidSimilar(inA,inB):
    return 1.0/(1.0+np.linalg.norm(inA-inB))
#余弦相似度
def cosSimilar(inA,inB):
    inA=np.mat(inA)
    inB=np.mat(inB)
    numerator=float(inA*inB.T)
    denominator=np.linalg.norm(inA)*np.linalg.norm(inB)
    return numerator / denominator

# # 读取CSV文件获得数据
# filename = "book_csv.csv"
# with open(filename, mode="r") as f:
#     reader = csv.reader(f)
#     # print(list(reader))
#     book = list(reader)

# user_tag=book[0][2]#得到用户的书籍的标签

for i in range(len(book)):
    # print(user_tag.split())
    item_tag=book[i][2]
    # print(item_tag.split())
    all_tag=user_tag+item_tag
    # print(all_tag.split())

    user_list=user_tag.split()
    item_list=item_tag.split()
    all_list=all_tag.split()
    # print("没有去掉不相关标签的时候---",user_list)#没有去掉不相关标签的时候
    # user_list=seg_sentence(user_list)
    # item_list = seg_sentence(item_list)
    # all_list = seg_sentence(all_list)
    # user_list = user_list.split()
    # item_list = item_list.split()
    # all_list = all_list.split()
    # print("去掉不相关的标签-----------",user_list)#去掉不相关的标签

    uv=[]
    for val in all_list :
        if val in user_list:
            uv.append(1)
        else:uv.append(0)
    User_Profiles=np.array(uv)#创建用户向量
    # print("User_Profiles: ",User_Profiles)

    iv=[]
    for val in all_list :
        if val in item_list:
            iv.append(1)
        else:iv.append(0)
    Item_Profiles=np.array(iv)#创建物品的向量
    # print("Item_Profiles: ",Item_Profiles)
    # print("用户的书为:\t",book[0][0])
    # print(i,"比较的书名：",book[i][0],"\t余弦相似度为：",cosSimilar(User_Profiles,Item_Profiles),"--------0.5代表相似度最低\n")

    recommend_list.append([cosSimilar(User_Profiles,Item_Profiles)*10,book[i][0],book[i][1]])
recommend_list.sort(reverse=True)
# print("向您推荐以下书籍:(未加入评分权重)")
# for i in range(5):
#     print(recommend_list[i])

weight=[]
for i in recommend_list:
    weight.append([i[0]+float(i[2])/2,i[1]])
# print("向您推荐以下书籍:(加入评分权重)")
weight.sort(reverse=True)

for i in range(3):
    print(weight[i+1][1])



