import difflib
import sys
import pymysql

def SQL():
    try:
        # 打开数据库连接
        db = pymysql.connect(host="localhost", port=3306, user="root", passwd="", db="py_book", charset="utf8")

        # 使用 cursor() 方法创建一个游标对象 cursor
        cursor = db.cursor()

        # 使用 execute()  方法执行 SQL 查询
        cursor.execute("SELECT `book_name` FROM `book` WHERE 1")

        # 使用 fetchone() 方法获取单条数据.
        data = cursor.fetchall()
        # 关闭数据库连接
        db.close()
        return data
    except:
        pass

if __name__ == '__main__':
    bookname = []
    for i in range(1, len(sys.argv)):
        bookname.append(sys.argv[i])
    bookname=''.join(bookname)
    list=[]
    score={}
    for row in SQL():
        list.append(row[0])
    for i in list:
        s=difflib.SequenceMatcher(None, bookname, i).quick_ratio()
        score[i]=s
    print(max(score, key=score.get))
