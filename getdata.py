import pandas as pd
import pymysql
import xlwt

config = {
    'host': 'localhost',
    'port': 3306,
    'db': 'gxdc',
    'user': 'root',
    'passwd': '123456',
    'charset': 'utf8',
    'cursorclass': pymysql.cursors.DictCursor}


def sql2df(sql):
    conn = pymysql.connect(**config)
    cursor = conn.cursor()
    cursor.execute(sql)
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    df = pd.DataFrame(list(result))
    return df


# 得到活跃网点某时段间活跃单车数
# sql = "select startlat,startlon,count(*) from dcset where starttime between '2017-05-10 00:00:01' and '2017-05-10 01:00:00' group by startlat,startlon;"
#
# 得到网点活跃单车总计数
# sql = 'select startlat,startlon,count(*) from dcset group by startlat,startlon;'
#
# 得到某时刻网点停靠单车计数
# sql = "select startlat,startlon,count(*) from (select bikeid,startlat,startlon,max(starttime) from dcset where starttime<='2017-05-11 00:00:00' group by bikeid) tb1 group by startlat,startlon;"


xlspath = "C:/Users/Administrator/Desktop/test4.xls"  # 导出xls的绝对路径

df = sql2df(sql)
df.to_excel(xlspath)
print("xls exported successful")
