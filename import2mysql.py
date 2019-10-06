import csv
import pymysql
from tqdm import tqdm

config={
	'host':'localhost',
	'port':3306,
	'db':'gxdc',
	'user':'root',
	'passwd':'123456',
	'charset':'utf8',
	'cursorclass':pymysql.cursors.DictCursor}

data=[]
csvf="D:\\BaiduNetdiskDownload\\new_train_data.csv"

with open(csvf)as f:
	f_csv=csv.reader(f)
	for row in f_csv:
		data.append((row[1],row[2],row[3],row[4],row[6],row[7],row[8],row[9],row[10]))

conn=pymysql.connect(**config)
cursor=conn.cursor()
sql = "insert into dcset values(%s,%s,%s,%s,%s,%s,%s,%s,%s)" # 要插入的数据

for row in data:
	i+=1
	k=cursor.execute(sql,row)
	if i%10000==0:
		conn.commit()
		print(i)

conn.commit()
cursor.close()
conn.close()