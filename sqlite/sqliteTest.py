#! /usr/bin/env python
# -*- coding: UTF-8 -*-

import sqlite3, os

conn = sqlite3.connect("test.db")
cursor = conn.cursor()

cursor.execute(r"SELECT count(*) FROM sqlite_master WHERE type='table' AND name='user'")
if not cursor.rowcount:
    cursor.execute(r'create table user (id varchar(20) primary key, name varchar(20))')


cursor.execute(r"SELECT count(*) FROM user WHERE id=?",('1',))
if not cursor.rowcount:
    cursor.execute(r'insert into user (id,name) values ("1","zzw")')

print cursor.rowcount

cursor.close()
conn.commit()
conn.close()



#query data from db

conn = sqlite3.connect("test.db")
cursor = conn.cursor()
cursor.execute('select * from user where id = ?',('1',))
values = cursor.fetchall()
print values

cursor.close()
conn.close()