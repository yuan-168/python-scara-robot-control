

import sqlite3
import struct

# 数据库文件路径
db_path = '/Users/shiyuanwang/Documents/5:5/my_bag_file2/my_bag_file2_0.db3'

# 连接到SQLite数据库
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 查询数据
cursor.execute("SELECT data FROM messages")
rows = cursor.fetchall()

# 打印结果
print("Retrieved data:")
for row in rows:
    binary_data = row[0]
    if binary_data and len(binary_data) == 24:  # 确保数据长度为24字节
        floats = struct.unpack('6f', binary_data)  # 解码为6个float32
        print(floats)
    else:
        print(f"Data error with length {len(binary_data) if binary_data else 'None'}")

# 关闭连接
conn.close()