# coding:utf-8

from impala.dbapi import connect

def get_connection_impala(is_test=False):
    host = '192.168.9.91' if is_test else '192.168.9.91'
    conn = connect(host=host, port=25001, timeout=3600)
    return conn

	
def load_data(sql, is_test=False):
	conn = get_connection_impala(True)
    cursor = conn.cursor()

    #print(sql)

    sec_quote = []
    try:
       # 执行SQL语句
       cursor.execute(sql)
       # 获取所有记录列表
       sec_quote = cursor.fetchall()
    except Exception as e:
       loginfo.logger.error("Error: unable to fecth data: %s ,%s" % (repr(e), sql))
    finally:
        # 关闭数据库连接
        cursor.close()
        conn.close()

    return sec_quote


if __name__ == '__main__':
   pass