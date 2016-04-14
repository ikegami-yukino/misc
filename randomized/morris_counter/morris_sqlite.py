from __future__ import unicode_literals
import os
import random
import sqlite3


class MorrisSqlite(object):
    def __init__(self, path, radix=2):
        self.path = path
        if not os.path.exists(path):
            self._send_sql('CREATE TABLE word_df (word text unique, df integer)')
        self.radix = radix
        self.max_exponent = 255        # maximum of unsigned char

    def _send_sql(self, sql):
        conn = sqlite3.connect(self.path)
        c = conn.cursor()
        c.execute(sql)
        conn.commit()
        c.close()
        conn.close()

    def _get_exponent(self, word):
        exponent = 0
        try:
            conn = sqlite3.connect(self.path)
            c = conn.cursor()
            for row in c.execute('SELECT df FROM word_df WHERE word="%s"' % word):
                exponent = row[0]
                break
        finally:
            c.close()
            conn.close()
        return exponent

    def delta(self, exponent):
        return self.radix**-exponent

    def incr(self, word):
        exponent = self._get_exponent(word)
        if (exponent < self.max_exponent and random.random() < self.delta(exponent)):
            if exponent == 0:
                exponent += 1
                self._send_sql("INSERT INTO word_df VALUES ('%s', %d)" % (word, exponent))
            else:
                exponent += 1
                self._send_sql("UPDATE word_df SET df = %d WHERE word = '%s';" % (exponent, word))

    def get(self, word):
        exponent = self._get_exponent(word)
        return self.radix**exponent / (self.radix - 1)

    def get_all(self):
        conn = sqlite3.connect(self.path)
        c = conn.cursor()
        for row in c.execute('SELECT * FROM word_df ORDER BY df DESC'):
            print(row)
        c.close()
