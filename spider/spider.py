
#! /usr/bin/env python
# -*- coding: UTF-8 -*-

from urlparse import urlsplit
from os.path import basename
import urllib2
import re
import requests
import os
import json
import urllib



if not os.path.exists('images'):
    os.mkdir("images")

page_size = 50
offset = 0
x=0

id=37787176
url = 'https://www.zhihu.com/question/37787176'
url_content = urllib2.urlopen(url).read()
answers = re.findall('meta itemprop="answerCount" content="(.*?)"', url_content)
limits = int(answers[0])

func = lambda x,y:x if y in x else x + [y]

while offset < limits:

    get_url = 'https://www.zhihu.com/api/v4/questions/37787176/answers?sort_by=default&include=data%5B%2A%5D.is_normal%2Cis_sticky%2Ccollapsed_by%2Csuggest_edit%2Ccomment_count%2Ccan_comment%2Ccontent%2Ceditable_content%2Cvoteup_count%2Creshipment_settings%2Ccomment_permission%2Cmark_infos%2Ccreated_time%2Cupdated_time%2Crelationship.is_authorized%2Cis_author%2Cvoting%2Cis_thanked%2Cis_nothelp%2Cupvoted_followees%3Bdata%5B%2A%5D.author.badge%5B%3F%28type%3Dbest_answerer%29%5D.topics&limit=20&offset='+str(offset)

    header = {
        'User-Agent': "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:34.0) Gecko/20100101 Firefox/34.0",
        'Host': "www.zhihu.com",
        'Referer': "https://www.zhihu.com/question/37787176",
        'Authorization'     :"Bearer Mi4wQUFEQTRCWWJBQUFBRUFKRDRWUUNEQmNBQUFCaEFsVk53dTJIV1FESXIzYlk4cnNRRGpYUEVrVjhIQzAtdXNXUmVB|1499488450|83b917a3f308f6d9275d29d5a35c3a4e41ad2663"
    }
  
    
    req = urllib2.Request(get_url,  headers = header)
    response=urllib2.urlopen(req).read()
    txt=json.loads(response)
    offset += 20


    #parse data
    data = txt["data"]
    for answer in data:
        if answer["author"]["gender"]==0: 
            content = answer["content"]
            urls = re.findall('img .*?data-original="(.*?_r.*?)"', content.encode('utf-8').strip())
            img_urls = reduce(func, [[], ] + urls)#去重
            for img_url in img_urls:
                try:
                    x+= 1
                    img_data = urllib2.urlopen(img_url).read()
                    # file_name = basename(urlsplit(img_url)[2])
                    output = open('images/' +str(x), 'wb')
                    output.write(img_data)
                    output.close()
                except:
                    pass