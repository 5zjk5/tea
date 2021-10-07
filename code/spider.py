import requests
import re
import csv
import tqdm
from lxml import etree
from multiprocessing import Pool



def create_tea_csv():
    '''
    创建保存茶信息的文件
    :return:
    '''
    with open('../data/tea.csv','w+',encoding='utf8',newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['标题','评分','品牌','产地','茶类','详情链接','id','荐指数','总评','茶语排行','茶语分类排行',
                     '综合评分排行','热搜排行'])


def create_comment_csv():
    '''
    创建保存评鉴的文件
    :return:
    '''
    with open('../data/comment.csv','w+',encoding='utf8',newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['id','评论人','评论人等级','评分','评论内容','评论时间'])


def get_html(url):
    '''
    请求
    :param url:
    :return:
    '''
    headers = {
        'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.164 Safari/537.36'
    }
    count = 0
    while True:
        response = requests.get(url,headers=headers)
        if response.status_code == 200:
            response.encoding = 'utf8'
            return response.text
        else:
            count += 1
            if count == 3:
                return ''
            continue


def basic_info(html):
    '''
    提取基本信息
    标题，评分，品牌，产地，茶类，链接，id
    :param html:
    :return:
    '''
    infos = []
    html = etree.HTML(html)
    label = html.xpath('//div[@class="search_list tea_comment_list"]/div')
    for l in label:
        # 标题
        title = l.xpath('.//h5')[0]
        title = title.xpath('string(.)')
        # 评分
        score = l.xpath('.//span[@class="score Yahei"]')[0]
        score = score.xpath('string(.)')
        score = score.replace('\t','').replace('\n','').replace('\t','').replace(' ','').replace('\r','')
        # 品牌
        brand = l.xpath('.//div[@class="param"]/p[2]')[0]
        brand = brand.xpath('string(.)')
        brand = brand.replace('品牌：', '')
        # 产地
        place = l.xpath('.//div[@class="param"]/p[3]')[0]
        place = place.xpath('string(.)')
        place = place.replace('产地：', '')
        # 茶类
        kind = l.xpath('.//p[@class="blue"]')[0]
        kind = kind.xpath('string(.)')
        kind = kind.replace('\t','').replace('\n','').replace('\t','').replace(' ','').replace('\r','')
        # 链接
        link = l.xpath('.//div[@class="fr comment_info"]/a/@href')[0]
        # id
        ID =  re.findall('https://chaping\.chayu\.com/tea/(\d+)',link)[0]

        infos.append([title,score,brand,place,kind,link,ID])

    return infos


def rank_info(info_url):
    '''
    提取总评及排行信息
    推荐指数，总评，茶语排行，茶语分类排行，综合评分排行，热搜排行
    :param info_url:每一页的详情链接
    :return:
    '''
    infos = []
    for u in info_url:
        html =  get_html(u)
        html = etree.HTML(html)
        # 推荐指数
        recommend = html.xpath('//i[@class="recommend_full"]')
        if recommend == []:
            recommend = 0
        else:
            recommend = len(recommend)
        # 总评
        general = html.xpath('//div[@class="fl con"]/text()')[0]
        # 茶语排行
        chayu_rank = html.xpath('//div[@class="ranklist_box"]/ul/li[1]/strong/text()')[0]
        # 茶语分类排行
        chayu_kind_rank = html.xpath('//div[@class="ranklist_box"]/ul/li[2]/strong/text()')[0]
        # 综合评分排行
        zonghe_rank = html.xpath('//div[@class="ranklist_box"]/ul/li[3]/strong/text()')[0]
        # 热搜排行
        resou_rank = html.xpath('//div[@class="ranklist_box"]/ul/li[4]/strong/text()')[0]

        infos.append([recommend,general,chayu_rank,chayu_kind_rank,zonghe_rank,resou_rank])

    return infos


def write_tea_csv(basic_infos,rank_infos):
    '''
    保存茶数据
    :param basic_infos:
    :param rank_infos:
    :return:
    '''
    # 数据合并
    for i in range(len(basic_infos)):
        basic_infos[i] = basic_infos[i] + rank_infos[i]

    with open('../data/tea.csv','a+',encoding='utf8',newline='') as f:
        wr = csv.writer(f)
        wr.writerows(basic_infos)


def comment_info(ids):
    '''
    提取评论信息
    评论人	评论人等级	评分	评论内容	评论时间
    :param ids:
    :return:
    '''
    comm = []
    url = 'https://chaping.chayu.com/tea/review?id={}&sort=0&p={}'
    for i in ids:
        u = url.format(i,1)
        page = comment_page(u)
        for p in range(1,eval(page)+1):
            u = url.format(i,str(p))
            html = get_html(u)
            html = etree.HTML(html)
            label = html.xpath('//ul/li')
            for l in label:
                # 评论人
                user = l.xpath('./div[2]/div[@class="nickname"]/a/text()')[0]
                # 评论人等级
                try: # 没等级
                    level = l.xpath('./div[2]/div[@class="member_rank"]/span[2]/text()')[0]
                except:
                    level = ''
                # 评分
                score = l.xpath('./div[2]/div[@class="score"]/span[1]/i/text()')[0]
                # 评论
                comment = l.xpath('./div[2]/div[@class="content"]')[0]
                comment = comment.xpath('string(.)')
                # 评论时间
                pub_time = l.xpath('./div[2]/div[@class="time fltime"]/text()')[0]
                pub_time = pub_time.replace('\r','').replace('\t','').replace('\n','').replace(' ','')
                comm.append([i,user,level,score,comment,pub_time])

    return comm


def comment_page(url):
    '''
    请求评论的第一页，获得有几页评论需要爬取
    :param url:
    :return:
    '''
    html = get_html(url)
    html = etree.HTML(html)
    page = html.xpath('//div[@class="paging_box"]/a/@href')[-1]
    page = re.findall('p=(\d+)',page)[0]
    return page


def write_comment_csv(comment_infos):
    '''
    评论数据保存
    :param comment_infos:
    :return:
    '''
    with open('../data/comment.csv','a+',encoding='utf8',newline='') as f:
        wr = csv.writer(f)
        wr.writerows(comment_infos)


def main(url):
    '''
    主逻辑
    :param url: 页数链接
    :return:
    '''
    html = get_html(url)
    basic_infos = basic_info(html)
    info_url = [i[-2] for i in basic_infos]
    rank_infos = rank_info(info_url)
    write_tea_csv(basic_infos,rank_infos)
    ids = [i[-7] for i in basic_infos]
    comment_infos = comment_info(ids)
    write_comment_csv(comment_infos)


if __name__ == '__main__':
    # 初始化
    create_tea_csv()
    create_comment_csv()
    urls = ['https://chaping.chayu.com/?p={}'.format(str(i)) for i in range(1,444)]

    # 配置好进度条
    pbar = tqdm.tqdm(total=len(urls))
    pbar.set_description(' Task Has Completed ')
    update = lambda *args: pbar.update()

    # 多进程爬取
    p = Pool()
    for url in urls:
        p.apply_async(main, args=(url,), callback=update)
        p.apply_async(main, args=(url,))
    p.close()
    p.join()













