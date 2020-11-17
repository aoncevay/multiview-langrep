#wit3_langs = {
wit3_tag2name = {
    'ar': 'Arabic',
    'bg': 'Bulgarian',
    'cs': 'Czech',
    'de': 'German',
    'el': 'Greek',
    'es': 'Spanish',
    'eu': 'Basque',
    'fa': 'Farsi',
    'fr': 'French',
    'hu': 'Hungarian',
    'id': 'Indonesian',
    'it': 'Italian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'nl': 'Dutch',
    'pl': 'Polish',
    'pt': 'Portuguese',
    'ro': 'Romanian',
    'ru': 'Russian',
    'sk': 'Slovak',
    'sl': 'Slovenian',
    'th': 'Thai',
    'tr': 'Turkish',
    'vi': 'Vietnamese',
    'zh': 'Chinese',
    'he': 'Hebrew'
}

wit3_langs_name2id = {
    'Arabic': 'ar',
    'Hebrew': 'he',
    'Bulgarian': 'bg',
    'Czech': 'cs',
    'Polish': 'pl',
    'Russian': 'ru',
    'Slovak': 'sk',
    'Slovenian': 'sl',
    'German': 'de',
    'Dutch': 'nl',
    'Greek': 'el',
    'Spanish': 'es',
    'Portuguese': 'pt',
    'Romanian': 'ro',
    'Basque': 'eu',
    'Farsi': 'fa',
    'French': 'fr',
    'Italian': 'it',
    'Hungarian': 'hu',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Thai': 'th',
    'Turkish': 'tr',
    'Vietnamese': 'vi',
    'Chinese': 'zh',
    'Indonesian': 'id'
}

wit3_map = {
    'ar': 'arb', #'ara',
    'bg': 'bul',
    'cs': 'ces',
    'de': 'deu',
    'el': 'ell',
    'es': 'spa',
    'eu': 'eus',
    'fa': 'pes', #'fas',
    'fr': 'fra',
    'hu': 'hun',
    'id': 'ind',
    'it': 'ita',
    'ja': 'jpn',
    'ko': 'kor',
    'nl': 'nld',
    'pl': 'pol',
    'pt': 'por',
    'ro': 'ron',
    'ru': 'rus',
    'sk': 'slk',
    'sl': 'slv',
    'th': 'tha',
    'tr': 'tur',
    'vi': 'vie',
    'zh': 'cmn', #'zho'
    'he': 'heb'
}

#wit3_iso2tag
wit3_map_inv = {
    #'ara': 'ar',
    'arb': 'ar',
    'bul': 'bg',
    'ces': 'cs',
    'deu': 'de',
    'ell': 'el',
    'spa': 'es',
    'eus': 'eu',
    #'fas': 'fa',
    'pes': 'fa',
    'fra': 'fr',
    'hun': 'hu',
    'ind': 'id',
    'ita': 'it',
    'jpn': 'ja',
    'kor': 'ko',
    'nld': 'nl',
    'pol': 'pl',
    'por': 'pt',
    'ron': 'ro',
    'rus': 'ru',
    'slk': 'sk',
    'slv': 'sl',
    'tha': 'th',
    'tur': 'tr',
    'vie': 'vi',
    'zho': 'zh', #cmn
    'heb': 'he',
    'cmn': 'zh'
}

langs_others = {
    #'ara': 'Arabic', 
    'arb': 'StandardArabic', 
    #'fas': 'Persian', 
    'prs': 'DariPersian', 
    'pes': 'WesternPersian', 
    #'zho': 'Chinese', 
    'cdo': 'MinDongChinese', 
    'cjy': 'JinyuChinese', 
    'cmn': 'MandarinChinese', 
    'cpx': 'PuXianChinese', 
    'czh': 'HuizhouChinese', 
    'czo': 'MinZhongChinese', 
    'gan': 'GanChinese', 
    'hak': 'HakkaChinese', 
    'hsn': 'XiangChinese', 
    'lzh': 'LiteraryChinese', 
    'mnp': 'MinBeiChinese', 
    'nan': 'MinNanChinese', 
    'wuu': 'WuChinese', 
    'yue': 'YueChinese'
}

langs_map = {
    'ara' : ['arb'],
    'fas' : ['prs', 'pes'],
    'zho' : ['cdo', 'cjy', 'cmn', 'cpx', 'czh', 'czo', 'gan', 'hak', 'hsn', 'lzh', 'mnp', 'nan', 'wuu', 'yue']
}

#without "eus" (Basque)
wit3_uriel_ids = ['arb', 'bul', 'ces', 'deu', 'ell', 'spa', 'pes', 'fra', 'hun', 'ind', 'ita', 'jpn', 
                  'kor', 'nld', 'pol', 'por', 'ron', 'rus', 'slk', 'slv', 'tha', 'tur', 'vie', 'cmn', 'heb']
wit3_learn_ids = ['arb', 'bul', 'ces', 'deu', 'ell', 'spa', 'pes', 'fra', 'hun', 'ind', 'ita', 'jpn', 
                  'kor', 'nld', 'pol', 'por', 'ron', 'rus', 'slk', 'slv', 'tha', 'tur', 'vie', 'zho']
wit3_both_ids  = wit3_uriel_ids[:-2]
            #['arb', 'bul', 'ces', 'deu', 'ell', 'spa', 'pes', 'fra', 'hun', 'ind', 'ita', 'jpn',
            # 'kor', 'nld', 'pol', 'por', 'ron', 'rus', 'slk', 'slv', 'tha', 'tur', 'vie']

#emnlp'19 paper from Tan et al. (23 langs)
wit3_emnlp_ids = ['arb', 'bul', 'ces', 'deu', 'ell', 'spa', 'pes', 'fra', 'hun', 'ita', 'jpn', 
                  'nld', 'pol', 'por', 'ron', 'rus', 'slk', 'slv', 'tha', 'tur', 'vie', 'cmn', 'heb']

wit3_25_langs = ['arb', 'bul', 'ces', 'deu', 'ell', 'spa', 'pes', 'fra', 'hun', 'ind', 'ita', 'jpn', 
                 'kor', 'nld', 'pol', 'por', 'ron', 'rus', 'slk', 'slv', 'tha', 'tur', 'vie', 'cmn', 'heb']

#print(" ".join(["<%s>|<%s>" % (wit3_map_inv[l], wit3_map_inv[l]) for l in wit3_emnlp_ids]))