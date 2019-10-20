import os
import re
import requests
import fitz

languages = ['BG','ES','CS','DA','DE','ET','EL',
             'EN','FR','HR','IT','LV','LT','HU',
             'MT','NL','PL','PT','RO','SK','SL',
             'FI','SV']

DA_dict = dict({
                'BG': 'Официален вестник на Европейския съюз',
                'CS': 'Úřední věstník Evropské unie',
                'DA': 'Den Europæiske Unions Tidende',
                'DE': 'Amtsblatt der Europäischen Union',
                'EL': 'Επίσημη Εφημερίδα της Ευρωπαϊκής Ένωσης',
                'EN': 'Official Journal of the European Union',
                'ES': 'Diario Oficial de la Unión Europea',
                'ET': 'Euroopa Liidu Teataja',           
                'FI': 'Euroopan unionin virallinen lehti',
                'FR': "Journal officiel de l'Union européenne",
                'HR': 'Službeni list Europske unije',         
                'HU': 'Az Európai Unió Hivatalos Lapja',      
                'IT': "Gazzetta ufficiale dell'Unione europea",
                'LT': 'Europos Sąjungos oficialusis leidinys',
                'LV': 'Eiropas Savienības Oficiālais Vēstnesis',
                'MT': 'Il-Ġurnal Uffiċjali tal-Unjoni Ewropea',
                'NL': 'Publicatieblad van de Europese Unie',  
                'PL': 'Dziennik Urzędowy Unii Europejskiej',  
                'PT': 'Jornal Oficial da União Europeia',     
                'RO': 'Jurnalul Oficial al Uniunii Europene', 
                'SK': 'Úradný vestník Európskej únie',        
                'SL': 'Uradni list Evropske unije',            
                'SV': 'Europeiska unionens officiella tidning'})

art_dict= dict({'BG': ['Член',      'pre'],
                'CS': ['Článek',    'pre'],
                'DA': ['Artikel',   'pre'],
                'DE': ['Artikel',   'pre', 'TITEL|KAPITEL|ABSCHNITT|Unterabschnitt'],
                'EL': ['Άρθρο',     'pre'],
                'EN': ['Article',   'pre', 'TITLE|CHAPTER|SECTION|Subsection'],
                'ES': ['Artículo',  'pre'],
                'ET': ['Artikkel',  'pre'],
                'FI': ['artikla',   'post'],
                'FR': ['Article',   'pre', 'TITRE|CHAPITRE|SECTION|Sous-section'],
                'HR': ['Članak',    'pre'],
                'HU': ['cikk',      'postdot'],
                'IT': ['Articolo',  'pre'],
                'LT': ['straipsnis','post'],
                'LV': ['pants',     'postdot'],
                'MT': ['Artikolu',  'pre'],
                'NL': ['Artikel',   'pre', 'TITEL|HOOFDSTUK|AFDELING|Onderafdeling'],
                'PL': ['Artykuł',   'pre'],
                'PT': ['Artigo',    'pre'],
                'RO': ['Articolul', 'pre'],
                'SK': ['Článok',    'pre'],
                'SL': ['Člen',      'pre'],
                'SV': ['Artikel',   'pre']})

legislation_urls =  [   
                ('Solvency II Delegated Acts',
                ['https://eur-lex.europa.eu/legal-content/'+lang+
                '/TXT/PDF/?uri=OJ:L:2015:012:FULL&from='+lang 
                for lang in languages]),

                ('Solvency II Directive',
                ['https://eur-lex.europa.eu/legal-content/'+lang+
                '/TXT/PDF/?uri=CELEX:32009L0138&from='+lang
                for lang in languages])
                    ]

def download_legislation(path = '/10_central_data/legislation/', languages = languages):
    # Obtaining the Delegated acts
    for urls in legislation_urls:
        print("Retrieving " + urls[0] + " - language ", end='')
        for index in range(len(urls[1])):
            filename = urls[0] + ' - '+languages[index]+ '.pdf'
            if not(os.path.isfile(path + filename)):
                print(languages[index] + " ", end='')
                r = requests.get(urls[1][index])
                f = open(path + filename,'wb')
                f.write(r.content) 
                f.close()
    return None

def read_legislation(path = '/10_central_data/legislation/', name = "Solvency II Delegated Acts", languages = languages):
    legislation = dict()
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]    
    print("Retrieving " + name + " - language ", end='')
    for language in languages:
        print(language + " ", end='')
        if not(name + " - " + language + ".txt" in files):
            # reading pages from pdf file
            legislation_pdf = fitz.open(filename = path + name + ' - ' + language + '.pdf')
            legislation_pages = [page.getText(output = "text") for page in legislation_pdf]
            legislation_pdf.close()
            legislation[language] = ''.join(legislation_pages)
            # saving txt file
            legislation_txt = open(path + name + " - " + language + ".txt", "wb")
            legislation_txt.write(legislation[language].encode('utf-8'))
            legislation_txt.close()
        else:
            # loading txt file
            legislation_txt = open(path + name + ' - ' + language + ".txt", "rb")
            legislation[language] = legislation_txt.read().decode('utf-8')
            legislation_txt.close()
            # deleting page headers
            if name == "Solvency II Delegated Acts":
                header = "\\d+\\.\\d+\\.\\d+\\s+L\\s+\\d+/\\d+\\s+" + DA_dict[language].replace(' ','\\s+') + "\\s+" + language + "\\s+"
                legislation[language] = re.sub(header, '', legislation[language])
            elif name == "Solvency II Directive":
                header1 = '\n' + language[1] + '\n' + language[0]+'\n\d\n\d\n\d\n\d\.\d\n\d\.\d\n\d\n' + DA_dict[language].replace(' ','\s+') +'\n\d+\.\d+\.\d+'
                legislation[language] = re.sub(header1, '', legislation[language])
                header2 = '\n' + language[1] + '\n' + language[0]+'(\n\d)*\n\/(\n\d)*\nL\n' + DA_dict[language].replace(' ','\s+') +'\nL\s\d+\/\d+'
                legislation[language] = re.sub(header2, '', legislation[language])
            else:
                print("Unknown text")
            #legislation[language] = ' '.join(legislation[language].split())
            # some preliminary cleaning -> should be more 
            legislation[language] = legislation[language].replace('\xad', '')
    return legislation

def article_regex(language, num):
    order = art_dict[language][1]
    heading_ids = art_dict[language][2]
    art_id = art_dict[language][0]
    if order == 'pre':
        string = art_id+'\s('+str(num)+')\s\n(.*?)\n([A-Z|1].*?\n)(\s'+heading_ids+'|'+art_id+'\s'+str(num+1)+')'
    elif order == 'post':
        string = str(num)+'\s('+art_id+')\s(.*?)'+str(num+1)+' '+art_id
    elif order == 'postdot':
        string = str(num)+'.\s('+art_id+')\s(.*?)'+str(num+1)+'. '+art_id
    return re.compile(string, re.DOTALL)

def retrieve_article(DA, language, num):
    art_re = article_regex(language, num)
    art_text = art_re.search(DA[language])
    if art_text != []:
        art_num = int(art_text[1])
        art_title = ' '.join(art_text[2].split())
        art_body = art_text[3]
        if art_body[0:2] == '1.': 
            # if the article start with '1.' then it has numbered paragraphs
            paragraph_re = re.compile('((\d+)\.\n)(.*?)(?=((\d+)\.\n)|$)', re.DOTALL)
            art_paragraphs = [(int(p[1]), p[2]) for p in paragraph_re.findall(art_body)]
        else:
            art_paragraphs = [(0, ' '.join(art_body.split()))]
        return (art_num, art_title, art_paragraphs)
    else:
        print("Article not found")
        return None
