"""Polish author memo, expanded empirical results and outlet assessment."""
from pathlib import Path
import json
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
import pandas as pd
ROOT=Path(__file__).parent
pdfmetrics.registerFont(TTFont('DejaVu','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'));pdfmetrics.registerFont(TTFont('DejaVuBold','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='PL',fontName='DejaVu',fontSize=10.1,leading=14.8,spaceAfter=10));styles.add(ParagraphStyle(name='PLh',fontName='DejaVuBold',fontSize=14,leading=19,spaceBefore=12,spaceAfter=9,textColor=colors.HexColor('#174f74')));styles.add(ParagraphStyle(name='PLtitle',fontName='DejaVuBold',fontSize=20,leading=26,spaceAfter=16));styles.add(ParagraphStyle(name='PLsmall',fontName='DejaVu',fontSize=8.5,leading=11.8,spaceAfter=7));story=[]
def p(t,sty='PL'):story.append(Paragraph(t,styles[sty]))
def tbl(data,widths):
 t=Table([[Paragraph(str(c),styles['PLsmall']) for c in row] for row in data],colWidths=widths);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f0f5')),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#174f74')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),3)]));story.append(t);story.append(Spacer(1,9))
p('Artykuł po rozszerzeniu analizy GPW','PLtitle');p('Marek Gałązka i Hanna Wdowicka | wersja 0.4','PLsmall')
p('Finite-sample edge-order inference and certified degree comparisons in correlation spanning trees.')
p('Zachowano zmianę metodologiczną z poprzedniej rewizji: dokładne testy McNemara dla porównań Kendalla na z góry ustalonych, rozłącznych parach niezależnych obserwacji, korektę Holma, bezpośrednie granice różnic stopni, ograniczenia ścieżek oraz sumę stopni drzewa. Twierdzenie o przedziałach prostokątnych jest metodą porównawczą; klasyczne składniki są przypisane literaturze.')
p('Co dodano teraz','PLh')
for t in ['Dłuższe okna: 252, 504 i 1008 sesji. Grupy 10, 20 i 40 spółek są zagnieżdżone i wybierane według mediany iloczynu dostępnej ceny zamknięcia i wolumenu na wcześniejszym, najdłuższym oknie. To przybliżenie płynności z danych dostawcy.', 'Porównanie przy stałym składzie. W obrębie danego roku te same spółki są używane przy wszystkich długościach okien. Dane przyszłe nie uczestniczą w wyborze spółek. Rozmiar grupy zmienia badane drzewo, dlatego hub małej grupy nie jest automatycznie hubem całego rynku.', 'Pełna ustalona siatka. Wybrano ostatni zachowany termin obserwacji w każdym z 19 lat. Z 513 planowanych konfiguracji 81 wykluczono z powodu niedostatecznej historii lub zbyt małej liczby kwalifikujących się spółek. Obliczono wszystkie 432 dostępne analizy.', 'Sprawdzenia dodatkowe: 34 analizy długości bloków 5 i 20 oraz diagnostykę w następnych 252 sesjach. Uwzględniono pełną listę braków danych i wszystkie wyniki, także nierozstrzygające.']:p(t)
p('To retrospektywna analiza wrażliwości, ustalona przed obliczeniem nowych wyników. Nie jest prerejestracją ani potwierdzeniem poprawności nominalnego poziomu bootstrapu.','PLsmall')
story.append(PageBreak());p('Wyniki rzeczywistych danych','PLtitle')
d=pd.read_csv(ROOT/'results/gpw_extended_paired_summary.csv');data=[['Spółki','Wspólne terminy','Sesje','Średnia szerokość stopnia','Obowiązkowe krawędzie']]
for _,r in d.iterrows():data.append([int(r.cap),int(r.origins),int(r.lookback),f'{r.degree_width:.2f}',f'{r.forced:.2f}'])
tbl(data,[55,88,60,142,142]);p('Tabela dotyczy surowych sieci, tych samych spółek oraz terminów dostępnych dla wszystkich trzech długości próby. Liczba obowiązkowych krawędzi wynika z nominalnego regionu porządków bootstrapowych.','PLsmall')
p('Wydłużenie próby z 252 do 1008 sesji zawęża przedziały stopni średnio o 13,9% dla 10 spółek, 14,1% dla 20 i 10,3% dla 40. Jest to mierzalna poprawa rozdzielczości stopni, przy kontrolowanym składzie grup.')
p('Rangi nadal pozostają nierozstrzygnięte','PLh');p('We wszystkich 432 analizach podstawowych i 34 sprawdzeniach długości bloków przedziały rang pozostają [1,p]. Nie uzyskano certyfikatu jednego huba ani dokładnego stopnia pojedynczej spółki. Ten wynik został zachowany w artykule. Nie deklarujemy, że samo wydłużenie okna rozwiązało problem.')
p('Łącznie dodano 466 analiz, 9710 rekordów spółka-konfiguracja i 24 419 poprawiających certyfikatów wymiernych. Każdy certyfikat przeszedł niezależne sprawdzenie. Wcześniejszy pełny panel 231 terminów i jego 693 analizy również pozostają w artykule.')
p('Dłuższe okno daje więcej obserwacji, ale może łączyć różne reżimy rynkowe. Korzyść obliczeniowa nie ustanawia stacjonarności procesu ani prawidłowego pokrycia statystycznego.','PLsmall')
story.append(PageBreak());p('Stabilność w kolejnej próbie','PLtitle')
f=pd.read_csv(ROOT/'results/gpw_extended_forward_summary.csv');label={'raw':'Surowe','demeaned':'Po demeaning','factor_residual':'Reszty WIG'};data=[['Sieć','Dostępne konfiguracje','Zgodność kierunków','Hub nadal na szczycie']]
for _,r in f.iterrows():data.append([label[r.network],int(r.configurations),f'{100*r.mean_order_agreement:.1f}%',f'{int(r.anchor_still_top)}/{int(r.configurations)}'])
tbl(data,[130,95,120,142]);p('Hub nadal na szczycie obejmuje remisy. Zgodność kierunków jest średnią odsetków porządków krawędzi zgodnych z korelacjami oszacowanymi w następnych 252 sesjach.','PLsmall')
p('Dla każdej reprezentacji dostępne są 93 ze 144 konfiguracji podstawowych. Pozostałe 51 ma braki przyszłych historii lub za krótki okres obserwacji. Okna i grupy częściowo się pokrywają, więc nie są to niezależne próby. Badany skład spółek pozostaje stały w części przyszłej.')
p('W kierunkach porównań krawędzi widać dużą zgodność, lecz tożsamość empirycznego huba znacznie częściej się zmienia. To porównanie różnych cech sieci, a nie test pokrycia populacyjnego. Nawet duża zgodność pojedynczych kierunków nie oznacza zachowania wszystkich porównań jednocześnie.')
p('Ilustracja z 2008 r.','PLh');p('Dla 10 spółek i 504 sesji PKO BP ma najwyższy stopień, z uwzględnieniem remisów, w 91,2% drzew bootstrapowych. Przedział jego rangi nadal wynosi [1,10]. W następnej próbie szacowana ranga wynosi [3,10], mimo 100% zgodności kierunków wskazanych porządków krawędzi. Przykład wybrano retrospektywnie według jawnej reguły; pełna siatka wyników jest udostępniona.')
p('Weryfikacja i ograniczenia','PLh');p('Niezależnie odtworzono wybór 50 grup spółek, sprawdzono pięć sum kontrolnych wejściowych plików, wszystkie nowe rekordy wyników, 303 dostępne diagnostyki przyszłe oraz 24 419 certyfikatów. Wszystkie kontrole zakończyły się PASS. Wcześniejszy problem kalibracji bootstrapu, w tym pokrycie 22% w trudnym scenariuszu AR(1), pozostaje otwarty. Ścisła gwarancja Kendalla dotyczy niezależnych obserwacji.')
story.append(PageBreak());p('Moje typy do publikacji','PLtitle');p('Kolejność wynika z dopasowania aktualnej treści, a nie z deklarowanej szansy przyjęcia. Wszystkie trzy mają 100 punktów w ostatnim opublikowanym wykazie ministerialnym (2024). Model subskrypcyjny pozwala publikować bez APC; otwarty dostęp jest opcją płatną.','PLsmall')
tbl([['Kolejność / czasopismo','Dopasowanie i główne ryzyko'],['1. Journal of Complex Networks','Najbardziej naturalny odbiorca: niepewność cech sieci, drzewa korelacyjne, algorytmy i rzeczywiste dane. Ryzyko: nadal trzeba wykazać istotny wkład w analizę sieci.'],['2. Computational Statistics & Data Analysis','Dobre dopasowanie metod obliczeniowych, symulacji, projekcji i walidacji. Ryzyko: połączenie klasycznych narzędzi może zostać ocenione jako niewystarczająco nowe.'],['3. Journal of Statistical Planning and Inference','Alternatywa dla nacisku na dokładne wnioskowanie, remisy i pokrycie. Wymaga szczególnie przekonującego własnego wkładu statystycznego; części finansowej nie należy używać jako jego substytutu.']],[184,303])
p('Pierwszy wybór: Journal of Complex Networks. Nie dokonałem zgłoszenia. Rozszerzenie empiryczne jest ukończone, ale nie uzasadnia twierdzenia o rozwiązaniu problemu populacyjnego huba na GPW ani gwarancji akceptacji.')
p('Statistics and Computing nie traktuję jako domyślnie bezpłatnej alternatywy: nowe zgłoszenia od lipca 2026 są objęte APC po akceptacji, z możliwością zwolnienia z opłaty. Physica A i Computational Statistics mają po 70 punktów, więc nie spełniają progu.','PLsmall')
p('Źródła oceny czasopism','PLh')
for url,text in [
('https://www.gov.pl/attachment/62739f3a-6a1d-4659-98ad-59dd63108ee7','Wykaz ministerialny: punktacja (2024)'),
('https://academic.oup.com/comnet/pages/About','Journal of Complex Networks: zakres'),
('https://academic.oup.com/comnet/pages/General_Instructions','Journal of Complex Networks: standardowa licencja i opcjonalne OA'),
('https://shop.elsevier.com/journals/computational-statistics-and-data-analysis/0167-9473','CSDA: zakres i model subskrypcyjny'),
('https://www.elsevier.com/publishing/publish-in-a-journal/journal-selection','Elsevier: brak opłaty publikacyjnej w modelu subskrypcyjnym'),
('https://www.sciencedirect.com/journal/journal-of-statistical-planning-and-inference/about/insights','JSPI: opcja subskrypcyjna bez opłaty publikacyjnej'),
('https://link.springer.com/journal/11222','Statistics and Computing: zmiana zasad opłat')]:p(f'<link href="{url}" color="#174f74">{text}</link>','PLsmall')
def footer(canvas,doc):canvas.setFont('DejaVu',8);canvas.setFillColor(colors.HexColor('#607080'));canvas.drawString(52,28,'Rewizja empiryczna | wersja 0.4');canvas.drawRightString(543,28,str(doc.page))
SimpleDocTemplate(str(ROOT/'REWIZJA_MERYTORYCZNA_PL.pdf'),pagesize=(595.28,841.89),rightMargin=52,leftMargin=52,topMargin=46,bottomMargin=48,title='Rozszerzona analiza GPW i wybór czasopisma',author='Marek Gałązka i Hanna Wdowicka').build(story,onFirstPage=footer,onLaterPages=footer)
