from pathlib import Path
import json
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
root=Path(__file__).parent
pdfmetrics.registerFont(TTFont('DejaVu','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'));pdfmetrics.registerFont(TTFont('DejaVuBold','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='PL',fontName='DejaVu',fontSize=10.2,leading=15,spaceAfter=10));styles.add(ParagraphStyle(name='PLh',fontName='DejaVuBold',fontSize=14,leading=19,spaceBefore=14,spaceAfter=10,textColor=colors.HexColor('#174f74')));styles.add(ParagraphStyle(name='PLtitle',fontName='DejaVuBold',fontSize=21,leading=27,spaceAfter=17));styles.add(ParagraphStyle(name='PLsmall',fontName='DejaVu',fontSize=8.6,leading=12,spaceAfter=8))
story=[]
def p(t,sty='PL'):story.append(Paragraph(t,styles[sty]))
p('Przebudowa merytoryczna artykułu','PLtitle')
p('Marek Gałązka i Hanna Wdowicka | wersja 0.3','PLsmall')
p('Nowy tytuł: Finite-sample edge-order inference and certified degree comparisons in correlation spanning trees.')
p('Zmieniony jest rdzeń metody, dowody i doświadczenia obliczeniowe. Wcześniejsze twierdzenie o prostokątnych przedziałach zostało przesunięte do roli metody porównawczej. Nie przedstawiamy klasycznego argumentu wymiany dla matroidów jako nowego głównego wyniku.')
p('Co stanowi zmianę','PLh')
for t in [
'1. Bezpośrednie testy porządku korelacji Kendalla. Z góry ustalone, rozłączne pary niezależnych obserwacji dają dokładne warunkowe testy McNemara. Korekta Holma kontroluje prawdopodobieństwo choć jednego fałszywego porównania. Dowód obejmuje zmienne dyskretne dzięki niezależnemu rozstrzyganiu remisów i zachowuje cel Kendall tau-a.',
'2. Wspólne ograniczenia drzewa. Obliczamy różnice stopni w jednym dopuszczalnym drzewie, dodajemy zakazane konfiguracje ścieżek i rozwiązujemy silniejsze zadania liniowe. Poprawione granice mają certyfikaty wymierne, które można sprawdzić bez ufania dokładności solvera.',
'3. Ostrzejsze ograniczenia rang. Oprócz bezpośrednich porównań wykorzystujemy tożsamość sumy stopni 2p-2. Podajemy dowód dokładności obliczenia rang w obrębie relaksacji stopni; nie przypisujemy mu dokładności dla całego pierwotnego problemu.',
'4. Literatura i zakres nowości. Dodane zostały prace Zaffalona i Huttera o odpornym wnioskowaniu o drzewach, literatura o możliwej i koniecznej optymalności, analiza wrażliwości MST i combinatorial inference. Test McNemara, korekta Holma i własności matroidów są wyraźnie przypisane wcześniejszej literaturze.'
]:p(t)
p('Istotna granica: ścisła gwarancja dotyczy niezależnych obserwacji. Nie przenosimy jej na zależne w czasie stopy zwrotu z GPW.','PLsmall')
story.append(PageBreak())
p('Wyniki nowych obliczeń','PLtitle')
p('Przeprowadzono 4400 prób dla metody dokładnej i 2400 prób walidujących bootstrap. Ponownie obliczono wszystkie 231 okien GPW w trzech reprezentacjach: 693 analizy i 35 979 rekordów spółka-okno-reprezentacja.')
data=[['Scenariusz, n = 4096','Nowa metoda','Poprzednia'],['12 spółek, model Gaussa','100%','0%'],['12 spółek, model t(5)','100%','0%'],['52 spółki, model Gaussa','79%','0%'],['52 spółki, model t(5)','78%','0%']]
t=Table([[Paragraph(c,styles['PLsmall']) for c in row] for row in data],colWidths=[250,95,95]);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f0f5')),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#174f74')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),4)]));story.append(t);story.append(Spacer(1,10))
p('Tabela: odsetek prób, w których certyfikowano prawdziwy hub jako jedyny pierwszy w rankingu; po 200 prób w każdym scenariuszu. Nowa metoda porównań wykorzystuje rozłączne pary, a poprzednia pełną estymatę Kendalla i jednoczesne pasmo Hoeffdinga. Zmienione są zarówno kalibracja, jak i projekcja. To nie jest twierdzenie o przewadze nad wszystkimi możliwymi metodami.','PLsmall')
p('Przy krótkich próbach i niemal związanych hubach metoda najczęściej pozostaje nierozstrzygająca. W jednym eksperymencie zerowym zaobserwowano 14 fałszywych porządków na 200 prób (7%); podajemy przedział Monte Carlo, obejmujący 5%, i zachowujemy ten wynik w tabeli.')
p('Bootstrap nadal wymaga ostrożnej interpretacji','PLh')
p('W najtrudniejszym scenariuszu AR(1), przy 52 zmiennych i n = 252, jednoczesne pokrycie przedziałów kontrastów wynosi tylko 22%, przy nominalnych 95%. Częstość choć jednego fałszywego porównania wynosi 78%. Tego wariantu nie można przedstawiać jako naprawionej procedury 95%. Wynik został zachowany i opisany jako istotne niepowodzenie kalibracji.')
p('Dobre pokrycie szerokich przedziałów stopni nie usuwa tego problemu: zbyt szeroka projekcja może ukrywać błędy wejściowych testów. Dlatego artykuł osobno raportuje kalibrację i wyniki końcowej projekcji.')
story.append(PageBreak())
p('GPW, weryfikacja i status','PLtitle')
p('W surowej sieci GPW średnia szerokość przedziału stopnia spada z 49,89 po projekcji na minor do 45,99 po dodaniu ograniczeń ścieżek, czyli o około 7,8%. Dla demeaning spada z 49,90 do 46,51, a dla reszt względem WIG z 49,91 do 47,63. To rzeczywiste zawężenie granic stopni.')
p('Jednocześnie wszystkie 35 979 przedziałów rang nadal wynoszą [1,p]. Nie uzyskano certyfikatu jednego huba ani przynależności do górnego decyla. Artykuł podaje ten wynik wprost. Obliczenia GPW pozostają diagnostyką nominalnego bootstrapu, bez gwarancji wynikającej z twierdzenia dla próby niezależnej.')
p('Sprawdzenie implementacji','PLh')
p('Pełna enumeracja małych drzew potwierdziła 300 przypadków ekstremów stopni i ich różnic oraz 300 przypadków ograniczeń rang. Dodatkowo sprawdzono 32 przypadki projekcji z ograniczeniami ścieżek, 242 poprawiające certyfikaty dla małych grafów, 30 porównań pełnej estymaty Kendalla i 30 porównań bezpośrednich liczników niezgodności. Dla GPW niezależny weryfikator sprawdził wszystkie 85 129 zapisanych certyfikatów wymiernych. Wszystkie przeszły weryfikację; końcowy status to PASS.')
p('Granice wkładu','PLh')
p('Zmiana odpowiada na zarzuty merytoryczne, ale nie gwarantuje akceptacji. Nowa konstrukcja nadal łączy znane narzędzia. Jej wartość polega na dokładnej statystycznie ścieżce od testów Kendalla do hubów, mocniejszej projekcji i sprawdzonych wynikach. Dla zależnych danych finansowych pozostaje otwarty problem poprawnej kalibracji. Nie należy go maskować zmianą tytułu lub deklaracją nowości.')
p('Przekazywana wersja zawiera tekst angielski, dowody, bibliografię, kod, komplet wyników i instrukcje odtworzenia. Zachowano oboje autorów oraz krótką deklarację rzeczywistego zakresu pomocy AI. Poprzednio zgłoszone wersje pozostają zachowane. Nie dokonano nowego zgłoszenia do czasopisma.')
p('Repozytorium: https://github.com/mgalazka84/gpw-hub-reliability<br/>Gałąź: contrast-revision-v0.3; katalog: contrast_inference.','PLsmall')
def footer(canvas,doc):
 canvas.setFont('DejaVu',8);canvas.setFillColor(colors.HexColor('#607080'));canvas.drawString(52,28,'Rewizja merytoryczna | wersja 0.3');canvas.drawRightString(543,28,str(doc.page))
SimpleDocTemplate(str(root/'REWIZJA_MERYTORYCZNA_PL.pdf'),pagesize=(595.28,841.89),rightMargin=52,leftMargin=52,topMargin=48,bottomMargin=48,title='Przebudowa merytoryczna artykułu',author='Marek Gałązka i Hanna Wdowicka').build(story,onFirstPage=footer,onLaterPages=footer)
