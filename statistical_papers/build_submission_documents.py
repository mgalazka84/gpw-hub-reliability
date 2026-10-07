"""Create the editor-facing letter and a separate Polish author review note."""
from pathlib import Path
import argparse,json,shutil,re
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,KeepTogether

ROOT=Path(__file__).parent
FONT='/usr/share/fonts/truetype/dejavu'
pdfmetrics.registerFont(TTFont('DV',FONT+'/DejaVuSerif.ttf'))
pdfmetrics.registerFont(TTFont('DV-Bold',FONT+'/DejaVuSerif-Bold.ttf'))
pdfmetrics.registerFontFamily('DV',normal='DV',bold='DV-Bold',italic='DV',boldItalic='DV-Bold')
STYLE=getSampleStyleSheet()
STYLE.add(ParagraphStyle(name='BodyPL',fontName='DV',fontSize=10.2,leading=14.6,spaceAfter=8))
STYLE.add(ParagraphStyle(name='HeadingPL',fontName='DV-Bold',fontSize=12.3,leading=16.2,spaceBefore=10,spaceAfter=7))
STYLE.add(ParagraphStyle(name='TitlePL',fontName='DV-Bold',fontSize=17,leading=22,spaceAfter=16))
STYLE.add(ParagraphStyle(name='SmallPL',fontName='DV',fontSize=8.5,leading=11.6,spaceAfter=5))

def p(text,style='BodyPL'):return Paragraph(text,STYLE[style])
def footer(canvas,doc):
    canvas.setFont('DV',8);canvas.setFillColor(colors.HexColor('#555555'))
    canvas.drawString(45,28,'Gałązka · Wdowicka | Statistical Papers | v0.2')
    canvas.drawRightString(A4[0]-45,28,str(doc.page))

def doc(path,story,title):
    SimpleDocTemplate(str(path),pagesize=A4,rightMargin=45,leftMargin=45,topMargin=43,bottomMargin=44,
       title=title,author='Marek Gałązka; Hanna Wdowicka').build(story,onFirstPage=footer,onLaterPages=footer)

def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    title='Simultaneous degree and rank inference for hubs in correlation spanning trees'
    repo='https://github.com/mgalazka84/gpw-hub-reliability/tree/statpapers-v0.2.0'
    letter=[p('Cover letter','TitlePL'),p('Dear Editors of <i>Statistical Papers</i>,'),
       p('Please consider our manuscript, “'+title+'”, for publication as a regular research article.'),
       p('The paper studies uncertainty in the degree and rank of vertices in estimated correlation spanning trees. Its central construction maps a simultaneous edge-weight band to exact coordinatewise degree extrema over the rectangular region, using two lexicographic spanning-tree computations per vertex. It then gives simultaneous outer rank intervals that retain all population tree and degree ties.'),
       p('The manuscript states precisely which ingredients come from existing interval optimization, bootstrap and rank-inference literature. It provides the projection proof, a finite-sample Kendall implementation for independent observations, and an asymptotic block-bootstrap Pearson implementation. The simulation study evaluates both input-band coverage and projected inference, with 10500 bootstrap-based data sets and a further 3000 independent data sets for the Kendall construction.'),
       p('The application uses 231 historical Polish equity windows. Under the primary rectangular procedure, every vertex retains the full range of possible ranks. We report this lack of resolution directly. We also disclose finite-sample undercoverage of the bootstrap input band in some simulation designs; the finite-sample Kendall guarantee has a separate independence assumption.'),
       p('The statistical target, explicit computational result and assessment of coverage and conservatism are intended to fit the journal’s focus on methodological foundations and applications. Code, seeds, derived results and verification records are available in the versioned public repository:<br/><link href="'+repo+'" color="#174B72">'+repo+'</link>'),
       p('No dedicated funding was received, and the authors have no relevant competing interests. ChatGPT (OpenAI) assisted with literature retrieval, methodological development, coding, numerical verification and manuscript drafting; AI assistance is also disclosed in the manuscript.'),
       p('Sincerely,<br/>Marek Gałązka<br/>Corresponding author for Marek Gałązka and Hanna Wdowicka<br/>Faculty of Mathematics and Computer Science<br/>Adam Mickiewicz University, Poznań, Poland<br/>galazka@amu.edu.pl')]
    doc(out/'SP_cover_letter.pdf',letter,'Cover letter - Statistical Papers')
    s=pd.read_csv(ROOT/'results/simulation_summary.csv')
    sh=pd.read_csv(ROOT/'results/simulation_shapes_summary.csv')
    mc=pd.concat([s,sh],ignore_index=True)
    bandmin=100*mc.band_coverage.min();degree_min=100*mc.degree_coverage.min()
    abstract=re.search(r'\\abstract\{(.*?)\}\s*\\keywords',(ROOT/'manuscript/SP_manuscript.tex').read_text(),re.S).group(1)
    abstract_words=len(abstract.split())
    report=[p('Przegląd przed zgłoszeniem','TitlePL'),
       p('<b>Cel: Statistical Papers. Status: kompletna wersja do merytorycznej oceny autorów przed zgłoszeniem.</b> Nowy tytuł: <i>'+title+'</i>. Autorzy: Marek Gałązka i Hanna Wdowicka.'),
       p('Co zmieniono po decyzji IRFA','HeadingPL'),
       p('IRFA odrzuciło poprzednią pracę z powodu niewystarczającej nowości. W decyzji nie było szczegółowych raportów recenzentów ani zarzutu błędu w obliczeniach. Nowa wersja odpowiada na tę jedną wskazaną uwagę przez zmianę głównego problemu badawczego i dodanie wnioskowania o stopniach i rangach.'),
       p('Rdzeniem pracy jest obliczanie dokładnych skrajnych stopni węzła wśród wszystkich drzew optymalnych dla wag dopuszczonych przez przedziały. Wystarczają dwa drzewa dla każdego węzła. Dalej wyprowadzono jednoczesne przedziały rang, obejmujące także remisy, oraz twierdzenie przenoszące pokrycie wspólnego obszaru ufności na wynik sieciowy.'),
       p('Dodatkowo podano wariant Kendalla z gwarancją dla skończonej próby przy niezależnych obserwacjach. Wariant Pearsona z bootstrapem blokowym ma uzasadnienie asymptotyczne pod zapisanymi założeniami. Dotychczasowe proste własności centrowania i modelu czynnikowego pozostają tłem interpretacyjnym. Dawne prognozy GPW są opisane jako odziedziczony wynik eksploracyjny.'),
       p('Granice deklarowanej nowości','HeadingPL'),
       p('Klasyczne argumenty wymiany krawędzi, optymalizacja przy wagach przedziałowych, bootstrap oraz projekcja obszaru ufności są znane. Manuskrypt to zaznacza. Wkład jest sformułowany jako jawna konstrukcja projekcji dla stopni i rang, jej dowód, obsługa remisów oraz badanie własności i ograniczeń. Nie deklaruje pierwszeństwa dla wymienionych klasycznych składników.'),
       p('Najbliższe sprawdzone prace: Yaman, Karaşan i Pınar (2001), optymalizacja drzew przy wagach przedziałowych; Kasperski i Zieliński (2007), matroidy z niepewnymi wagami; Hall i Miller (2009), bootstrap rankingów; Kalyagin i współautorzy (2022), identyfikacja drzew korelacyjnych; Mogstad i współautorzy (2024), jednoczesne wnioskowanie o rangach. Pełne dane bibliograficzne i DOI są w artykule.'),
       p('<b>Ocena naukowa:</b> przebudowa jest istotna względem wersji IRFA. Nadal istnieje ryzyko uznania wkładu za zbyt przyrostowy. Ukierunkowany przegląd literatury nie dowodzi pierwszeństwa. Przed wysłaniem szczególnej oceny autorskiej wymaga stopień nowości wyniku o skrajnych konfiguracjach na tle szerszej teorii optymalizacji matroidowej.'),
       PageBreak(),p('Wyniki i kontrola jakości','TitlePL'),
       p('Wykonane obliczenia','HeadingPL'),
       p('Łącznie wykonano 13 500 replikacji Monte Carlo: 7500 w pięciu podstawowych modelach, 3000 w sieciach z dwoma hubami i w łańcuchach oraz 3000 dla wariantu Kendalla. Część bootstrapowa używa 999 losowań na próbę. Analiza GPW obejmuje 231 okien, trzy reprezentacje zwrotów i 155 różnych spółek. Dodatkowe długości bloków 5 i 20 sprawdzono na 47 równomiernie wybranych oknach.'),
       p(f'<b>Kalibracja:</b> wśród modeli bootstrapowych najniższe pokrycie wspólnego przedziału wag wynosi {bandmin:.1f}%, przy poziomie nominalnym 95%. Najniższe pokrycie całego celu stopni wynosi {degree_min:.1f}%. Wysokie pokrycie szerokich przedziałów stopni nie naprawia słabej kalibracji przedziałów wejściowych. Wyniki oraz przedziały Wilsona są udostępnione w repozytorium.'),
       p('<b>Precyzja:</b> przy wyraźnym hubie i 1008 obserwacjach wariant Gaussowski identyfikuje lidera we wszystkich 500 replikacjach; przy 252 obserwacjach nie certyfikuje go w żadnej. W modelu bliskiego remisu przy 252 obserwacjach zwykłe percentylowe przedziały stopni pokrywają cały prawdziwy wektor jedynie w 3,0% replikacji. Nowe przedziały są w tym modelu szerokie.'),
       p('<b>GPW:</b> we wszystkich oknach przedziały rang mają postać [1, liczba spółek]. Nie potwierdzają żadnego huba. Wniosek dotyczy ograniczonej precyzji tej procedury, a nie nieistnienia hubów. Zastosowanie finansowe jest więc ilustracją ograniczeń metody, nie pozytywnym dowodem jej przydatności prognostycznej.'),
       p('Sprawdzenie matematyczne i programistyczne','HeadingPL'),
       p('Dowody zapisano w manuskrypcie. Algorytm porównano z pełną enumeracją wszystkich drzew na 3-6 wierzchołkach w 1040 przypadkach wag przedziałowych. Zgodność obejmuje 11 244 dopuszczalne drzewa. Sprawdzono też definicję Kendalla przy remisach, zgodność ważonych kowariancji z bezpośrednim bootstrapem i ponownym dopasowaniem regresji oraz wszystkie zapisane agregaty. Odtworzono pełną replikację z ustalonego ziarna w każdym z 21 scenariuszy bootstrapowych.'),
       p('Zgodność z profilem czasopisma','HeadingPL'),
       p(f'Praca jest złożona w szablonie Springera, z nazwiskami autorów zgodnie z procedurą single-blind, abstraktem liczącym {abstract_words} słów, sześcioma słowami kluczowymi, klasyfikacją MSC i bibliografią autor-rok. Dołączono deklaracje braku dedykowanego finansowania i konfliktów interesów zgodnie z odpowiedzią autora. Krótka deklaracja AI obejmuje faktyczny rozszerzony zakres pomocy, w tym rozwój metody i tekstu; nie podaje dat przygotowania.'),
       PageBreak(),p('Pliki i sposób zgłoszenia','TitlePL'),
       p('Pakiet zawiera','HeadingPL'),
       p('<b>SP_manuscript.pdf</b> - pełny manuskrypt z autorami i figurami.<br/><b>SP_cover_letter.pdf</b> - list przewodni po angielsku.<br/><b>SP_LaTeX_source.zip</b> - plik TEX, bibliografia, klasa, styl i trzy figury.<br/><b>SP_przeglad_PL.pdf</b> - niniejsza notatka dla autorów, której nie należy wysyłać jako części artykułu.'),
       p('Kod i wyniki','HeadingPL'),
       p('<link href="'+repo+'" color="#174B72">'+repo+'</link>'),
       p('Katalog statistical_papers zawiera kod, wyniki, ziarna, instrukcje odtwarzania i raporty sprawdzeń. Surowe notowania podmiotów trzecich pozostają poza publicznym repozytorium. Artykuł wyraźnie opisuje brak pełnej walidacji dywidend i zmian składu indeksów, niedostępne historie, selekcję na kompletnych danych oraz nominalny charakter przedziałów GPW.'),
       p('Przed wysłaniem','HeadingPL'),
       p('Autorzy powinni przeczytać dowody i zaakceptować zakres roszczenia o nowości, przyjmując do wiadomości słabą precyzję na GPW oraz niedopokrycie bootstrapu w części symulacji. W formularzu należy podać rzeczywisty wkład każdego autora, ORCID-y jeśli są dostępne i potwierdzić wymagane przez redakcję oświadczenia. Tych osobistych oświadczeń nie zastępuje weryfikacja programu.'),
       p('Zgłoszenie w Statistical Papers','HeadingPL'),
       p('Do głównego zgłoszenia służy pełny, podpisany manuskrypt oraz edytowalne źródła; nie ma potrzeby przygotowywania anonimowej wersji dla procedury single-blind. List przewodni można wkleić do formularza lub dołączyć w przewidzianym miejscu. Wybór publikacji subskrypcyjnej pozwala uniknąć APC; wybór modelu wydawca oferuje po przyjęciu artykułu.'),
       p('Źródła wymagań','HeadingPL'),
       p('<link href="https://link.springer.com/journal/362/submission-guidelines" color="#174B72">Instrukcje dla autorów Statistical Papers</link><br/><link href="https://link.springer.com/journal/362/how-to-publish-with-us" color="#174B72">Modele publikacji i opłaty</link>','SmallPL'),
       p('<b>Nie wykonano zgłoszenia w systemie redakcyjnym.</b> Pakiet umożliwia ocenę nowej wersji i przygotowanie zgłoszenia. Poprzednia decyzja IRFA była ostateczna; nowa wersja jest przeznaczona do innego czasopisma.','SmallPL')]
    doc(out/'SP_przeglad_PL.pdf',report,'Przegląd przed zgłoszeniem - Statistical Papers')
    shutil.copy2(ROOT/'manuscript/SP_manuscript.pdf',out/'SP_manuscript.pdf')
    (out/'START_HERE_PL.txt').write_text('Statistical Papers - wersja do oceny autorów przed zgłoszeniem\n\n1. Przeczytaj SP_manuscript.pdf i SP_przeglad_PL.pdf.\n2. Do czasopisma przeznaczone są SP_manuscript.pdf, źródła w SP_LaTeX_source.zip i SP_cover_letter.pdf.\n3. SP_przeglad_PL.pdf jest notatką wewnętrzną dla autorów.\n4. Kod i wyniki: '+repo+'\n5. Żadne zgłoszenie do czasopisma nie zostało wykonane.\n',encoding='utf-8')
    print(json.dumps(dict(output=str(out),documents=['SP_manuscript.pdf','SP_cover_letter.pdf','SP_przeglad_PL.pdf']),indent=2))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',required=True);main(a.parse_args().out)
