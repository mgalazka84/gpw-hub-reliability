"""Polish point-by-point review response, title page and concise Highlights."""
from pathlib import Path
import json
import pandas as pd
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors

ROOT=Path(__file__).parent;OUT=ROOT.parents[1]/'revision_deliverables_v05';OUT.mkdir(exist_ok=True)
pdfmetrics.registerFont(TTFont('DV','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('DVB','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
pdfmetrics.registerFontFamily('DV',normal='DV',bold='DVB')
body=ParagraphStyle('body',fontName='DV',fontSize=10,leading=14.2,spaceAfter=9)
heading=ParagraphStyle('heading',fontName='DVB',fontSize=13.4,leading=18,spaceAfter=12,textColor=colors.HexColor('#153953'))
title=ParagraphStyle('title',fontName='DVB',fontSize=19,leading=25,spaceAfter=16,textColor=colors.HexColor('#153953'))
small=ParagraphStyle('small',fontName='DV',fontSize=8.6,leading=12.3,spaceAfter=7)

def foot(canvas,doc):
    canvas.setFont('DV',8);canvas.setFillColor(colors.HexColor('#607080'))
    canvas.drawString(48,26,'Rewizja merytoryczna | wersja 0.5');canvas.drawRightString(548,26,str(doc.page))

def table(story,data,widths):
    rows=[[Paragraph(str(v),small) for v in row] for row in data]
    t=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f0f5')),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#7d9aad')),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#d9e1e6')),
        ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
    story.extend([t,Spacer(1,10)])

def run():
    numbers=json.loads((ROOT/'results/review_report_numbers.json').read_text());checks=json.loads((ROOT/'results/review_verification.json').read_text());assert checks['status']=='PASS'
    summary=pd.read_csv(ROOT/'results/comparative/summary_all.csv');nodes=pd.read_csv(ROOT/'results/cross_section/nodes_all.csv')
    story=[]
    def p(text,style=body):story.append(Paragraph(text,style))
    def page(label):story.append(PageBreak());p(label,title)
    p('Odpowiedź na recenzję i zakres rewizji',title)
    p('Finite-sample edge-order inference and certified degree comparisons in correlation spanning trees',heading)
    p('Marek Gałązka i Hanna Wdowicka',small)
    p('Rewizja obejmuje zmianę konstrukcji statystycznej, pełne porównania na tych samych danych, audyt kalibracji oraz rzeczywistą aplikację do pomiarów przekrojowych. Ten dokument prowadzi po punktach recenzji i wskazuje założenia wymagające ponownej oceny.')
    p('Najważniejsza zmiana merytoryczna',heading)
    p('Dodano dokładne w sensie gwarancji dla skończonej próby wnioskowanie przez średnią e-wartości. Każda e-wartość jest nieujemnym iloczynem czynników 1 + λZ dla niezależnych par obserwacji. Przy hipotezie zerowej jej wartość oczekiwana jest najwyżej 1. Średnia zachowuje tę własność mimo zależności między sparowaniami; odwrotność daje poprawną p-wartość do Holma. Surowe jądra tau-a zachowują zera przy remisach.')
    p('Siedem z góry określonych frakcji obejmuje min(1/2, 1/√N), co zapewnia zgodność dla każdej ustalonej dodatniej luki. Porównano 8 i 64 sparowania, obok średniej p-wartości, pojedynczego McNemara, dwóch metod Hoeffdinga i dwóch kalibracji mnożnikowych. Klasyczna teoria e-wartości jest wyraźnie przypisana literaturze.')
    p('Rzeczywisty wynik, którego wcześniej brakowało',heading)
    p('W zbiorze Abalone (4177 okazów, 8 pomiarów numerycznych), przy jawnym modelu IID, E64 daje dla masy całkowitej stopień [3,5] i rangę [1,3]. Wyklucza cztery zmienne z pozycji huba. Rings ma stopień [1,1] i rangę [3,8]. Średnia szerokość stopni wynosi 1,375 wobec 2,625 dla pojedynczego dokładnego testu po tej samej projekcji. Bootstrap rozstrzyga więcej, lecz jego wynik jest opisany oddzielnie jako przybliżony.')
    p('Materiał do ponownej recenzji',heading)
    p('Manuskrypt zawiera nowe dowody, 24 komórki porównań i 7800 wspólnych danych symulowanych, oddzielny audyt na 3000 danych, 192 analizy pomiarów oraz skróconą główną część GPW. Pełne rekordy, projekcje, źródła i weryfikatory udostępniono wraz z kodem.')
    page('Uwagi główne 1-2')
    p('1. Strata efektywności przez rozłączne pary',heading)
    p('Potwierdzono mnożnik wariancji 4,5 względem pełnej U-statystyki w modelu ciągłych niezależnych współrzędnych: Var(h)=2, ζ₁=2/9. Jest to rachunek dla konkretnego modelu zerowego, a nie uniwersalny przelicznik mocy. Dla średniej estymatorów z R losowych sparowań odpowiadający stosunek asymptotyczny wynosi 1+3,5/R; nie jest to automatycznie efektywność testu z połączonych p-wartości.')
    p('Poprawna reguła min(1, 2×średnia p) została wdrożona i zweryfikowana. Wyniki wykazały jej konserwatyzm, dlatego nie przedstawiono jej jako rozwiązania problemu. Dodano zamiast tego agregację dowodów E8/E64 z gwarancją FWER i osobnym dowodem zgodności. Porównania E8 i M8 korzystają z tych samych pierwszych ośmiu sparowań.')
    p('W rozdziale o dokładnych porównaniach znajdują się: rachunek efektywności, Propozycja dotycząca p-wartości oraz Propozycja dotycząca e-wartości i zgodności. Nowy dowód opiera się na ograniczeniu wartości oczekiwanej iloczynu i nierówności Markowa. Pełna U-statystyka służy również jako komparator i filtr zgodności kierunków.')
    p('2. Siła komparatorów i rozdzielenie efektów',heading)
    p('Zestaw obejmuje osiem metod: pojedynczy McNemar-Holm, M8, E8, E64, bezpośredni Hoeffding dla różnic pełnych U-statystyk, prostokąt Hoeffdinga, mnożnikowe różnice Kendalla oraz mnożnikowy prostokąt. Warianty mnożnikowe wykorzystują wszystkie pary i empiryczne projekcje pierwszego rzędu, zgodnie z literaturą Chena i CCK.')
    p('Każda kalibracja otrzymuje tę samą projekcję. Przy ustalonych testach porównano marginesowe rangi, bezpośrednie różnice stopni, zachowanie sumy stopni, ich przecięcie oraz LP ze ścieżkami. Zysk z kalibracji i z ograniczeń drzewa jest zatem rozdzielony. Częstości 499 zwykłych bootstrapów stanowią osobny praktyczny punkt odniesienia na 2800 tych samych próbach.')
    p('W modelach rozdzielonej gwiazdy przy p=52 i n=4096 E8/E64 rozstrzygają huba w 100/100 prób w każdym z dwóch rozkładów; pojedynczy McNemar daje 67/100 i 72/100. Przy n=1008 i p=12 dokładne warianty pozostają słabsze od pełnych U-statystyk. Moc bootstrapu jest oceniana razem z jego rzeczywistym FWER, a nie wyłącznie z nominalnym poziomem.')
    page('Uwagi główne 3-4')
    p('3. Aplikacja zgodna z twierdzeniem',heading)
    p('Główną aplikacją są oryginalne pomiary Abalone, z celem określonym przez Kendall tau-a. Zastosowano w niej bezpośrednio procedury z gwarancją dla skończonej próby, w tym nową agregację e-wartości. Raport obejmuje wszystkie pomiary numeryczne, trzy dodatkowe straty płci/statusu i obie rodziny porównań.')
    rows=[['Procedura','Masa całkowita: stopień','Masa całkowita: ranga','Rings: stopień / ranga']]
    for method,label in [('paired_exact','McNemar'),('evalue_8','E8'),('evalue_64','E64'),('multiplier_contrast','Mnożnikowy')]:
        sub=nodes[(nodes.dataset=='Abalone')&(nodes.stratum=='pooled')&(nodes.family=='local')&(nodes.method==method)]
        a=sub[sub.name=='Whole weight'].iloc[0];b=sub[sub.name=='Rings'].iloc[0]
        rows.append([label,f'[{int(a.degree_lower)},{int(a.degree_upper)}]',f'[{int(a.rank_lower)},{int(a.rank_upper)}]',f'[{int(b.degree_lower)},{int(b.degree_upper)}] / [{int(b.rank_lower)},{int(b.rank_upper)}]'])
    table(story,rows,[105,125,120,149])
    p('Repozytorium źródłowe nie podaje identyfikatorów okazów ani klastrów pobrania wystarczających do wykazania IID z samych plików. Gwarancja jest zatem warunkowa względem jawnego modelu próbkowania. Osobne analizy mają osobny nominalny poziom; nie deklarujemy wspólnego pokrycia wszystkich metod i grup. Dry Bean, z 68 powtórzonymi wierszami, pozostaje testem obliczeniowym. GPW skrócono do studium wrażliwości dla Pearsona i zależnych stóp zwrotu.')
    p('Dodano także znany model nieeliptyczny: dodatnie przekształcenia wykładnicze zachowują wszystkie jądra Kendalla, ale zmieniają stopień wierzchołka 0 z 11 w drzewie Kendalla do 1 w drzewie Pearsona. Weryfikator sprawdza oba cele populacyjne. Nienakładające się okresy finansowe same nie zapewniają niezależności.')
    p('4. Audyt bootstrapu Pearsona przy IID',heading)
    table(story,[['p=52, n=1008; 1000 prób','FWER'],['Oryginalny, 499 replikacji','0,107'],['Oryginalny, 1999 replikacji','0,086'],['Bootstrap wierszy, skala z funkcji wpływu','0,067'],['Mnożniki Gaussa, skala z funkcji wpływu','0,076']],[370,129])
    p('Różnica sparowana między 499 a 1999 replikacjami wynosi 0,021 z przedziałem Monte Carlo [0,007;0,035]. Liczba losowań ma znaczenie, lecz pozostałe warianty także nie osiągają nominalnego poziomu w tej komórce. Nie przypisano zatem całości problemu wyłącznie 499 replikacjom ani bootstrapowi jako klasie metod. Obliczenia kowariancji sprawdzono przez bezpośrednie losowanie indeksów; funkcje wpływu i maksima zweryfikowano osobno.')
    page('Uwagi główne 5-8')
    p('5. Precyzja Monte Carlo w komórkach zerowych',heading)
    p('Wszystkie nowe komórki zerowe mają po 1000 prób, także pełna rodzina krawędzi przy p=12. Raportowane są przedziały Wilsona i osobno FWER, pokrycie stopni oraz rang. Dawne 14/200 ma prawdopodobieństwo ogonowe około 0,130 przy prawdziwym błędzie 0,05, więc nie traktowano go jako obalenia twierdzenia. W nowym zestawie najwyższy błąd pojedynczego dokładnego testu wynosi 0,035.')
    p('6. LP w Monte Carlo',heading)
    p(f'Dla pierwszych 100 prób każdej komórki p=12 zastosowano wszystkie cele stopni i uporządkowanych różnic stopni. Łącznie wykonano {checks["runtime_lp_solves"]:,} celów LP i sprawdzono {checks["runtime_verified_simulation_certificates"]:,} poprawiających ograniczeń z niezależną arytmetyką całkowitą. Puste relacje i przypadki wymuszonego jednego drzewa mają znane ostre końce i pomijają zbędne LP. Porównania etapów korzystają z tych samych podzbiorów prób.'.replace(',',' '))
    p(f'Zapisane zastosowania rzeczywiste dostarczają {checks["saved_measurement_certificates"]:,} certyfikatów; po zakończeniu obliczeń ponownie sprawdzono ścieżki, ograniczenia dualne i odtworzono wszystkie poprawione końce stopni i rang. Poprzednie {109548:,} certyfikatów GPW zachowano z ich audytami.'.replace(',',' '))
    p('7. Lokalna rodzina porównań',heading)
    p('Dodano pełną rodzinę krawędzi i badanie na wspólnych próbach p=12 oraz wszystkich pomiarach rzeczywistych. Rodzina pełna dostarcza więcej potencjalnie przydatnych porównań, lecz ponosi większy koszt korekty. Stosunek liczby uporządkowanych hipotez wynosi (p+1)/4; przy p=52 to 132600 wobec 1756950. Wyniki pokazują trade-off zależny od modelu, zamiast zakładać dominację którejś rodziny. Domknięcie przechodnie lokalnych porządków nie odzyskuje z definicji całej informacji.')
    p('8. Styl i ekspozycja nowości',heading)
    p('Przepisano abstrakt i początek wstępu wokół pytania o rangę populacyjną, konstrukcji e-wartości oraz rzeczywistego pozytywnego wyniku. Ograniczenia znajdują się przy właściwych założeniach. Usunięto powtarzane formuły o prerejestracji i twierdzeniach niemożliwości. Tekst opisuje finalną metodę, zamiast odtwarzać rozmowę z recenzentem.')
    page('Drobne poprawki i dalsza ocena')
    p('Uwagi drobne',heading)
    for text in [
        'Warunek w Propozycji o mocy uproszczono do maksimum log(K/α) i log(2s/η). Dowód używa osobno warunku odrzucenia i sumy prawdopodobieństw złych zdarzeń.',
        'Zastąpiono szablon Springera neutralnym formatem artykułu; usunięto pluralizowany opis autora korespondencyjnego i poprawiono odstępy afiliacji.',
        'Bibliografię sortuje styl alfabetyczny. Nie zawiera tagu poprzedniego zgłoszenia ani opisu preceding version. Pochodzenie wejść GPW zachowano poprzez neutralny, niezmienny identyfikator kodu.',
        'Dodano literaturę o pełnych i niekompletnych U-statystykach, mnożnikach i łączeniu e-wartości. Odróżniono własną konstrukcję od klasycznych zasad wymiany i optymalizacji przedziałowej.',
        'Deklaracja AI podaje narzędzie, cel pomocy i odpowiedzialność autorów; metody opisują rzeczywisty zakres pomocy badawczej. Nie zawiera dokładnych dat użycia i nie ukrywa prac nad formułami.',
        'Przygotowano pięć Highlights o długości najwyżej 85 znaków, oddzielną stronę tytułową i manuskrypt bez danych autorów.'
    ]:p('• '+text)
    p('Co warto sprawdzić w ponownej recenzji',heading)
    p('Najważniejsza jest ocena, czy konstrukcja surowych kontrastów Kendalla, ważna agregacja dowodów, jej zgodność i sprawdzalna projekcja stanowią wystarczający własny wkład. Nie przypisujemy sobie zasad łączenia e-wartości ani klasycznych argumentów matroidowych. Potrzebna jest również ocena wiarygodności modelu IID dla zastosowania i wystarczającej użyteczności uzyskanych rang.')
    p('Pozostają jawne granice: bootstrap mnożnikowy nie ma tutaj gwarancji dokładnej kalibracji w każdej próbie; ceny GPW nie tworzą kompletnej licencjonowanej bazy; fizycznej niezależności okazów nie można dowieść z udostępnionych metadanych. Dokładne stwierdzenia populacyjne dotyczą procedur i modelu, które są wyraźnie wskazane. Near-tie ma lukę malejącą z n, więc zgodność dla ustalonej dodatniej luki nie obiecuje tam rozstrzygnięcia.')
    p('Kod i pełne wyniki',heading)
    p('<link href="https://github.com/mgalazka84/gpw-hub-reliability/tree/csda-revision-v0.5/contrast_inference" color="#155b8b">github.com/mgalazka84/gpw-hub-reliability - csda-revision-v0.5</link>',small)
    p('Weryfikacja końcowa: 446400 rekordów Monte Carlo; 27200 sparowanych rekordów etapu ścieżkowego; 118231 sprawdzonych prostych ścieżek w zapisanych zastosowaniach; wszystkie wymienione kontrole zakończone PASS.',small)
    SimpleDocTemplate(str(OUT/'REWIZJA_MERYTORYCZNA_PL.pdf'),pagesize=(595.28,841.89),leftMargin=48,rightMargin=48,topMargin=44,bottomMargin=44,
        title='Odpowiedź na recenzję i rewizja merytoryczna',author='Marek Gałązka i Hanna Wdowicka').build(story,onFirstPage=foot,onLaterPages=foot)
    details=[]
    def q(t,style=body):details.append(Paragraph(t,style))
    q('Title page',title);q('Finite-sample edge-order inference and certified degree comparisons in correlation spanning trees',heading)
    q('Marek Gałązka<super>1</super> and Hanna Wdowicka<super>2</super>',heading)
    q('<super>1</super> Faculty of Mathematics and Computer Science,<br/>Adam Mickiewicz University, Poznań, Poland.<br/>Email: galazka@amu.edu.pl')
    q('<super>2</super> Department of Statistics,<br/>Poznań University of Economics and Business, Poznań, Poland.<br/>Email: hanna.wdowicka@ue.poznan.pl')
    q('Corresponding author',heading);q('Marek Gałązka<br/>Email: galazka@amu.edu.pl')
    q('Funding',heading);q('No dedicated funding was received for this study.')
    q('Competing interests',heading);q('The authors declare no relevant competing interests.')
    SimpleDocTemplate(str(OUT/'CSDA_title_page.pdf'),pagesize=(595.28,841.89),leftMargin=48,rightMargin=48,topMargin=52,bottomMargin=48,
        title='Title page with author details',author='Marek Gałązka and Hanna Wdowicka').build(details)
    highlights=[
        'Exact edge-order tests cover discrete and continuous Kendall correlations.',
        'Evidence averaging over matchings retains finite-sample familywise control.',
        'Cycle-path bounds carry independently checked rational certificates.',
        'Matched simulations separate calibration, projection and family effects.',
        'Real morphometric measurements yield informative degree and rank bounds.'
    ];assert all(len(s)<=85 for s in highlights)
    (OUT/'Highlights.txt').write_text('\n'.join('- '+s for s in highlights)+'\n')
    print(json.dumps(dict(response_pdf=str(OUT/'REWIZJA_MERYTORYCZNA_PL.pdf'),title_pdf=str(OUT/'CSDA_title_page.pdf'),highlight_lengths=[len(s) for s in highlights])),flush=True)

if __name__=='__main__':run()
