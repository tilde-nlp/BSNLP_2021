# coding='utf-8'
# -*- coding: utf-8 -*-
import unittest
from Document import Document, Entity
import os

TEST_TXT = "testDoc.txt"
TEST_ANN = "testDoc.out"
CONLL_F = "testDoc.conll"

def prep_doc():
    with open(TEST_TXT, 'w', encoding='utf-8') as testDoc:
        testDoc.write("""bg-1
bg
2018-11-08
https://dnes.dir.bg/svyat/pakistanskiyat-sluchay-na-bogohulstvo-hristiyankata-asiya-bibi-e-osvobodena
Пакистанският случай на богохулство: християнката Асия Биби е освободена

Оправданата от пакистанския върховен съд пакистанска християнка Асия Биби, 8 години след като беше осъдена на смърт за богохулство и лежа в затвор, е била прехвърлена снощи на неизвестно място в столицата Исламабад от съображения за сигурност. Това заявиха двама високопоставени правителствени представители, цитирани от Асошиейтед прес и БТА. Каква ще е съдбата на "освободената" Биби. Асия Биби беше арестувана през 2009 г. по обвинения в обида към пророка Мохамед и оскверняване на исляма. Биби бе осъдена на смърт през 2010 г. според показанията на нейни съседки, че казала обидни думи по адрес на исляма, след като те не й дали да пие вода от тяхната чаша, защото не е мюсюлманка. Преди седмица Върховният съд на Пакистан обаче излезе с историческо решение, в което оневини Биби и нареди тя да бъде освободена. Това предизвика протести в цял Пакистан. Семейството на Биби винаги е твърдяло, че тя е невинна и че никога не е обиждала пророка Мохамед.

Вчера, седмица след решението на Върховния съд, бе съобщено, че Биби е освободена от затвора. Тя е на сигурно място на територията на страната, каза днес висш представител на властта, който отрече медийни твърдения, че жената е заминала за чужбина. Делото е много чувствително и министърът на информацията на Пакистан Фауад Хюсеин заяви, че журналистите са "изключително безотговорни", когато съобщават, че е напуснала страната без официално потвърждение.

""")
    with open(TEST_ANN, 'w', encoding='utf-8') as testDoc:
        testDoc.write("""bg-1
Асия Биби	Асия Биби	PER	PER-Asia-Bibi
Асошиейтед прес	Асошиейтед прес	ORG	ORG-AP-The-Associated-Press
БТА	БТА	ORG	ORG-Bulgarian-News-Agency
Биби	Биби	PER	PER-Asia-Bibi
Върховният съд на Пакистан	Върховен съд на Пакистан	ORG	ORG-Supreme-Court-of-Pakistan
Исламабад	Исламабад	LOC	GPE-Islamabad
Мохамед	Мохамед	PER	PER-Prophet-Muhammad
Пакистан	Пакистан	LOC	GPE-Pakistan
Фауад Хюсеин	Фауад Хюсеин	PER	PER-Fawad-Hussain
пакистанския върховен съд	Пакистански върховен съд	ORG	ORG-Supreme-Court-of-Pakistan""")

def cleanup_doc():
    try:
        os.remove(TEST_TXT)
        os.remove(TEST_ANN)
        os.remove(CONLL_F)
        
    except OSError:
        pass

class Test1(unittest.TestCase):
    def setUp(self):
        prep_doc()

    def tearDown(self):
        cleanup_doc()

    def test_text_load(self):
        doc = Document(TEST_TXT)
        self.assertEqual(doc.doc_id, 'bg-1')
        self.assertEqual(doc.doc_language, 'bg')
        self.assertEqual(doc.doc_creation_date, '2018-11-08')
        self.assertEqual(doc.doc_url, 'https://dnes.dir.bg/svyat/pakistanskiyat-sluchay-na-bogohulstvo-hristiyankata-asiya-bibi-e-osvobodena')
        self.assertEqual(doc.doc_title, 'Пакистанският случай на богохулство: християнката Асия Биби е освободена')
        self.assertEqual(len(doc.segmented_text_lines), 12) # 1 title + 11 text
        # with open("dump.txt", 'w', encoding="utf-8") as dump:
        #     dump.write("\n".join(doc.segmented_text_lines))

    def test_ann_load(self):
        doc = Document(TEST_TXT)
        doc.load_entities_from_ann(TEST_ANN)
        self.assertEqual(len(doc.entities), 10)
        self.assertEqual(doc.entities[4].text, "Върховният съд на Пакистан")
        self.assertEqual(doc.entities[4].lemma, "Върховен съд на Пакистан")
        self.assertEqual(doc.entities[4].type, "ORG")
        self.assertEqual(doc.entities[4].linked_name, "ORG-Supreme-Court-of-Pakistan")
        
    def test_save_conll(self):
        doc = Document(TEST_TXT)
        doc.load_entities_from_ann(TEST_ANN)
        IOBstr = doc.save_as_conll(CONLL_F)
        IOBstr_len = len(IOBstr.split('\n'))
        self.assertEqual(IOBstr_len, 274) # A bit arbitrary, depends on tokenization
                

if __name__ == "__main__":
    unittest.main()