from anyio import Event
import nltk
import pythainlp
from transformers import pipeline
from pythainlp import word_tokenize
from fastapi import FastAPI, Form, HTTPException

import random
import nltk
from pythainlp.tokenize import word_tokenize
from pythainlp.util import dict_trie
from pythainlp.corpus.common import thai_words
import pythainlp
from transformers import pipeline
from fastapi import FastAPI, Request, Response
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
import random
from pythainlp.tokenize import word_tokenize
import numpy as np

from linebot.models import MessageEvent, TextMessage, TextSendMessage
from linebot.exceptions import LineBotApiError, InvalidSignatureError
from linebot import LineBotApi, WebhookHandler
from fastapi.responses import JSONResponse
from starlette.status import HTTP_405_METHOD_NOT_ALLOWED


line_bot_api = LineBotApi("iG+0gpcSlIKhgLvq58cUNST9Ul7LEGSpPiKODOB4zYYubNNvO/nCnJjv8zX5kXojJ9fBrp+z9wqy1yX6byNI/vLx5tEsFPfcIXBzauv4F+RB5Kqja/1//0i3svByRztUkOzI8rinJ2l/JKh/g/y7IgdB04t89/1O/w1cDnyilFU=")
handler = WebhookHandler("6cc421b87c9df5ec33037076a056e15f")
# Download NLTK resources
nltk.download('punkt')

# Initialize FastAPI app
app = FastAPI()

# Load the MRC pipeline
mrcpipeline = pipeline("question-answering", model="MyMild/finetune_iapp_thaiqa")

university_rules = {
     "เกียรตินิยม": "เกียรตินิยม คือ ผู้ที่เรียนจนสำเร็จการศึกษา และไม่เคยสอบได้ D+, D หรือ F ในรายวิชาใดเลย และมีเกรดเฉลี่ยไม่ต่ำกว่า 3.25",
     "เกียรตินิยมอันดับ2" : "ต้องเรียนครบตามหลักสูตร 4 ปี ได้เกรดเฉลี่ยสะสม ตั้งแต่ 3.25 ขึ้นไป และไม่เคยสอบได้ D+, D หรือ F ในรายวิชาใดเลย",
     "เกียรตินิยมอันดับ1" : "ต้องเรียนครบตามหลักสูตร 4 ปี ได้เกรดเฉลี่ยสะสม ตั้งแต่ 3.60 ขึ้นไป และไม่เคยสอบได้ D+, D หรือ F ในรายวิชาใดเลย",
     "F เอฟ" : "ถ้าติด F นิสิตจะต้องลงทะเบียนเรียนรายวิชานั้นซ้ำอีก",
     "ตัดเกรด":"การให้ระดับคะแนนเป็นมี 8 ระดับ A, B+, B, C+, C, D+, D, F",
     "I ไอ": "I (Incomplete) หมายถึงการทำงานยังไม่เรียบร้อย",
     "W": "W (Withdrawn) หมายถึง การถอนรายวิชาในช่วงเวลาที่มหาวิทยาลัยกำหนด ซึ่งกรณีนี้จะไม่ได้รับค่าหน่วยกิตคืน ส่วนในระดับคะแนนอื่น ๆ เช่น Au (ลงทะเบียนเรียนรายวิชาโดยไม่นับหน่วยกิต(Audit))",
     "S": "S (ผ่านตามเกณฑ์ (Satisfactory))",
     "U": "U (ไม่ผ่านตามเกณฑ์(Unsatisfactory)) คะแนนที่ไม่ได้เป็นระดับขั้น และอาจจะใช้ในบางรายวิชาในระดับปริญญาโทหรือเอก",
     "MOU": "บริษัท คลิกเน็กซ์ จำกัด (ClickNext), ศูนย์เทคโนโลยีอิเล็กทรอนิกส์และคอมพิวเตอร์แห่งชาติ (NECTEC), บริษัท เอ็ม เอฟ อี ซี จำกัด (มหาชน) (MFEC), ซีดีจี และกลุ่มเครือบริษัทจีเอเบิล, INET - Internet Thailand Public Co., Ltd, เอโฮสต์, ทูเฟลโลส์, สำนักงานพัฒนาเทคโนโลยีอวกาศ, แกรนด์ลีนุกซ์, ทรู คอร์ปอเรชั่น, เจเนอรัล อิเลคทรอนิค, ซี.เอส.ไอ., อัฟวาแลนท์, เว็ลธ์ แมเนจเม้นท์, โมทีฟ เทคโนโลยี, พริซึ่มโซลูชั่นส์, สิริวัฒนาคอนสตรัคชั่น, ไดกิ้น อินดัสทรีส์, สยาม เด็นโซ่, สึบาคิโมโตะ, เพาว์เวอร์ ยูต้า, แวมสแตค, TOT, ห้องปฏิบัติการวิจัย MADI",
     "คอมพิวเตอร์ CS วิดคอม วิทย์คอม จบ":"นักวิจัยด้านวิทยาการคอมพิวเตอร์  และวิทยาศาสตร์เชิงคำนวณ, นักวิชาการคอมพิวเตอร์  (Computer  Technical  Officer), ผู้เขียนชุดคำสั่ง  (Programmer), ผู้เขียนชุดคำสั่งบนโทรศัพท์มือถือ  (Mobile  Programmer  IOS /Android), นักวิเคราะห์และออกแบบระบบ  (System  Analysis  and  Design), นักทดสอบโปรแกรมคอมพิวเตอร์  (Software  Tester), นักออกแบบระบบฐานข้อมูล  (Database  Designer), นักพัฒนาธุรกิจอัจฉริยะ  (Business  Intelligence  Developer), เจ้าหน้าที่ทำงานทางด้านการประยุกต์ใช้ปัญญาประดิษฐ์และข้อมูลดิจิทัลในองค์กร, วิศวกรข้อมูล  (Data  Engineer), นักวิเคราะห์ข้อมูล  (Data  Analysts)",
     "ค่า คอมพิวเตอร์ CS วิดคอม วิทย์คอม": "ค่าใช้จ่ายตลอดหลักสูตรแบบเหมาจ่าย 184,000 บาท (ภาคการศึกษาละ 23,000 บาท)",
     "อาจารย์ คอมพิวเตอร์ CS วิดคอม วิทย์คอม": "ผศ.ดร.โกเมศ อัมพวัน, ดร.พิเชษ วะยะลุน, อาจารย์ภูสิต กุลเกษม, อาจารย์เบญจภรณ์ จันทรกองกุล, อาจารย์วรวิทย์ วีระพันธุ์, อาจารย์จรรยา อ้นปันส์",
     "ปริญญา CS วิทยาการคอมพิวเตอร์ วิทคอม": "ภาษาไทย: วิทยาศาสตรบัณฑิต (วิทยาการคอมพิวเตอร์), ชื่อย่อภาษาไทย: วท.บ. (วิทยาการคอมพิวเตอร์), ภาษาอังกฤษ: Bachelor of Science (Computer Science), ชื่อย่อภาษาอังกฤษ: B.Sc. (Computer Science)",
     "CS วิทยาการคอมพิวเตอร์ วิทคอม คือ เกี่ยวกับ อธิบาย": "วิทยาการคอมพิวเตอร์ (Computer Science: CS) เป็นศาสตร์เกี่ยวกับการศึกษาค้นคว้าทฤษฏีการคำนวณสำหรับคอมพิวเตอร์ และทฤษฏีการประมวลผลสารสนเทศ ทั้งด้านซอฟต์แวร์ ฮาร์ดแวร์ และ เครือข่าย ประกอบด้วยหลายหัวข้อที่เกี่ยวข้อง เช่น การวิเคราะห์และสังเคราะห์ขั้นตอนวิธี ทฤษฎีภาษาโปรแกรม ทฤษฏีการพัฒนาซอฟต์แวร์ ทฤษฎีฮาร์ดแวร์คอมพิวเตอร์ และทฤษฏีเครือข่าย",
     "หลักสูตร CS วิทยาการคอมพิวเตอร์ วิทคอม":"เป็นหลักสูตรระดับปริญญาตรี หลักสูตร 4 ปี และ ภาษาที่ใช้จัดการเรียนการสอนเป็นภาษาไทย",
     "ภาษา สอน": "ภาษาที่ใช้จัดการเรียนการสอนเป็นภาษาไทย",
     "หน่วยกิต CS วิทยาการคอมพิวเตอร์ วิทคอม": "มีทั้งหมด 123 หน่วยกิต ประกอบด้วย 1) หมวดวิชาศึกษาทั่วไป จำนวน 30 หน่วยกิต 2) หมวดวิชาเฉพาะ จำนวน 81 หน่วยกิต 3) หมวดวิชาเลือกเสรี ไม่น้อยกว่า 6 หน่วยกิต 4) หมวดวิชาประสบการณภาคสนาม 6 หน่วยกิต อ้างอิงจากหลักสูตร 65",
     "ปริญญา ITDI ไอที เทคโนโลยีสารสนเทศเพื่ออุตสาหกรรมดิจิทัล": "ภาษาไทย: วิทยาศาสตรบัณฑิต (เทคโนโลยีสารสนเทศเพื่ออุตสาหกรรมดิจิทัล) ชื่อย่อภาษาไทย: วท.บ. (เทคโนโลยีสารสนเทศเพื่ออุตสาหกรรมดิจิทัล) ภาษาอังกฤษ: Bachelor of Science (Information Technology for Digital Industry) ชื่อย่อภาษาอังกฤษ: B.Sc. (Information Technology for Digital Industry)",
     "อาจารย์ ITDI ไอที เทคโนโลยีสารสนเทศเพื่ออุตสาหกรรมดิจิทัล": "ผศ.ดร.อังศุมาลี สุทธภักติ, ผศ.ดร.ประจักษ์ จิตเงินมะดัน, ดร. คนึงนิจ กุโบลา, อาจารย์วิทวัส พันธุมจินดา, อาจารย์เหมรัศมิ์ วชิรหัตถพงศ์ ผศ.เอกภพ บุญเพ็ง",
     "หลักสูตร  ITDI ไอที เทคโนโลยีสารสนเทศเพื่ออุตสาหกรรมดิจิทัล":"เป็นหลักสูตรระดับปริญญาตรี หลักสูตร 4 ปี และ ภาษาที่ใช้จัดการเรียนการสอนเป็นภาษาไทย",
     "อธิบาย เกี่ยวกับ คือ  ITDI ไอที เทคโนโลยีสารสนเทศเพื่ออุตสาหกรรมดิจิทัล": "เทคโนโลยีสารสนเทศเพื่ออุตสาหกรรมดิจิทัล (Information Technology for Digital Industry : ITDI) เป็นศาสตร์เกี่ยวกับการประยุกต์ใช้เทคโนโลยีในการประมวลผลสารสนเทศ ซึ่งครอบคลุมถึงการรับ-ส่ง การแปลง การจัดเก็บ การประมวลผล และการค้นคืนสารสนเทศ เป็นการประยุกต์ใช้ทฤษฎีและขั้นตอนวิธีจากวิทยาการคอมพิวเตอร์ในการทำงาน การศึกษาอุปกรณ์ต่าง ๆ ทางเทคโนโลยีสารสนเทศ การวางโครงสร้างสถาปัตยกรรมองค์กรด้วยเทคโนโลยีสารสนเทศอย่างมีประสิทธิภาพสูงสุดกับสังคม ธุรกิจ องค์กร หรืออุตสาหกรรม",
     "หน่วยกิต ITDI ไอที เทคโนโลยีสารสนเทศเพื่ออุตสาหกรรมดิจิทัล":"มีทั้งหมด 123 หน่วยกิต ประกอบด้วย 1) หมวดวิชาศึกษาทั่วไป จำนวน 30 หน่วยกิต 2) หมวดวิชาเฉพาะ จำนวน 87 หน่วยกิต 3) หมวดวิชาเลือกเสรี ไม่น้อยกว่า 6 หน่วยกิต",
     "ITDI ไอที เทคโนโลยีสารสนเทศเพื่ออุตสาหกรรมดิจิทัล อาชีพ จบทำงาน": "นักวิเคราะห์และออกแบบระบบงานสารสนเทศ (Information System Analyst) นักวิชาการคอมพิวเตอร์ (Computer Technical Officer) ผู้ดูแลระบบเครือข่ายและเครื่องแม่ข่าย (System Administrator) นักออกแบบและพัฒนาเว็บไซต์และระบบสารสนเทศ (Web Developer) นักออกแบบและพัฒนาซอฟต์แวร์ (Software Designer and Developer) นักออกแบบและพัฒนาสื่อมัลติมีเดียเชิงโต้ตอบ (Interactive Media Creator) นักออกแบบและพัฒนาส่วนติดต่อผู้ใช้งานเชิงโต้ตอบ (Interactive User Interface Designer) นักออกแบบประสบการณ์ผู้ใช้งาน (User Experience Designer) ผู้ประกอบการที่ใช้เทคโนโลยีดิจิทัลเป็นฐาน (Digital Technology Startup)",
     "ค่า ITDI ไอที เทคโนโลยีสารสนเทศเพื่ออุตสาหกรรมดิจิทัล": "ค่าใช้จ่ายตลอดหลักสูตรแบบเหมาจ่าย 184,000 บาท (ภาคการศึกษาละ 23,000 บาท)",
     "ปริญญา SE วิศวกรรมซอฟต์แวร์ ":"ภาษาไทย: วิทยาศาสตรบัณฑิต (วิศวกรรมซอฟต์แวร์) ชื่อย่อภาษาไทย: วท.บ. (วิศวกรรมซอฟต์แวร์) ชื่อภาษาอังกฤษ: Bachelor of Science (Software Engineering) ชื่อย่อภาษาอังกฤษ: B.Sc. (Software Engineering)",
     "อธิบาย เกี่ยวกับ คือ SE วิศวกรรมซอฟต์แวร์ ":"วิศวกรรมซอฟต์แวร์ (Software Engineering: SE) เป็นศาสตร์เกี่ยวกับวิศวกรรมด้านซอฟต์แวร์ เกี่ยวข้องกับการใช้กระบวนการทางวิศวกรรมในการดูแลการผลิตซอฟต์แวร์ที่สามารถปฏิบัติงานตามเป้าหมาย ภายใต้เงื่อนไขที่กำหนด โดยเริ่มตั้งแต่การเก็บความต้องการ การตั้งเป้าหมายของระบบ การออกแบบ กระบวนการพัฒนา การตรวจสอบ การประเมินผล การติดตามโครงการ การประเมินต้นทุน การรักษาความปลอดภัย ไปจนถึงการคิดราคาซอฟต์แวร์ เป็นต้น",
     "SE วิศวกรรมซอฟต์แวร์ อาชีพ จบทำงาน": "วิศวกรรมซอฟต์แวร์ (Software Engineer) หรือนักเขียนโปรแกรม (Programmer/Developer) วิศวกรความต้องการ  (Requirement Engineer) นักประกันคุณภาพซอฟต์แวร์ (Software Quality Assurance) วิศวกรปรับปรุงกระบวนการซอฟต์แวร์ (Software Process Improvement Engineer) นักทดสอบระบบ (Software Tester) นักบูรณาการระบบ (System Integrator) นักวิเคราะห์ระบบหรือนักออกแบบระบบ (System Analyst/Designer) ผู้จัดการโครงการซอฟต์แวร์ (Software Project Manager)",
     "MOU เครือข่าย SE วิศวกรรมซอฟต์แวร์ ": "บริษัท เอ็กซ์เวนชั่น จำกัด, บริษัท เอ - โฮสต์ จำกัด, บริษัท คลิกเน็กซ์ จำกัด (ClickNext) บริษัท สยาม เด็นโซ่ แมนูแฟคเจอริ่ง จำกัด, บริษัท ซอฟต์สแควร์ อินเตอร์เนชั่นแนล จํากัด, บริษัท ไอวี ซอฟต์ จำกัด, บริษัท ทีทีทีบราเธอร์ส จำกัด, ศูนย์เทคโนโลยีอิเล็กทรอนิกส์และคอมพิวเตอร์แห่งชาติ (NECTEC), เขตอุตสาหกรรมซอฟต์แวร์ภาคตะวันออก (Eastern Software Park), บริษัท กรีน ฮับ จำกัด, บริษัท ไอเอ็มซี เอ้าท์ซอร์สซิ่ง (ประเทศไทย) จํากัด, บริษัท อะเฮด จำกัด, บริษัท เน็ตก้า ซิสเต็ม จำกัด, บริษัท บิทคับ ออนไลน์ จำกัด ซีดีจี และกลุ่มเครือบริษัทจีเอเบิล, บริษัท ซี.เอส.ไอ. (ประเทศไทย) จำกัด, บริษัท เอ็ม เอฟ อี ซี จำกัด (มหาชน) (MFEC) ห้องปฏิบัติการวิจัยวิศวกรรมระบบสารสนเทศ (ISERL), บริษัท แคนส์คอมมิวนิเคชั่น จำกัด, บริษัท นิว ดอน จำกัด บริษัท พายซอฟท์ จำกัด (PieSoft), บริษัท เอนี่ไอ จำกัด, บริษัท คิว คอนซัลติ้ง จำกัด INET - Internet Thailand Public Co., Ltd, Advanced Info Service Public Company Limited, บริษัท กสิกร เทคโนโลยี กรุ๊ป เซเครเทเรียต จำกั (KTBG), บริษัท ล็อกซเล่ย์ ออบิท จำกัด (มหาชน), บริษัท เว็ลธ์ แมเนจเม้นท์ ซิสเท็ม จำกัด, บริษัท ไอโคเน็กซ์ จำกัด, บริษัท PRISM จำกัด, บริษัท แบ็กซ์เตอร์ เมนูแฟคเจอริ่ง (ประเทศไทย) จํากัด, บริษัท สยามคอมเพรสเซอร์อุตสาหกรรม จำกัด, บริษัท ซีล เทค อินเตอร์เนชั่นแนล จำกัด, บริษัท ไทรอัมพ์ เอวิเอชั่น เซอร์วิสเซส เอเชีย จำกัด, บริษัท คอมพิวเตอร์โลจี จำกัด, บริษัท ไทยโพลีคาร์บอเนต จำกัด, บริษัท เคหกสี่ เอ็นจิเนียริ่ง จำกัด",
     "ค่า SE วิศวกรรมซอฟต์แวร์ ": "ค่าใช้จ่ายตลอดหลักสูตรแบบเหมาจ่าย 184,000 บาท (ภาคการศึกษาละ 23,000 บาท)",
     "อาจารย์ประจำ SE วิศวกรรมซอฟต์แวร์ " : "ผศ.พีระศักดิ์ เพียรประสิทธิ์,ดร.ณัฐพร ภักดี,ดร.อธิตา อ่อนเอื้อน,อาจารย์วันทนา ศรีสมบูรณ์,อาจารย์อภิสิทธิ์ แสงใส",
     "ปริญญา AI เอไอ ปัญญาประดิษฐ์ประยุกต์และเทคโนโลยีอัจฉริยะ": "ภาษาไทย: วิทยาศาสตรบัณฑิต (ปัญญาประดิษฐ์ประยุกต์และเทคโนโลยีอัจฉริยะ), ชื่อย่อภาษาไทย: วท.บ. (ปัญญาประดิษฐ์ประยุกต์และเทคโนโลยีอัจฉริยะ), ภาษาอังกฤษ: Bachelor of Science (Applied Artificial Intelligence and Smart Technology), ชื่อย่อภาษาอังกฤษ: B.Sc. (Applied Artificial Intelligence and Smart Technology)",
     "อธิบาย เกี่ยวกับ คือ AI เอไอ ปัญญาประดิษฐ์ประยุกต์และเทคโนโลยีอัจฉริยะ": "ปัญญาประดิษฐ์ประยุกต์และเทคโนโลยีอัจฉริยะ (Applied Artificial Intelligence and Smart Technology: AAI) ได้พัฒนาขึ้นเพื่อตอบสนองต่อความต้องการบุคลากรด้านเทคโนโลยีสารสนเทศเพื่อการเปลี่ยนรูปองค์การไปสู่องค์กรอัจฉริยะที่ขับเคลื่อนด้วยข้อมูล (Data-driven Business) บนพื้นฐานของเทคโนโลยีปัญญาประดิษฐ์ ตลอดถึงการพัฒนากำลังคนในธุรกิจดิจิทัล และระบบอัจฉริยะ เช่น โรงงานอัจฉริยะ (Smart Factory),เกษตรอัจฉริยะ (Smart Agriculture), ฟาร์มอัจฉริยะ (Smart Farming), เมืองอัจฉริยะ (Smart City),การบริการอัจฉริยะ (Smart Services), การท่องเที่ยวอัจฉริยะ (Smart Tourisms) และ โลจิสติกส์อัจฉริยะ (Smart Logistrics) สอดคล้องกับโครงการเขตพัฒนาพิเศษภาคตะวันออก (EEC) ภายใต้แผนยุทธศาสตร์ประเทศไทย 4.0",
     "หลักสูตร AI เอไอ ปัญญาประดิษฐ์ประยุกต์และเทคโนโลยีอัจฉริยะ": "เป็นหลักสูตรระดับปริญญาตรี หลักสูตร 4 ปี และ ภาษาที่ใช้จัดการเรียนการสอนเป็นภาษาไทย",
     "หน่วยกิต AAI AI เอไอ ปัญญาประดิษฐ์ประยุกต์และเทคโนโลยีอัจฉริยะ": "มีทั้งหมด 121 หน่วยกิต ประกอบด้วย 1) หมวดวิชาศึกษาทั่วไป ไม่น้อยกว่า จำนวน 24 หน่วยกิต 2) หมวดวิชาเฉพาะ จำนวน 91 หน่วยกิต  3) หมวดวิชาเลือกเสรี ไม่น้อยกว่า 6 หน่วยกิต (หลักสูตร 2567)",
     "ค่า AI เอไอ ปัญญาประดิษฐ์ประยุกต์และเทคโนโลยีอัจฉริยะ":"สาขาวิชาปัญญาประดิษฐ์ประยุกต์และเทคโนโลยีอัจฉริยะ (4 ปี) ภาคต้นและภาคปลาย ภาคเรียนละ 25,000 บาท ภาคฤดูร้อน ภาคเรียนละ 12,500 บาท",
     "อาจารย์ประจำ AI เอไอ ปัญญาประดิษฐ์ประยุกต์และเทคโนโลยีอัจฉริยะ":"ผศ.ดร. สุภาวดี ศรีคำดี,อาจารย์ประวิทย์ บุญมี,รศ.ดร. กฤษณะ ชินสาร,อาจารย์จอห์น เกตวูต แฮม,ดร.วัชรพงศ์ อยู่ขวัญ,อาจารย์พลวัต ช่อผูก",
     "คณะวิทยาการสารสนเทศ สาขา":"คณะวิทยาการสารสนเทศมี 4 สาขา ประกอบด้วย สาขาวิชาวิทยาการคอมพิวเตอร์, สาขาวิชาเทคโนโลยีสารสนเทศเพื่ออุตสาหกรรมดิจิทัล, สาขาวิชาวิศวกรรมซอฟต์แวร์ และสาขาวิชาปัญญาประดิษฐ์ประยุกต์และเทคโนโลยีอัจฉริยะ",
     "ผลงาน รางวัล CS  วิทยาการคอมพิวเตอร์ วิทคอม":"ผลงานสาขา CS รางวัลที่ 1 ประเภทโปรแกรมเพื่อการประยุกต์ใช้งานบนเครือข่ายสำหรับอุปกรณ์คอมพิวเตอร์เคลื่อนที่ (Mobile Application) (นิสิต นักศึกษา) จากการแข่งขันพัฒนาโปรแกรมคอมพิวเตอร์แห่งประเทศไทย ครั้งที่ 21 (NSC-2019)และได้เข้ารับพระราชทานเกียรติบัตร และถ้วยรางวัล จากสมเด็จพระเทพรัตนราชสุดาฯ สยามบรมราชกุมารี รางวัลการนำเสนอผลงานแบบบรรยาย (ORAL) การประชุมวิชาการปริญญาตรีด้านคอมพิวเตอร์ภูมิภาคอาเซียน (AUCC 2019), Proceedings Click Click: The Automatic International Conference Proceedings Management System",
     "สวัสดี Hi Hello หวัดดี ดีจ้า" : "สวัสดีครับ/ค่ะ นี่คือ InformaticsBot ที่จะช่วยตอบคำถามเกี่ยวกับคณะวิทยาการสารสนเทศ จัดทำโดยนิสิตสาขาวิทยการคอมพิมเตอร์ชั้นปีที่ 3  Hello there This is InformationBot that'll help to answer question about our faculty",
     "ผลงาน รางวัล SE วิศวกรรมซอฟต์แวร์": "ผลงานสาขา SE รางวัลการนำเสนอผลงานแบบบรรยาย (ORAL) จำนวน 9 ผลงาน ในการประชุมวิชาการ The 11th ASEAN Undergraduate Conference In Computing (AUCC) 2022 ได้รับรางวัล Excellent จำนวน 1 ผลงาน จากนิสิตในที่ปรึกษาของอาจารย์อภิสิทธิ์ แสงใส เรื่อง Identity Server System ระบบยืนยันตัวตนและกำหนดสิทธิ์การใช้งานผ่านเครือข่ายกลาง นอกจากนี้ยังมีรางวัล Very good จำนวน 3 รางวัล รางวัล Good จำนวน 3 รางวัล และ participant จำนวน 2 รางวัล นิสิตสาขาวิชาวิศวกรรมซอฟต์แวร์ได้รับทุนส่งเสริมการศึกษา นิสิตสหกิจศึกษาดีเด่น ที่เข้าปฎิบัติงานสหกิจศึกษากับทางบริษัท คลิกเน็กซ์ จำกัด รางวัลทุนโครงการการแข่งขันพัฒนาโปรแกรมคอมพิวเตอร์แห่งประเทศไทย ครั้งที่ 24 (NSC 2022) “โครงการไก่กังวล” ที่ผ่านรับทุนรอบนำเสนอผลงาน รางวัลการนำเสนอผลงานแบบบรรยาย (ORAL) จำนวน 6 ผลงาน ในการประชุมวิชาการ The 10th ASEAN Undergraduate Conference In Computing (AUCC) 2022 นิสิตสาขาวิชาวิศวกรรมซอฟต์แวร์ นางสาวลลิตา เฉินจุณวรรณ ได้รับทุนส่งเสริมการศึกษาดีเด่น จากบริษัท คลิกเน็กซ์ จำกัด ทุนละ 20,000 บาท (สองหมื่นบาทถ้วน) รางวัลการนำเสนอผลงานแบบบรรยาย (ORAL) จำนวน 3 ผลงาน ในการประชุมวิชาการ The 9th ASEAN Undergraduate Conference In Computing (AUCC) 2021 รางวัลการนำเสนอผลงานแบบบรรยาย (ORAL) จำนวน 20 ผลงาน แบบ Poster 3 ผลงาน ในการประชุมวิชาการ The 7th ASEAN Undergraduate Conference In Computing (AUC2) 2019 นิสิตสาขาวิชาวิศวกรรมซอฟต์แวร์ได้รับทุนในโครงการ  Sakura Science Program จากประเทศญี่ปุ่น จำนวน 2 คน รางวัล Cloud Credit ในการแข่งขัน Chiangmai HackaTrain 2019 รางวัลชนะเลิศ และรองชนะเลิศจากค่ายอบรมเชิงปฏิบัติการ Thailand Networking Group ครั้งที่ 8 นิสิตสาขาวิชาวิศวกรรมซอฟต์แวร์จำนวน 106 คน ได้รับประกาศนียบัตรระดับสากล หลักสูตร compTIA A+",
     "ขอบคุณ THANKS THX": "ขอบคุณครับ/ค่ะ ที่ใช้งาน InformaticsBot, Thanks you for using our InformaticsBot"
}

# Preprocess university rules keys for better matching
preprocessed_university_rules = {pythainlp.util.normalize(keyword): answer for keyword, answer in university_rules.items()}

# Create context from all answers in university_rules
university_context = ' '.join(preprocessed_university_rules.values())

# Function to handle university questions
def handle_university_question(question: str) -> str:
    question = question.upper()
    matched_key = None
    max_matched_tokens = 0
    
    if question in university_rules:
        return university_rules[question]
    else:
        # Tokenize the question
        tokens = word_tokenize(question)
        # Check for substring matches
        for key in university_rules:
            matched_tokens = sum(1 for token in tokens if token in key)
            if matched_tokens > max_matched_tokens:
                max_matched_tokens = matched_tokens
                matched_key = key
        
        # If a matching key is found, return its corresponding value
        if matched_key:
            return university_rules[matched_key]
        
        # If no exact or substring match is found, use the model
        answer = mrcpipeline(question=question, context=university_context)
        return answer['answer']

    
@app.get("/")  # Handles GET requests for verification
async def verify_line_webhook(request: Request):
    print(request)
    # LINE sends a verification challenge as a query parameter
    challenge = request.query_params.get("hub.challenge")
    if challenge:
        # Echo the challenge back in the response
        return JSONResponse(content={"challenge": challenge}, status_code=200)
    else:
        # No challenge found, return an appropriate response
        return JSONResponse(content={"message": "Challenge not found"}, status_code=400)

@app.post("/webhook")
async def line_webhook(request: Request):
    # Get request body as text
    body = await request.body()
    body_text = body.decode("utf-8")

    # Get X-Line-Signature header value
    signature = request.headers["X-Line-Signature"]

    # Handle the webhook body
    try:
        handler.handle(body_text, signature)
    except InvalidSignatureError:
        return {"error": "Invalid signature."}
    except LineBotApiError as e:
        return {"error": str(e)}
    return "OK"

@app.post("/university")
def university_chatbot(message: str = Form(...)):
    # Check if message is empty or None
    if not message:
        raise HTTPException(status_code=400, detail="Empty message. Please provide a valid question.")
    
    # Use the function to answer the question
    answer = handle_university_question(message)
    
    return {"answer": answer}

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_message = event.message.text

    # Check if message is empty or None
    if not user_message:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="Empty message. Please provide a valid question.")
        )
        return

    # Use the function to answer the question
    answer = handle_university_question(user_message)

    # Respond to the user
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=answer)
    )
