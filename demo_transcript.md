# Demo Transcript

These 5 conversations were produced by actually running `src/predict.py`
on 2026-10-25. The replies are chosen at random from the matching intent's
responses in `intents.json`, so the same question can get a different
(but still correct) reply on another run.

## 1. Admissions

- **Input:** how can I apply for admission in UET Peshawar
- **Predicted intent:** admissions
- **Confidence:** 0.23
- **Bot reply:** Admissions at UET Peshawar usually open after the intermediate results, through an online application on the university website. You need your matric and FSc marks, domicile, CNIC or B-Form, and photographs. Merit is based on your academic marks and the entry test score, and it changes every year by department.

## 2. Library

- **Input:** what are the library timings
- **Predicted intent:** library
- **Confidence:** 0.17
- **Bot reply:** The central library is open on working days, usually from morning to evening, with shorter hours on Saturday. You can borrow books with your library card, and late returns carry a small fine per day.

## 3. Hostel

- **Input:** how can I apply for hostel
- **Predicted intent:** hostel
- **Confidence:** 0.15
- **Bot reply:** For hostel issues, contact your hostel warden or the Provost office. Mess timings and hostel rules are displayed in the hostel notice board, and students are expected to follow them.

## 4. NAVTTC courses

- **Input:** what are NAVTTC courses at UET Peshawar
- **Predicted intent:** navttc_courses
- **Confidence:** 0.22
- **Bot reply:** NAVTTC courses at UET Peshawar are short, skill-based training courses run with the National Vocational and Technical Training Commission. They cover practical technical skills, usually last a few months, and students who complete them get a certificate.

## 5. NCBC

- **Input:** what is NCBC at UET Peshawar
- **Predicted intent:** ncbc
- **Confidence:** 0.19
- **Bot reply:** NCBC is the National Center for Big Data and Cloud Computing at UET Peshawar. It works on big data, cloud computing, machine learning, and related research, and undergraduate students can work there as research assistants on live projects.

Note: the confidences look low, but that is normal here — with 18 intents,
random guessing would be about 0.06, and this small dataset spreads the
probability across similar intents. All 5 demo predictions were correct.
