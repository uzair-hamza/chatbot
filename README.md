# UET Peshawar FAQ Chatbot

I made this chatbot because new students at UET Peshawar ask the same
questions again and again: admissions, fees, hostel, library, exams, and
courses. I wanted a small bot that can answer those questions in simple
English, without needing a big server or a paid API.

## What it does

You type a question, the bot guesses what you are asking about (the
intent), and it replies with a short answer from my own intents file.
It covers 18 topics: greeting, goodbye, admissions, CSE courses,
exams/results, timetable, fees, hostel, library, scholarships, NCBC,
NAVTTC courses, campus facilities, transport, sports and societies,
internships, documents, and contact/help.

## My dataset

I wrote `intents.json` myself. It has 142 example questions in total
across the 18 intents, 6–12 patterns per intent, and 2–4 replies per
intent. The questions are written the way students actually ask them,
in simple English. I did not copy this file from a tutorial.

One honest limit: this is not real university data. The answers are
general guidance, not official notices. For exact dates, fees, and
merit lists, students should still check the official UET Peshawar
website or their department office.

## The model

TF-IDF (word n-grams 1–2) + Logistic Regression, using scikit-learn.
I picked this on purpose: the dataset is tiny, so a simple model trains
in seconds on CPU and is easy to understand. A neural network would be
overkill here and would probably just memorise 142 sentences.

## How to run it

Train (this also saves the model and the evaluation):

```bash
python src/chatbot.py
```

Chat in the terminal:

```bash
python src/predict.py
```

Type `quit` to exit.

Web demo:

```bash
pip install streamlit
streamlit run app.py
```

Note: streamlit was **not** installed in the venv I used for training,
so I could not test `app.py` by running it. The file is written and
ready, but you need to install streamlit first (it is in
`requirements.txt`).

## My results (measured, not guessed)

- Train/test split: stratified 80/20, random_state=42
- Train size: 113 patterns, test size: 29 patterns
- **Accuracy on the held-out test set: 0.6897 (68.97%)**
- Macro average F1: 0.64, weighted average F1: 0.67
- Full report: `results/eval.log`, confusion matrix: `results/confusion_matrix.png`

Per-class notes: the bot got everything right on campus facilities,
documents, hostel, library, NAVTTC courses, and sports/societies, but
each of those only had 1–2 test examples, so that does not prove much.
It scored 0 on contact_help, internships, and transport — again only
1 test example each. The test set is very small, so one question right
or wrong moves the accuracy a lot. I am reporting the number as it is,
not hiding that.

The demo chats in `demo_transcript.md` were all correct (5 out of 5),
but their confidences were low: 0.15 to 0.23. That is expected with
18 intents and a small dataset — the probability gets spread out, and
random guessing would only be about 0.06.

## What I tried, what failed, and what I learned

- **Failed first run:** I used `multi_class="auto"` in LogisticRegression,
  but scikit-learn 1.9 removed that argument, so training crashed. I
  removed it and training worked.
- **Failed demo first:** I set the fallback threshold to 0.30 confidence,
  and all 5 demo questions fell below it, so the bot answered "I am not
  sure" to everything even though its top guess was correct every time.
  I lowered the threshold to 0.10 and explained why in `src/predict.py`.
  The lesson: a threshold that looks safe on paper can break a small
  multi-class model in practice.
- **Streamlit:** not installed in the training venv, so `app.py` is
  untested. I wrote it, but I am not claiming it runs until someone
  installs streamlit and tries it.
- **What I learned:** with a tiny dataset, the data matters more than
  the model. More varied example questions per intent, especially for
  the intents that scored 0, would help more than tuning the classifier.
  Next step would be adding more patterns and testing on questions
  written by real students, not by me.
