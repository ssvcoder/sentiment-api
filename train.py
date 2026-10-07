#!/usr/bin/env python3
"""Train a sentiment classification pipeline and save it to model.pkl.

Pipeline: TF-IDF (word unigrams + bigrams) -> Logistic Regression.
Dataset: 72 hand-labeled product/movie reviews, balanced positive/negative.

Usage:
    python train.py [--test-size 0.2] [--seed 42] [--model-out model.pkl]
"""

from __future__ import annotations

import argparse
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Labeled training data: (text, label) with label 1 = positive, 0 = negative.
# ---------------------------------------------------------------------------
DATA: list[tuple[str, int]] = [
    # ---- positive ----
    ("This movie was an absolute delight from start to finish, with brilliant performances and a clever script.", 1),
    ("I loved every minute of it; the story kept me hooked and the ending was perfect.", 1),
    ("A masterpiece of modern cinema - beautiful cinematography and a moving score.", 1),
    ("The headphones I bought exceeded my expectations; the sound quality is outstanding.", 1),
    ("Fantastic customer service, my order arrived early and everything was exactly as described.", 1),
    ("This blender works like a dream and crushes ice with no effort at all.", 1),
    ("An inspiring and heartfelt film that left the whole theater applauding.", 1),
    ("The plot twists were brilliant and the acting felt so genuine.", 1),
    ("I was smiling the entire time; easily one of the best comedies I have seen in years.", 1),
    ("Superb build quality on this laptop, fast, lightweight, and the battery lasts all day.", 1),
    ("The restaurant's new menu is wonderful - every dish was fresh and full of flavor.", 1),
    ("A gripping thriller with smart writing and edge-of-your-seat suspense.", 1),
    ("These shoes are incredibly comfortable; I walked all day with zero pain.", 1),
    ("The camera takes stunning photos even in low light. Totally worth the price.", 1),
    ("Charming, funny, and beautifully animated - my kids ask to watch it every weekend.", 1),
    ("The book was impossible to put down; the author writes with such warmth and wit.", 1),
    ("Excellent value for money. It does everything advertised and more.", 1),
    ("The director crafted a visually stunning world with characters I truly cared about.", 1),
    ("My skin has never looked better since I started using this moisturizer.", 1),
    ("A powerful documentary that is both informative and deeply emotional.", 1),
    ("The keyboard feels amazing to type on and the backlighting looks great at night.", 1),
    ("Warm, funny, and uplifting - the perfect feel-good movie for a rainy day.", 1),
    ("Delivery was fast and the packaging was thoughtful and eco-friendly.", 1),
    ("The vacuum cleaner picks up everything, even pet hair, and it is so quiet.", 1),
    ("A beautifully told story with outstanding performances from the entire cast.", 1),
    ("This app is intuitive and fast; it saved me hours of work this week.", 1),
    ("The coffee maker brews a rich, flavorful cup every single morning.", 1),
    ("I laughed, I cried - a rare film that truly has it all.", 1),
    ("Great fit, great fabric, and it arrived sooner than expected. Highly recommend.", 1),
    ("The soundtrack alone is worth the price of admission; the film is a triumph.", 1),
    ("Reliable, sturdy, and easy to assemble - exactly what I needed.", 1),
    ("A delightful romantic comedy with real chemistry between the leads.", 1),
    ("The smartwatch tracks everything accurately and the screen is gorgeous.", 1),
    ("Thought-provoking and beautifully shot, with a haunting final scene.", 1),
    ("Best purchase I have made this year, no regrets whatsoever.", 1),
    ("The sequel is even better than the original - sharper, funnier, bolder.", 1),
    # ---- negative ----
    ("This was the worst movie I have ever seen; boring, predictable, and painfully long.", 0),
    ("I hated every minute of it - terrible acting and a script full of holes.", 0),
    ("A complete waste of time and money. I walked out halfway through.", 0),
    ("The headphones broke after two days; the sound is tinny and awful.", 0),
    ("Terrible customer service - rude staff and my refund still has not arrived.", 0),
    ("This blender barely crushes anything and smells like burning plastic.", 0),
    ("A dull, lifeless film with wooden dialogue and zero emotional impact.", 0),
    ("The plot made no sense and the characters were impossible to care about.", 0),
    ("I have never been so bored in a theater; I kept checking my watch.", 0),
    ("Poor build quality on this laptop - it overheats and the screen flickers.", 0),
    ("The food was cold, bland, and overpriced. I will not be coming back.", 0),
    ("A mess of a thriller with a confusing plot and laughable twists.", 0),
    ("These shoes gave me blisters within an hour. Total disappointment.", 0),
    ("The camera is a disaster in low light and the battery dies in an hour.", 0),
    ("Ugly animation, weak jokes, and a story that goes nowhere.", 0),
    ("The book dragged on forever; I gave up after a hundred dull pages.", 0),
    ("Overpriced junk. It stopped working the same week I bought it.", 0),
    ("Visually muddy and narratively empty - two hours I will never get back.", 0),
    ("This moisturizer irritated my skin and caused an awful rash.", 0),
    ("A shallow documentary that repeats the same point for ninety minutes.", 0),
    ("Keys stick constantly and half the backlight LEDs are already dead.", 0),
    ("Depressing, slow, and pretentious - the opposite of entertainment.", 0),
    ("Delivery took three weeks and the box arrived crushed and open.", 0),
    ("The vacuum is loud, heavy, and barely picks up anything.", 0),
    ("Forgettable performances in a story that feels stitched together.", 0),
    ("This app crashes constantly and the interface is a confusing nightmare.", 0),
    ("The coffee tastes burnt no matter what beans I use. Avoid.", 0),
    ("I cringed through the whole thing - unfunny and painfully awkward.", 0),
    ("Shrank two sizes after one wash and the color faded immediately.", 0),
    ("An obnoxious soundtrack over an empty, style-over-substance film.", 0),
    ("Flimsy, wobbly, and missing screws - a nightmare to assemble.", 0),
    ("No chemistry between the leads and jokes that fall completely flat.", 0),
    ("The watch constantly loses sync and the battery barely lasts a day.", 0),
    ("Pretentious and hollow, with an ending that insults the audience.", 0),
    ("Regret buying this more than anything I have purchased in years.", 0),
    ("The sequel ruins everything the original built - lazy and cynical.", 0),
    # ---- additional positive (vocabulary overlap for generalization) ----
    ("An excellent film with a wonderful cast and a story that stays with you.", 1),
    ("Amazing performance by the lead actor; truly one of the greats.", 1),
    ("I absolutely love this phone - the camera is amazing and it is so fast.", 1),
    ("The food was fantastic and the service was warm and attentive.", 1),
    ("Great movie! I had a wonderful time from beginning to end.", 1),
    ("This is the best coffee I have ever tasted, rich and smooth.", 1),
    ("Perfect in every way. I could not be happier with this purchase.", 1),
    ("Awesome product, works exactly as promised and looks great.", 1),
    ("Brilliant writing and outstanding acting make this a must-see.", 1),
    ("I am so pleased with how easy this was to set up and use.", 1),
    ("Highly recommend this to anyone looking for quality and value.", 1),
    ("Beautiful design and delightful details throughout the whole experience.", 1),
    ("Such a fun and exciting adventure, perfect for the entire family.", 1),
    ("Comfortable, durable, and stylish - I love these boots.", 1),
    ("Fast shipping and excellent packaging, everything arrived in perfect condition.", 1),
    ("The concert was amazing, the band played with so much energy.", 1),
    ("Wonderful flavors and generous portions; we will definitely return.", 1),
    ("This software is excellent - clean interface and very reliable.", 1),
    ("I love how quiet and powerful this machine is.", 1),
    ("A fantastic sequel that is even better than the first one.", 1),
    ("Great value, superb quality, and outstanding customer support.", 1),
    ("The hotel staff were wonderful and the room was beautiful.", 1),
    ("Amazing views and a delightful stay from check-in to check-out.", 1),
    ("This game is so much fun, with excellent graphics and gameplay.", 1),
    ("I am delighted with the results; it works better than expected.", 1),
    ("Perfect gift - beautifully wrapped and arrived right on time.", 1),
    ("The battery life is excellent and it charges incredibly fast.", 1),
    ("Such an enjoyable evening, with great music and great company.", 1),
    ("Impressive attention to detail and superb craftsmanship.", 1),
    ("Love the color, love the fit, love everything about it.", 1),
    ("The plot was excellent and the ending was deeply satisfying.", 1),
    ("A wonderful experience I would happily repeat any time.", 1),
    ("Fast, friendly service and fantastic food - five stars.", 1),
    ("This novel is brilliant, with wonderful characters and prose.", 1),
    ("Excellent sound, comfortable fit, and a very fair price.", 1),
    ("I had an amazing time and would recommend it to everyone.", 1),
    # ---- additional negative (vocabulary overlap for generalization) ----
    ("An awful film with terrible acting and a boring story.", 0),
    ("Horrible experience; the staff were rude and unhelpful.", 0),
    ("I hate this phone - it is slow, buggy, and the battery is awful.", 0),
    ("The food was disgusting and the service was incredibly slow.", 0),
    ("Boring movie with a predictable plot and flat characters.", 0),
    ("This is the worst restaurant I have ever visited. Avoid it.", 0),
    ("Poor quality and terrible design. A complete waste of money.", 0),
    ("Useless product, broke within days and customer support was awful.", 0),
    ("I was so disappointed - dull, slow, and totally forgettable.", 0),
    ("The worst purchase I have ever made. I regret it completely.", 0),
    ("Bland food, rude waiters, and absurd prices. Never again.", 0),
    ("A pathetic excuse for a comedy - not a single funny moment.", 0),
    ("Cheap materials and faulty wiring; it stopped working in a week.", 0),
    ("Dreadful service and a dirty room. I checked out early.", 0),
    ("The game is boring, buggy, and not worth half the price.", 0),
    ("I hate how complicated and unreliable this software is.", 0),
    ("Awful noise, weak suction, and it overheats after ten minutes.", 0),
    ("A horrible sequel that ruins a once-great franchise.", 0),
    ("Lame plot, wooden acting, and dialogue that made me cringe.", 0),
    ("Defective on arrival and the return process was a nightmare.", 0),
    ("Slow, unresponsive, and full of ads - I deleted it immediately.", 0),
    ("The worst hotel stay of my life: noisy, dirty, and overpriced.", 0),
    ("Boring characters and a story that drags without purpose.", 0),
    ("Terrible value - cheap, flimsy, and already falling apart.", 0),
    ("I found it dull and pretentious, with nothing to say.", 0),
    ("Rude driver, filthy car, and a fare twice what was quoted.", 0),
    ("The fabric is awful and it fell apart after two washes.", 0),
    ("Disappointing sequel with lazy writing and zero imagination.", 0),
    ("Horrible battery life and it gets painfully hot to touch.", 0),
    ("A waste of an evening - boring, loud, and overpriced.", 0),
    ("The plot was dreadful and the acting was even worse.", 0),
    ("Faulty sensor, useless app, and support that never replies.", 0),
    ("I regret every dollar spent on this miserable experience.", 0),
    ("Ugly, uncomfortable, and poorly made in every way.", 0),
    ("Slow shipping, damaged box, and missing parts. Awful.", 0),
    ("Boring, predictable, and twice as long as it needed to be.", 0),
]

LABEL_NAMES = {0: "negative", 1: "positive"}


def build_pipeline() -> Pipeline:
    """TF-IDF (unigrams + bigrams) feeding a logistic regression classifier."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=5000,
                    stop_words="english",
                    sublinear_tf=True,
                    min_df=2,  # ignore hapax legomena: with a small corpus they only add noise
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    C=1.0,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def top_features(pipeline: Pipeline, n: int = 10) -> tuple[list[str], list[str]]:
    """Most indicative tokens for each class from the logistic regression weights."""
    vectorizer: TfidfVectorizer = pipeline.named_steps["tfidf"]
    clf: LogisticRegression = pipeline.named_steps["clf"]
    names = vectorizer.get_feature_names_out()
    coefs = clf.coef_[0]
    order = coefs.argsort()
    neg = [names[i] for i in order[:n]][::-1]
    pos = [names[i] for i in order[-n:]][::-1]
    return pos, neg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the sentiment classifier.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", default="model.pkl")
    args = parser.parse_args()

    texts = [t for t, _ in DATA]
    labels = [y for _, y in DATA]
    print(f"Dataset: {len(texts)} examples "
          f"({sum(labels)} positive, {len(labels) - sum(labels)} negative)")

    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=args.test_size, random_state=args.seed, stratify=labels
    )
    print(f"Train: {len(x_train)}  Test: {len(x_test)} (stratified split, seed={args.seed})")

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    train_acc = accuracy_score(y_train, pipeline.predict(x_train))
    test_acc = accuracy_score(y_test, pipeline.predict(x_test))
    print(f"\nTrain accuracy: {train_acc:.3f}")
    print(f"Test  accuracy: {test_acc:.3f}\n")
    print("Classification report (test set):")
    print(classification_report(y_test, pipeline.predict(x_test),
                                target_names=["negative", "positive"]))
    print("Confusion matrix (rows=true, cols=pred) [negative, positive]:")
    print(confusion_matrix(y_test, pipeline.predict(x_test)))

    pos, neg = top_features(pipeline)
    print("\nTop positive indicators:", ", ".join(pos))
    print("Top negative indicators:", ", ".join(neg))

    out_path = os.path.abspath(args.model_out)
    joblib.dump({"pipeline": pipeline, "labels": LABEL_NAMES}, out_path)
    print(f"\nSaved model to {out_path}")


if __name__ == "__main__":
    main()
