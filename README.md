# Intent-Based Chatbot (TF-IDF + Logistic Regression)

A simple command-line chatbot built with Python and scikit-learn. It does not generate text. It detects the **intent** of your message (greeting, thanks, help, ...) and replies with a random ready-made answer for that intent.

## How It Works

```
User message -> TF-IDF (words + characters) -> Logistic Regression -> intent + confidence -> random reply
```

1. The message is converted to numbers with two TF-IDF vectorizers combined:
   - **Word level:** single words and word pairs (`ngram_range=(1, 2)`)
   - **Character level:** 2 to 4 character pieces inside words (`char_wb`), which helps with typos like `helo` or `thanx`
2. A **Logistic Regression** classifier predicts the probability of each intent.
3. The intent with the highest probability wins.
4. If the confidence is below the threshold (`0.25`), the intent is set to `unknown`.
5. A random reply is chosen from that intent's list of responses.

## Intents

| Intent | Example |
|---|---|
| `greeting` | hello, good morning |
| `goodbye` | bye, see you later |
| `thanks` | thank you, thx |
| `help` | can you help me |
| `small_talk` | how are you |
| `identity` | who are you |
| `unknown` | what is the weather today |

The `unknown` class contains questions the bot cannot answer, so the model learns to say "I don't know" instead of forcing a wrong intent.

## Requirements

- Python 3.9+
- pandas, numpy, scikit-learn

```bash
pip install pandas numpy scikit-learn
```

## Usage

```bash
python chatbot.py
```

The script trains the model, prints the evaluation report, then starts the chat:

```
YOU:thanks a lot
BOT :Happy to help!
[Predicted Intent:thanks] | [Predicted Confidence:0.99]
```

Type `quit` or `exit` to stop.

## Training Details

| Setting | Value |
|---|---|
| Dataset | 114 short sentences, 7 intents |
| Split | 80% train / 20% test (`random_state=42`) |
| Classifier | Logistic Regression (`max_iter=2000`) |
| Tuning | `GridSearchCV`, `C` in [1, 10, 100], 5-fold cross-validation |
| Confidence threshold | 0.25 |

Results with this setup: best `C` = 100, cross-validation accuracy ~0.69, test accuracy ~0.87. The test set is only 23 sentences, and the gap between the two numbers shows how much the score depends on which sentences land in the test set, so treat it as a rough guide.

## Project Structure

```
.
├── Project.py          # Data, training, evaluation, and chat loop
└── README.md
```

## Limitations

- The dataset is very small, so the bot works best on sentences similar to the training examples.
- Some intents overlap (for example `hi how are you` is both a greeting and small talk).
- The bot only picks pre-written replies. It has no memory of the conversation.

## Ideas for Improvement

- Add many more examples per intent (50 to 100 each).
- Add new intents (time, jokes, weather) with their own responses.
- Use `stratify=Data['intent']` in `train_test_split` so each intent appears evenly in the test set.
- Try an embedding-based model or a small neural network.
- Save the trained model with `joblib` so it does not retrain on every run.
