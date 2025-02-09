# 📜 Roman Urdu Poetry Generator

Generate poetry in Roman Urdu using an LSTM-based word-level language model.

## ✨ Features

- Generates poetry based on a given seed word.
- Uses an LSTM model trained on a Roman Urdu poetry dataset.
- Provides a simple and user-friendly Streamlit UI.

## 📂 Dataset

The dataset used for training consists of Roman Urdu poetry, stored in `data/Roman-Urdu-Poetry.csv`. It contains poetic verses in Roman Urdu, which serve as input for training the LSTM model.

## 📂 Project Structure
```
Roman-Urdu-Poetry-Generator/
│-- data/
│   ├── Roman-Urdu-Poetry.csv  # Dataset
│-- model/
│   ├── word_rnn_model_updated.pth  # Trained LSTM model
│-- app.py  # Streamlit application
│-- README.md  # Project documentation
```

## 🤝 Contributing

Contributions are welcome! Feel free to submit a pull request or open an issue.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Acknowledgments

- Inspired by classic Urdu poetry and deep learning techniques.
- Built using PyTorch and Streamlit.
- Thanks to all contributors and supporters!

